import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning import LightningModule

from sklearn.metrics import roc_auc_score

from utils import regularize, spread_return_sharpe_from_weights



class MLP(LightningModule):

    def __init__(self, width, mult=2, lr=3e-3, num_of_cat_feats=31):

        super(MLP, self).__init__()

        self.nocf = num_of_cat_feats
        self.lr = lr

        self.batch_norm0 = nn.BatchNorm1d(width-self.nocf)
        self.batch_norm1 = nn.BatchNorm1d(width*mult)
        self.batch_norm2 = nn.BatchNorm1d(width*mult)

        self.dense0 = nn.Linear(width, width*mult)
        self.dense1 = nn.Linear(width*mult, width*mult)
        self.dense2 = nn.Linear(width*mult, width*mult)

        self.dropout0 = torch.nn.Dropout(p=0.35)
        self.dropout1 = torch.nn.Dropout(p=0.4)
        self.dropout2 = torch.nn.Dropout(p=0.45)

        self.linear = nn.Linear(width*mult, 1)

    def forward(self, x):

        x[:,:,:-self.nocf] = self.batch_norm0(x[:,:,:-self.nocf].permute(0,2,1)).permute(0,2,1)
        x = self.dense0(x)
        x = torch.relu(x)
        x = self.dropout0(x)

        x = self.batch_norm1(x.permute(0,2,1)).permute(0,2,1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.batch_norm2(x.permute(0,2,1)).permute(0,2,1)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        return self.linear(x).squeeze(-1)

    def training_step(self, train_batch, batch_idx):

        return self.process_batch(*train_batch, 'train', batch_idx)

    def validation_step(self, val_batch, batch_idx):

        with torch.no_grad():
        
            self.process_batch(*val_batch, 'val', batch_idx)



class ReturnsDeltaClassifier(MLP):

    def __init__(self, width):

        super(ReturnsDeltaClassifier, self).__init__(width)

        self.name = 'returns_classification'

    def process_batch(self, x, y, w, prefix, batch_idx):

        if prefix == 'train':

            x, y, w = regularize(x, y, w,  self.nocf)

            logits = self.forward(x)
        else:

            with torch.no_grad():
                logits = self.forward(x)

        targets = 1.*(y - torch.mean(y, dim=1).unsqueeze(dim=-1) > 0)

        loss = F.binary_cross_entropy_with_logits(logits, targets, w)

        with torch.no_grad():

            cats_alpha = targets.detach().cpu().numpy().flatten()
            probs_alpha = torch.sigmoid(logits).detach().cpu().numpy().flatten()
            momentum_roc_auc_alpha = roc_auc_score(y_true=cats_alpha, y_score=probs_alpha)

            self.log(prefix+'_loss', loss.item())
            self.log(prefix+'_roc_auc', momentum_roc_auc_alpha)

        return loss

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        sccheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=1/3, verbose=True, min_lr=1e-5)
        return [optimizer], {'scheduler': sccheduler, 'monitor': 'val_roc_auc'}




class VolatilityDeltaClassifier(MLP):

    def __init__(self, width):

        super(VolatilityDeltaClassifier, self).__init__(width)

        self.name = 'volatility_classification'

    def process_batch(self, x, y, w, prefix, batch_idx):

        if prefix == 'train':
            logits = self.forward(x)
        else:

            with torch.no_grad():
                logits = self.forward(x)

        targets = 1.*(y - torch.mean(y, dim=1).unsqueeze(dim=-1) > 0)

        loss = F.binary_cross_entropy_with_logits(logits, targets, w)

        with torch.no_grad():

            cats_alpha = targets.detach().cpu().numpy().flatten()
            probs_alpha = torch.sigmoid(logits).detach().cpu().numpy().flatten()
            volatility_roc_auc_alpha = roc_auc_score(y_true=cats_alpha, y_score=probs_alpha)

            self.log(prefix+'_loss', loss.item())
            self.log(prefix+'_roc_auc', volatility_roc_auc_alpha)
 
        return loss

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        sccheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=1/3, verbose=True, min_lr=1e-5)
        return [optimizer], {'scheduler': sccheduler, 'monitor': 'val_roc_auc'}



class PortfolioOptimizer(MLP):

    def __init__(self, width, mult=2):

        super(PortfolioOptimizer, self).__init__(width, mult)

        self.name = 'portfolio_optimization'

    def forward(self, x):

        portfolio_weights = F.softmax(super().forward(x), dim=1)
        
        portfolio_weights += F.softmax(x[:,:,2], dim=1) / 100

        selected_extra_weights = torch.zeros(portfolio_weights.shape, device='cuda', requires_grad=True)

        for i in range(10):

            m = 0.1 if i>0 else 1

            temp = portfolio_weights - torch.quantile(portfolio_weights, 0.9 + i*0.01, dim=1, keepdim=True)
            temp = torch.clip(temp,0,m)
            temp = temp * 1e10
            selected_extra_weights = selected_extra_weights + torch.clip(temp,0,m)

            temp = -portfolio_weights + torch.quantile(portfolio_weights, 0.1 - i*0.01, dim=1, keepdim=True)
            temp = torch.clip(temp,0,m)
            temp = temp * 1e10
            selected_extra_weights = selected_extra_weights - torch.clip(temp,0,m)

        return portfolio_weights + selected_extra_weights

    def process_batch(self, x, y, w, prefix, batch_idx):

        if prefix == 'train':

            # x, y, w = regularize(x, y, w,  self.nocf, mix_days=False)
            weights = self.forward(x)

        else:

            with torch.no_grad():
                weights = self.forward(x)

        daily_spread_return = (y * weights * w).sum(dim=1)
        loss = -torch.mean(daily_spread_return) / (torch.std(daily_spread_return) + 1e-6)

        with torch.no_grad():

            self.log(prefix+'_loss', loss.item())
            # self.log(prefix+'_sharpe_ratio', -loss.item())
            self.log(prefix+'_sharpe_ratio', spread_return_sharpe_from_weights(y, weights)[0])

        return loss

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        sccheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=6, factor=1/3, verbose=True, min_lr=1e-5)
        return [optimizer], {'scheduler': sccheduler, 'monitor': 'val_sharpe_ratio'}

