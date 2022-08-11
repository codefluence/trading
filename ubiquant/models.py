from sklearn.metrics import roc_auc_score
from scipy.stats import beta

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning import LightningModule



class UbiquantMultiTask(LightningModule):

    def __init__(self, input_width):

        super(UbiquantMultiTask, self).__init__()

        hidden1 = int(input_width*1.2)
        hidden2 = int(input_width*0.6)

        self.batch_norm1 = nn.BatchNorm1d(input_width)
        self.batch_norm2 = nn.BatchNorm1d(hidden1)
        self.batch_norm3 = nn.BatchNorm1d(hidden2)

        self.dense1 = nn.Linear(input_width, hidden1)
        self.dense2 = nn.Linear(hidden1, hidden2)
        self.dense3 = nn.Linear(hidden2, hidden2)

        self.dropout1 = torch.nn.Dropout(p=0.5)
        self.dropout2 = torch.nn.Dropout(p=0.5)
        self.dropout3 = torch.nn.Dropout(p=0.5)

        self.linear = nn.Linear(hidden2, 6)

    def forward(self, x):

        x = self.batch_norm1(x)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.batch_norm2(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        x = self.batch_norm3(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dropout3(x)

        return self.linear(x)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=3e-3)
        sccheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=1, factor=1/3, verbose=True, min_lr=1e-3)
        return [optimizer], {'scheduler': sccheduler, 'monitor': 'val_corr_resp'}

    def training_step(self, train_batch, batch_idx):

        x, context, strat_corrs = train_batch

        return self.process_batch(x, context, strat_corrs, 'train', batch_idx)

    def validation_step(self, val_batch, batch_idx):

        x, context, strat_corrs = val_batch

        self.process_batch(x, context, strat_corrs, 'val', batch_idx)

    def process_batch(self, x, context, strat_corrs, prefix, batch_idx):

        targets_slow     = context[:,3]
        targets_slow_std = context[:,4]
        targets_fast     = context[:,5]
        targets_fast_std = context[:,6]

        targets_responses = torch.column_stack((    targets_slow, targets_fast, 
                                                    torch.normal(mean=targets_slow,std=targets_slow_std),
                                                    torch.normal(mean=targets_fast,std=targets_fast_std)))

        targets_corrs = torch.column_stack((strat_corrs,torch.normal(mean=strat_corrs,std=1)))

        time_ids = context[:,0].reshape(-1,1)

        if prefix == 'train':
            x, targets_responses, targets_corrs, time_ids = self.blend(x, targets_responses, targets_corrs, time_ids, ab=0.2)

        targets_slow = targets_responses[:,0].reshape(-1,1)
        targets_fast = targets_responses[:,1].reshape(-1,1)
        targets_resp = targets_slow + targets_fast

        noisy_targets_slow = targets_responses[:,2].reshape(-1,1)
        noisy_targets_fast = targets_responses[:,3].reshape(-1,1)

        targets_strat_corrs = targets_corrs[:,:3]
        noisy_targets_strat_corrs = targets_corrs[:,3:]

        if prefix == 'train':
            logits = self.forward(x)
        else:
            with torch.no_grad():
                logits = self.forward(x)

        estimate_resp = logits[:,0].reshape(-1,1)
        estimate_slow = logits[:,1].reshape(-1,1)
        estimate_fast = logits[:,2].reshape(-1,1)
        estimate_strat_corrs = logits[:,3:]

        noisy_targets_slow = torch.normal(mean=noisy_targets_slow,std=0.1*torch.abs(noisy_targets_slow))
        noisy_targets_fast = torch.normal(mean=noisy_targets_fast,std=0.1*torch.abs(noisy_targets_fast))
        noisy_targets_resp = torch.normal(mean=noisy_targets_slow+noisy_targets_fast,std=0.1*torch.abs(noisy_targets_slow+noisy_targets_fast))
        noisy_targets_strat_corrs = torch.normal(mean=noisy_targets_strat_corrs,std=1*torch.abs(noisy_targets_strat_corrs))

        noisy_stratcorr_0 = 1.*(noisy_targets_strat_corrs[:,0]>0)
        stratcorr_0_ce = F.binary_cross_entropy_with_logits(estimate_strat_corrs[:,0], noisy_stratcorr_0)
        try:
            stratcorr_0 = 1.*(targets_strat_corrs[:,0]>0)
            stratcorr_0_probs = torch.sigmoid(estimate_strat_corrs[:,0]).detach().cpu()
            strat_roc_auc_0 = roc_auc_score(y_true=stratcorr_0.detach().cpu().numpy(), y_score=stratcorr_0_probs.detach().cpu().numpy())
        except Exception as ex:
            strat_roc_auc_0 = 0.5

        noisy_stratcorr_1 = 1.*(noisy_targets_strat_corrs[:,1]>0)
        stratcorr_1_ce = F.binary_cross_entropy_with_logits(estimate_strat_corrs[:,1], noisy_stratcorr_1)
        try:
            stratcorr_1 = 1.*(targets_strat_corrs[:,1]>0)
            stratcorr_1_probs = torch.sigmoid(estimate_strat_corrs[:,1]).detach().cpu()
            strat_roc_auc_1 = roc_auc_score(y_true=stratcorr_1.detach().cpu().numpy(), y_score=stratcorr_1_probs.detach().cpu().numpy())
        except Exception as ex:
            strat_roc_auc_1 = 0.5

        noisy_stratcorr_2 = 1.*(noisy_targets_strat_corrs[:,2]>0)
        stratcorr_2_ce = F.binary_cross_entropy_with_logits(estimate_strat_corrs[:,2], noisy_stratcorr_2)
        try:
            stratcorr_2 = 1.*(targets_strat_corrs[:,2]>0)
            stratcorr_2_probs = torch.sigmoid(estimate_strat_corrs[:,2]).detach().cpu()
            strat_roc_auc_2 = roc_auc_score(y_true=stratcorr_2.detach().cpu().numpy(), y_score=stratcorr_2_probs.detach().cpu().numpy())
        except Exception as ex:
            strat_roc_auc_2 = 0.5

        noisy_corr_slow_loss, best_fit_slow = self.day_metric_mean(estimate_slow, noisy_targets_slow, time_ids, clip_value=0.55, is_loss=True)
        noisy_corr_fast_loss, best_fit_fast = self.day_metric_mean(estimate_fast, noisy_targets_fast, time_ids, clip_value=0.45, is_loss=True)
        noisy_corr_resp_loss, best_fit_resp = self.day_metric_mean(estimate_resp, noisy_targets_resp, time_ids, clip_value=0.3, is_loss=True)

        loss_daycorr = noisy_corr_resp_loss + noisy_corr_slow_loss + noisy_corr_fast_loss
        loss_stratcorr = stratcorr_0_ce + stratcorr_1_ce + stratcorr_2_ce
        loss = loss_daycorr + loss_stratcorr

        self.log(prefix+'_loss', loss)
        self.log(prefix+'_loss_daycorr', loss_daycorr)
        self.log(prefix+'_loss_stratcorr', loss_stratcorr)

        self.log(prefix+'_corr_slow', self.day_metric_mean(estimate_slow, targets_slow, time_ids)[0])
        self.log(prefix+'_corr_fast', self.day_metric_mean(estimate_fast, targets_fast, time_ids)[0])
        self.log(prefix+'_corr_resp', self.day_metric_mean(estimate_resp, targets_resp, time_ids)[0])

        self.log(prefix+'_strat_pc0_roc_auc',strat_roc_auc_0)
        self.log(prefix+'_strat_pc1_roc_auc',strat_roc_auc_1)
        self.log(prefix+'_strat_pc2_roc_auc',strat_roc_auc_2)

        # self.log(prefix+'_best_fit_slow',best_fit_slow)
        # self.log(prefix+'_best_fit_fast',best_fit_fast)
        # self.log(prefix+'_best_fit_resp',best_fit_resp)

        return loss


    def blend(self, x, signals, day_pcs, time_ids, only_from_same_day=False, ab=0.4):

        def blend_aux(feat, resp, dpcs, tids, ab=0.4):

            blended = [], [], [], []

            if len(feat) < 2:
                blended[0].append(feat)
                blended[1].append(resp)
                blended[2].append(dpcs)
                blended[3].append(tids)
            else:

                if len(feat) % 2 > 0:
                    feat = feat[:-1]
                    resp = resp[:-1]
                    dpcs = dpcs[:-1]
                    tids = tids[:-1]

                b = torch.tensor(beta.rvs(ab, ab, size=len(feat)//2), device='cuda', dtype=torch.float32).reshape(-1,1)

                blended[0].append(b * feat[::2] + (1-b) * feat[1::2])
                blended[1].append(b * resp[::2] + (1-b) * resp[1::2])
                blended[2].append(b * dpcs[::2] + (1-b) * dpcs[1::2])
                blended[3].append( torch.where(b > 0.5, tids[::2], tids[1::2]) )

            return torch.vstack(blended[0]), torch.vstack(blended[1]), torch.vstack(blended[2]), torch.vstack(blended[3])

        if only_from_same_day:

            blended = [], [], [], []

            for time_id in torch.unique(time_ids):

                idx = torch.where(time_ids == time_id)[0]

                if len(idx) < 2:
                    blended[0].append(x[idx])
                    blended[1].append(signals[idx])
                    blended[2].append(day_pcs[idx])
                    blended[3].append(time_ids[idx])
                else:
                    
                    day_bends = blend_aux(x[idx], signals[idx], day_pcs[idx], time_ids[idx], ab)
                    blended[0].append(day_bends[0])
                    blended[1].append(day_bends[1])
                    blended[2].append(day_bends[2])
                    blended[3].append(day_bends[3])

            return torch.vstack(blended[0]), torch.vstack(blended[1]), torch.vstack(blended[2]), torch.vstack(blended[3])
        else:

            return blend_aux(x, signals, day_pcs, time_ids, ab)


    def day_metric_mean(self, estimates, targets, day_ids, clip_value=1, is_loss=False, method='pearson'):

        day_metrics = []
        day_weights = []

        for day_id in torch.unique(day_ids):

            idx = day_ids == day_id

            e = estimates[idx]
            t = targets[idx]

            if len(e) < 2:
                continue

            try:
                if is_loss:

                    if method=='pearson':
                        day_metric = -self.pearson_corr(e,t)
                        day_metric = torch.clip(day_metric, -clip_value, 1)
                    elif method=='mse':
                        day_metric = F.mse_loss(e, t)
                    else:
                        print('unknown loss function')
                else:
                    if method=='pearson':
                        day_metric = self.pearson_corr(e,t)
                    elif method=='mse':
                        day_metric = F.l1_loss(e, t)
                    else:
                        print('unknown loss function')
            except Exception as ex:
                print('day_metric_mean error:',ex)
                continue

            day_metrics.append(day_metric)

            length = torch.tensor(len(e), dtype=torch.float32, device=torch.device('cuda'), requires_grad=is_loss)
            day_weights.append(length)

        if len(day_metrics) == 0:

            return torch.tensor(0, device='cuda', requires_grad=is_loss), torch.tensor(0, device='cuda', requires_grad=is_loss)

        day_metrics = torch.stack(day_metrics)
        day_weights = torch.stack(day_weights)

        return torch.sum(day_metrics * day_weights) / torch.sum(day_weights), torch.min(day_metrics)

    def pearson_corr(self, x, y):

        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        return cos(x - x.mean(dim=0, keepdim=True), y - y.mean(dim=0, keepdim=True))

    def pearson_corr_weighted(self, x, y, w):

        raise NotImplementedError()

        vx = x - torch.mean(x)
        vy = y - torch.mean(y)

        num = torch.sum(vx * vy)
        dem = (torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)))

        return num / dem







class PastResponseRegressorDataSet(Dataset):
    
    def __init__(self, features, targets):

        super(PastResponseRegressorDataSet, self).__init__()

        self.features = features
        self.targets = targets

    def __len__(self):

        return len(self.features)

    def __getitem__(self, ndx):

        return self.features[ndx], self.targets[ndx]

class PastResponseRegressorData(pl.LightningDataModule):

    def __init__(self, feats, targets, is_train):

        super(PastResponseRegressorData, self).__init__()

        self.train_ds = PastResponseRegressorDataSet(feats[is_train], targets[is_train])
        self.val_ds   = PastResponseRegressorDataSet(feats[~is_train], targets[~is_train])

    def train_dataloader(self):

        return DataLoader(self.train_ds, batch_size=8192, shuffle=True)

    def val_dataloader(self):

        return DataLoader(self.val_ds, batch_size=8192, shuffle=False)

class PastResponseRegressor(LightningModule):

    def __init__(self, input_width):

        super(PastResponseRegressor, self).__init__()

        hidden1 = input_width//2
        hidden2 = input_width//4

        self.dense1 = nn.Linear(input_width, hidden1)
        self.dense2 = nn.Linear(hidden1, hidden2)

        self.linear = nn.Linear(hidden2, 2)

    def forward(self, x):

        x = torch.relu(self.dense1(x))
        x = torch.relu(self.dense2(x))

        return self.linear(x)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=3e-3)
        sccheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1, factor=1/3, verbose=False)
        return [optimizer], {'scheduler': sccheduler, 'monitor': 'val_mae'}

    def training_step(self, train_batch, batch_idx):

        x, c = train_batch
        l = self.forward(x)

        return F.mse_loss(l[:,0].reshape(-1,1), c[:,0].reshape(-1,1)) + F.mse_loss(l[:,1].reshape(-1,1), c[:,1].reshape(-1,1))

    def validation_step(self, val_batch, batch_idx):

        x, c = val_batch
        l = self.forward(x)
        
        self.log('val_mae',F.l1_loss(l[:,0].reshape(-1,1), c[:,0].reshape(-1,1)) + F.l1_loss(l[:,1].reshape(-1,1), c[:,1].reshape(-1,1)))

