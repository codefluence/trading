import json
import os
import gc
import pandas as pd
from tqdm import tqdm

import numpy as np
import torch

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from data import JPXdataModule
from models import PortfolioOptimizer, ReturnsDeltaClassifier, VolatilityDeltaClassifier
from utils import spread_return_sharpe_from_weights


def get_trainer(name, monitor, mode):

    early_stop_callback = EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=7,
        min_delta=0.0001,
        verbose=True
    )

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        mode=mode,
        dirpath='./weights/',
        filename='jpx_'+name,
        save_top_k=1,
        verbose=True
    )

    trainer = pl.Trainer(   logger=pl_loggers.TensorBoardLogger('./logs/',name=name),
                            gpus=1,
                            max_epochs=30,
                            log_every_n_steps=100,
                            checkpoint_callback=True,
                            callbacks=[early_stop_callback,checkpoint_callback]
                            )

    file_name = './weights/jpx_'+name+'.ckpt'
    if os.path.isfile(file_name):
        os.remove(file_name)

    return trainer, early_stop_callback, checkpoint_callback


def training(mode, fold, model_class, monitor, nocf):

        data = JPXdataModule(mode=mode, fold=fold)
        name = '{}_{}'.format(mode, fold)

        if os.path.exists('./weights/jpx_'+name+'.ckpt'):

            model = model_class.load_from_checkpoint('./weights/jpx_'+name+'.ckpt', width=data.x.shape[-1], num_of_cat_feats=nocf)
        else:

            model = model_class(width=data.x.shape[-1])
            trainer, _, _ = get_trainer(name=name, monitor=monitor, mode='max')
            trainer.fit(model, data)

        model.eval()
        model.cuda()

        outputs = []

        for x, _, _ in tqdm(data.all_dataloader()):

            outputs.append(torch.sigmoid(model(x.cuda())).detach().cpu())

        outputs = torch.cat(outputs, dim=0)

        torch.save(outputs, data.settings['CACHE_DIR'] + 'pred_{}.pt'.format(name))

        del data, model
        gc.collect()
        torch.cuda.empty_cache()


def eval(fold = 'EVAL'):

    mode = 'portfolio_optimization'
    data = JPXdataModule(mode=mode, fold=fold)

    model_por = PortfolioOptimizer.load_from_checkpoint('./weights/jpx_'+'portfolio_optimization_{}.ckpt'.format(fold), width=data.x.shape[-1], num_of_cat_feats=42)
    model_por.eval()
    model_por.cuda()

    targets, weights = [], []

    print('val dataset evaluation...')
    for x,y,_ in tqdm(data.val_dataloader()):

        weights.append(model_por(x.cuda()))
        targets.append(y.cuda())

    sharpe, buf_mean, buf_std = spread_return_sharpe_from_weights(torch.cat(targets,dim=0), torch.cat(weights,dim=0))

    print('num of days: ',len(weights[0]))
    print('sharpe ratio:',np.round(sharpe,2))
    print('sharpe ratio (annualized):',np.round(sharpe*np.sqrt(len(weights[0])),2))
    print('daily spread return mean: ',np.round(buf_mean,2))
    print('daily spread return std:  ',np.round(buf_std,2))


def inference(fold = 'EVAL'):

    print('INFERENCE TIME')

    data = JPXdataModule(mode='inference', fold=fold)

    with open('./settings.json') as f:
        settings = json.load(f)

    prices_csv = pd.read_csv(settings['SUPP_DIR'] + 'stock_prices.csv').iloc[:,1:]
    financials_csv = pd.read_csv(settings['SUPP_DIR'] + 'financials.csv')

    model_ret = ReturnsDeltaClassifier.load_from_checkpoint('./weights/jpx_'+'returns_classification_{}.ckpt'.format(fold), width=117, num_of_cat_feats=31)
    model_ret.eval()
    model_ret.cuda()

    model_vol = VolatilityDeltaClassifier.load_from_checkpoint('./weights/jpx_'+'volatility_classification_{}.ckpt'.format(fold), width=117, num_of_cat_feats=31)
    model_vol.eval()
    model_vol.cuda()

    model_por = PortfolioOptimizer.load_from_checkpoint('./weights/jpx_'+'portfolio_optimization_{}.ckpt'.format(fold), width=117, num_of_cat_feats=42)
    model_por.eval()
    model_por.cuda()

    for day in tqdm(np.unique(prices_csv.Date)):

        day_prices_csv = prices_csv[prices_csv.Date == day]
        day_financials_csv = financials_csv[financials_csv.Date == day]

        sample_prediction = pd.DataFrame(day_prices_csv.SecuritiesCode, columns=['SecuritiesCode'])
        sample_prediction['Date'] = day_prices_csv.Date.iloc[0]
        sample_prediction['Rank'] = 0

        sto, ret, vol, fin = data.process_day_for_inference(day_prices_csv, day_financials_csv)

        input_ret = torch.cat((ret, sto), dim=-1)
        input_vol = torch.cat((vol, sto), dim=-1)

        output_ret = torch.sigmoid(model_ret(input_ret)).unsqueeze(-1)
        output_vol = torch.sigmoid(model_vol(input_vol)).unsqueeze(-1)

        input_por = torch.cat((output_ret - 0.5, 0.5 - output_vol, fin, sto), dim=-1)

        weights = model_por(input_por)

        ranking = np.zeros(len(sample_prediction))

        for i in range(len(sample_prediction)):

            sec_id = np.argwhere(data.unique_secus==sample_prediction.iloc[i].SecuritiesCode)

            if len(sec_id) == 1:

                ranking[i] = weights[0,sec_id.item()].item()

        sample_prediction.Rank = ranking
        sample_prediction.sort_values(by=['Rank'], ascending=False, inplace=True)
        sample_prediction.Rank = np.arange(len(sample_prediction))

        sample_prediction.sort_index(ascending=True, inplace=True)


if __name__ == '__main__':

    # training('returns_classification', 'EVAL', ReturnsDeltaClassifier, 'val_roc_auc', 31)
    # training('volatility_classification', 'EVAL', VolatilityDeltaClassifier, 'val_roc_auc', 31)
    training('portfolio_optimization', 'EVAL', PortfolioOptimizer, 'val_sharpe_ratio', 42)
    #TODO: finetuning with validation data block

    # eval()

    # inference()


