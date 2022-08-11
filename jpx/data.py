import math
import os
import gc
import json
from tqdm import tqdm

import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from utils import torch_nanmean, torch_nanstd, torch_nansum, torch_nanmax, torch_nanmin, transform_num_days
from utils import impute, impute_with_medians, extend, from_date_to_int

import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

np.set_printoptions(threshold=2000, linewidth=140, precision=5, edgeitems=20, suppress=1)
pd.set_option('display.max_rows', 600)



class JPXdataModule(pl.LightningDataModule):

    def __init__(self, mode=None, fold='EVAL', secondary_data=False, settings=None):

        super(JPXdataModule, self).__init__()

        self.mode = mode
        self.fold = fold

        print('loading settings...')
        if settings is None:
            with open('./settings.json') as f:
                self.settings = json.load(f)
        else:
            self.settings = settings

        self.create_channel_files(secondary=secondary_data)
        torch.cuda.empty_cache()
        gc.collect()

        if mode is None:
            return

        if mode == 'inference':

            self.sto = torch.load(self.settings['CACHE_DIR'] + 'data_stocks.pt')
            self.last_date = np.max(np.unique(self.sto[:,:,0].flatten().detach().numpy())[:-1])
            self.unique_secus = np.unique(self.sto[:,:,1].flatten().detach().numpy())[:-1]
            self.sto = self.sto[-1].unsqueeze(0).cuda()

            self.pri = torch.load(self.settings['CACHE_DIR'] + 'last_prices.pt').cuda()
            self.upd = torch.load(self.settings['CACHE_DIR'] + 'last_updates.pt')[-1].unsqueeze(0).cuda()

            self.prepro_means_por = torch.load(self.settings['CACHE_DIR'] + 'prepro_means_por_{}.pt'.format(fold)).cuda()
            self.prepro_means_ret = torch.load(self.settings['CACHE_DIR'] + 'prepro_means_ret_{}.pt'.format(fold)).cuda()
            self.prepro_means_vol = torch.load(self.settings['CACHE_DIR'] + 'prepro_means_vol_{}.pt'.format(fold)).cuda()

            self.prepro_stds_por  = torch.load(self.settings['CACHE_DIR'] + 'prepro_stds_por_{}.pt'.format(fold)).cuda()
            self.prepro_stds_ret  = torch.load(self.settings['CACHE_DIR'] + 'prepro_stds_ret_{}.pt'.format(fold)).cuda()
            self.prepro_stds_vol  = torch.load(self.settings['CACHE_DIR'] + 'prepro_stds_vol_{}.pt'.format(fold)).cuda()

        else:

            sto = torch.load(self.settings['CACHE_DIR'] + 'data_stocks.pt').cuda()
            tar = torch.load(self.settings['CACHE_DIR'] + 'data_targets.pt').cuda()

            is_active = sto[:,:,-1]

            if fold == 'EVAL':

                lenght = len(sto)
                block_size = 126

                is_train = np.ones(lenght)
                is_train[:(lenght % block_size)] = 0
                is_train[-block_size:] = 0
                is_train = is_train==1

                is_valid = np.zeros(lenght)
                is_valid[-block_size:] = 1
                is_valid = is_valid==1

            elif fold.startswith('CV'):

                jump  = int(fold[-2])
                shift = int(fold[-1])

                is_train = (np.arange(len(tar)) + shift) % jump != 0
                is_valid = ~is_train

            else:

                raise Exception('Unknown split method')

            if mode == 'returns_classification':

                self.batch_size = 3

                ret = torch.load(self.settings['CACHE_DIR'] + 'data_returns.pt').cuda()
                volatility = torch.load(self.settings['CACHE_DIR'] + 'data_volatility.pt').cuda()[:,:,3]

                self.x  = torch.cat(self.pre_process(ret, sto, is_train, 'ret'), dim=-1)
                self.y  = tar[:,:,0]
                self.w  = is_active * torch.clip(0.025 - torch.abs(volatility - 0.025), 0)

            elif mode == 'volatility_classification':

                self.batch_size = 3

                vol = torch.load(self.settings['CACHE_DIR'] + 'data_volatility.pt').cuda()
                self.x  = torch.cat(self.pre_process(vol, sto, is_train, 'vol'), dim=-1)
                self.y  = tar[:,:,1]
                self.w  = is_active

            elif mode == 'portfolio_optimization':

                self.batch_size = 126

                fin = torch.load(self.settings['CACHE_DIR'] + 'data_finances.pt').cuda()
                ret = torch.load(self.settings['CACHE_DIR'] + 'data_returns.pt').cuda()
                vol = torch.load(self.settings['CACHE_DIR'] + 'data_volatility.pt').cuda()
                ret_probs = torch.load(self.settings['CACHE_DIR'] + 'pred_returns_classification_{}.pt'.format(fold)).cuda().unsqueeze(-1)
                vol_probs = torch.load(self.settings['CACHE_DIR'] + 'pred_volatility_classification_{}.pt'.format(fold)).cuda().unsqueeze(-1)

                # fin tensor features:
                # quaterly_diff, annual_diff, forecasts_diff, forecasts_ann, eq_to_asset,  
                # days_since_last_update, days_since_last_annual_update, days_since_last_forecast, 
                # profit_type, market_cap, is_year_update, is_forecast, update_types

                fin[:,:,-8:-5] = transform_num_days(fin[:,:,-8:-5])  # days_since_last_update
                fin[:,:,-4] = 1. *(fin[:,:,-4] < 8e9)  # is_illiquid

                fin = torch.cat((   ret[:,:,:8], 
                                    vol[:,:,:8], 
                                    fin),
                                dim=-1)

                feats, feats_common = self.pre_process(fin[:,:,:-5], sto, is_train, 'por')

                update_types = F.one_hot(fin[:,:,-1].long())
                profit_types = F.one_hot(fin[:,:,-5].long())
                binary_feats = fin[:,:,-4:-1]

                vol_weights = - torch.clip(vol[:,:,3],0,0.05).unsqueeze(-1)

                self.x  = torch.cat((   ret_probs - 0.5, 0.5 - vol_probs, vol_weights,
                                        feats, profit_types, binary_feats, update_types, feats_common), dim=-1)
                self.y  = tar[:,:,0]
                self.w  = is_active

            else:

                raise Exception('Unknown training_mode')

            self.x  = self.x.detach().cpu().numpy()
            self.y  = self.y.detach().cpu().numpy()

            gc.collect()
            torch.cuda.empty_cache()

            print('train len:',len(self.x[is_train]))
            print('val len:',len(self.x[~is_train]))

            self.train_ds = JpxDataSet(self.x[is_train], self.y[is_train], self.w[is_train])
            self.val_ds   = JpxDataSet(self.x[is_valid], self.y[is_valid], self.w[is_valid])
            self.all_ds   = JpxDataSet(self.x, self.y, self.w)

    def pre_process(self, feats, sto, is_train, cache_label):

        feats = torch.clip(feats, -5000, 5000)

        feats = torch.nan_to_num(feats)
        sto = torch.nan_to_num(sto)

        day_means = torch.mean(feats, dim=1).unsqueeze(1)
        day_means = torch.repeat_interleave(day_means, feats.shape[1], dim=1)
        feats = torch.cat((feats - day_means, day_means), dim=-1)

        stds, means = torch.std_mean(feats[is_train], dim=(0,1))
        feats = (feats - means) / (stds + 1e-6)

        torch.save(means.detach().cpu(),  self.settings['CACHE_DIR'] + 'prepro_means_{}_{}.pt'.format(cache_label,self.fold))
        torch.save(stds.detach().cpu(), self.settings['CACHE_DIR'] + 'prepro_stds_{}_{}.pt'.format(cache_label,self.fold))

        exchange_segment_id = F.one_hot(sto[:,:,3].long() - 1)
        sector_id           = F.one_hot(sto[:,:,4].long() - 1)
        index_id            = F.one_hot(sto[:,:,5].long())

        sto = torch.cat((   exchange_segment_id,
                            sector_id,
                            index_id,
                            sto[:,:,-3:]
                        ), dim=-1)

        return feats, sto

    def pre_process_for_inference_feat(self, feats, cache_label):

        feats = torch.clip(feats, -5000, 5000)
        feats = torch.nan_to_num(feats)

        day_means = torch.mean(feats, dim=1).unsqueeze(1)
        day_means = torch.repeat_interleave(day_means, feats.shape[1], dim=1)
        feats = torch.cat((feats - day_means, day_means), dim=-1)

        if cache_label == 'ret':
            feats = (feats - self.prepro_means_ret) / (self.prepro_stds_ret + 1e-6)
        elif cache_label == 'vol':
            feats = (feats - self.prepro_means_vol) / (self.prepro_stds_vol + 1e-6)
        elif cache_label == 'por':
            feats = (feats - self.prepro_means_por) / (self.prepro_stds_por + 1e-6)
        else:
            raise Exception()

        return feats

    def pre_process_for_inference_sto(self, sto):

        sto = torch.nan_to_num(sto)

        exchange_segment_id = F.one_hot(sto[:,:,3].long() - 1, num_classes=3)
        sector_id           = F.one_hot(sto[:,:,4].long() - 1, num_classes=17)
        index_id            = F.one_hot(sto[:,:,5].long(), num_classes=8)

        sto = torch.cat((   exchange_segment_id,
                            sector_id,
                            index_id,
                            sto[:,:,-3:]
                        ), dim=-1)

        return sto

    def process_day_for_inference(self, prices_csv, financials_csv):

        prices_csv = prices_csv[[   'Date', 'SecuritiesCode', 'Open', 'High', 'Low', 'Close', 'Volume', 
                                    'AdjustmentFactor', 'ExpectedDividend', 'SupervisionFlag']]

        prices_csv.Date = from_date_to_int(prices_csv.Date)
        prices_csv = prices_csv.to_numpy(np.float32)

        # series tensor dims: time, security, channel
        new_date = prices_csv[-1,0].item()

        # this is just for kaggle public LB which uses old validatin data
        if new_date <= self.last_date:
            self.last_date = new_date - 1

        d0 = int(new_date - self.last_date)
        d1 = len(self.unique_secus)

        self.last_date = new_date

        ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

        self.sto[:,:,0] = new_date
        self.sto[:,:,-3:] = 0  # expected_dividend, under_supervision, is_active

        tensor_pri = np.empty((d0, d1, 5), dtype=np.float32);  tensor_pri[:] = np.nan

        for sec_id in range(len(self.unique_secus)):

            sec = self.unique_secus[sec_id]
            sec_rows_prices = prices_csv[prices_csv[:,1] == sec]

            adj_factor = self.sto[-1,sec_id,2].item()

            if len(sec_rows_prices) > 0:

                tensor_pri[-1,sec_id,:] = sec_rows_prices[:,2:7]
                tensor_pri[-1,sec_id,:4] /= adj_factor
                tensor_pri[-1,sec_id,4] *= adj_factor

                #!!! keep this code block AFTER adj_factor is applied to prices and volume
                self.sto[:,sec_id,2] *= np.nan_to_num(sec_rows_prices[:,-3], nan=1).item()  # adj_factor   
                self.sto[:,sec_id,-3] = 1.*(sec_rows_prices[:,-2] > 0).item()  # expected_dividend
                self.sto[:,sec_id,-2] = 1.*(sec_rows_prices[:,-1] ==1).item()  # under_supervision
                self.sto[:,sec_id,-1] = 1  # is_active
            else:
                tensor_pri[-1,sec_id,:] = np.nan
                self.sto[:,sec_id,2] *= 1   
                self.sto[:,sec_id,-3:] = 0

        self.pri = torch.cat((self.pri, torch.from_numpy(tensor_pri).cuda()), dim=0)[-112:]

        ret, vol = self.get_ta_channels(self.pri)
        ret = ret[-1].unsqueeze(0)
        vol = vol[-1].unsqueeze(0)

        ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

        # financials_csv = financials_csv[[   'Date', 'SecuritiesCode',
        #                                     'AverageNumberOfShares','EquityToAssetRatio','UpdateType',
        #                                     'NetSales','OperatingProfit','OrdinaryProfit','EarningsPerShare',
        #                                     'ForecastNetSales','ForecastOperatingProfit','ForecastOrdinaryProfit','ForecastEarningsPerShare']]

        financials_csv = self.process_financials_csv(financials_csv)
        financials_csv = financials_csv.to_numpy(np.float32)
        financials_csv = torch.from_numpy(financials_csv).cuda()

        # statements_last, forecasts_last, quaterly_updates, eq_to_asset,  
        # days_since_last_update, days_since_last_annual_update, days_since_last_forecast, 
        # profit_type, market_cap, is_year_update, is_forecast, update_types

        tensor_upd = torch.empty((d0, d1, 33), dtype=torch.float32, device=self.upd.device);  tensor_upd[:] = np.nan

        # incrementar day counters;  resettear update flags
        tensor_upd[:] = self.upd[-1].unsqueeze(0)
        tensor_upd[-1,:,-8:-5] += d0  # day counters
        tensor_upd[-1,:,-5] = 0  # profit_type
        tensor_upd[-1,:,-3:] = 0  # is_year_update, is_forecast, update_types

        close = self.pri[-1,:,3]

        # calcular diferencia entre self.fin y financials; copiar diferencia y num_days = 0;  actualizar self.fin
        for i in range(len(financials_csv)):

            update = financials_csv[i]
            sec = update[1].item()

            if not sec in self.unique_secus:
                continue

            sec_id = np.argwhere(self.unique_secus==sec).item()
            num_of_shares = float(update[2])
            eq_to_asset   = float(update[3])
            update_type   = float(update[4])
            profit        = float(update[7])
            market_cap    = num_of_shares * close[sec_id]
            is_forecast   = 0

            if math.isnan(eq_to_asset):
                eq_to_asset = tensor_upd[-1,sec_id,-9]

            if market_cap.isnan().item():
                market_cap = tensor_upd[-1,sec_id,-4]

            if (~update[9:13].isnan()).all():

                forecasts = update[9:13]
                forecasts[:3] /= market_cap
                forecasts[-1] /= close[sec_id]

                tensor_upd[-1,sec_id,-6] = 0
                tensor_upd[-1,sec_id,4:8] = forecasts

                is_forecast = 1

            if update_type > 0:

                statements = update[5:9]
                statements[:3] /= market_cap
                statements[-1] /= close[sec_id]

                profit_type = 0
                if profit > 0 and statements[2] < 0: profit_type = 1
                if profit < 0 and statements[2] > 0: profit_type = 2

                tensor_upd[-1,sec_id,-5] = profit_type
                tensor_upd[-1,sec_id,-8] = 0

                previous_values = tensor_upd[-1,sec_id,:24].clone()
                tensor_upd[-1,sec_id,:4] = statements

                if update_type == 1:
                    tensor_upd[-1,sec_id,8:12] = statements
                elif update_type == 2:
                    tensor_upd[-1,sec_id,12:16] = statements
                elif update_type == 3:
                    tensor_upd[-1,sec_id,16:20] = statements
                elif update_type == 4:
                    tensor_upd[-1,sec_id,20:24] = statements
                    tensor_upd[-1,sec_id,-7] = 0

                tensor_upd[-1,sec_id,:24] = torch.nan_to_num(tensor_upd[-1,sec_id,:24]) + torch.isnan(tensor_upd[-1,sec_id,:24]) * previous_values

            tensor_upd[-1,sec_id,-9] = eq_to_asset
            tensor_upd[-1,sec_id,-4] = market_cap
            tensor_upd[-1,sec_id,-3] = 1. * (update_type==4)
            tensor_upd[-1,sec_id,-2] = is_forecast
            tensor_upd[-1,sec_id,-1] = update_type

        self.upd = tensor_upd[-1].unsqueeze(0)
        fin = self.get_fi_channels(tensor_upd)[-1].unsqueeze(0)

        fin[:,:,-8:-5] = transform_num_days(fin[:,:,-8:-5])  # days_since_last_update
        fin[:,:,-4] = 1. *(fin[:,:,-4] < 8e9)  # is_illiquid

        update_types = F.one_hot(fin[:,:,-1].long(), num_classes=5)
        profit_types = F.one_hot(fin[:,:,-5].long(), num_classes=3)
        binary_feats = fin[:,:,-4:-1]

        fin = torch.cat((   ret[:,:,:8],
                            vol[:,:,:8],
                            fin),
                        dim=-1)

        fin = self.pre_process_for_inference_feat(fin[:,:,:-5], 'por')

        vol_weights = - torch.clip(vol[:,:,3],0,0.05).unsqueeze(-1)
        fin = torch.cat((vol_weights, fin, profit_types, binary_feats, update_types), dim=-1)

        sto = self.pre_process_for_inference_sto(self.sto)
        ret = self.pre_process_for_inference_feat(ret, 'ret')
        vol = self.pre_process_for_inference_feat(vol, 'vol')

        return sto, ret, vol, fin

    def process_financials_csv(self, financials_csv):

        financials_csv = financials_csv[~financials_csv.TypeOfDocument.isna()]

        financials_csv['UpdateType'] = -1
        financials_csv.loc[financials_csv.TypeOfCurrentPeriod.str.startswith('1Q'), 'UpdateType'] = 1
        financials_csv.loc[financials_csv.TypeOfCurrentPeriod.str.startswith('2Q'), 'UpdateType'] = 2
        financials_csv.loc[financials_csv.TypeOfCurrentPeriod.str.startswith('3Q'), 'UpdateType'] = 3
        financials_csv.loc[financials_csv.TypeOfCurrentPeriod.str.startswith('FY'), 'UpdateType'] = 4
        financials_csv.loc[financials_csv.TypeOfDocument.str.startswith('Forecast'), 'UpdateType'] = 0
        financials_csv = financials_csv[financials_csv.UpdateType > -1]

        financials_csv = financials_csv[[   'Date', 'SecuritiesCode',
                                            'AverageNumberOfShares','EquityToAssetRatio','UpdateType',
                                            'NetSales','OperatingProfit','OrdinaryProfit','EarningsPerShare',
                                            'ForecastNetSales','ForecastOperatingProfit','ForecastOrdinaryProfit','ForecastEarningsPerShare']]
        financials_csv.Date = from_date_to_int(financials_csv.Date)
        financials_csv = financials_csv.replace('－',np.nan)

        return financials_csv

    def get_stock_info(self, sec):

        if not hasattr(self, 'stock_dico'):

            self.stock_dico = pd.read_csv(self.settings['ROOT_DIR'] + 'stock_list.csv')
            self.stock_dico = self.stock_dico.set_index('SecuritiesCode')

        try:
            segment_id = self.stock_dico.loc[sec]['NewMarketSegment']

            if segment_id.startswith('Prime'): segment_id = 1.
            elif segment_id.startswith('Standard'): segment_id = 2.
            elif segment_id.startswith('Growth'): segment_id = 3.
            else: segment_id = 0.
        except:
            segment_id = 0.

        try:
            # if self.use33SectorCode:
            #     sector_id = float(self.stock_dico.loc[sec]['33SectorCode'])
            # else:
            sector_id = float(self.stock_dico.loc[sec]['17SectorCode'])
        except:
            sector_id = 0.

        try:
            index_id = float(self.stock_dico.loc[sec]['NewIndexSeriesSizeCode'])
        except:
            index_id = 0.

        return segment_id, sector_id, index_id

    def create_channel_files(self, secondary):

        filename = 'data_stocks_sec' if secondary else 'data_stocks'
        cached_stock_info_path = self.settings['CACHE_DIR'] + filename + '.pt'
        files_already_created = os.path.exists(cached_stock_info_path)

        filename = 'data_targets_sec' if secondary else 'data_targets'
        cached_targets_path = self.settings['CACHE_DIR'] + filename + '.pt'
        files_already_created &= os.path.exists(cached_targets_path)

        filename = 'data_returns_sec' if secondary else 'data_returns'
        cached_returns_path = self.settings['CACHE_DIR'] + filename + '.pt'
        files_already_created &= os.path.exists(cached_returns_path)

        filename = 'data_volatility_sec' if secondary else 'data_volatility'
        cached_volatility_path = self.settings['CACHE_DIR'] + filename + '.pt'
        files_already_created &= os.path.exists(cached_volatility_path)

        filename = 'data_finances_sec' if secondary else 'data_finances'
        cached_finances_path = self.settings['CACHE_DIR'] + filename + '.pt'
        files_already_created &= os.path.exists(cached_finances_path)

        filename = 'last_prices_sec' if secondary else 'last_prices'
        cached_prices_path = self.settings['CACHE_DIR'] + filename + '.pt'
        files_already_created &= os.path.exists(cached_prices_path)

        filename = 'last_updates_sec' if secondary else 'last_updates'
        cached_updates_path = self.settings['CACHE_DIR'] + filename + '.pt'
        files_already_created &= os.path.exists(cached_updates_path)

        if files_already_created:
            return

        print('reading prices csv file...')
        filename = 'secondary_stock_prices' if secondary else 'stock_prices'
        stock_prices_csv = pd.concat((  pd.read_csv(self.settings['DATA_DIR'] + filename + '.csv').iloc[:,1:],
                                        # pd.read_csv(self.settings['SUPP_DIR'] + filename + '.csv').iloc[:,1:] 
                                    ))

        stock_prices_csv.Date = from_date_to_int(stock_prices_csv.Date)
        stock_prices_csv = stock_prices_csv.to_numpy(np.float32)

        print('reading financials csv file...')
        financials_csv = pd.concat((pd.read_csv(self.settings['DATA_DIR'] + 'financials.csv', low_memory=False),
                                    # pd.read_csv(self.settings['SUPP_DIR'] + 'financials.csv', low_memory=False),
                                    ))

        financials_csv = self.process_financials_csv(financials_csv)
        financials_csv = financials_csv.to_numpy(np.float32)

        unique_dates = np.unique(stock_prices_csv[:,0])
        unique_secus = np.unique(stock_prices_csv[:,1])
        min_date = min(unique_dates)

        # series tensor dims: day, security, channel
        d0 = int(max(unique_dates) - min(unique_dates)) + 1
        d1 = len(unique_secus)

        tensor_sto = np.empty((d0, d1, 9),  dtype=np.float32);  tensor_sto[:] = np.nan
        tensor_pri = np.empty((d0, d1, 5),  dtype=np.float32);  tensor_pri[:] = np.nan
        tensor_upd = np.empty((d0, d1, 11), dtype=np.float32);  tensor_upd[:] = np.nan
        tensor_tar = np.empty((d0, d1, 2),  dtype=np.float32);  tensor_tar[:] = np.nan

        print('creating tensors from csv files...')
        for sec_idx in tqdm(range(len(unique_secus))):

            sec = unique_secus[sec_idx].item()

            sec_rows_prices = stock_prices_csv[stock_prices_csv[:,1] == sec]
            sec_rows_financials = financials_csv[financials_csv[:,1] == sec]

            days_prices = list(map(int,sec_rows_prices[:,0] - min_date))
            days_financials = list(map(int,sec_rows_financials[:,0] - min_date))

            segment_id, sector_id, index_id = self.get_stock_info(sec)

            tensor_sto[days_prices,sec_idx,:2] = sec_rows_prices[:,:2]  # date, security_id
            tensor_sto[days_prices,sec_idx,2]  = sec_rows_prices[:,7]  # adjustment_factor
            tensor_sto[:,sec_idx,3] = segment_id
            tensor_sto[:,sec_idx,4] = sector_id
            tensor_sto[:,sec_idx,5] = index_id
            tensor_sto[days_prices,sec_idx,6] = 1.*(sec_rows_prices[:,-3] > 0)  # expected_dividend
            tensor_sto[days_prices,sec_idx,7] = sec_rows_prices[:,-2]  # under_supervision
            tensor_sto[days_prices,sec_idx,8] = 1  # is_active

            tensor_pri[days_prices,sec_idx,:] = sec_rows_prices[:,2:7]  # prices, volume
            tensor_tar[days_prices,sec_idx,0] = sec_rows_prices[:,-1]  # target
            tensor_upd[days_financials,sec_idx,:] = sec_rows_financials[:,2:]  # num_of_shares, eq_to_asset, type of update, statement, forecasts

        # filling nans with 0s for expected_dividend, under_supervision and is_active
        tensor_sto[:,:,-3:] = np.nan_to_num(tensor_sto[:,:,-3:])
        # filling a few targets with missing values with 0
        tensor_tar = np.nan_to_num(tensor_tar)

        print('adjusting prices after share splits...')
        tensor_sto[:,:,2] = np.nan_to_num(tensor_sto[:,:,2], nan=1)
        tensor_sto[:,:,2] = np.cumprod(tensor_sto[:,:,2], axis=0)
        adj_factor = np.expand_dims(tensor_sto[:-1,:,2], axis=-1)
        tensor_pri[1:,:,:4] /= adj_factor  # prices
        tensor_pri[1:,:,4:5] *= adj_factor  # volume

        print('imputing prices and creating rolling channels...')
        tensor_pri = torch.from_numpy(tensor_pri).cuda()
        tensor_ret, tensor_vol = self.get_ta_channels(tensor_pri)
        torch.cuda.empty_cache()

        close = tensor_pri[:,:,3:4]
        tensor_upd = torch.from_numpy(tensor_upd).cuda()
        tensor_upd = self.get_report_updates(tensor_upd, close)
        print('processing updates and forecasts diffs...')
        tensor_fin = self.get_fi_channels(tensor_upd)
        torch.cuda.empty_cache()

        tensor_sto = torch.from_numpy(tensor_sto).cuda()
        tensor_tar = torch.from_numpy(tensor_tar).cuda()

        is_active = tensor_sto[:,:,-1]
        is_bank_holiday = (is_active==0).all(dim=-1)
        tensor_sto = tensor_sto[~is_bank_holiday]
        tensor_ret = tensor_ret[~is_bank_holiday]
        tensor_vol = tensor_vol[~is_bank_holiday]
        tensor_fin = tensor_fin[~is_bank_holiday]
        tensor_tar = tensor_tar[~is_bank_holiday]
        torch.cuda.empty_cache()

        missing_targets = tensor_tar[:,:,0].sum(dim=-1)==0
        tensor_sto = tensor_sto[~missing_targets]
        tensor_ret = tensor_ret[~missing_targets]
        tensor_vol = tensor_vol[~missing_targets]
        tensor_fin = tensor_fin[~missing_targets]
        tensor_tar = tensor_tar[~missing_targets]

        returns_windows = tensor_ret[:,:,0].unfold(0,5,1)
        return_stds = torch_nanstd(returns_windows, dim=-1).unsqueeze(-1)
        return_stds = extend(return_stds, len(tensor_ret))
        return_stds_future = torch.roll(return_stds,-4,0)
        return_stds_future[-4:] = np.nan
        return_stds_future[-5:] = impute(return_stds_future[-5:], backpass=False)
        tensor_tar[:,:,1] = return_stds_future.squeeze()

        print('saving tensor files...')
        torch.save(tensor_sto.detach().cpu(), cached_stock_info_path)
        torch.save(tensor_pri[-112:].detach().cpu(), cached_prices_path)
        torch.save(tensor_upd[-1].unsqueeze(0).detach().cpu(), cached_updates_path)
        torch.save(tensor_tar.detach().cpu(), cached_targets_path)
        torch.save(tensor_ret.detach().cpu(), cached_returns_path)
        torch.save(tensor_vol.detach().cpu(), cached_volatility_path)
        torch.save(tensor_fin.detach().cpu(), cached_finances_path)

    def get_report_updates(self, tensor_upd, close):

        # num_of_shares, eq_to_asset, type of update, statement (4), forecast (4)

        print('processing num_of_shares, eq_to_asset, market_cap...')
        tensor_upd[:,:,:2] = impute(tensor_upd[:,:,:2])
        num_of_shares = tensor_upd[:,:,0:1]
        num_of_shares = torch.nan_to_num(num_of_shares, nan=torch.nanmedian(num_of_shares))
        eq_to_asset = tensor_upd[:,:,1:2]
        eq_to_asset = torch.nan_to_num(eq_to_asset, nan=torch.nanmedian(eq_to_asset))
        market_cap = num_of_shares * impute(close)
        market_cap = torch.nan_to_num(market_cap, nan=torch.nanmedian(market_cap))

        print('processing profits...')
        statements = tensor_upd[:,:,3:7]
        statements[:,:,:3]  /= market_cap
        statements[:,:,3:4] /= close
        statements_last = impute(statements)
        for i in range(4):
            statements_last[:,:,i] = impute_with_medians(statements_last[:,:,i])
        profit = statements_last[:,:,2:3]

        update_types = tensor_upd[:,:,2:3]
        is_forecast = 1. * ~update_types.isnan()
        update_types = torch.nan_to_num(update_types)
        is_year_update = 1. * (update_types == 4)

        quaterly_updates = []

        print('processing quaterly updates...')
        for update_type in range(1,5):

            quaterly_upd = torch.empty_like(update_types);  quaterly_upd[:] = np.nan
            quaterly_upd[update_types==update_type] = 1.
            quaterly_upd = quaterly_upd * statements

            quaterly_upd = impute(quaterly_upd)

            for i in range(4):
                quaterly_upd[:,:,i] = impute_with_medians(quaterly_upd[:,:,i]) 

            quaterly_updates.append(quaterly_upd)

        quaterly_updates = torch.cat(quaterly_updates, dim=-1)

        print('processing forecasts...')
        forecasts_last = tensor_upd[:,:,7:]
        forecasts_last[:,:,:3]  /= market_cap
        forecasts_last[:,:,3:4] /= close
        forecasts_last = impute(forecasts_last)
        # Some securities don't provide forecast, we use the median of the market last forecast:
        for i in range(4):
            forecasts_last[:,:,i] = impute_with_medians(forecasts_last[:,:,i])

        days_since_last_update = torch.empty_like(update_types);  days_since_last_update[:] = np.nan
        days_since_last_annual_update = torch.empty_like(update_types);  days_since_last_annual_update[:] = np.nan
        days_since_last_forecast = torch.empty_like(update_types);  days_since_last_forecast[:] = np.nan
        profit_type = torch.zeros_like(update_types)

        print('processing number of days since last update and profit types...')
        for day in range(len(update_types)):

            if day > 0:

                days_since_last_update[day] = days_since_last_update[day-1] + 1
                days_since_last_annual_update[day] = days_since_last_annual_update[day-1] + 1
                days_since_last_forecast[day] = days_since_last_forecast[day-1] + 1

                break_even  = (profit[day-1] < 0) & (profit[day] > 0)
                going_south = (profit[day-1] > 0) & (profit[day] < 0)
                profit_type[day] = torch.where(break_even,  torch.ones_like(profit_type[day]), torch.where(going_south, 2., 0.))

            days_since_last_update[day, update_types[day]>0] = 0
            days_since_last_annual_update[day, is_year_update[day]==1] = 0
            days_since_last_forecast[day, is_forecast[day]==1] = 0

        days_since_last_update = impute_with_medians(days_since_last_update)
        days_since_last_annual_update = impute_with_medians(days_since_last_annual_update)
        days_since_last_forecast = impute_with_medians(days_since_last_forecast)

        return torch.cat((  statements_last, forecasts_last, quaterly_updates, eq_to_asset,  
                            days_since_last_update, days_since_last_annual_update, days_since_last_forecast, 
                            profit_type, market_cap, is_year_update, is_forecast, update_types), dim=-1)

    def get_fi_channels(self, tensor_upd):

        updates = tensor_upd[:,:,4:]

        quaterly_upd = updates[:,:,4:20]
        annual_upd   = updates[:,:,16:20].clone()
        forecasts    = updates[:,:,:4]

        update_types = updates[:,:,-1].unsqueeze(-1)
        is_forecast  = updates[:,:,-2].unsqueeze(-1)

        last_upd_type = torch.empty_like(update_types);  last_upd_type[:] = np.nan
        last_upd_type[update_types>0] = update_types[update_types>0]
        last_upd_type = impute(last_upd_type)

        quaterly_diff = torch.diff(quaterly_upd, dim=0)
        quaterly_diff = extend(quaterly_diff, len(updates), blank=0)

        quaterly_diff[:,:,:4]   = quaterly_diff[:,:,:4]   * torch.where(update_types==1., 1., np.nan)
        quaterly_diff[:,:,4:8]  = quaterly_diff[:,:,4:8]  * torch.where(update_types==2., 1., np.nan)
        quaterly_diff[:,:,8:12] = quaterly_diff[:,:,8:12] * torch.where(update_types==3., 1., np.nan)
        quaterly_diff[:,:,12:]  = quaterly_diff[:,:,12:]  * torch.where(update_types==4., 1., np.nan)

        quaterly_diff = torch.nan_to_num(impute(quaterly_diff))

        quaterly_diff[:,:,12:] -= quaterly_diff[:,:,8:12]
        quaterly_diff[:,:,8:12] -= quaterly_diff[:,:,4:8]
        quaterly_diff[:,:,4:8] -= quaterly_diff[:,:,:4]

        quaterly_diff = quaterly_diff[:,:,:4] * (last_upd_type==1) + \
                        quaterly_diff[:,:,4:8] * (last_upd_type==2) + \
                        quaterly_diff[:,:,8:12] * (last_upd_type==3) + \
                        quaterly_diff[:,:,12:] * (last_upd_type==4)

        annual_diff = torch.diff(annual_upd, dim=0)
        annual_diff = extend(annual_diff, len(updates))
        annual_diff *= torch.where(update_types==4., 1., np.nan)
        annual_diff = torch.nan_to_num(impute(annual_diff))

        forecasts_diff = torch.diff(forecasts, dim=0)
        forecasts_diff = extend(forecasts_diff, len(updates))
        forecasts_diff *= torch.where(is_forecast==1., 1., np.nan)
        forecasts_diff = torch.nan_to_num(impute(forecasts_diff))

        forecasts_ann = torch.nan_to_num(forecasts - annual_upd)

        return torch.cat((quaterly_diff, annual_diff, forecasts_diff, forecasts_ann, updates[:,:,20:]), dim=-1)

    def get_ta_channels(self, tensor_pri):

        num_days = len(tensor_pri)

        tensor_ret = torch.empty((num_days, tensor_pri.shape[1], 43),  dtype=torch.float32, device=tensor_pri.device);  tensor_ret[:] = np.nan
        tensor_vol = torch.empty((num_days, tensor_pri.shape[1], 43),  dtype=torch.float32, device=tensor_pri.device);  tensor_vol[:] = np.nan

        open   = tensor_pri[:,:,0:1]
        high   = tensor_pri[:,:,1:2]
        low    = tensor_pri[:,:,2:3]
        close  = tensor_pri[:,:,3:4]
        volume = tensor_pri[:,:,4:5]

        prev_close = torch.roll(close,1,0)
        prev_close[0] = prev_close[1]

        prev_high = torch.roll(high,1,0)
        prev_high[0] = prev_high[1]

        prev_low = torch.roll(low,1,0)
        prev_low[0] = prev_low[1]

        exec_cap = volume * (high + low + close) / 3
        zeros = torch.zeros_like(exec_cap)

        returns = extend(torch.diff(torch.log(impute(close)), dim=0), num_days, blank=0)
        returns = torch.clip(returns, -0.2, 0.2)
        returns_prev = torch.roll(returns,1,0)
        returns_prev[0] = 0
        returns[close.isnan()] = np.nan
        returns_prev[close.isnan()] = np.nan
        tensor_ret[:,:,0] = returns.squeeze()
        tensor_ret[:,:,1] = returns_prev.squeeze()

        close_open_unbalance = close / open - 1
        close_open_unbalance = torch.clip(close_open_unbalance, -0.2, 0.2)
        tensor_ret[:,:,2] = close_open_unbalance.squeeze()

        returns_windows = returns.unfold(0,7,1)
        vol = torch_nanstd(returns_windows, dim=-1)
        vol = extend(vol, num_days)
        past_vol = torch.roll(vol,7,0)
        past_vol[:7] = np.nan
        vol_delta = vol - past_vol
        vol_delta = torch.nan_to_num(vol_delta, nan=0, posinf=0, neginf=0)
        tensor_vol[:,:,0] = vol_delta.squeeze()

        volume_growth = extend(torch.diff(torch.log(impute(volume)),dim=0), num_days, blank=0)
        volume_growth[volume.isnan()] = np.nan
        volume_growth = torch.clip(volume_growth, -3, 3)
        tensor_vol[:,:,1] = volume_growth.squeeze()

        high_low_unbalance = high / low - 1
        high_low_unbalance = torch.clip(high_low_unbalance, 0, 3)
        tensor_vol[:,:,2] = high_low_unbalance.squeeze()

        for f in range(4):

            window_size = 7 * 2**f

            # ROLLING

            returns_windows = returns.unfold(0,window_size,1)
            close_windows = close.unfold(0,window_size,1)
            high_windows = high.unfold(0,window_size,1)
            low_windows = low.unfold(0,window_size,1)
            volume_growth_windows = volume_growth.unfold(0,window_size,1)

            rolling_close_means = extend(torch_nanmean(close_windows,dim=-1), num_days)
            rolling_close_stds = extend(torch_nanstd(close_windows,dim=-1), num_days)
            rolling_close_max = extend(torch_nanmax(close_windows, dim=-1), num_days)
            rolling_high_max = extend(torch_nanmax(high_windows, dim=-1), num_days)
            rolling_low_min = extend(torch_nanmin(low_windows, dim=-1), num_days)

            # returns moving avg
            rolling_returns_means = torch_nanmean(returns_windows, dim=-1)
            rolling_returns_means = extend(rolling_returns_means, num_days)
            rolling_returns_means = torch.nan_to_num(rolling_returns_means, nan=0, posinf=0, neginf=0)
            tensor_ret[:,:,3+f] = rolling_returns_means.squeeze()

            # last x days max return
            rolling_returns_max = torch_nanmax(returns_windows, dim=-1)
            rolling_returns_max = extend(rolling_returns_max, num_days)
            rolling_returns_max = impute_with_medians(rolling_returns_max)
            tensor_ret[:,:,3+4+f] = rolling_returns_max.squeeze()

            # last x days volatility
            rolling_returns_stds = torch_nanstd(returns_windows, dim=-1)
            rolling_returns_stds = extend(rolling_returns_stds, num_days)
            rolling_returns_stds = impute_with_medians(rolling_returns_stds)
            tensor_vol[:,:,3+f] = rolling_returns_stds.squeeze()

            # last x days volume growth mean
            rolling_volume_growth_means = torch_nanmean(volume_growth_windows, dim=-1)
            rolling_volume_growth_means = extend(rolling_volume_growth_means, num_days)
            rolling_volume_growth_means = torch.clip(rolling_volume_growth_means, -1, 1)
            rolling_volume_growth_means = torch.nan_to_num(rolling_volume_growth_means, nan=0, posinf=0, neginf=0)
            tensor_vol[:,:,3+4+f] = rolling_volume_growth_means.squeeze()


            # VOLATILITY

            # Average True Range: https://www.investopedia.com/terms/a/atr.asp    #Keltner Channel
            TR = torch.topk(torch.cat((high - low, torch.abs(high - prev_close), torch.abs(low - prev_close)), dim=-1), dim=-1, k=1)[0]
            TR_windows = TR.unfold(0,window_size,1)
            ATR = torch_nanmean(TR_windows, dim=-1)
            ATR = extend(ATR, num_days)
            deviation_ATR = (close - rolling_close_means) / ATR
            deviation_ATR = torch.clip(deviation_ATR, -10, 10)
            deviation_ATR = torch.nan_to_num(deviation_ATR, nan=0, posinf=0, neginf=0)
            tensor_vol[:,:,3+8+f] = deviation_ATR.squeeze()

            # Bollinger Band: https://www.investopedia.com/terms/b/bollingerbands.asp
            deviation_bollinger = (close - rolling_close_means) / rolling_close_stds
            deviation_bollinger = torch.clip(deviation_bollinger, -5, 5)
            deviation_bollinger = torch.nan_to_num(deviation_bollinger, nan=0, posinf=0, neginf=0)
            tensor_vol[:,:,3+12+f] = deviation_bollinger.squeeze()

            # Donchian Channels: https://www.investopedia.com/terms/d/donchianchannels.asp
            middle_channel = (rolling_high_max + rolling_low_min) / 2
            deviation_donchian = close / middle_channel - 1
            deviation_donchian = torch.clip(deviation_donchian, -5, 5)
            deviation_donchian = torch.nan_to_num(deviation_donchian, nan=0, posinf=0, neginf=0)
            tensor_vol[:,:,3+16+f] = deviation_donchian.squeeze()

            # Ulcer Index: https://www.investopedia.com/terms/u/ulcerindex.asp
            R = 100 * (close - rolling_close_max) / rolling_close_max
            R2_windows = torch.pow(R,2).unfold(0,window_size,1)
            ulcer_index = torch.sqrt(torch_nansum(R2_windows, dim=-1) / (~R2_windows.isnan()).sum(dim=-1))
            ulcer_index = torch.clip(ulcer_index, 0, 50)
            ulcer_index = impute_with_medians(extend(ulcer_index, num_days))
            tensor_vol[:,:,3+20+f] = ulcer_index.squeeze()


            # VOLUME

            # Money Flow Index: https://www.investopedia.com/terms/m/mfi.asp
            exec_cap_pos_windows = torch.where(returns > 0, exec_cap, zeros).unfold(0,window_size,1)
            exec_cap_neg_windows = torch.where(returns < 0, exec_cap, zeros).unfold(0,window_size,1)
            money_flow_pos = torch_nansum(exec_cap_pos_windows, dim=-1)
            money_flow_neg = torch_nansum(exec_cap_neg_windows, dim=-1)
            MFI = money_flow_pos / (money_flow_pos + money_flow_neg) - 0.5
            MFI = torch.nan_to_num(extend(MFI, num_days), nan=0, posinf=0, neginf=0)
            tensor_vol[:,:,3+24+f] = MFI.squeeze()

            # Chaikin Oscillator: https://www.investopedia.com/terms/c/chaikinoscillator.asp    #Accumulation/Distribution Indicator
            mfv = volume * ((close - low) - (high - close)) / (high - low)
            mfv_slow = torch_nanmean(mfv.unfold(0,window_size,1), dim=-1)
            CO = mfv / extend(mfv_slow, num_days) - 1
            CO = torch.clip(CO, -100, 100)
            CO = torch.nan_to_num(CO, nan=0, posinf=0, neginf=0)
            tensor_vol[:,:,3+28+f] = CO.squeeze()

            # On-Balance Volume: https://www.investopedia.com/terms/o/onbalancevolume.asp
            obv = torch.where(returns > 0, volume, zeros) - torch.where(returns < 0, volume, zeros)
            obv_slow = torch_nanmean(obv.unfold(0,window_size,1), dim=-1)
            OBV = obv / extend(obv_slow, num_days) - 1
            OBV = torch.clip(OBV, -100, 100)
            OBV = torch.nan_to_num(OBV, nan=0, posinf=0, neginf=0)
            tensor_vol[:,:,3+32+f] = OBV.squeeze()

            # Force Index: https://www.investopedia.com/terms/f/force-index.asp
            fi = volume * (close - prev_close)
            fi_slow = torch_nanmean(fi.unfold(0,window_size,1), dim=-1)
            FI = fi / extend(fi_slow, num_days) - 1
            FI = torch.clip(FI, -100, 100)
            FI = torch.nan_to_num(FI, nan=0, posinf=0, neginf=0)
            tensor_vol[:,:,3+36+f] = FI.squeeze()


            # TREND

            # Average Directional Index: https://www.investopedia.com/terms/a/adx.asp
            dm_pos = high - prev_high
            dm_neg = prev_low - low
            smooth_dm_pos = torch_nanmean(dm_pos.unfold(0,window_size,1), dim=-1)
            smooth_dm_neg = torch_nanmean(dm_neg.unfold(0,window_size,1), dim=-1)
            di_pos = extend(smooth_dm_pos, num_days) / ATR
            di_neg = extend(smooth_dm_neg, num_days) / ATR
            DX = (di_pos - di_neg) / (di_pos + di_neg)
            DX_windows = DX.unfold(0,window_size,1)
            ADX = torch_nanmean(DX_windows, dim=-1)
            ADX = extend(ADX, num_days)
            ADX = torch.clip(ADX, -100, 100)
            ADX = torch.nan_to_num(ADX, nan=0, posinf=0, neginf=0)
            tensor_ret[:,:,3+8+f] = ADX.squeeze()

            # Vortex Indicator: https://www.investopedia.com/terms/v/vortex-indicator-vi.asp
            vm_pos = torch.abs(high - prev_low)
            vm_neg = torch.abs(low - prev_high)
            sum_vm_pos = torch_nansum(vm_pos.unfold(0,window_size,1), dim=-1)
            sum_vm_neg = torch_nansum(vm_neg.unfold(0,window_size,1), dim=-1)
            VI = sum_vm_pos / sum_vm_neg - 1
            VI = extend(VI, num_days)
            VI = torch.clip(VI, -1, 5)
            VI = torch.nan_to_num(VI, nan=0, posinf=0, neginf=0)
            tensor_ret[:,:,3+12+f] = VI.squeeze()

            # Mass Index: https://www.investopedia.com/terms/m/mass-index.asp
            high_low_diff = high - low
            high_low_diff_slow = torch_nanmean(high_low_diff.unfold(0,window_size,1), dim=-1)
            high_low_diff_fast = torch_nanmean(high_low_diff.unfold(0,window_size//3,1), dim=-1)
            MI = extend(high_low_diff_fast, num_days) / extend(high_low_diff_slow, num_days) - 1
            MI = extend(MI, num_days)
            MI = torch.clip(MI, -1, 1.5)
            MI = torch.nan_to_num(MI, nan=0, posinf=0, neginf=0)
            tensor_ret[:,:,3+16+f] = MI.squeeze()


            # MOMENTUM

            # Relative Strength Index: https://www.investopedia.com/terms/r/rsi.asp
            returns_pos_windows = torch.where(returns > 0, returns, zeros).unfold(0,window_size,1)
            returns_neg_windows = torch.where(returns < 0, returns, zeros).unfold(0,window_size,1)
            returns_pos_mean = torch_nanmean(returns_pos_windows, dim=-1)
            returns_neg_mean = torch_nanmean(returns_neg_windows, dim=-1)
            RSI = returns_pos_mean / (returns_pos_mean + returns_neg_mean) - 0.5
            RSI = extend(RSI, num_days)
            RSI = torch.clip(RSI, -100, 100)
            RSI = torch.nan_to_num(RSI, nan=0, posinf=0, neginf=0)
            tensor_ret[:,:,3+20+f] = RSI.squeeze()

            # Stochastic RSI: https://www.investopedia.com/terms/s/stochrsi.asp
            RSI_windows = RSI.unfold(0,window_size,1)
            RSI_min = extend(torch_nanmin(RSI_windows, dim=-1), num_days)
            RSI_max = extend(torch_nanmax(RSI_windows, dim=-1), num_days)
            SRSI = (RSI - RSI_min) / (RSI_max - RSI_min) - 0.5
            SRSI = torch.nan_to_num(SRSI, nan=0, posinf=0, neginf=0)
            tensor_ret[:,:,3+24+f] = SRSI.squeeze()

            # True Strength Index: https://www.investopedia.com/terms/t/tsi.asp
            pc_windows = (close - prev_close).unfold(0,window_size,1)
            price_change = extend(torch_nanmean(pc_windows, dim=-1), num_days)
            price_change_abs = extend(torch_nanmean(torch.abs(pc_windows), dim=-1), num_days)
            TSI = price_change / price_change_abs
            TSI = torch.nan_to_num(TSI, nan=0, posinf=0, neginf=0)
            tensor_ret[:,:,3+28+f] = TSI.squeeze()

            # Ultimate Oscillator: https://www.investopedia.com/terms/u/ultimateoscillator.asp
            buying_pressure = close - torch.min(low, prev_close)
            true_high = torch.max(high, prev_close) - torch.min(low, prev_close)
            buying_pressure_windows = buying_pressure.unfold(0,window_size,1)
            true_high_windows = true_high.unfold(0,window_size,1)
            UAV = torch_nansum(buying_pressure_windows, dim=-1) / torch_nansum(true_high_windows, dim=-1) - 0.5
            UAV = torch.nan_to_num(extend(UAV, num_days), nan=0, posinf=0, neginf=0)
            tensor_ret[:,:,3+32+f] = UAV.squeeze()

            # Stochastic Oscillator: https://www.investopedia.com/terms/s/stochasticoscillator.asp
            SO = (close - rolling_low_min) / (rolling_high_max - rolling_low_min) - 0.5
            SO = torch.nan_to_num(SO, nan=0, posinf=0, neginf=0)
            tensor_ret[:,:,3+36+f] = SO.squeeze()

        tensor_ret[:,:,:3] = torch.nan_to_num(tensor_ret[:,:,:3], nan=0, posinf=0, neginf=0)
        tensor_vol[:,:,:3] = torch.nan_to_num(tensor_vol[:,:,:3], nan=0, posinf=0, neginf=0)

        return tensor_ret, tensor_vol

    def train_dataloader(self):

        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self):

        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, drop_last=True)

    def all_dataloader(self):

        return DataLoader(self.all_ds, batch_size=100, shuffle=False)


class JpxDataSet(Dataset):
    
    def __init__(self, x, y, w):

        super(JpxDataSet, self).__init__()

        self.x = x
        self.y = y
        self.w = w

    def __len__(self):

        return len(self.x)

    def __getitem__(self, idx):

        return self.x[idx], self.y[idx], self.w[idx]



if __name__ == '__main__':

    data = JPXdataModule()




