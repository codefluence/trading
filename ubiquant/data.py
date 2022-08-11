
import gc
import math
import pickle
import os.path
from tqdm import tqdm

import pandas as pd
import numpy as np

import scipy.signal
from scipy import stats
from pykalman import KalmanFilter

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from models import PastResponseRegressor, PastResponseRegressorData


np.set_printoptions(threshold=2000, linewidth=120, precision=3, edgeitems=20, suppress=1)

DATA_PATH = 'D:/data/ubiquant/train.csv'
CAHE_DIR = 'D:/ubiquant_cache/'


def kalman_filter(series, tc=0.2):

    kf = KalmanFilter(transition_matrices = [1],
    observation_matrices = [1],
    initial_state_mean = 0,
    initial_state_covariance = 1,
    observation_covariance = 1,
    transition_covariance = tc)

    return kf.filter(series)[0].squeeze()

def butterworth_filter(series, f=0.2):

    if len(series) < 13:
        return series

    b, a = scipy.signal.butter(3, f)
    return scipy.signal.filtfilt(b, a, series)

def clip_corrs(corrs, thr_min=0.1, thr_max=0.4):

    signs = np.sign(corrs)
    corrs = np.clip(np.abs(corrs),thr_min,thr_max)
    corrs = signs * (corrs - thr_min)

    print(np.sum(np.abs(corrs)>0,axis=0) / len(corrs))
    return corrs



class UbiquantDataModule(pl.LightningDataModule):

    def __init__(self, split, feat_num_pcs=210, day_corr_num_pcs=81, strat_corr_num_pcs=3, degradation_level=3):

        super(UbiquantDataModule, self).__init__()

        np.random.seed(0)
        torch.manual_seed(0)

        original_features_path = CAHE_DIR+'original_features.pkl'
        original_context_path = CAHE_DIR+'original_context.pkl'
        context_path = CAHE_DIR+'context.pkl'
        features_split_path = CAHE_DIR+'features_'+split+'.pkl'
        strat_corrs_split_path = CAHE_DIR+'strat_corrs_'+split+'.pkl'


        if os.path.isfile(original_features_path) and os.path.isfile(original_context_path):
            
            print('loading original features from',original_features_path)
            with open(original_features_path,'rb') as f:
                original_features = pickle.load(f)

            print('loading original context from',original_context_path)
            with open(original_context_path,'rb') as f:
                original_context = pickle.load(f)
        else:

            print('reading data file...')
            original_features = pd.read_csv(DATA_PATH).to_numpy(dtype=np.float32)
            gc.collect()
            original_context  = original_features[:,1:4]
            original_features = original_features[:,4:]

            print('saving original_features...')
            with open(original_features_path,'wb') as f:
                pickle.dump(original_features, f, protocol=4)

            print('saving original_context...')
            with open(original_context_path,'wb') as f:
                pickle.dump(original_context, f, protocol=4)


        if os.path.isfile(context_path):
            
            print('loading data from',context_path)

            with open(context_path,'rb') as f:
                context = pickle.load(f)
        else:

            context = np.empty((len(original_context),11), dtype=np.float32);  context[:] = np.nan
            context[:,:3] = original_context

            strat_ids = context[:,1]
            targets   = context[:,2]

            print('computing series...')
            for strat_id in tqdm(np.unique(strat_ids)):

                idx = strat_ids==strat_id
                strat_targets = targets[idx]

                context[idx,3] = np.diff(kalman_filter(strat_targets.cumsum()),prepend=0)  # slow signal
                context[idx,4] = pd.Series(context[idx,3]).rolling(window=5, min_periods=1).std().to_numpy(dtype=np.float32)  # slow signal rolling std

                context[idx,5] = strat_targets - context[idx,3]  # fast signal
                context[idx,6] = pd.Series(context[idx,5]).rolling(window=5, min_periods=1).std().to_numpy(dtype=np.float32)  # fast signal rolling std

                context[idx,7] = np.roll(np.diff(butterworth_filter(strat_targets.cumsum()),prepend=0),1); context[0,7] = 0  # filtered signal lagged
                context[idx,8] = pd.Series(context[idx,7]).rolling(window=5, min_periods=1).std().to_numpy(dtype=np.float32)  # filtered signal lagged rolling std

                context[idx,9] = pd.Series(context[idx,7]).rolling(window=5, min_periods=1).mean().to_numpy(dtype=np.float32)  # filtered signal lagged rolling mean
                context[idx,10] = pd.Series(context[idx,9]).rolling(window=5, min_periods=1).std().to_numpy(dtype=np.float32)  # filtered signal lagged rolling mean rolling std

            context = np.nan_to_num(context, posinf=0., neginf=0.)

            print('saving context...')
            with open(context_path,'wb') as f:
                pickle.dump(context, f, protocol=4)

        time_ids  = context[:,0]
        strat_ids = context[:,1]
        targets_slow = context[:,3]
        targets_past = context[:,[7,9]]


        if split.startswith('CV'):

            jump = int(split[-2])
            shift = int(split[-1])

            block_size = (max(time_ids)+1) // 20

            # is_train = ((time_ids//block_size + shift) % jump != 0) & ((strat_ids + shift) % jump != 0)
            # is_val = ((time_ids//block_size + shift) % jump == 0) & ((strat_ids + shift) % jump == 0)

            is_train = (time_ids//block_size + shift) % jump != 0
            is_val   = ~is_train

        elif split.startswith('EVAL'):

            is_train = time_ids < 1020
            is_val = ~is_train

        else:

            raise('Unknown split method')

        print('split:',split)
        print('validation days:',np.unique(time_ids[is_val]))
        print(round(len(time_ids[is_val])/len(time_ids),2),'% of total')
        

        if os.path.isfile(features_split_path):

            print('loading data from',features_split_path)
            with open(features_split_path,'rb') as f:
                features = pickle.load(f)
        else:

            print('scaling...')
            scaler = StandardScaler().fit(original_features[is_train])
            pickle.dump(scaler, open('./preprocessing/scaler_{}.pkl'.format(split),'wb'))
            features = scaler.transform(original_features)

            print('computing PCA...')
            pca_fea = PCA(n_components=feat_num_pcs).fit(features[is_train])
            pickle.dump(pca_fea, open('./preprocessing/pca_fea_{}.pkl'.format(split),'wb'))
            features = pca_fea.transform(features).astype(np.float32)

            past_targets_regressor_path = './preprocessing/regressor_{}.ckpt'.format(split)

            # target information in recent past is extracted from the features
            if not os.path.isfile(past_targets_regressor_path):

                regressor = PastResponseRegressor(feat_num_pcs)
                trainer = pl.Trainer( logger=False, gpus=1, max_epochs=5, checkpoint_callback=True, callbacks=[
                    EarlyStopping( monitor='val_mae', mode='min', patience=1, min_delta=0.001, verbose=True ),
                    ModelCheckpoint( monitor='val_mae', mode='min', dirpath='./preprocessing/', filename='regressor_{}'.format(split), save_top_k=1, verbose=True)
                ] )
                trainer.fit(regressor, PastResponseRegressorData(features, targets_past, is_train))
            else:
                regressor = PastResponseRegressor.load_from_checkpoint(past_targets_regressor_path,input_width=feat_num_pcs)

            print('computing past responses estimates...')
            f_gpu = torch.from_numpy(features).to('cuda')
            regressor.eval(); regressor.cuda()
            estimate_targets_past = regressor(f_gpu).detach().cpu().numpy()
            estimate_targets_past = np.nan_to_num(estimate_targets_past, posinf=0., neginf=0.)
            print(np.corrcoef(np.hstack((targets_past[is_val],estimate_targets_past[is_val])).T))

            max_time_id = int(max(time_ids))+1
            time_index = np.array(range(max_time_id))
            day_means = np.empty((max_time_id,original_features.shape[1]), dtype=np.float32);  day_means[:] = np.nan
            pears_to_target = np.empty((max_time_id,(day_corr_num_pcs+2)*2), dtype=np.float32);  pears_to_target[:] = np.nan
            spear_to_target = np.empty((max_time_id,(day_corr_num_pcs+2)*2), dtype=np.float32);  spear_to_target[:] = np.nan

            # features and targets values are degradated for correlations to avoid overfitting
            features_de = np.clip(np.round(features[:,:day_corr_num_pcs] * degradation_level), -2*degradation_level, 2*degradation_level)
            targets_de  = 1.*(estimate_targets_past>0)

            print('computing day means and correlations of day features to estimated past targets...')
            for time_id in tqdm(np.unique(time_ids)):

                day_means[int(time_id)] = np.nan_to_num(np.mean(original_features[time_ids==time_id],axis=0), posinf=0., neginf=0.)

                features_daygroup_de = features_de[time_ids==time_id]
                targets_daygroup_de  = targets_de[time_ids==time_id]

                pears = np.corrcoef(np.hstack((targets_daygroup_de,features_daygroup_de)).T)[:,:2]
                spear = stats.spearmanr(np.hstack((targets_daygroup_de,features_daygroup_de)))[0][:,:2].astype(np.float32)

                pears_to_target[int(time_id)] = np.nan_to_num(pears, posinf=0., neginf=0.).flatten()
                spear_to_target[int(time_id)] = np.nan_to_num(spear, posinf=0., neginf=0.).flatten()

            idx = ~np.isnan(day_means).any(axis=1)
            time_index = time_index[idx]
            day_means = day_means[idx]
            pears_to_target = pears_to_target[idx]
            spear_to_target = spear_to_target[idx]

            pears_to_target = clip_corrs(pears_to_target)
            spear_to_target = pears_to_target - clip_corrs(spear_to_target)

            print('computing PCAs...')

            pca_day = PCA(n_components=6).fit(day_means)
            print('pca_day explained var:',pca_day.explained_variance_ratio_.sum())
            pickle.dump(pca_day, open('./preprocessing/pca_day_{}.pkl'.format(split),'wb'))
            day_means = pca_day.transform(day_means)

            pca_pears_target = PCA(n_components=16).fit(pears_to_target)
            print('pca_pears_target explained var:',pca_pears_target.explained_variance_ratio_.sum())
            pickle.dump(pca_pears_target, open('./preprocessing/pca_pears_{}.pkl'.format(split),'wb'))
            pears_to_target = pca_pears_target.transform(pears_to_target)
        
            pca_spear_target = PCA(n_components=16).fit(spear_to_target)
            print('pca_spear_target explained var:',pca_spear_target.explained_variance_ratio_.sum())
            pickle.dump(pca_spear_target, open('./preprocessing/pca_spear_{}.pkl'.format(split),'wb'))
            spear_to_target = pca_spear_target.transform(spear_to_target)

            daily_feats  = np.hstack(( day_means, pears_to_target, spear_to_target ))
            features_ext = np.empty((len(features),daily_feats.shape[1]), dtype=np.float32);  features_ext[:] = np.nan

            print('extending features with day features...')
            for i in tqdm(range(len(time_index))):

                features_ext[time_ids==time_index[i]] = daily_feats[i]

            features = np.hstack(( features,features_ext,estimate_targets_past ))

            print('saving features...')
            with open(features_split_path,'wb') as f:
                pickle.dump(features, f, protocol=4)


        if os.path.isfile(strat_corrs_split_path):

            print('loading data from',strat_corrs_split_path)
            with open(strat_corrs_split_path,'rb') as f:
                strat_corrs = pickle.load(f)
        else:

            # features and targets values are degradated for correlations to avoid overfitting
            features_de = np.clip(np.round(features[:,:strat_corr_num_pcs] * degradation_level), -2*degradation_level, 2*degradation_level)
            targets_de  = 1.*(targets_slow>0)

            strat_corrs = np.empty((int(max(strat_ids))+1,strat_corr_num_pcs), dtype=np.float32); strat_corrs[:] = np.nan
            strat_index = np.array(range(len(strat_corrs)))

            print('computing correlations of strat features to target...')
            for strat_id in tqdm(np.unique(strat_ids)):

                feats_stratgroup_de = features_de[strat_ids==strat_id]
                targs_stratgroup_de = targets_de[strat_ids==strat_id]

                corrs = np.corrcoef(np.hstack((targs_stratgroup_de.reshape(-1,1),feats_stratgroup_de)).T)[1:,0]
                strat_corrs[int(strat_id)] = np.nan_to_num(corrs, posinf=0., neginf=0.)
                strat_index[int(strat_id)] = strat_id

            idx = ~np.isnan(strat_corrs).any(axis=1)
            strat_corrs = strat_corrs[idx]
            strat_index = strat_index[idx]

            strat_corrs = clip_corrs(strat_corrs, thr_min=0.1, thr_max=0.3)
            strat_corrs = StandardScaler().fit_transform(strat_corrs)

            strat_corrs_tmp = np.empty((len(features),strat_corrs.shape[1]), dtype=np.float32);  strat_corrs_tmp[:] = np.nan

            print('creating strat_corrs array...')
            for i in tqdm(range(len(strat_index))):

                strat_corrs_tmp[strat_ids==strat_index[i]] = strat_corrs[i]

            strat_corrs = strat_corrs_tmp

            print('saving strat_corrs...')
            with open(strat_corrs_split_path,'wb') as f:
                pickle.dump(strat_corrs, f, protocol=4)


        print('creating datasets...')

        self.train_ds = UbiquantDataSet(features[is_train], context[is_train], strat_corrs[is_train])
        self.val_ds   = UbiquantDataSet(features[is_val], context[is_val], strat_corrs[is_val])
  
        print('train size:', len(self.train_ds))
        print('val size:', len(self.val_ds))

    def train_dataloader(self):

        return DataLoader(self.train_ds, shuffle=False, batch_sampler=UbiquantSampler(self.train_ds, 8))

    def val_dataloader(self):

        return DataLoader(self.val_ds, shuffle=False, batch_sampler=UbiquantSampler(self.val_ds, 61))



class UbiquantDataSet(Dataset):
    
    def __init__(self, features, context, strat_corrs):

        super(UbiquantDataSet, self).__init__()

        self.features = features
        self.context = context
        self.strat_corrs = strat_corrs

    def __len__(self):

        return len(self.features)

    def __getitem__(self, ndx):

        return self.features[ndx], self.context[ndx], self.strat_corrs[ndx]



class UbiquantSampler(Sampler):

    """ Yields indices of a subset of times selected randomy,
        so each batch during training will contain a few concentrated time_ids (around 6).
        Otherwise the correlation values in the loss function would be inconsistent.
    """

    def __init__(self, dataset, times_per_batch):

        self.time_ids = dataset.context[:,0]
        self.times_per_batch = times_per_batch
        self.num_batches = math.ceil(len(np.unique(self.time_ids)) / times_per_batch)

    def __len__(self):

        return self.num_batches

    def __iter__(self):

        unique_time_ids = np.unique(self.time_ids,return_counts=True)[0]
        rand_idx = np.random.permutation(unique_time_ids)

        for i in np.arange(self.num_batches):

            if i+1 == self.num_batches:
                random_times = rand_idx[self.times_per_batch*i:]
            else:
                random_times = rand_idx[self.times_per_batch*i:self.times_per_batch*(i+1)]

            idx = np.where(np.isin(self.time_ids,random_times))[0]

            np.random.shuffle(idx)
            yield  idx




if __name__ == '__main__':

    UbiquantDataModule(split='EVAL')

    # for i in range(5):
    #     UbiquantDataModule(split='CV5'+str(i))

