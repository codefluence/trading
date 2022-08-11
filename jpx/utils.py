import numpy as np
import pandas as pd
import torch
import math


def regularize( features, targets, weights, number_of_categorical_features, 
                mix_days=False, mix_with_secondary=False):

    '''normalized features are expected'''
    '''last feature is expected as is_active'''
    '''number_of_categorical_features at the end'''

    noise_level=0.2
    degradation_level=0.2
    beta_a = 4.

    nocf = number_of_categorical_features
    is_active = features[:,:,-1].squeeze(-1)

    def add_noise(m):

        m = torch.round(m / degradation_level) * degradation_level
        m = torch.normal(mean = m, std = torch.abs(m)/3).cuda()
        m += torch.normal(mean = m, std = noise_level).cuda()

        return m

    features[:,:,:-nocf] = add_noise(features[:,:,:-nocf])
    targets = torch.normal(mean = targets, std = torch.abs(targets)/2).cuda()

    if mix_days and len(features) > 2:

        mp = math.ceil(len(features)/2)
        poi = -2

        mixed_feature = []
        mixed_targets = []
        mixed_weights = []

        while abs(poi) <= math.ceil(len(features)/2):

            sampler = torch.distributions.beta.Beta(torch.tensor([beta_a]), torch.tensor([1.]))
            w = sampler.rsample((features.shape[1],)).cuda().squeeze(-1)
            w[(is_active[poi+1]==0) | (is_active[poi]==0)] = 1

            #features[poi,:,:-nocf] = w.unsqueeze(-1) * features[poi,:,:-nocf] + (1-w.unsqueeze(-1)) * features[poi+1,:,:-nocf]

            c_feats = w.unsqueeze(-1) * features[poi,:,:-nocf] + (1-w.unsqueeze(-1)) * features[poi+1,:,:-nocf]
            mixed_feature.append(torch.cat((c_feats,features[poi,:,-nocf:]),dim=-1).unsqueeze(0))
            mixed_targets.append((w * targets[poi] + (1-w) * targets[poi+1]).unsqueeze(0))
            mixed_weights.append((w * weights[poi] + (1-w) * weights[poi+1]).unsqueeze(0))

            poi -= 2

        features = torch.cat((features[:math.floor(len(features)/2)], torch.cat(mixed_feature, dim=0)),dim=0)
        targets  = torch.cat((targets[:math.floor(len(targets)/2)],   torch.cat(mixed_targets, dim=0)),dim=0)
        weights  = torch.cat((weights[:math.floor(len(weights)/2)],   torch.cat(mixed_weights, dim=0)),dim=0)

    # if mix_with_secondary:

    #     is_active_sec = features_sec[:,:,-1].squeeze(-1)

    #     for i in range(18):

    #         sector_channels = features[0,:,i-nocf] == 1
    #         sector_channels_sec = features_sec[0,:,i-nocf] == 1

    #         sector_features = features[:,sector_channels]
    #         sector_features_sec = features_sec[:,sector_channels_sec]

    #         sector_is_active = is_active[:,sector_channels].clone()
    #         sector_is_active_sec = is_active_sec[:,sector_channels_sec].clone()

    #         # print(sector_features.shape[1], sector_features_sec.shape[1])

    #         if sector_features.shape[1] == 0:
    #             continue

    #         sec_idx = torch.randperm(sector_features.shape[1]).tolist()
    #         sec_idx_sec = torch.randperm(sector_features_sec.shape[1]).tolist()

    #         kk = min(len(sec_idx),len(sec_idx_sec))
    #         sec_idx = sec_idx[:kk]
    #         sec_idx_sec = sec_idx_sec[:kk]=

    #         dim = len(features), len(sec_idx), 1
    #         sampler = torch.distributions.beta.Beta(torch.tensor([beta_a]), torch.tensor([1.]))
    #         w = sampler.rsample(dim).cuda().squeeze(-1)

    #         # w[torch.randint(0, 4, dim).cuda() == 0] = 1.
    #         w[sector_is_active[:,sec_idx]==0] = 1.
    #         w[sector_is_active_sec[:,sec_idx_sec]==0] = 1.

    #         features[:,sec_idx,:-nocf] = w*features[:,sec_idx,:-nocf] + (1-w)*features_sec[:,sec_idx_sec,:-nocf]
    #         targets[:,sec_idx] = w.squeeze(-1)*targets[:,sec_idx] + (1-w.squeeze(-1))*targets_sec[:,sec_idx_sec]

    return features, targets, weights



# def torch_diff(tensor, hack=False):

#     if hack:
#         return torch.tensor(np.diff(tensor.cpu().numpy()))
#     else:
#         return torch.diff(tensor)

# def torch_nan_to_num(tensor, nan=0., posinf=0., neginf=0., hack=False):

#     if hack:
#         return torch.tensor(np.nan_to_num(tensor.cpu().numpy(), nan=nan, posinf=posinf, neginf=neginf))
#     else:
#         return torch.nan_to_num(tensor, nan=nan, posinf=posinf, neginf=neginf)

def transform_num_days(num_days):

    return torch.log(num_days + 1) * 20

def from_date_to_int(date_series):

    '''returns the number of days since 1970/01/01'''

    return pd.to_datetime(date_series).values.astype(np.int64) // 10**9 // 86400

def torch_nansum(tensor, dim):

    all_nans = tensor.isnan().all(dim=dim)

    sums = torch.sum(torch.nan_to_num(tensor), dim=dim)
    sums[all_nans] = np.nan

    return sums

def torch_nanmax(tensor, dim):

    maxs = torch.max(torch.nan_to_num(tensor, nan=-np.inf), dim=dim)[0]
    maxs[maxs==-np.inf] = np.nan

    return maxs

def torch_nanmin(tensor, dim):

    mins = torch.min(torch.nan_to_num(tensor, nan=np.inf), dim=dim)[0]
    mins[mins==np.inf] = np.nan

    return mins

def torch_nanmean(tensor, dim):

    num = torch.sum(torch.nan_to_num(tensor), dim=dim)
    den = torch.sum(~torch.isnan(tensor), dim=dim)

    return num/den
    #TODO: comparar con torch.nanmean(tensor, dim=dim)

def torch_nanstd(tensor, dim, unbiased=False):

    '''Unbiased.'''

    m = torch_nanmean(tensor, dim=dim)
    s = (tensor - m.unsqueeze(-1))**2

    num = torch.sum(torch.nan_to_num(s), dim=dim)
    den = torch.sum(~torch.isnan(tensor), dim=dim)

    if unbiased:
        den -= 1

    return torch.sqrt(num / den)
    #TODO: comparar con torch.tensor(np.nanstd(tensor.cpu().numpy(), axis=dim))

def impute(tensor, backpass=True, in_place=False):

    '''Performs forward and backward imputation along dim 0'''

    if not in_place:
        tensor = tensor.clone()

    for shift in range(1,len(tensor)):

        if tensor[shift:].isnan().sum() == 0:
            break

        zeros = torch.zeros(tensor[shift:].shape, device=tensor.device, dtype=tensor.dtype)

        tensor[shift:] = torch.nan_to_num(tensor[shift:]) + torch.where(torch.isnan(tensor[shift:]), tensor[:-shift], zeros)

    if backpass:

        flipped = impute(torch.flip(tensor,[0]), backpass=False, in_place=True)
        return torch.flip(flipped,[0])
    else:

        return tensor

def impute_with_medians(tensor):

    ''' Performs median imputation through dim=0
        If nans remain, another imputation is performed in dim=1'''

    tensor = torch.nan_to_num(tensor) + torch.isnan(tensor) * torch.nanmedian(tensor, dim=0)[0]

    if tensor.isnan().sum() > 0:
        tensor = torch.nan_to_num(tensor, nan=torch.nanmedian(tensor))

    return tensor

def extend(series, final_len, blank=np.nan):

    gap = final_len - series.shape[0]

    if gap < 1:
        return series

    head_dim = gap, series.shape[1], series.shape[2]
    head = torch.empty(head_dim, device=series.device, dtype=series.dtype)
    head[:] = blank
    return torch.cat((head, series), dim=0)

# annual_sharp = daily_sharp * 252**(1/2)

# from the competition host: https://www.kaggle.com/code/smeitoma/jpx-competition-metric-definition
def calc_spread_return_sharpe(df: pd.DataFrame, portfolio_size: int = 200, toprank_weight_ratio: float = 2) -> float:
    """
    Args:
        df (pd.DataFrame): predicted results
        portfolio_size (int): # of equities to buy/sell
        toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
    Returns:
        (float): sharpe ratio
    """
    def _calc_spread_return_per_day(df, portfolio_size, toprank_weight_ratio):
        """
        Args:
            df (pd.DataFrame): predicted results
            portfolio_size (int): # of equities to buy/sell
            toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
        Returns:
            (float): spread return
        """
        assert df['Rank'].min() == 0
        assert df['Rank'].max() == len(df['Rank']) - 1
        weights = np.linspace(start=toprank_weight_ratio, stop=1, num=portfolio_size)
        purchase = (df.sort_values(by='Rank')['Target'][:portfolio_size] * weights).sum() / weights.mean()
        short = (df.sort_values(by='Rank', ascending=False)['Target'][:portfolio_size] * weights).sum() / weights.mean()
        return purchase - short

    buf = df.groupby('Date').apply(_calc_spread_return_per_day, portfolio_size, toprank_weight_ratio)
    buf_mean = buf.mean()
    buf_std  = buf.std()
    sharpe_ratio = buf_mean / buf_std
    return sharpe_ratio, buf_mean, buf_std

def spread_return_sharpe_from_weights(targets, weights):

    dates = []
    ranks = []
    targs = []

    for i in torch.arange(len(weights)):

        targs.append(targets[i].squeeze())

        w = weights[i].squeeze()

        r = torch.empty(len(w), dtype=torch.int64)
        r[torch.flip(torch.argsort(w),dims=[0]).tolist()] = torch.arange(len(w))

        ranks.append(r)

        day = torch.empty(len(w))
        day[:] = i
        dates.append(day)

    daily_portfolio = pd.DataFrame(data={
        "Date": torch.cat(dates).detach().cpu().numpy(),
        "Rank": torch.cat(ranks).detach().cpu().numpy(),
        "Target": torch.cat(targs).detach().cpu().numpy()
    })

    return calc_spread_return_sharpe(daily_portfolio)

def utility_score(weights, returns, device='cuda', mode='loss'):

    if mode == 'loss':
        returns = torch.normal(mean=returns, std=torch.abs(returns)/2)

    daily_profit = []

    for d in range(len(weights)):
        pnl = torch.mul(weights[d],returns[d]).sum()
        daily_profit.append(pnl)
        
    p = torch.tensor(0, dtype=torch.float32, requires_grad=False, device=device)
    vol = torch.tensor(0, dtype=torch.float32, requires_grad=False, device=device)
        
    for dp in daily_profit:
        p = p + dp
        vol = vol + dp**2
    
    t = p / vol**.5 * (250/len(weights))**.5

    ceiling = torch.tensor(6, dtype=torch.float32, requires_grad=False, device=device)
    floor = torch.tensor(0, dtype=torch.float32, requires_grad=False, device=device)
    t = torch.min(torch.max(t, floor), ceiling)

    # if profit is negative the utility score is not clipped to 0 in loss mode (for learning purposes)
    if mode == 'loss' and p < 0.0:
        u = p
    else:
        u = torch.mul(p, t)

    if mode == 'loss':
        return -u
    else:
        return t.cpu().item(), p.cpu().item(), u.cpu().item()






