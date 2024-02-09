import numpy as np
import pandas as pd

def get_dataset(dataset_name, kwargs):
    if dataset_name.lower() == 'shell':
        return get_shell_dataset(**kwargs)
    elif dataset_name.lower() == 'europe':
        return get_europe_dataset(**kwargs)
    elif dataset_name.lower() == 'denmark':
        return get_denmark_dataset(**kwargs)
    else:
        raise ValueError('Unknown dataset: %s' % dataset_name)

def get_shell_dataset(N_total=1000, R=1.0, B=0.2, num_dims=2):
    u = np.random.normal(0,1,(N_total,num_dims))  # an array of d normally distributed random variables
    norm = np.sum(u**2, axis=1, keepdims=True) **(0.5)
    r = np.random.uniform((R-B/2)**num_dims,(R+B/2)**num_dims,(N_total,1))**(1.0/num_dims)
    data = (r*u/norm).astype(np.float32)
    return data

def get_europe_dataset(num_dims=None, temporal_aggregate_level=24):
    num_users = num_dims
    raw_data = pd.read_csv('data/entsoe_europe_60min_15countries.csv',index_col=0, parse_dates=True, sep=';', low_memory=False)
    raw_data = raw_data.apply(pd.to_numeric,errors='coerce')
    if num_dims is not None:
        user_id = np.random.choice(raw_data.shape[1],num_users,replace=False)
        pair_data = raw_data.iloc[:,user_id]
    else:
        user_id = np.arange(raw_data.shape[1])
        pair_data = raw_data
    pair_data = pair_data.interpolate(method='linear', thresh=temporal_aggregate_level).dropna() # If the consequent 24 hours are all NaN, do not interpolate
    pair_data_daily = pair_data.groupby(pair_data.index.floor(f"{temporal_aggregate_level}h")).sum(min_count=temporal_aggregate_level).dropna()/temporal_aggregate_level # If there is at least one NaN in a day, drop the day
    # pair_data_daily = outlier_removal(pair_data_daily)
    pair_data_daily, data_mean, data_std = normalization(pair_data_daily)
    return pair_data_daily, [name[:2] for name in raw_data.columns[user_id]]

def get_denmark_dataset(year_start=2020, year_end=None, temporal_aggregate_level=24):
    if year_end is None:
        year_end = year_start + 1
    raw_data = pd.read_csv('data/entsoe_denmark_60min_2zones.csv', index_col=0, parse_dates=True, sep=';', low_memory=False)
    raw_data = raw_data.apply(pd.to_numeric,errors='coerce')
    raw_data.drop(columns=['DK_1_price_day_ahead','DK_2_price_day_ahead'], inplace=True)
    raw_data = raw_data.loc[f'{year_start}-01-01':f'{year_end}-01-01']
    raw_data = raw_data.interpolate(method='linear', thresh=temporal_aggregate_level).dropna() # If the consequent 24 hours are all NaN, do not interpolate
    raw_data = raw_data.groupby(raw_data.index.floor(f"{temporal_aggregate_level}h")).sum(min_count=temporal_aggregate_level).dropna()/temporal_aggregate_level

    # raw_data = raw_data.apply(pd.to_numeric,errors='coerce').dropna()
    # pair_data_daily = outlier_removal(raw_data)
    data, data_mean, data_std = normalization(raw_data)
    # data = raw_data.values.astype(np.float32)
    return data, data_mean, data_std

def split_dataset(data, test_set_ratio=0.2):
    N_total = data.shape[0]
    test_set_size = int(N_total*test_set_ratio)
    train_set_size = N_total - test_set_size
    rnd_idx = np.random.permutation(N_total)
    test_data = data[rnd_idx[train_set_size:]]
    train_data = data[rnd_idx[:train_set_size]]
    return train_data, test_data

def weekly_split_dataset(data, test_set_ratio=0.2):
    N_total = data.shape[0]
    num_weeks = N_total // 7
    weekly_data = data[:num_weeks*7].reshape(num_weeks, 7,-1)
    test_set_size = int(num_weeks*test_set_ratio)
    train_set_size = num_weeks - test_set_size
    rnd_idx = np.random.permutation(num_weeks)
    train_data = weekly_data[rnd_idx[:train_set_size]].reshape(-1, data.shape[1])
    test_data = weekly_data[rnd_idx[train_set_size:]].reshape(-1, data.shape[1])
    train_data = np.concatenate((train_data, data[num_weeks*7:]), axis=0)
    return train_data, test_data

def outlier_removal(df):
    quant = 0.99
    q_high = df.quantile(quant)
    q_low = df.quantile(1-quant)
    pair_data_daily_filtered = df[(df < q_high) & (df > q_low)].interpolate(method='linear')
    return pair_data_daily_filtered

def normalization(df):
    data = df.values.astype(np.float32)
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)
    data = (data - data_mean)/data_std
    return data, data_mean, data_std