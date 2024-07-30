import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import accuracy_score, r2_score

###### [ Scaler ] ######
def scaler_df(df, scaler, omni_param, auroral_param):
    """
    This code is used to apply scaling to the data. Deciding between a StandardScaler or RobustScaler. These two methods are similar, but RobustScaler works better for large data set
        1.- Choose the type of scaler to be used (RobustScaler or StandardScaler)
        2.- Perform scaling
        3.- Join the dataframes to obtain a single one
    """
    df_omni = df[omni_param]
    df_epoch = pd.DataFrame(df['Epoch'], columns=['Epoch'])
    df_auroral = pd.DataFrame(df[auroral_param], columns=auroral_param)

    # 1.- Choose the type of scaler to be used (RobustScaler or StandardScaler)
    match scaler:
        case 'robust':
            scaler = RobustScaler()
        case 'standard':
            scaler = StandardScaler()

    # 2.- Perform scaling
    df_omni = scaler.fit_transform(df_omni)
    df_omni = pd.DataFrame(df_omni, columns=omni_param)

     # 3.- Join the dataframes to obtain a single one
    df_omni.set_index(df_epoch.index, inplace=True)
    df_auroral.set_index(df_epoch.index, inplace=True)

    df_combined = df_epoch.join(df_omni).join(df_auroral)
    
    return df_combined


###### [ Train/Val/Test set ] ######
def create_set_prediction(df, omni_param, auroral_param, set_split, n_split_train_val_test, n_split_train_val, test_size, val_size, processing_file):
    """
    This function creates the Train/Val/Test set for the prediction. To do this, you can choose tree options, use a organized selection given for TimeSerieSplit(), use random selection given for train_test_split() or use the classification given by the storm_list.csv file.
    """
    match set_split:

        case "organized":
            tscv1 = TimeSeriesSplit(n_splits=n_split_train_val_test)
            tscv2 = TimeSeriesSplit(n_splits=n_split_train_val)

            for train_val, test in tscv1.split(df):
                train_val_df, test_df = df.iloc[train_val], df.iloc[test]
            
            for train, val in tscv2.split(train_val_df):
                train_df, val_df = train_val_df.iloc[train], train_val_df.iloc[val]

            return train_df, val_df, test_df

        case "random":
            train_val_df, test_df = train_test_split(df, test_size=test_size, shuffle=True)
            train_df, val_df = train_test_split(train_val_df, test_size=val_size, shuffle=True)

            return train_df, val_df, test_df
        
        case "list":
            file = processing_file + 'storm_list.csv'
            df_storm = pd.read_csv(file, sep=";")

            df_storm['start_date'] = pd.to_datetime(df_storm['start_date'], format='%Y-%m-%d') + pd.to_timedelta('00:00:00')
            df_storm['end_date'] = pd.to_datetime(df_storm['end_date'], format='%Y-%m-%d') + pd.to_timedelta('23:59:00')
            df_storm_train = df_storm[df_storm['pred'] == 'train']
            df_storm_val = df_storm[df_storm['pred'] == 'val']
            df_storm_test = df_storm[df_storm['pred'] == 'test']

            df['Epoch'] = pd.to_datetime(df['Epoch'])

            df_train_list = []
            df_val_list = []
            df_test_list = []
            for start, end in zip(df_storm_train['start_date'], df_storm_train['end_date']):
                temporal_df = df.loc[(df['Epoch'] >= start) & (df['Epoch'] <= end)].copy()
                df_train_list.append(temporal_df)
            train_df = pd.concat(df_train_list, axis=0, ignore_index=True)

            for start, end in zip(df_storm_val['start_date'], df_storm_val['end_date']):
                temporal_df = df.loc[(df['Epoch'] >= start) & (df['Epoch'] <= end)].copy()
                df_val_list.append(temporal_df)
            val_df = pd.concat(df_val_list, axis=0, ignore_index=True)

            for start, end in zip(df_storm_test['start_date'], df_storm_test['end_date']):
                temporal_df = df.loc[(df['Epoch'] >= start) & (df['Epoch'] <= end)].copy()
                df_test_list.append(temporal_df)
            test_df = pd.concat(df_test_list, axis=0, ignore_index=True)

            return train_df, val_df, test_df


###### [ Shift ] ######
def shifty(df, omni_param, auroral_index, shift_length, type_model):
    """
    This code creates a data delay for the neural networks. If it is an ANN it will be [m, n-t]. If it is a recurrent network it will be [m, [t,n]]
    """

    df_omni = df[omni_param].copy()
    df_index = df[auroral_index].copy()
    np_index = df_index.to_numpy()

    if type_model == 'ANN':
        for cols in df_omni.columns:
            for lag in range(1, shift_length + 1):
                df_omni[f'{cols}_{lag}'] = df_omni[cols].shift(lag).astype('float32')
                df_omni[f'{cols}_{lag}'].fillna(0, inplace=True)

        np_omni = df_omni.values

    else:
        sequence = []
        for i in range(len(df_omni) - shift_length + 1):
            seq = df_omni.iloc[i : i + shift_length].values
            sequence.append(seq)
        
        np_omni = np.array(sequence)
        np_index = np_index[shift_length-1:]
    
    return np_omni, np_index
        

###### [ DataTorch ] ######
class CustomDataset(Dataset):
    """
    This class converts train/val/test sets into torch tensors.
        1.- Convert the numpy array to a torch tensor, convert it to float32 type and move it to cuda
        2.- The total length of the delivered numpy array is obtained
        3.- A function is made that delivers the values ??of a given index
    """
    # 1.- Convert the numpy array to a torch tensor, convert it to float32 type and move it to cuda
    def __init__(self, omni, index, device):
        self.device = device
        
        self.x = torch.tensor(omni, dtype=torch.float32)
        self.y = torch.tensor(index, dtype=torch.float32).unsqueeze(1)
        self.x = self.x.to(self.device)
        self.y = self.y.to(self.device)

    # 2.- The total length of the delivered numpy array is obtained
    def __len__(self):
        return len(self.x)
    
    # 3.- A function is made that delivers the values ??of a given index
    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x, y
