import os
import time
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

device =(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


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
    if scaler == 'robust':
        scaler = RobustScaler()
    if scaler == 'standard':
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
def create_group_prediction(df, omni_param, auroral_param, group, storm_list, n_split_train_val_test, n_split_train_val, processing_file):
    """
    This function creates the Train/Val/Test group for the prediction. To do this, you can choose two options, make a random selection or use the classification given by the storm_list.csv file.
        1.- If storm_list=True, the storm_list.csv file is read and a dataframe is generated.
        2.- Change the format of the temporary columns of this dataframe and the parameters to study and create a classification according to the group you want to create (train/value/proof)
        3.- Make a loop with the dates of the previous classification and generate a new dataframe concatenating the results
        4.- Obtain the % of data of each group created
        5.- If random=False, perform the division with percentages given by n_split of the TimeSeriesSplit function (First the division of train_val and test set will be obtained,    and then train and val set will be obtained)
    """
    # 1.- If random=True, the storm_list.csv file is read and a dataframe is generated
    if storm_list:
        file = processing_file + 'storm_list.csv'
        df_storm = pd.read_csv(file, sep=";")

        # 2.- Change the format of the temporary columns of this dataframe and the parameters to study and create a classification according to the group you want to create (train/value/proof)   
        df_storm['start_date'] = pd.to_datetime(df_storm['start_date'], format='%Y-%m-%d')
        df_storm['end_date'] = pd.to_datetime(df_storm['end_date'], format='%Y-%m-%d')

        df_storm_group = df_storm[df_storm['pred'] == group]
        df['Epoch'] = pd.to_datetime(df['Epoch'])

        # 3.- Make a loop with the dates of the previous classification and generate a new dataframe concatenating the results
        df_group_list = []
        for start, end in zip(df_storm_group['start_date'], df_storm_group['end_date']):
            temporal_df = df.loc[(df['Epoch'] >= start) & (df['Epoch'] <= end)].copy()
            df_group_list.append(temporal_df)

        df_group = pd.concat(df_group_list, axis=0, ignore_index=True)

        return df_group

    # 5.- If random=False, perform the division with percentages given by n_split of the TimeSeriesSplit function (First the division of train_val and test set will be obtained,    and then train and val set will be obtained)
    else:
        x = df[omni_param]
        y = df[auroral_param]
        date = df['Epoch']

        tscv1 = TimeSeriesSplit(n_splits=n_split_train_val_test)
        tscv2 = TimeSeriesSplit(n_splits=n_split_train_val)

        for train_val, test in tscv1.split(x):
            train_val_x, test_x = x.iloc[train_val], x.iloc[test]
            train_val_y, test_y = y.iloc[train_val], y.iloc[test]
            train_val_date, test_date = date.iloc[train_val], date.iloc[test]
        
        for train, val in tscv2.split(train_val_x):
            train_x, val_x = train_val_x.iloc[train], train_val_x.iloc[val]
            train_y, val_y = train_val_y.iloc[train], train_val_y.iloc[val]
            train_date, val_date = train_val_date.iloc[train], train_val_date.iloc[val]
            
        train_df = pd.concat([train_date, train_x, train_y], axis=1)
        val_df = pd.concat([val_date, val_x, val_y], axis=1)
        test_df = pd.concat([test_date, test_x, test_y], axis=1)

        return train_df, val_df, test_df


###### [ Shift ] ######

####### 1-Dimension #######
def shifty_1d(df, omni_param, auroral_index, shifty):
    """ 
    This code creates a delay of the form X[t-m, n], where t is the delay and m,n are the original dimensions 
        1.- The columns are identified and a cycle is performed in the delay range
        2.- A series is created in Pandas that stores the delays generated by the 'shift' command
    """
    df_omni = df[omni_param].copy()
    df_index = df[auroral_index].copy()
    
    # 1.- The columns are identified and a cycle is performed in the delay range
    for cols in df_omni.columns:
        for lag in range(1, shifty + 1):
            # 2.- A series is created in Pandas that stores the delays generated by the 'shift' command
            df_omni[f'{cols}_{lag}'] = df_omni[cols].shift(lag).astype('float32')

    # Remove rows with NaN values that were created due to the shift
    df_omni = df_omni.dropna()

    # Align the auroral index data to match the delayed omni data
    df_index = df_index.loc[df_omni.index]

    return df_omni.values, df_index.values

####### 3-Dimension #######
def shifty_3d(df, omni_param, auroral_index, shifty):
    """
    This function creates sequences of the form X[[t-m], n], where t is the delay and m,n are the original dimensions
        1.- Extract necessary columns as separate numpy arrays
        2.- Extract the omni_param columns and create sequences
        3.- The cycle is performed in the delay range
        4.- A series is created in Pandas that stores the created arrays to later save them in a numpy array
    """
    # 1.- Extract necessary columns as separate numpy arrays
    np_index = df[auroral_index].to_numpy()
    
    # 2.- Extract the omni_param columns and create sequences
    df_omni = df[omni_param].copy()

    sequences = []
    # 3.- The columns are identified and a cycle is performed in the delay range
    for i in range(len(df_omni) - shifty + 1):
        # 4.- A series is created in Pandas that stores the created arrays to later save them in a numpy array
        seq = df_omni.iloc[i:i + shifty].values
        sequences.append(seq)

    np_omni = np.array(sequences)
    
    # Align the auroral index data to match the delayed omni data
    np_index = np_index[shifty - 1:]
        
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
        self.x = torch.tensor(omni, dtype=torch.float32)
        self.y = torch.tensor(index, dtype=torch.float32).unsqueeze(1)
        self.device = device
        self.x = self.x.to(self.device)
        self.y = self.y.to(self.device)

    # 2.- The total length of the delivered numpy array is obtained
    def __len__(self):
        return len(self.x)
    
    # 3.- A function is made that delivers the values ??of a given index
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
