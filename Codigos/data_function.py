import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, train_test_split

###### [ Epoch Storm Selection ] #####
def epoch_storm(df, processing_file):
    df['Epoch'] = pd.to_datetime(df['Epoch'])
    df.set_index('Epoch', inplace=True)
    
    storm_list = pd.read_csv(processing_file + 'storm_list.csv', header=None, names=['Epoch'])
    storm_list['Epoch'] = pd.to_datetime(storm_list['Epoch'])

    df_storm = pd.DataFrame()  

    for idx, row in storm_list.iterrows():
        start = row['Epoch'] - pd.Timedelta('24h')
        end = row['Epoch'] + pd.Timedelta('24h')

        if idx == 0:
            df_storm = df[start:end]
        else:
            df_storm = pd.concat([df_storm, df[start:end]], axis=0)
            
    return df_storm.reset_index()


###### [ Scaler ] ######
def scaler_df(df, scaler, omni_param, auroral_param):
    """
    This code is used to apply scaling to the data. Deciding between a StandardScaler or RobustScaler. These two methods are similar, but RobustScaler works better for large data set
    """
    df_omni = df[omni_param]
    df_epoch = df[['Epoch']]
    df_auroral = df[auroral_param]

    if scaler == 'robust':
        scaler = RobustScaler()
    elif scaler == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Scaler must be 'robust' or 'standard'")
    

    df_omni_scaled = scaler.fit_transform(df_omni)
    df_omni_scaled = pd.DataFrame(df_omni_scaled, columns=omni_param, index=df.index)

    df_combined = pd.concat([df_epoch, df_omni_scaled, df_auroral], axis=1)
    
    return df_combined


###### [ Train/Val/Test set ] ######
def create_set_prediction(df, set_split, n_split_train_val_test, n_split_train_val, test_size, val_size):
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

        case "random":
            train_val_df, test_df = train_test_split(df, test_size=test_size, shuffle=True)
            train_df, val_df = train_test_split(train_val_df, test_size=val_size, shuffle=True)

        case _:
            raise ValueError("set_split must be 'organized' or 'random'")

    return train_df, val_df, test_df
        

###### [ Shift ] ######
def shifty(df, omni_param, auroral_index, shift_length, type_model):
    """
    This code creates a data delay for the neural networks. If it is an ANN it will be [m, n-t]. If it is a recurrent network it will be [m, [t,n]]
    """

    df_omni = df[omni_param].copy()
    df_index = df[auroral_index].copy()
    df_epoch = df['Epoch'].iloc[shift_length:].reset_index(drop=True)
    np_index = df_index.to_numpy()

    if type_model == 'ANN':
        # Lista para almacenar todas las columnas nuevas
        shifted_columns = []

        for col in df_omni.columns:
            for lag in range(1, shift_length + 1):
                # Crear la columna desplazada
                shifted_col = df_omni[col].shift(lag).astype('float32')
                shifted_col.name = f'{col}_{lag}'
                shifted_columns.append(shifted_col)

        # Concatenar todas las columnas desplazadas de una sola vez
        df_omni = pd.concat([df_omni] + shifted_columns, axis=1)
        df_omni.dropna(inplace=True)

        np_omni = df_omni.values

    else:
        sequence = [df_omni.iloc[i : i + shift_length].values for i in range(len(df_omni) - shift_length + 1)]
        np_omni = np.array(sequence, dtype=np.float32)
        np_index = np_index[shift_length-1:]

    return np_omni, np_index, df_epoch
        

###### [ DataTorch ] ######
class CustomDataset(Dataset):
    def __init__(self, omni, index, device):
        self.device = device
        
        self.x = torch.tensor(omni, dtype=torch.float32).to(self.device)
        self.y = torch.tensor(index, dtype=torch.float32).unsqueeze(1).to(self.device)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x, y
