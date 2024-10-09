import time 
import os
from datetime import timedelta
import numpy as np
import pandas as pd
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


####### [ Check Folder ] #######
def check_folder(file):
    if not os.path.exists(file):
        os.makedirs(file)


####### [ Epoch Storm Selection ] #######
def epoch_storm(df, save_raw_file):
    """
    This function filters data based on a list of storm events and selects a time window of 48 hours around each storm for further analysis.

    Steps:
        1. Converts the 'Epoch' column of the input DataFrame `df` to datetime format and sets it as the index.
        2. Reads the list of storm events from a CSV file named 'storm_list.csv', converting the 'Epoch' column of the storm list to datetime format.
        3. Initializes an empty DataFrame to store the filtered storm data.
        4. For each storm event, extracts a 48-hour time window (24 hours before and 24 hours after the storm event).
        5. Concatenates the selected data from the input DataFrame into a new DataFrame.
        6. Finally, the concatenated DataFrame is returned with the index reset.

    Parameters:
        df: pandas.DataFrame
            The DataFrame containing the original dataset with an 'Epoch' column representing the timestamps.
        save_raw_file: str
            The path to the directory where the 'storm_list.csv' file is located, which contains the list of storm events.

    Returns:
        pandas.DataFrame
            A new DataFrame containing the data for the 48-hour period around each storm event.
    """
    df['Epoch'] = pd.to_datetime(df['Epoch'])
    df.set_index('Epoch', inplace=True)
    
    storm_list = pd.read_csv(save_raw_file + 'storm_list.csv', header=None, names=['Epoch'])
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


####### [ Scaler Data ] #######
def scaler_df(df, scaler, omni_param, auroral_param):
    """
    This function applies a specified scaling method to the OMNI parameters in the DataFrame.

    Steps:
        1. Extracts the OMNI parameters and the 'Epoch' column from the input DataFrame `df`.
        2. Checks the specified scaling method and initializes the corresponding scaler:
            - 'robust': Uses the RobustScaler to scale the data based on median and interquartile range.
            - 'standard': Uses the StandardScaler to standardize features by removing the mean and scaling to unit variance.
            - 'minmax': Uses the MinMaxScaler to transform features to a given range, usually between 0 and 1.
        3. If the scaler type is not recognized, a ValueError is raised.
        4. Applies the selected scaler to the OMNI parameters and creates a new DataFrame with the scaled values, preserving the original index.
        5. Concatenates the 'Epoch' column, the scaled OMNI parameters, and the original auroral parameters into a single DataFrame.
        6. Returns the new DataFrame containing the scaled OMNI parameters along with the original 'Epoch' and auroral parameters.

    Parameters:
        df: pandas.DataFrame
            The DataFrame containing the dataset, including 'Epoch', OMNI parameters, and auroral parameters.
        scaler: str
            The scaling method to apply; should be one of 'robust', 'standard', or 'minmax'.
        omni_param: list
            A list of column names representing the OMNI parameters to be scaled.
        auroral_param: list
            A list of column names representing the auroral parameters to be retained.

    Returns:
        pandas.DataFrame
            A new DataFrame with the scaled OMNI parameters, 'Epoch', and auroral parameters.
    """
    df_omni = df[omni_param]
    df_epoch = df[['Epoch']]
    df_auroral = df[auroral_param]

    if scaler == 'robust': 
        scaler = RobustScaler()
    elif scaler == 'standard': 
        scaler = StandardScaler()
    elif scaler == 'minmax': 
        scaler = MinMaxScaler()
    else: 
        raise ValueError('Scaler must be "robust", "standard" or "minmax" ')

    df_omni_scaled = scaler.fit_transform(df_omni)
    df_omni_scaled = pd.DataFrame(df_omni_scaled, columns=omni_param, index=df_omni.index)

    df = pd.concat([df_epoch, df_omni_scaled, df_auroral], axis=1)

    return df


####### [ Create set prediction ] #######
def create_set_prediction(df, set_split, test_size, val_size):
    """
    This function creates the Train/Val/Test set for the prediction.
    """
    match set_split:
        case 'organized':
            n = len(df)
            test_index = int(n * (1 - test_size))
            val_index = int(test_index * (1 - val_size))
            train_df = df[:val_index].copy()
            val_df = df[val_index:test_index].copy()
            test_df = df[test_index:].copy()

            train_df.reset_index(inplace=True, drop=True)
            val_df.reset_index(inplace=True, drop=True)
            test_df.reset_index(inplace=True, drop=True)
            
        case 'random':
            train_val_df, test_df = train_test_split(df, test_size=test_size, shuffle=True)
            train_df, val_df = train_test_split(train_val_df, test_size=val_size, shuffle=True)

        case _:
            raise ValueError('Set_split must be "organized" or "random"')
        
    train_len = round(len(train_df) / len(df), 2) * 100
    val_len = round(len(val_df) / len(df), 2) * 100
    test_len = round(len(test_df) / len(df), 2) * 100

    print('\n---------- [ Percentage Set ] ----------\n')
    print(f'Percentage Train Set: {train_len}%')
    print(f'Percentage Valid Set: {val_len}%')
    print(f'Percentage Test Set: {test_len}%\n')

    
    return train_df, val_df, test_df


####### [ Shift ] #######
def shifty(df, omni_param, auroral_index, shift_length, type_model, group):
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

    if group == 'test':
        return np_omni, np_index, df_epoch   
    else:
        return np_omni, np_index


####### [ DataTorch ] #######
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

