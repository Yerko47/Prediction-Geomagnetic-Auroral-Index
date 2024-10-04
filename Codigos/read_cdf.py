import os
import cdflib
import numpy as np
import pandas as pd

####### [ Read CDF ] #######
def cdf_read(file):
    """
    This function reads a CDF (Common Data Format) file and processes its data into a pandas DataFrame with appropriate formatting and column renaming.

    Steps:
    1. Uses the cdflib library to open and read the CDF file.
    2. Extracts the data for each variable in the CDF and creates a dictionary where the keys are the variable names.
    3. Converts the dictionary into a pandas DataFrame, transforming the 'Epoch' variable into a datetime format.
    4. Renames specific DataFrame columns for clarity ('E' to 'E_Field' and 'F' to 'B_Total').

    Parameters:
    file: str
        The file path to the CDF file to be read.

    Returns:
    pandas.DataFrame
        A DataFrame containing the CDF data, with the 'Epoch' column in datetime format and selected columns renamed.
    """
    
    cdf = cdflib.CDF(file)

    cdf_dict = {}
    info = cdf.cdf_info()
    for key in info.zVariables:
        cdf_dict[key] = cdf[key]

    cdf_df = pd.DataFrame(cdf_dict)
    if 'Epoch' in cdf_df.columns:
        cdf_df['Epoch'] = pd.to_datetime(cdflib.cdfepoch.encode(cdf_df['Epoch'].values))

    
    cdf_df.rename(columns={'E': 'E_Field', 'F': 'B_Total'}, inplace=True)

    return cdf_df


####### [ Data Cleaning ] #######
def bad_data(df, save_raw_file):
    """
    This function removes erroneous values from a DataFrame, which arise from sensor saturation during measurements.

    Steps:
        1. Identifies the maximum value for each column, based on data documentation (ignoring 'Epoch' or datetime columns).
        2. Replaces values that are equal to or exceed the identified maximum with NaN values.
        3. Counts the number of NaN values (erroneous values) for each column and saves this count to a CSV file for future graphing purposes.

    Parameters:
        df: pandas.DataFrame
            The DataFrame containing the data to be processed.
        save_raw_file: str
            The directory or path where the CSV file of NaN value counts will be saved.

    Returns:
        pandas.DataFrame
            The modified DataFrame with erroneous values replaced by NaN.
    """

    list_max = []
 
    for col in df.columns:
        
        if col == 'Epoch' or df[col].dtype == 'datetime64[ns]':
            continue
        
        max_val = df[col].max()
        val_max = int(max_val * 100) / 100
                
        df.loc[df[col] >= val_max, col] = np.nan

        count = df[col].isna().sum()
        list_max.append({'param': col, 'count': count})

    
    nan_df = pd.DataFrame(list_max)
    nan_df.to_csv(save_raw_file + 'count_nan.csv', index=False)

    return df


####### [ DataSet Building ] #######
def dataset(in_year, out_year, omni_file, save_feather, save_raw_file, processOMNI):
    
    """
This code is responsible for creating a database by applying previously defined functions to process and clean data.

Steps:
    1. Define the start and end dates, and create a date range array based on these dates.
    2. Check if the dataset file already exists. If it doesn't, proceed with data processing.
    3. Iterate over the date range to apply the 'cdf_read' function, which reads multiple files, and concatenate the resulting data arrays.
    4. Apply filters to clean the data and use the 'bad_data' function to remove erroneous values.
    5. Perform linear interpolation on the data parameters, replacing missing values using the 'bfill' method.

Parameters:
    in_year: int
        The starting year of the dataset.
    out_year: int
        The ending year of the dataset.
    save_feather: str
        The path where the processed dataset will be saved in Feather format.
    processOMNI: bool
        A flag to determine whether to process OMNI data.
    omni_file: str
        The directory path where the OMNI data files are located.
    save_raw_file: str
        The path where the count of NaN values will be saved in CSV format.

Returns:
    None
        This script saves the processed dataset to the specified Feather file path.
"""
    
    start_time = pd.Timestamp(in_year, 1, 1)
    end_time = pd.Timestamp(out_year, 3, 1)

    if not os.path.exists(save_feather):

        if processOMNI:
            print()
            print(f"Processing omni data. From {start_time} to {end_time.strftime('%Y%m%d 23:59:00')}")
            print(':::::::::::::::::::::::::::::::::::::::::::::::::::::')
            print()

        date_array = pd.date_range(start=start_time, end=end_time, freq='MS')
        o = []

        for date in date_array:
            name_file = f"omni_hro_1min_{date.strftime('%Y%m%d')}_v01.cdf"
            cdf = cdf_read(omni_file + f'{date.year}/{name_file}')
            print(f'The file {name_file} is loading')
            o.append(cdf)

        data = pd.concat(o, axis=0, ignore_index=True)

        data.index = data.Epoch
        data.drop(columns = ['YR', 'Day', 'HR', 'Minute', 'IMF', 'PLS', 'IMF_PTS', 'PLS_PTS',
                            'percent_interp', 'Timeshift', 'RMS_Timeshift', 'RMS_phase', 'Time_btwn_obs', 
                            'RMS_SD_B', 'RMS_SD_fld_vec','Mach_num', 'Mgs_mach_num', 'BSN_x', 'BSN_y',
                            'BSN_z', 'SYM_D', 'SYM_H', 'ASY_D', 'ASY_H', 'PC_N_INDEX','x','y','z'], inplace=True)
        data = bad_data(data, save_raw_file)

        for cols in data.columns:
            data[cols] = data[cols].interpolate(method='linear', limit=None)
            data[cols] = data[cols].fillna(method='bfill')

        data.reset_index(drop=True).to_feather(save_feather)
        print(f' ¡¡ Lets Go !! the {len(data)} data has been successfully saved to {save_feather}')

    else:
        print(f' The file already exists in {save_feather}')


####### [ Check Folder ] #######
def check_folder(file):
    if not os.path.exists(file):
        os.makedirs(file)


####### [ Check Shift Folder ] #######
def create_shift_folder(file, shift):
    if not os.path.exists(file + f'{shift}'):
        check_folder(file + f'{shift}')