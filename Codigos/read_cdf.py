import os
import cdflib
import numpy as np
import pandas as pd

####### [ Read CDF ] #######
def cdf_read(file):
    """
    This function reads the cdf file and identifies it with the name tag.
        1.- Using cdflib to identify and read the file
        2.- Identify the information in the CDF and place the labels for each column
        3.- Build a Dataframe using the previous information and the 'Epoch' tag transforms the format to datetime.
        4.- Some names of the columns of this DF are changed
    """
    # 1.- Using cdflib to identify and read the file
    cdf = cdflib.CDF(file)

    # 2.- Identify the information in the CDF and place the labels for each column
    cdf_dict = {}
    info = cdf.cdf_info()
    for key in info.zVariables:
        cdf_dict[key] = cdf[key]

    # 3.- Build a Dataframe using the previous information and the 'Epoch' tag transforms the format to datetime.
    cdf_df = pd.DataFrame(cdf_dict)
    if 'Epoch' in cdf_df.columns:
        cdf_df['Epoch'] = pd.to_datetime(cdflib.cdfepoch.encode(cdf_df['Epoch'].values))

    # 4.- Some names of the columns of this DF are changed
    cdf_df.rename(columns={'E': 'E_Field', 'F': 'B_Total'}, inplace=True)

    return cdf_df


####### [ Data Cleaning ] #######
def bad_data(df, processing_file):
    """
    This function eliminates bad values, due to erroneous measurements due to saturation of the sensors when measuring.
        1.- Identify the maximum value of each label (the maximum value is due to the data documentation)
        2.- Replace these maximum values with NaN values
        3.- Count the number of erroneous values for future graphs and save in csv
    """
    list_max = []

    # 1.- Identify the maximum value of each label (the maximum value is due to the data documentation)
    for col in df.columns:
        
        if col == 'Epoch' or df[col].dtype == 'datetime64[ns]':
            continue
        
        max_val = df[col].max()
        val_max = int(max_val * 100) / 100
        
        # 2.- Replace these maximum values with NaN values
        df.loc[df[col] >= val_max, col] = np.nan

        # 3.- Count the number of erroneous values for future graphs and save in csv
        count = df[col].isna().sum()
        list_max.append({'param': col, 'count': count})

    
    nan_df = pd.DataFrame(list_max)
    nan_df.to_csv(processing_file + 'count_nan.csv', index=False)

    return df


####### [ DataSet Building ] #######
def dataset(in_year, out_year, omni_file, save_feather, processing_file, processOMNI):
    """
    This code is responsible for creating a database by applying the previous functions.
        1.- The start and end dates will be defined, and an array will be created that contains the range of these dates.
        2.- The existence of the dataset file is checked. If it does not exist, the code will be crossed
        3.- It will be iterated over this range to apply the 'cdf_read' function for the multiple files to be read and the created arrays will be concatenated
        4.- Filters will be applied to the data and the 'bad_data' function will be used to clean the dataset
        5.- A linear interpolation method will be applied between the parameters and some values will be substituted using the bfill method
    """
    # 1.- The start and end dates will be defined, and an array will be created that contains the range of these dates.
    start_time = pd.Timestamp(in_year, 1, 1)
    end_time = pd.Timestamp(out_year, 3, 1)

    # 2.- The existence of the dataset file is checked. If it does not exist, the code will be crossed
    if not os.path.exists(save_feather):

        if processOMNI:
            print()
            print(f"Processing omni data. From {start_time} to {end_time.strftime('%Y%m%d 23:59:00')}")
            print(':::::::::::::::::::::::::::::::::::::::::::::::::::::')
            print()

        # 3.- It will be iterated over this range to apply the 'cdf_read' function for the multiple files to be read and the created arrays will be     concatenated
        date_array = pd.date_range(start=start_time, end=end_time, freq='MS')
        o = []

        for date in date_array:
            name_file = f"omni_hro_1min_{date.strftime('%Y%m%d')}_v01.cdf"
            cdf = cdf_read(omni_file + f'{date.year}/{name_file}')
            print(f'The file {name_file} is loading')
            o.append(cdf)

        data = pd.concat(o, axis=0, ignore_index=True)

        # 4.- Filters will be applied to the data and the 'bad_data' function will be used to clean the dataset
        data.index = data.Epoch
        data.drop(columns = ['YR', 'Day', 'HR', 'Minute', 'IMF', 'PLS', 'IMF_PTS', 'PLS_PTS',
                            'percent_interp', 'Timeshift', 'RMS_Timeshift', 'RMS_phase', 'Time_btwn_obs', 
                            'RMS_SD_B', 'RMS_SD_fld_vec','Mach_num', 'Mgs_mach_num', 'BSN_x', 'BSN_y',
                            'BSN_z', 'SYM_D', 'SYM_H', 'ASY_D', 'ASY_H', 'PC_N_INDEX','x','y','z'], inplace=True)
        data = bad_data(data, processing_file)

        # 5.- A linear interpolation method will be applied between the parameters and some values ??will be substituted using the bfill method
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
        print(f' The file has been created')
    else:
        print(f' The file exists')