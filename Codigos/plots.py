import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

#######################################################
############## [     Before Model      ] ##############
###### [ Plot Time serie ] ######
def time_serie_plot(df, omni_param, auroral_param, in_year, out_year, save_raw_file, omni_serie_file, auroral_serie_file):
    """
    Reduced version of time series plot.
    """
    storm_list = pd.read_csv(save_raw_file + 'storm_list.csv', header=None, names=['Epoch'])
    storm_list['Epoch'] = pd.to_datetime(storm_list['Epoch'])

    omni_len = len(omni_param)
    auroral_len = len(auroral_param)

    fig1, axs1 = plt.subplots(omni_len, 1, figsize=(12,12), sharex=True)  # Reduced size
    fig2, axs2 = plt.subplots(auroral_len, 1, figsize=(12,6), sharex=True)  # Reduced size

    fig1.subplots_adjust(hspace=0)
    fig2.subplots_adjust(hspace=0)

    for i, storm_date in enumerate(storm_list['Epoch']):
        start_time = storm_date - pd.Timedelta('24h')
        end_time = storm_date + pd.Timedelta('24h')

        fig1.suptitle(f'OMNI Parameters from {start_time} to {end_time}', y=0.91, fontsize=18)
        fig2.suptitle(f'Auroral Index from {start_time} to {end_time}', y=0.91, fontsize=18)

        period_data = df[(df['Epoch'] >= start_time) & (df['Epoch'] <= end_time)]

        if omni_len == 1:
            axs1 = [axs1]
        if auroral_len == 1:
            axs2 = [axs2]

        period_data.index = period_data['Epoch']

        for j, param in enumerate(omni_param):
            axs1[j].plot(period_data[param], color='teal', zorder=1, linewidth=1.5)  # Slightly thinner line
            axs1[j].axhline(0, color='red', zorder=2, linewidth=1.2, linestyle='--')

            axs1[j].set_ylabel(f'{param}', labelpad = 15, fontsize=12)
            axs1[j].tick_params(length=7, width=2, colors='black',
                                grid_color='black', grid_alpha=0.4,
                                axis='both', which='major', labelsize=10)
            
            axs1[j].set_xlim(min(period_data.index), max(period_data.index))
            axs1[j].grid(True)

        for j, param in enumerate(auroral_param):
            axs2[j].plot(period_data[param], color='teal', zorder=1, linewidth=1.5)  # Slightly thinner line
            axs2[j].axhline(0, color='red', zorder=2, linewidth=1.2, linestyle='--')

            axs2[j].set_ylabel(f'{param}', labelpad=15, fontsize=12)
            axs2[j].tick_params(length=7, width=2, colors='black',
                                grid_color='black', grid_alpha=0.4,
                                axis='both', which='major', labelsize=10)
            
            axs2[j].set_xlim(min(period_data.index), max(period_data.index))
            axs2[j].grid(True)


        axs1[-1].set_xlabel('Date', fontsize=12)
        axs2[-1].set_xlabel('Date', fontsize=12)

        fig1.savefig(omni_serie_file + f'Omni_Parameters_{start_time}__{end_time}.png')
        fig2.savefig(auroral_serie_file + f'Auroral_electrojet_index_{start_time}_{end_time}.png')
        plt.close(fig1)
        plt.close(fig2)


###### [ Correlation Plot ] ######
def corr_plot(df, correlation, stadistic_file):
    matrix = round(df.corr(method=correlation), 2)
    
    plt.figure(figsize=(12, 10))  # Reduced size
    plt.title(f'Correlation Map ({correlation.title()})', y=1.02, fontsize=18)
    plt.pcolor(matrix, cmap='RdBu', vmin=-1, vmax=1)

    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            color = 'white' if matrix.iloc[i, j] >= 0.8 or matrix.iloc[i, j] <= -0.8 else 'black'
            plt.text(j + 0.5, i + 0.5, f'{matrix.iloc[i, j]:.2f}',
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=10,
                     color=color)

    cbar = plt.colorbar()
    cbar.set_label('Correlation', rotation=-90, labelpad=25, fontsize=16)
    cbar.ax.tick_params(labelsize=11)

    plt.xticks(np.arange(0.5, len(matrix.columns), 1), matrix.columns, fontsize=12, rotation=35, ha='right')
    plt.yticks(np.arange(0.5, len(matrix.index), 1), matrix.index, fontsize=12)

    plt.savefig(stadistic_file + f'Correlation_Map_{correlation}.png')
    plt.close()





######################################################
############## [     After Model      ] ##############

######################################################
######################################################

############ [     Train/Val Model     ] ############

###### [ Plot Metric ] ######
def plot_metric(metrics_train_val, plot_metric_training_file, shift, auroral_index, type_model):
    metric_list = ['Loss', 'R2', 'RMSE', 'ExpVar']
    
    for metric in metric_list:
        plt.figure(figsize=(10, 6))  # Tamaño reducido
        plt.title(f'{metric} Plot {auroral_index.replace("_INDEX", " Index")} using {type_model} and Shifty {shift}', fontsize=20)  # Fuente ajustada
        for cols in metrics_train_val:
            if metric in cols:
                metric_value = metrics_train_val[cols].values
                color = 'teal' if 'Train' in cols else 'red'
                label_name = 'Train' if 'Train' in cols else 'Valid'
                plt.plot(metric_value, label=f'{label_name} {metric}', color=color, linewidth=2)

        plt.xlabel('Epoch', fontsize=20)  # Fuente ajustada
        plt.ylabel(f'{metric}', fontsize=20)  # Fuente ajustada
        plt.grid(True)
        plt.legend(fontsize=15)  # Fuente ajustada
        plt.tick_params(length=7, width=2, colors='black', grid_color='black', grid_alpha=0.4, axis='both', which='major', labelsize=10)  # Fuente ajustada

        plt.savefig(plot_metric_training_file + f'{shift}/{metric}_Plot_{auroral_index.replace("_INDEX", "_Index")}_using_{type_model}_and_Shifty_{shift}.png')
        plt.clf()
    plt.close()


############ [     Test Model     ] ############

###### [ Plot Time Model ] ######
def plot_time_model(results_df, save_raw_file, plot_time_serie_test_file, auroral_index, shift, len_test):
    """
    Reduced version of plot_time_model.
    """
    storm_list = pd.read_csv(save_raw_file + 'storm_list.csv', header=None, names=['Epoch'])    
    storm_list['Epoch'] = pd.to_datetime(storm_list['Epoch'])

    result_epoch = pd.to_datetime(results_df['Epoch'])
    result_min = result_epoch.min()
    result_min = result_min.strftime('%Y/%m/%d')

    storm_list =storm_list[storm_list['Epoch']>=result_min]
    
    for i, storm_date in enumerate(storm_list['Epoch']):
        start_time = storm_date - pd.Timedelta('24h')
        end_time = storm_date + pd.Timedelta('24h')

        period_data = results_df[(results_df['Epoch'] >= start_time) & (results_df['Epoch'] <= end_time)]
        period_data.index = period_data['Epoch']

        plt.figure(figsize=(12, 6))  # Reduced size
        plt.title(f'Real vs Prediction from {start_time} to {end_time}', fontsize=16)

        plt.plot(period_data['Test_Real'], label='Real', color='teal', zorder=1, linewidth=1.5)  # Slightly thinner line
        plt.plot(period_data['Test_Pred'], label='Pred', color='red', zorder=1, linewidth=1.5)  # Slightly thinner line

        plt.ylabel(f'{auroral_index.replace("_INDEX", "_Index")}', labelpad=15, fontsize=12)
        plt.xlabel('Date', fontsize=12)

        plt.tick_params(length=7, width=2, colors='black', grid_color='black', grid_alpha=0.4, axis='both', which='major', labelsize=10)
        plt.grid(True)

        plt.legend(loc='upper right', fontsize=12)

        plt.savefig(plot_time_serie_test_file + f'{shift}/Real_vs_Pred_{start_time.strftime("%Y-%m-%d_%H-%M")}_{end_time.strftime("%Y-%m-%d_%H-%M")}.png')
        plt.close()

##### [ Plot Compared Test Metrics ]
def plot_compared_test_metrics(metrics_test, shifty_set, plot_shift_metric_test_file, auroral_index, type_model):
    """
    Plots and compares different test metrics across varying shifts (delays).
    
    Args:
        metrics_test (pd.DataFrame): DataFrame containing the test metrics for different shift lengths.
        shifty_set (list or array): The list of shift (delay) values.
        plot_shift_metric_test_file (str): The file path where plots will be saved.
        auroral_index (str): The auroral index being used (for labeling).
        type_model (str): The type of model being used (for labeling).
    """
    metric_list = ['Loss', 'R2', 'RMSE', 'ExpVar']

    for metric in metric_list:
        plt.figure(figsize=(10, 6))  # Tamaño reducido
        
        for cols in metrics_test.columns:
            if metric in cols:
                metric_value = metrics_test[cols].values

                # Asegúrate de que la longitud de shifty_set coincida con metric_value
                if len(shifty_set) == len(metric_value):
                    plt.plot(shifty_set, metric_value, marker='o', color='teal', linewidth=2)
                else:
                    print(f"Longitud de shifty_set {len(shifty_set)} no coincide con longitud de {cols}: {len(metric_value)}")

        plt.title(f'{metric} Compared Plot {auroral_index.replace("_INDEX", " Index")} using {type_model}', fontsize=20)  # Fuente ajustada
        plt.ylabel(f'{metric}', fontsize=20)  # Fuente ajustada
        plt.xlabel('Shift', fontsize=20)  # Fuente ajustada
        plt.grid(True)
        plt.tick_params(length=7, width=2, colors='black', grid_color='black', grid_alpha=0.4, axis='both', which='major', labelsize=10)  # Fuente ajustada

        plt.savefig(plot_shift_metric_test_file + f'{metric}_Compared_Test.png')
        plt.close()


##### [ Plot Density Test ] #####
def density_plot(results_df, plot_density_file, metrics_test, auroral_index, type_model, shift):
    r_score = np.sqrt(metrics_test['Test_R2'].values[0])
    k = int(0.1*np.sqrt(len(results_df)))

    np_log_pred = np.log(np.abs(results_df['Test_Pred'].values))
    np_log_real = np.log(np.abs(results_df['Test_Real'].values))

    p2 = min(min(np_log_pred), min(np_log_real))
    p1 = max(max(np_log_pred), max(np_log_real))

    plt.figure(figsize=(10, 6))  # Tamaño reducido
    plt.title(f'Density Plot {auroral_index.replace("_INDEX", " Index")} using {type_model}', fontsize=20)  # Fuente ajustada

    plt.hist2d(np_log_real, np_log_pred, bins=k, norm=LogNorm())
    plt.plot([p2,p1], [p2,p1], color='black', label=f'R = {r_score:.2f}')
    cbar = plt.colorbar()
    cbar.set_label('Density', rotation=-90, labelpad=25, fontsize=15)  # Fuente ajustada

    plt.xlabel('Real Value (Log)', fontsize=20)  # Fuente ajustada
    plt.ylabel('Pred Value (Log)', fontsize=20)  # Fuente ajustada
    plt.legend(loc='best', fontsize=15)  # Fuente ajustada

    plt.grid(True)

    plt.savefig(f'{plot_density_file}Plot_Density_{type_model}_{auroral_index}_Shifty_{shift}.png')
    plt.close()
