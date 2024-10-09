import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm

# ----------      [   Before Model   ]      ---------- #

### [ Plot Time Data ] ###
def time_serie_plot(df, omni_param, auroral_param, auroral_historic_file, solar_historic_file, save_raw_file):
    """ 
    This code is responsible for graphing auroral indices and solar parameters.
    """
    storm_list = pd.read_csv(save_raw_file + 'storm_list.csv', header=None, names=['Epoch'])
    storm_list['Epoch'] = pd.to_datetime(storm_list['Epoch'])

    solar_len = len(omni_param)
    auroral_len = len(auroral_param)

    fig1, axs1 = plt.subplots(solar_len, 1, figsize=(12,15), sharex=True, layout='constrained') 
    fig2, axs2 = plt.subplots(auroral_len, 1, figsize=(12,6), sharex=True, layout='constrained')  

    for i, storm_date in enumerate(storm_list['Epoch']):
        start_time = storm_date - pd.Timedelta('24h')
        end_time = storm_date + pd.Timedelta('24h')

        fig1.suptitle(f'Solar Parameters from {start_time} to {end_time}', fontsize=18, fontweight='bold')
        fig2.suptitle(f'Auroral Index from {start_time} to {end_time}', fontsize=18, fontweight='bold')

        period_data = df[(df['Epoch'] >= start_time) & (df['Epoch'] <= end_time)]

        if solar_len == 1:
            axs1 = [axs1]
        if auroral_len == 1:
            axs2 = [axs2]
        
        for j, param in enumerate(omni_param):
            axs1[j].plot(period_data[param], color='teal', zorder=1, linewidth=1.5)

            if 'B' in param: 
                param = param.replace('_', ' ') + r' [nT]'
                if 'GSM' in param:
                    param = param.title().replace(' Gsm', ' GSM')
                if 'Total' in param:
                    param = param.replace(' Total', ' T')
            if 'proton' in param: 
                param = r'$\rho$ [#N/cm$^3$]'
            if 'V' in param: 
                param = param + r' [Km/s]'
            if 'T' in param and 'B' not in param: 
                param = param + r' [K]'
            if 'E_Field' in param: 
                param = 'E' + r' [mV/m]'
            if 'Pressure' in param: 
                param = 'P' + r' [nPa]'

            axs1[j].set_ylabel(f'{param}', labelpad = 30, fontsize=12, va='center')
            axs1[j].yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            axs1[j].yaxis.get_major_formatter().set_powerlimits((-3,4))
            axs1[j].tick_params(axis='y', length=7, width=2, colors='black',
                                grid_color='black', grid_alpha=0.4,
                                which='major', labelsize=15)
            axs1[j].set_xlim(min(period_data.index), max(period_data.index))
            axs1[j].grid(True)
        
        for j, param in enumerate(auroral_param):
            axs2[j].plot(period_data[param], color='teal', zorder=1, linewidth=1.5)
            param = param.replace('_INDEX', '').upper() + r' [nT]'
            axs2[j].set_ylabel(f'{param}', labelpad = 30, fontsize=12, va='center')
            axs2[j].yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            axs2[j].yaxis.get_major_formatter().set_powerlimits((-3,4))
            axs2[j].tick_params(axis='y', length=7, width=2, colors='black',
                                grid_color='black', grid_alpha=0.4,
                                which='major', labelsize=15)
            
            axs2[j].set_xlim(min(period_data.index), max(period_data.index))
            axs2[j].grid(True)

        axs1[-1].set_xlabel('Date', fontsize=15)
        axs2[-1].set_xlabel('Date', fontsize=15)
        axs1[-1].tick_params(axis='x', length=7, width=2, colors='black',
                                grid_color='black', grid_alpha=0.4,
                                which='major', labelsize=15)

        axs2[-1].tick_params(axis='x', length=7, width=2, colors='black',
                                grid_color='black', grid_alpha=0.4,
                                which='major', labelsize=15)
        
        plt.subplots_adjust(left=0.15)
        fig1.savefig(solar_historic_file + f'Omni_Parameters_{start_time}__{end_time}.png')
        fig2.savefig(auroral_historic_file + f'Auroral_electrojet_index_{start_time}_{end_time}.png')
        plt.close(fig1)
        plt.close(fig2)


### [ Plot Correlation Map ] ###
def correlation_plot(df, correlation, stad_file):
    matrix = round(df.corr(method=correlation), 2)
    column_mapping = {}

    plt.figure(figsize=(14,12))
    plt.title(f'{correlation.title()} Correlation Map', fontsize=18, fontweight='bold')
    heatmap = plt.pcolor(matrix, cmap='BrBG', vmin=-1, vmax=1, edgecolors='white', linewidth=1)

    
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            color = 'white' if matrix.iloc[i, j] >= 0.75 or matrix.iloc[i, j] <= -0.75 else 'black'
            plt.text(j + 0.5, i + 0.5, f'{matrix.iloc[i, j]:.2f}',
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=11, color=color, fontweight='bold')
            
    cbar = plt.colorbar(heatmap)
    cbar.set_label('Correlation', rotation=-90, labelpad=25, fontsize=16, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)

    for cols in matrix.columns:
        if 'INDEX' in cols:
            column_mapping[cols] = cols.replace('_INDEX', '')
        if 'GSM' in cols:
            column_mapping[cols] = cols.title().replace('_Gsm', ' GSM')
        if 'GSE' in cols:
            column_mapping[cols] = cols.title().replace('_Gse', 'GSE')
        if 'Total' in cols:
            column_mapping[cols] = cols.replace('_Total', 't')
        if 'proton' in cols:
            column_mapping[cols] = r'$\rho$'
        if 'Pressure' in cols:
            column_mapping[cols] = 'P'
        if 'E_Field' in cols:
            column_mapping[cols] = 'E'
        if 'Beta' in cols:
            column_mapping[cols] = r'$\beta$'
        if 'flow' in cols:
            column_mapping[cols] = 'Q'
        
    matrix.rename(columns=column_mapping, index=column_mapping, inplace=True)

    plt.xticks(np.arange(0.5, len(matrix.columns), 1), matrix.columns, fontsize=12, rotation=45, ha='right', fontweight='bold')
    plt.yticks(np.arange(0.5, len(matrix.index), 1), matrix.index, fontsize=12, fontweight='bold') 

    plt.tight_layout()

    plt.savefig(stad_file + f'Correlation_Map_{correlation}.png')
    plt.close()



# ----------      [   After Model   ]      ---------- #
### [ Plot Metric ] ###
def metric_plot(metrics_train_val, metric_train_file, shift, auroral_index, type_model):
    metric_list = ['Loss', 'R_Score', 'RMSE', 'ExpVar']

    for metric in metric_list:
        plt.figure(figsize=(12,10))
        plt.title(f'{metric} Plot {auroral_index.replace("_INDEX", " Index")} using {type_model} (Shifty = {shift})', fontsize=20)

        for cols in metrics_train_val:
            if metric in cols and ('Train' in cols or 'Valid' in cols):
                metric_value = metrics_train_val[cols].values
                color = 'teal' if 'Train' in cols else 'red'
                label_name = 'Train' if 'Train' in cols else 'Valid'
                plt.plot(metric_value, label=f'{label_name} {metric}', color=color, linewidth=2)
                plt.tick_params(axis='y', length=7, width=2, colors='black',
                                grid_color='black', grid_alpha=0.4,
                                which='major', labelsize=12)
                plt.tick_params(axis='x', length=7, width=2, colors='black',
                                grid_color='black', grid_alpha=0.4,
                                which='major', labelsize=12)
        
        plt.xlabel(f'Epoch', fontsize=15)
        plt.ylabel(f'{metric}', fontsize=15)
        plt.grid(True)
        plt.legend(fontsize=15)

        plt.savefig(metric_train_file + f'{metric}_Plot_{auroral_index.replace("_INDEX", "_Index")}_using_{type_model}_and_Shifty_{shift}.png')
        plt.close()


### [ Plot Metric Shift ] ###
def metric_shift_plot(metric_train_shift, metric_train_shifty_file, auroral_index, type_model):
    metric_list = ['Loss', 'R_Score', 'RMSE', 'ExpVar']
    process_list = ['Train', 'Valid']

    for metric in metric_list:
        for process in process_list:
            plt.figure(figsize=(12,10))
            plt.title(f'{metric} {process} Comparison Shift Plot {auroral_index.replace("_INDEX", " Index")} using {type_model}', fontsize=20)
            for cols in metric_train_shift.columns:
                if metric in cols and process in cols:
                    shift_label = cols.replace(f'{process}_{metric}_', ' Shift: ')
                    plt.plot(metric_train_shift[cols], label=shift_label, linewidth=2)
                    plt.tick_params(axis='x', length=7, width=2, colors='black',
                        grid_color='black', grid_alpha=0.4,
                        which='major', labelsize=15)

                    plt.tick_params(axis='y', length=7, width=2, colors='black',
                        grid_color='black', grid_alpha=0.4,
                        which='major', labelsize=15)
            

            plt.xlabel('Epoch', fontsize=20)
            plt.ylabel(f'{metric}', fontsize=20)
            plt.grid(True)
            plt.legend(fontsize=15)

            plt.savefig(metric_train_shifty_file + f"{metric}_{process}_Comparison_Plot_{auroral_index.replace('_INDEX', '_Index')}_using_{type_model}.png")
            plt.close()      


### [ Plot Density ] ###
def density_plot(results_df, shift, metrics_test, density_file, auroral_index, type_model):
    r_score = metrics_test['Test_R_Score'].values[0]
    k = int(0.1 * np.sqrt(len(results_df)))

    np_log_pred = np.log(np.abs(results_df[f'Test_Pred_{shift}'].values) + 1e-10)  
    np_log_real = np.log(np.abs(results_df['Test_Real'].values) + 1e-10)  

    p2 = min(min(np_log_pred), min(np_log_real))
    p1 = max(max(np_log_pred), max(np_log_real))

    plt.figure(figsize=(15, 12))  
    plt.title(f'Density Plot {auroral_index.replace("_INDEX", " Index")} using {type_model}', fontsize=20)

    hist = plt.hist2d(np_log_real, np_log_pred, bins=k, norm=LogNorm())
    
    plt.plot([p2, p1], [p2, p1], color='black', label=f'R = {r_score:.2f}', linewidth=2.5)

    cbar = plt.colorbar(hist[3], ax=plt.gca())  
    cbar.set_label('Density', rotation=-90, labelpad=25, fontsize=20, fontweight='bold')
    cbar.ax.tick_params(labelsize=15)

    plt.xlabel('Real Value (Log)', fontsize=20)
    plt.ylabel('Pred Value (Log)', fontsize=20)

    plt.tick_params(axis='y', length=7, width=2, colors='black',
                    grid_color='black', grid_alpha=0.4,
                    which='major', labelsize=15)
    plt.tick_params(axis='x', length=7, width=2, colors='black',
                    grid_color='black', grid_alpha=0.4,
                    which='major', labelsize=15)

    plt.legend(loc='best', fontsize=20)
    plt.grid(True)

    plt.savefig(f'{density_file}Plot_Density_{type_model}_{auroral_index}_Shifty_{shift}.png')
    plt.close()


### [ Metric Test Plot ] ###
def test_shift_plot(metrics_test, shifty_set, metric_test_shifty_file, auroral_index, type_model):
    """
    Plots and compares different test metrics across varying shifts (delays).
    
    Args:
        metrics_test (pd.DataFrame): DataFrame containing the test metrics for different shift lengths.
        shifty_set (list or array): The list of shift (delay) values.
        plot_shift_metric_test_file (str): The file path where plots will be saved.
        auroral_index (str): The auroral index being used (for labeling).
        type_model (str): The type of model being used (for labeling).
    """
    metric_list = ['Loss', 'R_Score', 'RMSE', 'ExpVar']   

    for metric in metric_list:
        plt.figure(figsize=(10, 6)) 
        plt.title(f'{metric} Compared Plot {auroral_index.replace("_INDEX", " Index")} using {type_model}', fontsize=20) 
        
        for cols in metrics_test.columns:
            if metric in cols:
                metric_value = metrics_test[cols].values

                if len(shifty_set) == len(metric_value):
                    plt.plot(shifty_set, metric_value, marker='o', color='teal', linewidth=2)
                else:
                    print(f"Longitud de shifty_set {len(shifty_set)} no coincide con longitud de {cols}: {len(metric_value)}")

        plt.ylabel(f'{metric}', fontsize=20) 
        plt.xlabel('Shift', fontsize=20)
        plt.grid(True)
        plt.tick_params(axis='x', length=7, width=3, colors='black',
                        grid_color='black', grid_alpha=0.4,
                        which='major', labelsize=15)

        plt.tick_params(axis='y', length=7, width=3, colors='black',
                        grid_color='black', grid_alpha=0.4,
                        which='major', labelsize=15)
        
        plt.savefig(metric_test_shifty_file + f'{metric}_Compared_Test.png')
        plt.close()


### [ Historic test Plot ] ###
def historic_shift_plot(metric_test_history, save_raw_file, historic_file, auroral_index, type_model):
    storm_list = pd.read_csv(save_raw_file + 'storm_list.csv', header=None, names=['Epoch'])    
    storm_list['Epoch'] = pd.to_datetime(storm_list['Epoch'])

    result_epoch = pd.to_datetime(metric_test_history['Epoch'])
    result_min = result_epoch.min()
    result_min = result_min.strftime('%Y/%m/%d')
    storm_list = storm_list[storm_list['Epoch'] >= result_min]

    for i, storm_date in enumerate(storm_list['Epoch']):
        start_time = storm_date - pd.Timedelta('24h')
        end_time = storm_date + pd.Timedelta('24h')

        # Filtrar datos para el periodo actual
        period_data = metric_test_history[(metric_test_history['Epoch'] >= start_time) & (metric_test_history['Epoch'] <= end_time)]
        period_data.index = period_data['Epoch']

        plt.figure(figsize=(20, 10))
        plt.title(f'Comparison Real vs Prediction ({type_model}) from {start_time} to {end_time}', fontsize=20)

        # Graficar la columna de valores reales
        plt.plot(period_data['Test_Real'], color='black', label='Real', zorder=1)

        # Graficar columnas de predicción
        for col in period_data.columns:
            if 'Test_Pred_' in col:  # Asegúrate de que este sea el prefijo correcto
                label_name = f'Shift: {col.replace("Test_Pred_", "")}'
                plt.plot(period_data[col], label=label_name, zorder=1)

        plt.ylabel(f'{auroral_index.replace("_INDEX", "")} [nT]', fontsize=15)
        plt.xlabel('Date', fontsize=15)

        plt.grid(True)

        plt.tick_params(axis='x', length=7, width=3, colors='black',
                        grid_color='black', grid_alpha=0.4,
                        which='major', labelsize=15)
        plt.tick_params(axis='y', length=7, width=3, colors='black',
                        grid_color='black', grid_alpha=0.4,
                        which='major', labelsize=15)
        plt.legend(fontsize=15)

        # Guardar la figura
        plt.savefig(f'{historic_file}Auroral_electrojet_index_{start_time.strftime("%Y%m%d")}_{end_time.strftime("%Y%m%d")}.png')
        plt.close() 
