import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

###### [ Time serie Plot ] ######
def time_serie_plot(df, in_year, out_year, omni_param, auroral_param, plot_file):
    df_epoch = df['Epoch']
    df.index = df['Epoch']
    omni_len = len(omni_param)
    auroral_len = len(auroral_param)

    fig1, axs1 = plt.subplots(omni_len, 1, figsize=(20, 25), sharex=True)
    fig2, axs2 = plt.subplots(auroral_len, 1, figsize=(20, 13), sharex=True)
    fig1.suptitle(f'OMNI Parameters from {in_year} to {out_year}', y=0.9, fontsize=20)
    fig2.suptitle(f'Auroral Index from {in_year} to {out_year}', y=0.95, fontsize=20)

    if omni_len == 1:
        axs1 = [axs1]
    if auroral_len == 1:
        axs2 = [axs2]

    for i, param in enumerate(omni_param):
        axs1[i].plot(df[param], color='teal', zorder=1)
        axs1[i].axhline(0, color='red', zorder=2, linewidth=3)
        axs1[i].set_ylabel(f'{param}', labelpad=17)
        axs1[i].set_xlim([min(df_epoch) - pd.Timedelta('100d'), max(df_epoch) + pd.Timedelta('100d')])
        axs1[i].grid(True)

    for i, param in enumerate(auroral_param):
        axs2[i].plot(df[param], color='teal', zorder=1)
        axs2[i].axhline(0, color='red', zorder=2, linewidth=3)
        axs2[i].set_ylabel(f'{param}', labelpad=17)
        axs2[i].set_xlim([min(df_epoch) - pd.Timedelta('100d'), max(df_epoch) + pd.Timedelta('100d')])
        axs2[i].grid(True)

    axs1[-1].set_xlabel('Date', fontsize=16)
    axs2[-1].set_xlabel('Date', fontsize=16)

    fig1.savefig(f'{plot_file}Omni_Parameters_{in_year}_to_{out_year}.png')
    fig2.savefig(f'{plot_file}Auroral_electrojet_index_{in_year}_to_{out_year}.png')
    plt.close(fig1)
    plt.close(fig2)
    df.reset_index(drop=True)


###### [ Correlation Plot ] ######
def corr_plot(df, correlation, plot_file):
    matrix = round(df.corr(method=correlation), 2)

    plt.figure(figsize=(15,10))
    plt.title(f'Heat Map Correlation ({correlation.title()})', y=1.01, fontsize=15)
    plt.pcolor(matrix, cmap='RdBu', vmin=-1, vmax=1)

    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            color = 'white' if i == j else 'black'
            plt.text(j + 0.5, i + 0.5, f'{matrix.iloc[i, j]:.2f}',
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=7,
                     color=color)
            
    plt.colorbar()
    plt.xticks(np.arange(0.5, len(matrix.columns), 1), matrix.columns, fontsize=10, rotation=35)
    plt.yticks(np.arange(0.5, len(matrix.index), 1), matrix.index[::-1], fontsize=10)
    plt.savefig(plot_file + f'Heat_map_correlation_{correlation}.png')
    plt.close()


###### [ Plot Metric ] ######
def plot_metric(metrics_train_val, save_plot_model, type_model, auroral_index):
    metrics_list = ["Loss", "Accuracy", "R2", "RMSE"]

    plt.figure(figsize=(15, 10))
    
    for metric in metrics_list:
        i=0
        while i <= len(metrics_list):
            i = 500
            plt.title(f'{metric} Plot {auroral_index.replace("_INDEX", " Index")} using {type_model}')
        
            for cols in metrics_train_val.columns:
                if cols.endswith(metric):
                    if cols.startswith("Train"):
                        metric_value = metrics_train_val[cols].values
                        plt.plot(metric_value, label=f'Train {metric}', color='teal')

                    elif cols.startswith("Valid"):                       
                        metric_value = metrics_train_val[cols].values
                        plt.plot(metric_value, label=f'Valid {metric}', color='red')
                        
                    else:
                        break

            plt.xlabel('Epoch', fontsize=10)
            plt.ylabel(f'{metric}', fontsize=10)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
        plt.savefig(save_plot_model + f'Plot_{metric}_{type_model}_{auroral_index}_Epoch_{len(metrics_train_val)}.png')
        print(f'--- [ Plot {metric} ] ---')
        plt.clf()
    plt.close()
            
###### [ Scatter Test Real/Prediction ] ######
def scatter_plot(df_real_pred, save_plot_model, type_model, auroral_index, metrics_test):
    R2 = metrics_test['Test_R2'].values[0]

    plt.figure(figsize=(10, 10))
    plt.title(f'Scatter Plot {auroral_index.replace("_INDEX", " Index")} using {type_model}', fontsize=15)

    plt.scatter(df_real_pred['Test_Real'], df_real_pred['Test_Pred'], c='teal', alpha=0.7, edgecolor='w', s=40)
    plt.xscale('log')
    plt.yscale('log')

    p1 = max(df_real_pred['Test_Pred'].max(), df_real_pred['Test_Real'].max())
    p2 = min(df_real_pred['Test_Pred'].min(), df_real_pred['Test_Real'].min())
    plt.plot([p1, p2], [p1, p2], color='black', linewidth=2, label=f'R={np.sqrt(R2):.2f}')
    
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')
    plt.legend(loc='best')

    plt.savefig(f'{save_plot_model}Plot_Scatter_{type_model}_{auroral_index}.png')
    print('--- [ Plot Scatter ] ---')
    plt.close()
