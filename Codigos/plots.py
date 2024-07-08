import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

###### [ Time serie Plot ] ######
def time_plot(df, in_year, out_year, auroral_param, plot_file):
    df.index = df['Epoch']
    auroral_subplot_count = len(auroral_param)
    omni_subplot_count = df.shape[1] - auroral_subplot_count-1

    ### Fig and axs OMNI Parameters plot
    fig1, ax1 = plt.subplots(omni_subplot_count, 1, figsize=(20,25), sharex=True)
    fig1.suptitle(f'OMNI Parameters from {in_year} to {out_year}', y=0.9, fontsize= 20)

    ### Fig and axs Auroral Index plot
    fig2, ax2 = plt.subplots(auroral_subplot_count, 1, figsize=(20,13), sharex=True)
    fig2.suptitle(f'Auroral Index from {in_year} to {out_year}', y=0.95, fontsize= 20)

    i, j = 0, 0

    for cols in df.columns:
        # OMNI Parameters plot
        if cols not in auroral_param and cols != 'Epoch':
            ax1[i].plot(df[cols], color='teal')
            ax1[i].set_ylabel(f'{cols}', labelpad=15)
            ax1[i].axhline(y=0, color='red', linestyle='dashed')
            ax1[i].grid(True)
            i += 1

        # Auroral Index plot
        if cols in auroral_param and cols != 'Epoch':
            ax2[j].plot(df[cols], color='teal')
            ax2[j].set_ylabel(f'{cols}', labelpad=15)
            ax2[j].axhline(y=0, color='red', linestyle='dashed')
            ax2[j].grid(True)
            j += 1            

    ax1[-1].set_xlabel('Date', fontsize=16)
    ax2[-1].set_xlabel('Date', fontsize=16)

    fig1.savefig(f'{plot_file}Omni_Parameters_{in_year}_to_{out_year}.png')
    fig2.savefig(f'{plot_file}Auroral_electrojet_index_{in_year}_to_{out_year}.png')
    plt.close(fig1)
    plt.close(fig2)
    df.reset_index(drop=True)


###### [ Correlation Plot ] ######
def corr_plot(df, correlation, plot_file):
    # Calculate correlation 
    matrix = round(df.corr(method=correlation), 2)

    ### Fig correlation plot
    plt.figure(figsize=(15, 10))
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
    plt.xticks(np.arange(0.5, len(matrix.columns), 1), matrix.columns, fontsize=8, rotation=35)
    plt.yticks(np.arange(0.5, len(matrix.index), 1), matrix.index[::-1], fontsize=8)
    plt.savefig(plot_file + f'Heat_map_correlation_{correlation}.png')
    plt.close()


###### [ Loss Plot ] ######
def plot_loss(num_epoch, train_loss, val_loss, plot_file, type_model, auroral_index):
    epochs = list(range(1,num_epoch+1))

    plt.figure(figsize=(15,10))
    plt.title(f'Loss Plot {auroral_index.replace("_INDEX", " Index")} using {type_model}')

    plt.plot(epochs, train_loss, color='teal', label='Train Loss')
    plt.plot(epochs, val_loss, color='red', label='Valid Loss')
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_file + f'Loss_{type_model}_{auroral_index}_Epoch_{num_epoch}.png')
    print('--- [ Loss Plot ] ---')
    plt.close()


###### [ Accuracy Plot ] ######
def plot_acc(num_epoch, train_acc, val_acc, plot_file, type_model, auroral_index):
    epochs = list(range(1,num_epoch+1))

    plt.figure(figsize=(15,10))
    plt.title(f'Accuracy Plot {auroral_index.replace("_INDEX", " Index")} using {type_model}')

    plt.plot(epochs, train_acc, color='teal', label='Train Accuracy')
    plt.plot(epochs, val_acc, color='red', label='Valid Accuracy')
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_file + f'Accuracy_{type_model}_{auroral_index}_Epoch_{num_epoch}.png')
    print('--- [ Accuracy Plot ] ---')
    plt.close()


###### [ R2 Score Plot ] ######
def plot_r2_score(num_epoch, train_r2, val_r2, plot_file, type_model, auroral_index):
    epochs = list(range(1,num_epoch+1))

    plt.figure(figsize=(15,10))
    plt.title(f'R2 Score Plot {auroral_index.replace("_INDEX", " Index")} using {type_model}')
    
    plt.plot(epochs, train_r2, color='teal', label='Train Accuracy')
    plt.plot(epochs, val_r2, color='red', label='Valid Accuracy')
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_file + f'R2_{type_model}_{auroral_index}_Epoch_{num_epoch}.png')
    print('--- [ Accuracy Plot ] ---')
    plt.close()


###### [ Density Plot ] ###### (Mejorar el gráfico)
def density_plot(real, pred, plot_file, type_model, auroral_index, group):
    
    norm = plt.Normalize(real.min(), real.max())
    cmap = plt.cm.plasma

    plt.figure(figsize=(15,10))
    plt.title(f'Density Plot {auroral_index.replace("_INDEX", " Index")} using {type_model}')

    sc = plt.scatter(real, pred, c=real, cmap=cmap, norm=norm, alpha=0.6, edgecolors='k')

    cbar = plt.colorbar(sc)
    cbar.set_label('Density')

    plt.plot([real.min(), real.max()], [real.min(), real.max()], color='red', linestyle='--', linewidth=2)
    plt.xlabel('Real')
    plt.ylabel('Prediction')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(plot_file + f'Density Plot_{group}_{type_model}_{auroral_index}.png')
    print('--- [ Density Plot ] ---')
    plt.close()