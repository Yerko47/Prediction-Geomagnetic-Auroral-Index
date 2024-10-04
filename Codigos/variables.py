###### [ Data ] ######
in_year = 1995
out_year = 2018


###### [ Geomagnetic Auroral Index ] ######
auroral_index = 'AE_INDEX'                      # AE_INDEX    |    AL_INDEX    |    AU_INDEX


###### [ Parameters ] ######
omni_param = ['B_Total',
              'BY_GSM',
              'BZ_GSM',
              'Vx',
              'Vy',
              'Vz',
              'proton_density',
              'T',
              'Pressure',
              'E_Field',
              ]

auroral_param = ['AE_INDEX', 
                 'AL_INDEX', 
                 'AU_INDEX'] 


###### [ Split Train/Val/Test Set  ] ######
set_split = 'organized'                         # organized => TimeSerieSplit()   |   random => train_test_split()    |   list => storm_list.csv
                             
test_size = 0.2                                 
val_size = 0.2                                  


###### [ Neural Networks Parameters ] ######
type_model = 'LSTM'                             # ANN   |   LSTM    |   CNN    |    etc
type_neural_network = 1                         # 1     |   2   |   3.... (eg: type_model = ANN and type_neural_network = 1  ==>   model = ANN_1)
scaler = 'robust'                             # robust => RobustScaler    |    standard => StandardScaler    |    minmax => MinMaxScaler()
shift_length = [30, 40, 45, 50, 55, 60]

## [ Loader ] ##
batch_train = 1040*2
batch_val = 1040*2
batch_test = 520*2

## [ Hyperparameters ] ##
num_epoch = 500
learning_rate = 1e-3
weight_decay = 1e-2
patience = 20
drop = 0.2
patience_scheduler = 10
scheduler_option = False
scheduler_patience = 100

# [ LSTM ] #
num_layer = 3


###### [ Plots and correlation ] ######
correlation = 'spearman'                        # spearman    |    kendall
processPLOT = False

###### [ Load ] ######
processOMNI = False


###### [ File ] ######
omni_file = f'/data/omni/hro_1min/'
desktop_file = f'/home/yerko/Desktop/'

result_file = desktop_file + f'result/'

save_raw_file = result_file + f'raw_data/'
save_feather = save_raw_file + f'omni_data_{in_year}_to_{out_year}.feather'

general_plot_file = result_file + 'general_plot/'
stadistic_file = general_plot_file + 'stadistic/'
time_serie_file = general_plot_file + 'plot_time_serie/'
omni_serie_file = time_serie_file + 'omni_serie/'
auroral_serie_file = time_serie_file + 'auroral_serie/'








