###### [ Data ] ######
in_year = 1995
out_year = 2018


###### [ Geomagnetic Auroral Index ] ######
auroral_index = 'AE_INDEX'      # AE_INDEX    |    AL_INDEX    |    AU_INDEX


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
set_split = 'organized'         # organized => TimeSerieSplit()   |   random => train_test_split()    |   list => storm_list.csv

## [ set_split = organized ] ##
n_split_train_val_test = 4
n_split_train_val = 3

## [ set_split = random ] ##
test_size = 0.2
val_size = 0.2


###### [ Neural Networks Parameters ] ######
type_model = 'ANN'              # ANN   |   LSTM    |   CNN    |    etc
type_neural_network = 2         # 1     |   2   |   3.... (eg: type_model = ANN and type_neural_network = 1  ==>   model = ANN_1)
scaler = 'robust'               # robust => RobustScaler    |    standard => StandardScaler
shift_length = 5
num_epoch = 100

## [ Loader ] ##
batch_train = 1040
batch_val = 1040
batch_test = 520

## [ Hyperparameters ] ##
learning_rate = 1e-4
weight_decay = 1e-2
scheduler_option = False
patience = 10
drop = 0.4


## [ ANN ] ##


## [ LSTM ] ##
num_layer = 3

## [ CNN ] ##


## [ RNN ] ##


###### [ Plots and correlation ] ######
correlation = 'spearman'        # spearman    |    kendall
processPLOT = False

###### [ Load ] ######
processOMNI = True


###### [ File ] ######
omni_file = f'/data/omni/hro_1min/'
desktop_file = f'/home/yerko/Desktop/'
save_raw_file = desktop_file + f'raw_data/'
processing_file = desktop_file + f'processing_data/'
model_file = processing_file + f'model/'
result_file = processing_file + f'result/'
plot_file = desktop_file + f'plots/'

save_feather = save_raw_file + f'omni_data_{in_year}_to_{out_year}.feather'

