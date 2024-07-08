""" Variable list """

####### [ Date ] #######
in_year = 1995
out_year = 2018


####### [ Geomagnétic Auroral Index ] #######
auroral_index = 'AE_INDEX'


####### [ Traget ] #######
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
              'flow_speed'
              ]

auroral_param = ['AU_INDEX',
                 'AE_INDEX',
                 'AL_INDEX']


####### [ Neural Network Model Parameters ] #######
type_model = 'ANN'
scaler = 'robust'
shifty = 3

num_epoch = 300
batch_train_val = 2080
barch_test = 1040
learning_rate = 1e-5
drop = 0.2
early_stop = 100

####### LSTM #######
num_layer = 3

####### CNN #######
kernel = 3
padding = 1

####### RNN #######


####### [ File ] #######
omni_file = f'/data/omni/hro_1min/'
desktop_file = f'/home/yerko/Desktop/'
save_raw_file = desktop_file + f'raw_data/'
processing_file = desktop_file + f'processing_data/'
model_file = processing_file + f'model/'
result_file = processing_file + f'result/'
plot_file = desktop_file + f'plots/'

save_feather = save_raw_file + f'omni_data_{in_year}_to_{out_year}.feather'


####### [ Train/Val/Test Set ] #######
storm_list = False
n_split_train_val_test = 4
n_split_train_val = 3


####### [ Plot ] #######
correlation = 'spearman'
processPLOT = False


####### [ Download ] #######
processOMNI = True





