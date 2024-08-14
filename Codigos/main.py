from read_cdf import *
from variables import *
from ANN_models import *
from LSTM_models import *
from performance import *
from save_info import *
from Training import *
from plots import time_serie_plot, corr_plot
from plots import plot_metric, scatter_plot

def main():

    today = datetime.now().strftime("%Y-%m-%d__%H:%M:%S")


    ###### [ Device cuda ] ######
    device = ("cuda" if torch.cuda.is_available() else "cpu")


    ## [ Save Model ] ##
    save_plot_model = plot_file + f'{type_model}_{auroral_index}{today}/'
    save_info_model = model_file + f'{type_model}_{auroral_index}{today}/'
    save_result_model = result_file + f'{type_model}{auroral_index}_{today}/'


    ##### [ Checks Folder ] ######
    check_folder(save_raw_file)
    check_folder(processing_file)
    check_folder(plot_file)
    check_folder(model_file)
    check_folder(result_file)        
    check_folder(save_plot_model)
    check_folder(save_info_model)
    check_folder(save_result_model)


    ###### [ Dataset Building ] ######
    dataset(in_year, out_year, omni_file, save_feather,processing_file, processOMNI)


    ###### [ Dataset Building ] ######
    dataset(in_year, out_year, omni_file, save_feather, processing_file, processOMNI)


    ###### [ Read CDF ] ######
    if os.path.exists(save_feather):
        df = pd.read_feather(save_feather)
    else:
        raise FileNotFoundError(f'The file {save_feather} doesnot  exist.')
    

    ###### [ Plot ] ######
    if processPLOT:
        time_serie_plot(df, in_year, out_year, omni_param, auroral_param, plot_file)
        corr_plot(df, correlation, plot_file)


    ###### [ Epoch Storm Selection ] #####
    df = epoch_storm(df, processing_file)


    ###### [ Scaler ] ######
    df = scaler_df(df, scaler, omni_param, auroral_param)


    ###### [ Percentage Set ] ######
    df_train, df_val, df_test = create_set_prediction(df,set_split,    n_split_train_val_test, n_split_train_val,test_size, val_size)
    train_len = round(len(df_train) / len(df), 2) * 100
    val_len = round(len(df_val) / len(df), 2) * 100
    test_len = round(len(df_test) / len(df), 2) * 100

    print('\n---------- [ Percentage Set ] ----------\n')
    print(f'Percentage Train Set: {train_len}%')
    print(f'Percentage Valid Set: {val_len}%')
    print(f'Percentage Test Set: {test_len}%\n')


    ###### [ Shift or Delay ] ######
    omni_train, index_train, df_epoch_train = shifty(df_train, omni_param, auroral_index, shift_length, type_model)
    omni_val, index_val, df_epoch_val = shifty(df_val, omni_param,auroral_index,     shift_length, type_model)
    omni_test, index_test, df_epoch_test = shifty(df_test, omni_param,    auroral_index, shift_length, type_model)

    print('\n---------- [ Dimension Set ] ----------\n')
    print(f'Dimension Train Set: OMNI--> {omni_train.shape}  |      {auroral_index.replace("_INDEX", " Index")}-->{index_train.shape}  ')
    print(f'Dimension Valid Set: OMNI--> {omni_val.shape}  |   {auroral_index.replace("_INDEX", " Index")}--> {index_val.shape}')
    print(f'Dimension Test Set: OMNI--> {omni_test.shape}  |   {auroral_index.replace("_INDEX", " Index")}--> {index_test.shape} \n')


    ###### [ DataTorch and DataLoader ] ######
    train_torch = CustomDataset(omni_train, index_train, device)
    val_torch = CustomDataset(omni_val, index_val, device)
    test_torch = CustomDataset(omni_test, index_test, device)
    train_loader = DataLoader(train_torch, shuffle=True,   batch_size=batch_train)
    val_loader = DataLoader(val_torch, shuffle=False,  batch_size=batch_val)
    test_loader = DataLoader(test_torch, shuffle=False,    batch_size=batch_test)
    

    ###### [ Neural Network Model ] ######
    model = type_nn(type_model, type_neural_network, omni_train,drop, num_layer, device)
    print()
    print(f'----- [ {type_model} Neural Network Model ] -----\n')
    summary(model)
    print()

    
    ###### [ HyperParameters ] ######
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)#,weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min', patience=patience, verbose=False)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,num_epoch * len(test_loader), verbose=False)

    start_time = time.time()


    ###### [ Training/Validation Model ] ######
    model, metrics_train_val, lr_scheduler = train_model(model,criterion, optimizer, train_loader, val_loader, num_epoch,save_result_model, type_model, auroral_index, scheduler,scheduler_option)


    ###### [ Test Model ] ######
    df_real_pred, metrics_test = test_model(model, criterion,test_loader, save_result_model, type_model, auroral_index,num_epoch, df_epoch_test)

    end_time = time.time()
    total_time = str(timedelta(seconds=(end_time - start_time)))


    ##### [ Save Info Model ] #####
    save_model_info(in_year, out_year, auroral_index, omni_param,set_split, type_model, type_neural_network, scaler,shift_length, num_epoch,
                train_len, val_len, test_len, batch_train,batch_val, batch_test, learning_rate,weight_decay, scheduler_option, patience, model, 
                criterion, optimizer, scheduler,metrics_train_val, metrics_test, lr_scheduler,today, total_time, df_real_pred, save_info_model)
    

    ###### [ Plot Metrics ] ######
    plot_metric(metrics_train_val, save_plot_model, type_model,auroral_index)

    ##### [ Scatter Plot ] #####
    scatter_plot(df_real_pred, save_plot_model, type_model,auroral_index, metrics_test)


    print('You have finished your path, you are a machine')


if __name__ == '__main__':
    main()
