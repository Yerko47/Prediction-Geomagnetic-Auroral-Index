from plots import *
from read_cdf import *
from training import *
from variables import *
from data_function import *


def main():
    today = datetime.now().strftime("%Y-%m-%d__%H:%M:%S")
    device = ("cuda" if torch.cuda.is_available() else "cpu")


    ##########################################################
    ############# [      File and Folders      ] #############

    model_file = result_file + f'Model_{type_model}_{auroral_index}_{today}/'
    result_model_file = model_file + 'Result_model/'
    plot_model_file = model_file + 'Plot_model/'
    plot_metric_training_file = plot_model_file + 'training/metrics_train_plot/' #######
    plot_shift_metric_training_file = plot_model_file + 'training/comparison_metrics_train_plot/'
    plot_density_file = plot_model_file + 'test/density_plot/'
    plot_shift_metric_test_file = plot_model_file + 'test/comparison_metrics_test_plot/'
    plot_time_serie_test_file = plot_model_file + 'test/comparison_time_serie_test_plot/'

    check_folder(save_raw_file)
    check_folder(stadistic_file)
    check_folder(omni_serie_file)
    check_folder(auroral_serie_file)
    check_folder(result_model_file)
    check_folder(plot_metric_training_file)
    check_folder(plot_shift_metric_training_file)
    check_folder(plot_density_file)
    check_folder(plot_shift_metric_test_file)
    check_folder(plot_time_serie_test_file)


    #########################################################
    ############## [      Read OMNI Set      ] ##############
    dataset(in_year, out_year, omni_file, save_feather, save_raw_file, processOMNI)

    if os.path.exists(save_feather):
        df = pd.read_feather(save_feather)
    else:
        raise FileNotFoundError(f'The file {save_feather} doesnot  exist')
    

    #########################################################
    ############## [     Selection Data      ] ##############
    df = epoch_storm(df, save_raw_file)


    ###############################################################
    ############## [     Time/Statistics Plot      ] ##############
    if processPLOT:
        time_serie_plot(df, omni_param, auroral_param, in_year, out_year, save_raw_file, omni_serie_file, auroral_serie_file)
        corr_plot(df, correlation, stadistic_file)


    ##################################################################
    ############## [     Scaler Solar Parameters      ] ##############
    df = scaler_df(df, scaler, omni_param, auroral_param)


    #######################################################
    ############## [     Divition Set      ] ##############
    train_df, val_df, test_df = create_set_prediction(df, set_split, test_size, val_size)

    shift_set = []
    test_real_pred = pd.DataFrame()

    for shift in shift_length:
        create_shift_folder(plot_metric_training_file, shift)
        create_shift_folder(plot_shift_metric_training_file, shift)
        create_shift_folder(plot_time_serie_test_file, shift)

        shift_set.append(shift)

        
        ####################################################
        ############## [     Shift Set      ] ##############
        print(f'\n---------- [ Shifty Set: {shift} ] ----------\n')
        omni_train, index_train = shifty(train_df, omni_param, auroral_index, shift, type_model, 'train')
        omni_val, index_val = shifty(val_df, omni_param,auroral_index, shift, type_model, 'val')
        omni_test, index_test, df_epoch_test = shifty(test_df, omni_param, auroral_index, shift, type_model, 'test')

        len_test = len(df) - len(index_test)
        ###################################################################
        ############## [     Torch Set and DataLoader      ] ##############
        train_torch = CustomDataset(omni_train, index_train, device)
        val_torch = CustomDataset(omni_val, index_val, device)
        test_torch = CustomDataset(omni_test, index_test, device)

        train_loader = DataLoader(train_torch, shuffle=True, batch_size=batch_train)
        val_loader = DataLoader(val_torch, shuffle=False, batch_size=batch_val)
        test_loader = DataLoader(test_torch, shuffle=False, batch_size=batch_test)


        ####################################################################
        ############## [     Model and Hyperparameters      ] ##############
        model = type_nn(type_model, type_neural_network, omni_train,drop, num_layer, device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min', patience=scheduler_patience, verbose=False)
        
        print(f'---------   [ {type_model} Neural Network and Shifty {shift} ]   ---------')
        start_time = time.time()

        ####################################################################
        ############## [     Training Model Prediction      ] ##############
        model, metrics_train_val, learning_rate_df, best_val_loss = train_model(model, criterion, optimizer, train_loader, val_loader, num_epoch, shift, type_model, auroral_index, scheduler, scheduler_option, result_model_file)
        results_df, metrics_test = test_model(model, criterion, test_loader, result_model_file, type_model, auroral_index, num_epoch, df_epoch_test, shift)


        end_time = time.time()
        total_time = str(timedelta(seconds=(end_time - start_time)))

        test_real_pred = pd.concat([test_real_pred, metrics_test], axis=0, ignore_index=True)
        ######################################################
        ############## [     Plot Model       ] ##############
        plot_compared_test_metrics(test_real_pred, shift_set, plot_shift_metric_test_file, auroral_index, type_model)
        plot_time_model(results_df, save_raw_file, plot_time_serie_test_file, auroral_index, shift, len_test)     
        density_plot(results_df, plot_density_file, metrics_test, auroral_index, type_model, shift)
        plot_metric(metrics_train_val, plot_metric_training_file, shift, auroral_index, type_model)



        del model, metrics_train_val, results_df

if __name__ == '__main__':
    main()