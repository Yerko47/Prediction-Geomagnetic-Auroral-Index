from read_cdf import *
from variables import *
from data_function import *
from plots import *
from training import *



def main():
    today = datetime.now().strftime("%Y-%m-%d__%H:%M:%S")
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    #########################################################################################
    ##############################     [     Folders     ]     ##############################

    neural_network_file = result_file +f'Model_{type_model}_{auroral_index}_{today}/'      # Model
    model_file = neural_network_file + 'model/'                                                     # Save Model
    plot_file = neural_network_file + 'plots/'                                                      # Save Plot
    metric_train_file = plot_file + 'training/metric_train_plot/'                                   # metric plot train vs valid    (training)
    metric_train_shifty_file = plot_file + 'training/metric_shifty_comparison_train/'               # metric plot shifty vs shifty  (training)
    density_file = plot_file + 'test/density/'                                                      # density plot                  (test)
    metric_test_shifty_file = plot_file + 'test/metric_shifty_comparison_test/'                     # metric plot  shifty vs shifty (test)
    historic_file = plot_file + 'test/comparison_historic/'                                         # historic plot                 (tes)

    auroral_historic_file = result_file + 'general_plot/historic_plot/auroral_index/'               # historic time plot auroral index (before)
    solar_historic_file = result_file + 'general_plot/historic_plot/solar_parameter/'               # historic time plot solar parameters (before)
    stad_file = result_file + 'general_plot/stad_plot/'

    check_folder(model_file)
    check_folder(metric_train_file)
    check_folder(metric_train_shifty_file)
    check_folder(density_file)
    check_folder(metric_test_shifty_file)
    check_folder(historic_file)
    check_folder(auroral_historic_file)
    check_folder(solar_historic_file)
    check_folder(stad_file)

    
    ##############################################################################################
    ##############################     [     Read OMNI Set    ]     ##############################

    dataset(in_year, out_year, omni_file, save_feather, save_raw_file, processOMNI)

    if os.path.exists(save_feather):
        df = pd.read_feather(save_feather)
    else:
        raise FileNotFoundError(f'The file {save_feather} doesnot  exist')
    
    df = epoch_storm(df, save_raw_file)


    #######################################################################################
    ##############################     [     Plots     ]     ##############################
    if processPLOT:
        time_serie_plot(df, omni_param, auroral_param, auroral_historic_file, solar_historic_file, save_raw_file)


    ##################################################################################################
    ##############################     [     Data Performance     ]     ##############################
    df = scaler_df(df, scaler, omni_param, auroral_param)
    train_df, val_df, test_df = create_set_prediction(df, set_split, test_size, val_size)

    shifty_set = []
    metric_train_shift = pd.DataFrame()
    metric_test_shift = pd.DataFrame()
    metric_test_history = pd.DataFrame()

    print(f'Model: {type_model}_{type_neural_network}    ')

    for shift in shift_length:
        print(f'[   Shift: {shift}   ]\n')
        shifty_set.append(shift)

        ###################################################################################################
        ##############################     [     Shift Performance     ]     ##############################
        omni_train, index_train = shifty(train_df, omni_param, auroral_index, shift, type_model, 'train')
        omni_val, index_val = shifty(val_df, omni_param,auroral_index, shift, type_model, 'val')
        omni_test, index_test, df_epoch_test = shifty(test_df, omni_param, auroral_index, shift, type_model, 'test')

        train_torch = CustomDataset(omni_train, index_train, device)
        val_torch = CustomDataset(omni_val, index_val, device)
        test_torch = CustomDataset(omni_test, index_test, device)

        train_loader = DataLoader(train_torch, shuffle=True, batch_size=batch_train)
        val_loader = DataLoader(val_torch, shuffle=False, batch_size=batch_val)
        test_loader = DataLoader(test_torch, shuffle=False, batch_size=batch_test)


        #########################################################################################################
        ##############################     [     Model & Hyperparameters     ]     ##############################
        model = type_nn(type_model, type_neural_network, omni_train,drop, num_layer, device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min', patience=scheduler_patience, verbose=False)
        

        #####################################################################################################
        ##############################     [     Training/Test Model     ]     ##############################
        start_time = time.time()

        model, metrics_train_val, learning_rate_df, best_val_loss = train_model(model, criterion, optimizer, train_loader, val_loader, num_epoch, shift, type_model,
                                                                                auroral_index, scheduler, scheduler_option,model_file)
        results_df, metrics_test = test_model(model, criterion, test_loader, model_file, type_model, auroral_index, num_epoch, df_epoch_test, shift)

        end_time = time.time()
        total_time = str(timedelta(seconds=(end_time - start_time)))

        #####################################################################################################
        ##############################     [     Plot Model     ]     ##############################
        metric_train_shift = pd.concat([metric_train_shift, metrics_train_val], axis=1, ignore_index=False)
        metric_test_shift = pd.concat([metric_test_shift, metrics_test], axis=1, ignore_index=False)
        metric_test_history = pd.concat([metric_test_history, results_df], axis=1, ignore_index=False)
        metric_test_history = metric_test_history.loc[:, ~metric_test_history.columns.duplicated()]
        metric_train_shift = pd.DataFrame(metric_train_shift)

        metric_shift_plot(metric_train_shift, metric_train_shifty_file, auroral_index, type_model)
        test_shift_plot(metrics_test, shifty_set, metric_test_shifty_file, auroral_index, type_model)
        metric_plot(metrics_train_val, metric_train_file, shift, auroral_index, type_model)
        density_plot(results_df, shift, metrics_test, density_file, auroral_index, type_model)
        historic_shift_plot(metric_test_history,save_raw_file, historic_file, auroral_index, type_model)


        
if __name__ == '__main__':
    main()

