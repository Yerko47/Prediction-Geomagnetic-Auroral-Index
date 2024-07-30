import time
import inspect
from torchinfo import summary
from tabulate import tabulate
from datetime import datetime, timedelta

def save_model_info(in_year, out_year, auroral_index, omni_param, set_split, type_model, type_neural_network, scaler, shift_length, num_epoch,
                    train_len, val_len, test_len, batch_train, batch_val, batch_test, learning_rate, weight_decay, scheduler_option, patience, model, 
                    criterion, optimizer, scheduler, metrics_train_val, metrics_test, lr_scheduler, today, total_time, save_info_model):

    file = save_info_model + f'Information_model_{type_model}_{auroral_index}.txt'
    with open(file, "w") as f:
        f.write(f'{auroral_index.replace("_", " ").upper()} Prediction ({in_year} to {out_year})\n')
        f.write(f'Date: {today}\n')
        f.write(f'Duration of training: {total_time}\n')
        f.write(f'Model: {type_model}\n')
        f.write(f'Epoch: {num_epoch}\n')
        f.write(f'Scaler: {scaler.title()}\n')
        f.write(f'Shift length: {shift_length}\n')
        f.write('\n')

        f.write(f'------- [ Omni Parameters ] -------\n')
        f.write('\n'.join(map(str, omni_param)))
        f.write('\n\n')

        f.write(f'------- [ Split Train/Val/Test Set ] -------\n')
        if set_split == "organized":
            f.write(f'TimeSerieSplit Function using split data\n')
        elif set_split == "random":
            f.write(f'Train_test_split Function using split data\n')
        elif set_split == "list":
            f.write(f'Storm_list.csv using split data\n')
        f.write(f'Train Set (%)={train_len}%      |      Valid Set (%)={val_len}%      |      Test Set (%)={test_len}%\n')
        f.write('\n')

        f.write(f'------- [ Hyperparameters ] -------\n')    
        f.write(f'Model Name: {type_model}_{type_neural_network}\n')
        f.write(f'Batch Train: {batch_train}      |      Batch Val: {batch_val}      |      Batch Test: {batch_test}\n')
        f.write(f'Criterion: {criterion.__class__.__name__}\n')
        f.write(f'Optimizer: {optimizer.__class__.__name__}\n')
        if scheduler_option:
            f.write(f'Scheduler Option is {scheduler_option} and using {scheduler.__class__.__name__} and Patience {patience}\n')
            lr_line = [f'Learning Rate: {lr}\n' for lr in lr_scheduler]
            f.writelines(lr_line) 
        else:
            f.write(f'Learning Rate: {learning_rate}\n')
        f.write(f'Weight Decay: {weight_decay}\n')
        f.write('\n')

        f.write(f'------- [ Model ] -------\n') 
        f.write(f'{summary(model)}\n')
        f.write('\n')

        f.write(f'------- [ Forward Model ] -------\n') 
        forward_method = inspect.getsource(model.forward)
        f.write(forward_method)
        f.write('\n')

        f.write(f'---------------------------------------------------\n')

        f.write(f'------- [ Metrics Training and Validation ] -------\n')
        f.write(tabulate(metrics_train_val, headers='keys', tablefmt='grid'))
        f.write('\n')

        f.write(f'------- [ Metrics Test ] -------\n')
        f.write(tabulate(metrics_test, headers='keys', tablefmt='grid'))
        f.write('\n')

