from datetime import datetime, timedelta

### Save Model txt
def save_model_txt(model, num_epoch, criterion, optimizer, train_loss, val_loss, test_loss, train_acc, val_acc, test_acc, train_r2, val_r2, test_r2, 
                df_train, df_val, df_test, time, auroral_index, omni_param, type_model, learning_rate, scaler, in_year, out_year, model_file):
    
    today = datetime.now()
    today = today.strftime("%Y-%m-%d--%H-%M-%S")

    len_train = len(df_train)
    len_val = len(df_val)
    len_test = len(df_test)
    len_total = len_train + len_val + len_test

    name = model_file + f'Model_{type_model}_{auroral_index}_{today}_Epoch_{num_epoch}'
    file = name + '.txt'
    with open(file, 'w') as f:
        f.write(f'{auroral_index.replace("_"," ").upper()} Prediction ({in_year} to {out_year}\n')
        f.write(f'Date: {today}\n')
        f.write(f'Length of time {time}\n')
        f.write(f'Model file name: {name}.pt\n')
        f.write(f'OMNI Parameters: ')
        for param in omni_param:
            f.write(f'  {param.replace("_", " ")}\n')
        f.write('\n')
        
        f.write('-----------[ Hyperparameters ]-----------\n')
        f.write('\n')
        f.write(f'Model Name: {type_model}\n')
        f.write(f'Scaler: {scaler.title()}')
        f.write(f'Training Set Length (%): {(round(len_train/len_total,2))*100}%\n')
        f.write(f'Validation Set Length (%): {(round(len_val/len_total,2))*100}%\n')
        f.write(f'Test Set Length (%): {(round(len_test/len_total,2))*100}%\n')
        f.write(f'Epoch: {num_epoch}\n')
        f.write(f'Learning Rate: {learning_rate}\n')
        f.write(f'Criterion: R{criterion.__class__.__name__}\n')
        f.write(f'Optimizer: {optimizer.__class__.__name__}\n')
        f.write('\n')
        f.write('-----------[ Model Parameters ]-----------\n')
        f.write('\n')
        for idx, module in enumerate(model.modules()):
            f.write(f' {idx+1} --> {module} \n')
        f.write('\n')
        f.write('-----------[ Loss, Accuracy and R2 Score ]-----------\n')
        f.write('\n')
        f.write('Train Loss/Accuracy:\n')
        for i, loss in enumerate(train_loss):
            if i % 10 == 0:
                f.write(f'  Epoch: {i+10}   |   Loss: {loss:.4f}   |   Accuracy: {train_acc[i]:.4f}   |   R2 Score: {train_r2[i]:.4f}\n')
        f.write('-----------------------------------------\n')
        f.write('Valid Loss/Accuracy:\n')
        for i, loss in enumerate(val_loss):
            if i % 10 == 0:
                f.write(f'  Epoch: {i+10}  |   Loss: {loss:.4f}  |   Accuracy: {val_acc[i]:.4f}   |   R2 Score: {val_r2[i]:.4f}\n')
        f.write('-----------------------------------------\n')
        f.write('Test Loss/Accuracy:\n')
        for i, loss in enumerate(test_loss):
            f.write(f'Loss: {loss:.4f}   |   Accuracy: {test_acc[i]:.4f}   |   R2 Score: {test_r2[i]:.4f}\n')
        f.write('\n')