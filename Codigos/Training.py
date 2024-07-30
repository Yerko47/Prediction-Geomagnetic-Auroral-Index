import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from torchinfo import summary
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score

from ANN_models import *
from LSTM_models import *

###### [ Type Neural Network ] ######
def type_nn(type_model, type_neural_network, omni_train, drop, num_layer, device):
    match type_model:
        case 'ANN':
            input_size = omni_train.shape[1]
            match type_neural_network:
                case 1: model = ANN_1(input_size, drop).to(device)
                case 2: model = ANN_2(input_size, drop).to(device)
                case 3: print('Your mission is to create ANN_3(input_size, drop).to(device)')
                case _:
                    print('Your mission is to create a new model and add case it to the "type_nn" function')

        case 'LSTM':
            input_size = omni_train.shape[2]
            match type_neural_network:
                case 1: model = LSTM_1(input_size, drop, num_layer)
                case 2: model = LSTM_2(input_size, drop, num_layer).to(device)
                case 3: print('Your mission is to create LSTM_3(input_size, drop).to(device)')
                case _: print('Your mission is to create a new model and add case it to the "type_nn" function')
                    

    if model is None:
        raise ValueError("Invalid type_model or type_neural_network specified")
        
    return model


###### [ Metrics ] ######
def calculate_metrics(output, target):
    output_np = output.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    rmse = np.sqrt(mean_squared_error(target_np, output_np))
    acc = accuracy_score(np.round(target_np), np.round(output_np))
    r2 = r2_score(target_np, output_np)
    if r2 < 0:
        r2 = 0

    return rmse, r2, acc


###### [ Training/Validation Model ] ######
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epoch, save_result_model, type_model, auroral_index, scheduler, scheduler_option):
    train_losses, val_losses = [], []
    train_accuracy, val_accuracy = [], []
    train_r2_score, val_r2_score = [], []
    train_rmse_score, val_rmse_score = [], []
    learning_rate = []

    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(num_epoch):
        ## [ Training ] ##
        model.train()
        train_loss, train_rmse, train_r2, train_acc = 0, 0, 0, 0
        all_real, all_pred = [], []

        for x, y in train_loader:
            yhat = model(x)
            loss = criterion(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            all_real.append(y.cpu())
            all_pred.append(yhat.cpu())

        train_loss /= len(train_loader.dataset)
        all_real = torch.cat(all_real)
        all_pred = torch.cat(all_pred)

        train_rmse, train_r2, train_acc = calculate_metrics(all_pred, all_real)

    
        ## [ Validation ] ##
        model.eval()
        val_loss, val_rmse, val_r2, val_acc = 0, 0, 0, 0
        all_real, all_pred = [], []
        with torch.inference_mode():
            for x, y in val_loader:
                yhat = model(x)
                loss = criterion(yhat, y)
                
                val_loss += loss.item() * x.size(0)

                all_real.append(y.cpu())
                all_pred.append(yhat.cpu())

        val_loss /= len(val_loader.dataset)

        all_real = torch.cat(all_real)
        all_pred = torch.cat(all_pred)

        val_rmse, val_r2, val_acc = calculate_metrics(all_pred, all_real)

        train_r2_score.append(train_r2), val_r2_score.append(val_r2)
        train_losses.append(train_loss), val_losses.append(val_loss)
        train_accuracy.append(train_acc), val_accuracy.append(val_acc)
        train_rmse_score.append(train_rmse), val_rmse_score.append(val_rmse)

        if epoch % 10 == 0:
            print(f'Epoch {epoch +10}/{num_epoch}: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train R2: {train_r2:.4f}, Train RMSE: {train_rmse:.4f}, '
                  f'Valid Loss: {val_loss:.4f}, Valid Acc: {val_acc:.4f}, Valid R2: {val_r2:.4f}, Valid RMSE: {val_rmse:.4f}') 

        if scheduler_option:
            current_lr = optimizer.param_groups[0]['lr']
            learning_rate.append(current_lr)
            if len(val_losses) > 1:
                if val_losses[-2] > val_losses[-1]:
                    scheduler.step(train_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

    ## [ Save ] ##
    torch.save(best_model_state, save_result_model + f'Model_{type_model}_{auroral_index}_Best.pt')
    torch.save(model.state_dict(), save_result_model + f'Model_{type_model}_{auroral_index}_Epoch_{num_epoch}.pt')                   

    metrics_df = pd.DataFrame({
        'Train_Loss': train_losses,
        'Train_RMSE': train_rmse_score,
        'Train_R2': train_r2_score,
        'Train_Accuracy': train_accuracy,
        'Valid_Loss': val_losses,
        'Valid_RMSE': val_rmse_score,
        'Valid_R2': val_r2_score,
        'Valid_Accuracy': val_accuracy,
    })

    metrics_df.to_csv(save_result_model + f'Metric_train_val_{type_model}_{auroral_index}.csv', index=False)

    learning_rate = pd.DataFrame(learning_rate)
    return model, metrics_df, learning_rate


###### [ Test Model ] ######
def test_model(model, criterion, test_loader, save_result_model, type_model, auroral_index, num_epoch, df_test):
    test_losses, test_rmse_score, test_r2_score, test_accuracy = [], [], [], []
    real, pred = [], []

    ## [ Test ] ##
    model.load_state_dict(torch.load(save_result_model + f'Model_{type_model}_{auroral_index}_Epoch_{num_epoch}.pt'))
    model.eval()

    test_loss, test_rmse, test_r2, test_acc = 0, 0, 0, 0

    with torch.inference_mode():
        for x, y in test_loader:
            yhat = model(x)
            loss = criterion(yhat, y)

            test_loss += loss.item() * x.size(0)

            real.append(y.detach().cpu())
            pred.append(yhat.detach().cpu())
    
    test_loss /= len(test_loader.dataset)

    # Convertir listas de tensores a un solo tensor
    all_real = torch.cat(real)
    all_pred = torch.cat(pred)

    test_rmse, test_r2, test_acc = calculate_metrics(all_pred, all_real)

    test_r2_score.append(test_r2)
    test_losses.append(test_loss)
    test_accuracy.append(test_acc)
    test_rmse_score.append(test_rmse)

    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test R2: {test_r2:.4f}, Test RMSE: {test_rmse:.4f}')

    ## [ Save ] ##
    metrics_df = pd.DataFrame({
        'Test_Loss': test_losses,
        'Test_RMSE': test_rmse_score,
        'Test_R2': test_r2_score,
        'Test_Accuarcy': test_accuracy,
    })

    df_test_epoch = df_test['Epoch'].reset_index(drop=True)

    results_df = pd.DataFrame({
        'Epoch': df_test_epoch,
        'Test_Real': all_real.numpy().flatten().tolist(),
        'Test_Pred': all_pred.numpy().flatten().tolist()
    })

    results_df.to_csv(save_result_model + f'Test_real_pred_{type_model}_{auroral_index}.csv', index=False)
    metrics_df.to_csv(save_result_model + f'Metric_test_{type_model}_{auroral_index}.csv', index=False)

    return results_df, metrics_df