import torch
import torchinfo
import numpy as np
import pandas as pd
import torch.optim as optim
from torchinfo import summary
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
                case 1: model = LSTM_1(input_size, drop, num_layer).to(device)
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
    r2 = r2_score(target_np, output_np)
    if r2 < 0:
        r2 = 0

    return rmse, r2


###### [ Training/Validation Model ] ######
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epoch, save_result_model, type_model, auroral_index, scheduler=None, scheduler_option=False):
    train_losses, val_losses = [], []
    train_r2_score, val_r2_score = [], []
    train_rmse_score, val_rmse_score = [], []
    learning_rate = []

    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(num_epoch):
        ## [ Training ] ##
        model.train()
        train_loss, train_rmse, train_r2 = 0, 0, 0
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

        train_rmse, train_r2 = calculate_metrics(all_pred, all_real)

        ## [ Validation ] ##
        model.eval()
        val_loss, val_rmse, val_r2 = 0, 0, 0
        all_real, all_pred = [], []
        with torch.no_grad():
            for x, y in val_loader:
                yhat = model(x)
                loss = criterion(yhat, y)
                
                val_loss += loss.item() * x.size(0)

                all_real.append(y.cpu())
                all_pred.append(yhat.cpu())

        val_loss /= len(val_loader.dataset)
        all_real = torch.cat(all_real)
        all_pred = torch.cat(all_pred)

        val_rmse, val_r2 = calculate_metrics(all_pred, all_real)

        train_r2_score.append(train_r2)
        val_r2_score.append(val_r2)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_rmse_score.append(train_rmse)
        val_rmse_score.append(val_rmse)

        if epoch % 10 == 0:
            print(f'Epoch {epoch + 10}/{num_epoch}: '
                  f'Train Loss: {train_loss:.4f}, Train R2: {train_r2:.4f}, Train RMSE: {train_rmse:.4f}, '
                  f'Valid Loss: {val_loss:.4f}, Valid R2: {val_r2:.4f}, Valid RMSE: {val_rmse:.4f}') 

        if scheduler_option and scheduler is not None:
            scheduler.step(val_loss)  
            current_lr = optimizer.param_groups[0]['lr']
            learning_rate.append(f'Epoch {epoch+1} --> Lr: {current_lr}')
                
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

    ## [ Save the best model ] ##
    if best_model_state is not None:
        torch.save(best_model_state, save_result_model + f'Model_{type_model}_{auroral_index}_Epoch_{num_epoch}.pt')                   

    metrics_df = pd.DataFrame({
        'Train_Loss': train_losses,
        'Train_RMSE': train_rmse_score,
        'Train_R2': train_r2_score,
        'Valid_Loss': val_losses,
        'Valid_RMSE': val_rmse_score,
        'Valid_R2': val_r2_score,
    })

    learning_rate = pd.DataFrame(learning_rate, columns=['Learning_Rate'])
    return model, metrics_df, learning_rate


###### [ Test Model ] ######
def test_model(model, criterion, test_loader, save_result_model, type_model, auroral_index, num_epoch, df_epoch_test):
    test_losses, test_rmse_score, test_r2_score = [], [], []
    real, pred = [], []

    ## [ Test ] ##
    model.load_state_dict(torch.load(save_result_model + f'Model_{type_model}_{auroral_index}_Epoch_{num_epoch}.pt'))
    model.eval()

    test_loss, test_rmse, test_r2 = 0, 0, 0

    with torch.no_grad():
        for x, y in test_loader:
            yhat = model(x)
            loss = criterion(yhat, y)

            test_loss += loss.item() * x.size(0)

            real.append(y.detach().cpu())
            pred.append(yhat.detach().cpu())
    
    test_loss /= len(test_loader.dataset)


    all_real = torch.cat(real)
    all_pred = torch.cat(pred)

    test_rmse, test_r2 = calculate_metrics(all_pred, all_real)

    test_r2_score.append(test_r2)
    test_losses.append(test_loss)
    test_rmse_score.append(test_rmse)

    print(f'Test Loss: {test_loss:.4f}, Test R2: {test_r2:.4f}, Test RMSE: {test_rmse:.4f}')

    ## [ Save ] ##
    metrics_df = pd.DataFrame({
        'Test_Loss': test_losses,
        'Test_RMSE': test_rmse_score,
        'Test_R2': test_r2_score
        })


    results_df = pd.DataFrame({
        'Epoch': df_epoch_test,
        'Test_Real': all_real.numpy().flatten().tolist(),
        'Test_Pred': all_pred.numpy().flatten().tolist()
    })


    return results_df, metrics_df
