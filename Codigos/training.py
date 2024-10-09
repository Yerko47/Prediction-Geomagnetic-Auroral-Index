import torch
import torchinfo
import numpy as np
import pandas as pd
import torch.optim as optim
from torchinfo import summary
from sklearn.metrics import root_mean_squared_error, r2_score, explained_variance_score

from ANN_models import *
from LSTM_models import *

####### [ Selection Model ] #######
def type_nn(type_model, type_neural_network, omni_train, drop, num_layer, device):
    """
    This code makes the selection of the neural network.
        Args:
            - type_model (str): The type of neural network you want to program.
            - type_neural_network (float): The neural network of the selected type you want to use.
            - omni_train (torch dataset): Training data to select the dimensions to be used as input size.
            - drop (float): Drop from the neural network.
            - num_layer (float): Number of layers of recurrent neural networks.
            - device (str): Selecting the memory to use (cuda).
    """

    match type_model:
        case 'ANN':
            input_size = omni_train.shape[1]
            match type_neural_network:
                case 1: model = ANN_1(input_size, drop).to(device)
                case 2: model = ANN_2(input_size, drop).to(device)
                case 3: print('Your mission is to create ANN_3(input_size, drop).to(device)')
                case _: ValueError("Invalid type_neural_network specified")
            
        case 'LSTM':
            input_size = omni_train.shape[2]
            match type_neural_network:
                case 1: model = LSTM_1(input_size, drop, num_layer, device).to(device)
                case 2: model = LSTM_2(input_size, drop, num_layer, device).to(device)
                case 3: print('Your mission is to create LSTM_2(input_size, drop, num_layer, device).to(device)')
                case _: ValueError("Invalid type_neural_network specified")
    
    if model is None:
        raise ValueError("Invalid type_model or type_neural_network specified")

    return model


####### [ Metrics ] #######
def calculate_metrics(real, pred):
    """
    This code calculate metrics for linear regression.
        Arg:
            - real (torch): Real value.
            - pred (torch): Prediction value.
    """
    real = real.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()

    rmse = root_mean_squared_error(real, pred)
    expv = explained_variance_score(real, pred)
    r = np.sqrt(max(0, r2_score(real, pred)))
    
    return rmse, expv, r


####### [ Train/Valid Model ]  ####### 
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epoch, shift, type_model, auroral_index, scheduler, scheduler_option, result_model_file):
    """
    Function to train a neural network model and evaluate it using training and validation datasets.
    
    Parameters:
    - model: the neural network model to be trained.
    - criterion: loss function to optimize the model.
    - optimizer: optimization algorithm for weight updates.
    - train_loader: data loader for the training dataset.
    - val_loader: data loader for the validation dataset.
    - num_epoch: number of epochs to run the training.
    - shift, type_model, auroral_index: identifiers for model saving.
    - scheduler: learning rate scheduler.
    - scheduler_option: boolean to control if the scheduler should be used.
    - result_model_file: path to save the best model.
    
    Returns:
    - model: the trained model loaded with the best weights.
    - metrics_df: dataframe containing training and validation metrics.
    - learning_rate_df: dataframe containing learning rates for each epoch.
    - best_val_loss: the best validation loss achieved during training.
    """
    train_losses, train_r_score, train_rmse_score, train_expv_score = [], [], [], []
    val_losses, val_r_score, val_rmse_score, val_expv_score = [], [], [], []
    learning_rate = []

    best_val_loss = float('inf')
    best_epoch = 0
    best_model_wts = None

    for epoch in range(num_epoch):
        ## [ Training Phase ] ##
        model.train()
        train_loss, train_rmse, train_r, train_expv = 0, 0, 0, 0
        all_real, all_pred = [], []

        # Training loop
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

        
        train_rmse, train_expv, train_r = calculate_metrics(all_real, all_pred)

        ## [ Validation Phase ] ##
        model.eval()
        val_loss, val_rmse, val_r, val_expv = 0, 0, 0, 0
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

        
        val_rmse, val_expv, val_r = calculate_metrics(all_real, all_pred)

        
        train_losses.append(train_loss), val_losses.append(val_loss)              # Loss
        train_rmse_score.append(train_rmse), val_rmse_score.append(val_rmse)      # RMSE
        train_expv_score.append(train_expv), val_expv_score.append(val_expv)      # Expv
        train_r_score.append(train_r), val_r_score.append(val_r)                  # R

        if epoch % 10 == 0:
            print(f'Epoch {epoch + 10}/{num_epoch} --> '
                  f'Train loss: {train_loss:.4f} | Train RMSE: {train_rmse:.4f} | Train R: {train_r:.4f} | Train Expv: {train_expv:.4f} | '
                  f'Val loss: {val_loss:.4f} | Val RMSE: {val_rmse:.4f} | Val R: {val_r:.4f} | Val Expv: {val_expv:.4f}')

        # Update learning rate if scheduler is used
        if scheduler_option:
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            learning_rate.append(f'Epoch {epoch + 1} --> Lr: {current_lr}')

        # Early Stopping Logic (save best model without stopping training)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_wts = model.state_dict()  

            
            torch.save(best_model_wts, f'{result_model_file}_Model_{type_model}_{auroral_index}_shift_{shift}.pt')

    metrics_df = pd.DataFrame({
        f'Train_Loss_{shift}': train_losses,
        f'Train_RMSE_{shift}': train_rmse_score,
        f'Train_R_Score_{shift}': train_r_score,
        f'Train_ExpVar_{shift}': train_expv_score,
        f'Valid_Loss_{shift}': val_losses,
        f'Valid_RMSE_{shift}': val_rmse_score,
        f'Valid_R_Score_{shift}': val_r_score,
        f'Valid_ExpVar_{shift}': val_expv_score,
    })

    learning_rate_df = pd.DataFrame(learning_rate, columns=['Learning_Rate'])

    model.load_state_dict(best_model_wts)

    return model, metrics_df, learning_rate_df, best_val_loss



###### [ Test Model ] ######
def test_model(model, criterion, test_loader, result_model_file, type_model, auroral_index, num_epoch, df_epoch_test, shift):
    """
    Function to evaluate a neural network model using a test dataset.
    
    Parameters:
    - model: the trained neural network model to be evaluated.
    - criterion: loss function used to compute the test loss.
    - test_loader: data loader for the test dataset.
    - save_result_model: path where the trained model was saved.
    - type_model, auroral_index: identifiers used in the saved model filename.
    - num_epoch: the epoch number corresponding to the best model.
    - df_epoch_test: list or dataframe of epoch numbers for the test set.
    - shift: identifier for data shift, used in the saved model filename.
    
    Returns:
    - results_df: dataframe containing real and predicted values for the test set.
    - metrics_df: dataframe containing test loss, RMSE, and R score.
    """
    test_losses, test_rmse_score, test_r_score, test_expv_score = [], [], [], []
    real, pred = [], []

    ## [ Load the Best Model ] ##
    model.load_state_dict(torch.load(f'{result_model_file}_Model_{type_model}_{auroral_index}_shift_{shift}.pt'))
    model.eval()  

    test_loss, test_rmse, test_r, test_expv = 0, 0, 0, 0

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

    
    test_rmse, test_expv, test_r = calculate_metrics(all_real, all_pred)

    
    test_r_score.append(test_r)
    test_losses.append(test_loss)
    test_rmse_score.append(test_rmse)
    test_expv_score.append(test_expv)

    print(f'Test Loss: {test_loss:.4f}, Test R: {test_r:.4f}, Test RMSE: {test_rmse:.4f}, Test ExpVar: {test_expv:.4f}')
    
    if len(all_real) > len(df_epoch_test):
        all_real = all_real[:-1]
        all_pred = all_pred[:-1]

    ## [ Save Metrics and Results ] ##
    # Create a dataframe for test metrics
    metrics_df = pd.DataFrame({
        'Test_Loss': test_losses,
        'Test_RMSE': test_rmse_score,
        'Test_R_Score': test_r_score,
        'Test ExpVar': test_expv_score
    })


    # Create a dataframe for real and predicted test values
    results_df = pd.DataFrame({
        'Epoch': df_epoch_test,
        f'Test_Real': all_real.numpy().flatten().tolist(),
        f'Test_Pred_{shift}': all_pred.numpy().flatten().tolist()
    })

    return results_df, metrics_df