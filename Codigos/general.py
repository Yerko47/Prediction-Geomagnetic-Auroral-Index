import time
from sklearn.metrics import r2_score, accuracy_score

from models import *
from read_cdf import *
from variables import *
from performance import *
from save_models_info import *

from plots import time_plot, corr_plot 
from plots import plot_loss, plot_acc, plot_r2_score, density_plot


###### [ Checks Folder ] ######
check_folder(save_raw_file)
check_folder(processing_file)
check_folder(model_file)
check_folder(result_file)
check_folder(plot_file)


###### [ Dataset Building ] ######
dataset(in_year, out_year, omni_file, save_feather, processing_file, processOMNI)


###### [ Read CDF ] ######
if os.path.exists(save_feather):
    df = pd.read_feather(save_feather)
else:
    print('The file does not exists')


###### [ Plot Option ] ######
if processPLOT:
    print('---- [ Statistical graphs are being created ] ----')
    time_plot(df, in_year, out_year, auroral_param, plot_file)
    print('[ Time Serie Plot has been created ]')
    corr_plot(df, correlation, plot_file) 
    print('[ Correlation Plot has been created ]')


###### [ Scaler ] ######
df = scaler_df(df, scaler, omni_param, auroral_param)


###### [ Porcentage Set ] ######
if storm_list:
    df_train = create_group_prediction(df, omni_param, auroral_param, 'train', storm_list, n_split_train_val_test, n_split_train_val, processing_file)
    df_val = create_group_prediction(df, omni_param, auroral_param, 'val', storm_list, n_split_train_val_test, n_split_train_val, processing_file)
    df_test = create_group_prediction(df, omni_param, auroral_param, 'test', storm_list, n_split_train_val_test, n_split_train_val, processing_file)

else:
    df_train, df_val, df_test = create_group_prediction(df, omni_param, auroral_param, 'non', storm_list, n_split_train_val_test, n_split_train_val, processing_file)

train_len = round(len(df_train)/len(df),2) * 100
val_len = round(len(df_val)/len(df),2) * 100
test_len = round(len(df_test)/len(df),2) * 100

print()
print('---------- [ Porcentage Set ] ----------')
print()
print(f'Porcentage Train Set: {train_len}% ')
print(f'Porcentage Valid Set: {val_len}% ')
print(f'Porcentage Test Set: {test_len}% ')
print()


###### [ Shift o Delay ] ######
if type_model == 'ANN':
    omni_train, index_train = shifty_1d(df_train, omni_param, auroral_index, shifty)
    omni_val, index_val = shifty_1d(df_val, omni_param, auroral_index, shifty)
    omni_test, index_test = shifty_1d(df_test, omni_param, auroral_index, shifty)
else:
    omni_train, index_train = shifty_3d(df_train, omni_param, auroral_index, shifty)
    omni_val, index_val = shifty_3d(df_val, omni_param, auroral_index, shifty)
    omni_test, index_test = shifty_3d(df_test, omni_param, auroral_index, shifty)   


print('---------- [ Dimension Set ] ----------')
print()
print(f'Dimension Train Set: OMNI--> {omni_train.shape}  |    {auroral_index.replace("_INDEX", " Index")}--> {index_train.shape}')
print(f'Dimension Valid Set: OMNI--> {omni_val.shape}  |    {auroral_index.replace("_INDEX", " Index")}--> {index_val.shape}')
print(f'Dimension Test Set: OMNI--> {omni_test.shape}  |    {auroral_index.replace("_INDEX", " Index")}--> {index_test.shape}')
print()


###### [ DataTorch and DataLoad ] ######
train = CustomDataset(omni_train, index_train, device)
val = CustomDataset(omni_val, index_val, device)
test = CustomDataset(omni_test, index_test, device)

train_loader = DataLoader(train, shuffle=True, batch_size=batch_train_val)
val_loader = DataLoader(val, shuffle=False, batch_size=batch_train_val)
test_loader = DataLoader(test, shuffle=False, batch_size=barch_test)


###### [ Neural Network Model ] ######
if type_model == 'ANN':
    print('----- [ ANN ] -----')
    input_size = omni_train.shape[1]
    model = ANN(input_size=input_size, drop=drop).to(device)
if type_model == 'LSTM':
    print('----- [ LSTM ] -----')
    input_size = omni_train.shape[2]
    model = LSTM(input_size=input_size, drop=drop, num_layer=num_layer, device=device).to(device)

if type_model == 'CNN':
    print('----- [ CNN ] -----')
    #model = CNN(input_size=input_size, drop=drop).to(device)

print('----- [ Neural Network Model ] -----')
print(model.parameters)
print()


###### [ HyperParameters ] ######
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


###### [ Training/Validation/Test Process ] ######
start_time = time.time()
print('------ [ Start to Training Model ] ------')
print()

train_loss, val_loss, test_loss = [], [], []
train_acc, val_acc, test_acc = [], [], []
train_r2, val_r2, test_r2 = [], [], []


for epoch in range(num_epoch):
    ### [ Training ] ###
    model.train()
    count_train_loss = 0
    label_train_real, label_train_pred = [], []

    for x, y in train_loader:
        optimizer.zero_grad()
        yhat = model(x)
        loss = criterion(yhat, y)  # Usar MSELoss directamente
        loss.backward()
        optimizer.step()

        count_train_loss += loss.item()
        label_train_real.append(y.detach().cpu().numpy())
        label_train_pred.append((yhat.detach().cpu().numpy()))

    train_real = pd.Series(np.concatenate(label_train_real).astype("float").flatten().reshape(-1))
    train_pred = pd.Series(np.round(np.concatenate(label_train_pred).astype("float").flatten()))

    train_accuracy = accuracy_score(train_real, train_pred)
    train_r2_score = r2_score(train_real, train_pred)
    
    # Calcular RMSE
    train_rmse = np.sqrt(count_train_loss / len(train_loader))
    train_loss.append(train_rmse)
    train_acc.append(train_accuracy)
    train_r2.append(train_r2_score)

    ### [ Validation ] ###
    model.eval()
    count_val_loss = 0
    label_val_real, label_val_pred = [], []

    with torch.no_grad():
        for x, y in val_loader:
            yhat = model(x)
            loss = criterion(yhat, y)

            count_val_loss += loss.item()
            label_val_real.append(y.detach().cpu().numpy())
            label_val_pred.append(yhat.detach().cpu().numpy())

    val_real = pd.Series(np.concatenate(label_val_real).astype("float").flatten().reshape(-1))
    val_pred = pd.Series(np.round(np.concatenate(label_val_pred).astype("float").flatten()))
    val_accuracy = accuracy_score(val_real, val_pred)
    val_r2_score = r2_score(val_real, val_pred)
    
    # Calcular RMSE
    val_rmse = np.sqrt(count_val_loss / len(val_loader))
    val_loss.append(val_rmse)
    val_acc.append(val_accuracy)
    val_r2.append(val_r2_score)

    if epoch % 10 == 0:
        print(f'Epoch {epoch+10}/{num_epoch}: '
              f'Train loss (RMSE): {train_loss[-1]:.3f}, Train acc: {train_accuracy:.3f}, Train R2 Score: {train_r2_score:.3f} | '
              f'Valid loss (RMSE): {val_loss[-1]:.3f}, Valid acc: {val_accuracy:.3f}, Valid R2 Score: {val_r2_score:.3f}')

    today = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

    if count_val_loss / len(val_loader) >= early_stop:
        print(f'Validation Loss value {val_loss[-1]:.4f} reached the limit in epoch {epoch}')
        num_epoch = epoch + 1
        break

### [ Save Model ] ###
torch.save(model.state_dict(), model_file + f'Model_{type_model}_{auroral_index}_{today}_Epoch_{num_epoch}.pt')

### [ Test ] ###
model.eval()
count_test_loss = 0
label_test_real, label_test_pred = [], []

with torch.no_grad():
    for x, y in test_loader:
        yhat = model(x)
        loss = criterion(yhat, y)

        count_test_loss += loss.item()
        label_test_real.append(y.detach().cpu().numpy())
        label_test_pred.append(yhat.detach().cpu().numpy())

test_real = pd.Series(np.concatenate(label_test_real).astype("float").flatten().reshape(-1))
test_pred = pd.Series(np.round(np.concatenate(label_test_pred).astype("float").flatten()))
test_accuracy = accuracy_score(test_real, test_pred)
test_r2_score = r2_score(test_real, test_pred)

# Calcular RMSE
test_rmse = np.sqrt(count_test_loss / len(test_loader))
test_loss.append(test_rmse)
test_acc.append(test_accuracy)
test_r2.append(test_r2_score)

print(f'Test Loss {test_loss[-1]:.4f} | Test Accuracy: {test_accuracy:.4f} | Test R2 Score: {test_r2_score:.4f}')  
print()
print('---------------------------------------------------------------------')
print('Lets Go! You win, the model has been trained, validated and tested')
print()

end_time = time.time()
total_time = end_time - start_time
diff_timedelta =timedelta(seconds=total_time)
format_diff = str(diff_timedelta)


#### [ Plot Model ] ####
plot_loss(num_epoch, train_loss, val_loss, plot_file, type_model, auroral_index)
plot_acc(num_epoch, train_acc, val_acc, plot_file, type_model, auroral_index)
plot_r2_score(num_epoch, train_r2, val_r2, plot_file, type_model, auroral_index)
density_plot(test_real, test_pred, plot_file, type_model, auroral_index, 'test') ### A mejorar


#### [ Save Info Model ] ####
save_model_txt(model, num_epoch, criterion, optimizer, train_loss, val_loss, test_loss, train_acc, val_acc, test_acc, train_r2, val_r2, test_r2, 
                df_train, df_val, df_test, format_diff, auroral_index, omni_param, type_model, learning_rate, scaler, in_year, out_year, model_file)



