# The program runs from here
# hardcoded is used to identify lines which are quick fix

from main_bert_tcn import NRC_Revenue_Prediction_Training

##############################parameters################################################
load_path = ".\TCN\saved_model\weight.pth" # save pretrained model
val_ratio = 0.32  # training set and validation set
load = False  # load the pretrained model
epochs = 100
data_path = "training_example.csv" # input data file
batch_size = 8
lr = 0.001  #learning rate
#####################################################################################

print('Start training')

NRC_Revenue_Prediction_Training (load_path,val_ratio,load,epochs,data_path,batch_size,lr)

print('End training')