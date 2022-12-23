import torch.nn as nn
import logging
import time
import torch
# from utils.metric import Metric
from Bert_TCN.finance.utils.test import Test
import numpy as np
from gui import *

logger = logging.getLogger(__name__)


class TCNTrainer:

    def __init__(self,trainloader,test_loader,model,optimizer,criterion,device,load_path,output_size):
        self.trainloader=trainloader
        self.testloader=test_loader
        self.load_path=load_path
        self.test = Test(criterion,output_size)
        Test.initial_time=time.time()
        self.model=model
        self.device=device
        self.optimizer=optimizer
        self.criterion = criterion

    def change_model(self,model):
        self.model=model
        


    def run(self,epochs,load,root):
        # metric = Metric()
        # if(load):
        #     epochs=0
        #     epoch=0
        train_loss_list = []
        validation_loss_list = []
        min_test_loss = 1e10
        for epoch in range(epochs):
        
            root.root.update_idletasks()
            gen_increment = 100/epochs
            root.gen_bar['value'] += gen_increment
            root.gen_label['text'] = "Epochs: " + str(epoch+1) + "/" + str(epochs)
            root.epoch_bar['value'] = 0
            
            logger.info("epoch number: {}".format(epoch))
            loss_train = 0
            total_num = 0
            for i, data in enumerate(self.trainloader, 0):
            
                root.root.update_idletasks()
                epoch_increment = 100/(len(self.trainloader)-1)
                root.epoch_bar['value'] += epoch_increment
                if root.epoch_bar['value'] > 100:
                    root.epoch_bar_label['text'] = "Epoch Progresss: 100.00%"
                else:
                    root.epoch_bar_label['text'] = "Epoch Progresss: {:.2f}%".format(root.epoch_bar['value'])
                
                
                inputs, labels = data
                # print(inputs.shape)
                inputs=inputs.to(self.device)
                self.model.train()
                self.optimizer.zero_grad()
                outputs =self.model(inputs.float())
                # print(outputs)
                loss = self.criterion(outputs, labels.to(self.device).float())

                # metric.update(outputs, labels.to(self.device), loss)
                loss.backward()
                self.optimizer.step()
                loss.detach_()
                loss_train+=loss.item()*labels.shape[0]
                total_num += labels.shape[0]
            print("loss train: {}".format(loss_train/total_num))
            
            
            #root.loss_plot.scatter(epoch, loss_train/total_num)
            root.train_loss_plot.scatter(epoch, loss_train/total_num, color = '#88c999') #'.r-'
            #root.train_loss_plot.plot(epoch, loss_train/total_num)
            #plt.show()
            
            #root.fig.pause(0.05)
            # train_loss_list.append(loss.cpu().detach().numpy())
            train_loss_list.append(loss_train/total_num)
            # metric.log(epoch)
            # metric.reset_params()
            test_loss, outputs, label_list = self.test.get_accuracy_test(self.testloader, self.model, self.device, epoch)
            validation_loss_list.append(test_loss)


            if test_loss < min_test_loss:
                min_test_loss = test_loss
                print('saving model...')
                model_name = ".\TCN\saved_model\weight_" + str(epoch) + '.pth'
                torch.save(self.model.state_dict(), self.load_path, _use_new_zipfile_serialization = False)
                # torch.save(self.model.state_dict(), model_name, _use_new_zipfile_serialization=False)
            print("Minimum L1 Loss: {}".format(min_test_loss))
            print('\n')
            
            root.val_loss_plot.scatter(epoch, test_loss, color = '#88c999') #'.r-'
            #plt.draw()
            #plt.pause(0.0001)
            #root.fig.show()
            #root.l1_loss_plot.plot(epoch, loss_train/total_num)
            

        # self.test.get_accuracy_test(self.testloader, self.model, self.device, epoch)
        # print('saving model...')
        # torch.save(self.model.state_dict(),self.load_path)

        train_loss_array = np.array(train_loss_list)
        validation_loss_array = np.array(validation_loss_list)
        
        root.train_loss_plot.plot(train_loss_array)
        root.val_loss_plot.plot(validation_loss_array)

        np.savetxt('train_loss.txt', train_loss_array)
        np.savetxt('validation_loss.txt', validation_loss_array)
