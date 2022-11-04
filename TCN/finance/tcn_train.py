import torch.nn as nn
import logging
import time
import torch
# from utils.metric import Metric
from TCN.finance.utils.test import Test



logger = logging.getLogger(__name__)

def update_progress_label(pb):
    return f"Current Progress: {pb['value']}%"


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

    def run(self, epochs, load):
        # metric = Metric()
        # if(load):
        #     epochs=0
        #     epoch=0
        
        root.select_dir()
    
        for epoch in range(epochs):
        
            
            root.gen_bar['value'] += (100/epochs)
            root.update_idletasks()
            
            logger.info("epoch number: {}".format(epoch))
            loss_train = 0
            for i, data in enumerate(self.trainloader, 0):
            
                root.epoch_bar['value'] += (100/len(data))
                root.update_idletasks()
                
                inputs, labels = data
                print("TESTTESTTESTTEST")
                # print(inputs.shape)
                inputs=inputs.to(self.device)
                self.model.train()
                self.optimizer.zero_grad()
                outputs =self.model(inputs.float())
                # print(outputs.shape)
                loss = self.criterion(outputs, labels.to(self.device).float())
                # metric.update(outputs, labels.to(self.device), loss)
                loss.backward()
                self.optimizer.step()
                loss.detach_()
                loss_train+=loss.item()
            print("loss train: {}".format(loss_train))
            # metric.log(epoch)
            # metric.reset_params()
            self.test.get_accuracy_test(self.testloader, self.model, self.device, epoch)
        self.test.get_accuracy_test(self.testloader, self.model, self.device, epoch)
        torch.save(self.model.state_dict(),self.load_path)

