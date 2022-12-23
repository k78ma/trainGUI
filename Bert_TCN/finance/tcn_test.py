import torch.nn as nn
import logging
import time
import torch
# from utils.metric import Metric
from Bert_TCN.finance.utils.test import Test

logger = logging.getLogger(__name__)


class TCNTester:

    def __init__(self,test_loader,model,optimizer,criterion,device,load_path,output_size):
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

    def run(self,epochs,load):
        # metric = Metric()
        # if(load):
        #     epochs=0
        #     epoch=0
        min_test_loss = 1e10
        epoch = 0
        logger.info("epoch number: {}".format(epoch))

        test_loss, outputs, label_list = self.test.get_accuracy_test(self.testloader, self.model, self.device, epoch)

        return outputs, label_list



        # self.test.get_accuracy_test(self.testloader, self.model, self.device, epoch)
        # print('saving model...')
        # torch.save(self.model.state_dict(),self.load_path)

