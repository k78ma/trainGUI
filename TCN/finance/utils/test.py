import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import logging
import numpy as np
from TCN.finance.utils.graphics import bar_plot
import time
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef

# from utils.file_manager import File_Manager
# from utils.tensor_writer import Tensor_Writer

logger = logging.getLogger(__name__)


class Test():
    initial_time = 0

    def __init__(self,criterion,output_size):
        self.results_test = {"accuracy": [], "epoch": []}
        self.results_valid = {"accuracy": [], "epoch": []}
        self.output_size=output_size
        self.criterion = nn.L1Loss()

    def get_accuracy_test(self, test_loader, model, device, epoch):
        model.eval()
        current_time = time.time()
        self.cal_correct(test_loader, model, device)
        passed_time = time.time() - current_time
        print("passed time so far: {}".format(current_time-Test.initial_time))
        self.results_test["epoch"].append(epoch)

    def cal_correct(self, loader, model, device):
        model.eval()
        correct = 0
        total = 0
        sum_loss = 0
        label_list=[[] for _ in range(self.output_size)]
        pred_list=[[] for _ in range(self.output_size)]
        with torch.no_grad():
            for data in loader:
                images, labels = data
                # import pdb;pdb.set_trace()
                outputs = model(images.to(device).float())
                sum_loss+=self.criterion(outputs,labels.float()).item()*labels.shape[0]
                total+=labels.shape[0]
                label_list=self.add_to_list(label_list,labels)
                pred_list=self.add_to_list(pred_list,outputs)
        print("L1 Loss: {}".format(sum_loss/total))
        # self.draw(pred_list,label_list)
        print(outputs[0])
        print(labels[0])

    def draw(self,pred,labels):
        pred=np.array(pred).mean(axis=1)
        labels = np.array(labels).mean(axis=1)
        years=[str(i) for i in range(2012,2018)]
        data = {
            "pred": pred,
            "target": labels
        }

        fig, ax = plt.subplots()
        bar_plot(ax, data, total_width=.8, single_width=.9)
        plt.xlabel("Year")
        plt.ylabel("Mean Revenue")
        plt.show()

    def add_to_list(self,l1,tensor):
        l2=tensor.numpy().T.tolist()
        for i in range(len(l1)):
            l1[i].extend(l2[i])
        return l1