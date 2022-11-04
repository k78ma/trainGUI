import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import logging
import copy
import numpy as np
from TCN.finance.utils.graphics import bar_plot
import time
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef

import math

# from utils.file_manager import File_Manager
# from utils.tensor_writer import Tensor_Writer

logger = logging.getLogger(__name__)


class Test():
    initial_time = 0

    def __init__(self,criterion,output_size):
        self.results_test = {"accuracy": [], "epoch": []}
        self.results_valid = {"accuracy": [], "epoch": []}
        self.output_size=output_size
        # self.criterion = nn.L1Loss()
        self.criterion = nn.MSELoss()

    def get_accuracy_test(self, test_loader, model, device, epoch):
        model.eval()
        current_time = time.time()
        test_loss, outputs, label_list = self.cal_correct(test_loader, model, device)
        passed_time = time.time() - current_time
        print("passed time so far: {}".format(current_time-Test.initial_time))
        self.results_test["epoch"].append(epoch)
        return test_loss, outputs, label_list

    def cal_correct(self, loader, model, device):
        model.eval()
        correct = 0
        total = 0
        sum_loss = 0

        A_cagr = []
        B_cagr = []

        TP = 0
        FP = 0
        FN = 0

        max_pred_cagr = -100000000
        min_pred_cagr = 10000000000


        label_list=[[] for _ in range(self.output_size)]
        pred_list=[[] for _ in range(self.output_size)]
        with torch.no_grad():
            for data in loader:
                images, labels = data
                # import pdb;pdb.set_trace()

                outputs = model(images.to(device).float())
                # print(outputs)

                output_copy = copy.deepcopy(outputs)


                labels = labels.to(device).float()
                # print(labels.shape)

                labels_copy = copy.deepcopy(labels)
                # print(labels_copy)


                for k in range(labels_copy.shape[0]):

                    # B_cagr.append(math.log(labels_copy[k, 2]))
                    # if math.log(labels_copy[k, 2]) > 0.2:
                    #
                    #     if math.log(output_copy[k, 2]) > 0.2:
                    #         TP += 1
                    #     else:
                    #         FN += 1
                    #
                    # if math.log(output_copy[k, 2]) > 0.2:
                    #     A_cagr.append(math.log(labels_copy[k, 2]))
                    #     if math.log(labels_copy[k, 2]) < 0.2:
                    #         FP += 1
                    #
                    # if math.log(output_copy[k, 2]) > max_pred_cagr:
                    #     max_pred_cagr = math.log(output_copy[k, 2])
                    # if math.log(output_copy[k, 2]) < min_pred_cagr:
                    #     min_pred_cagr = math.log(output_copy[k, 2])

                    B_cagr.append(labels_copy[k, 2].cpu())
                    if labels_copy[k, 2] > 0.2:

                        if output_copy[k, 2] > 0.2:
                            TP += 1
                        else:
                            FN += 1

                    if output_copy[k, 2] > 0.2:
                        A_cagr.append(labels_copy[k, 2].cpu())
                        if labels_copy[k, 2] < 0.2:
                            FP += 1












                # outputs = outputs/labels.float()
                # labels[:,:] = 1
                # print(labels.shape[0])
                sum_loss+=self.criterion(outputs,labels).item()*labels.shape[0]
                total+=labels.shape[0]
                label_list=self.add_to_list(label_list,labels)
                # print(len(label_list[0]))
                pred_list=self.add_to_list(pred_list,outputs)

        if (TP + FP) != 0 and (TP + FN) != 0:
            pre = TP/(TP + FP)
            recall = TP / (TP + FN)
            loss = 1/((pre*recall)/(pre+recall))

        print(total)
        print("Val Loss: {}".format(sum_loss/total))

        print('New metric:')
        print(np.mean(A_cagr)/np.mean(B_cagr))

        print("Precision:")
        print(TP)
        print(TP + FP)
        if (TP + FP) != 0 and (TP + FN) != 0:
            print(TP/(TP + FP))

        print("Recall:")
        print(TP)
        print(TP + FN)
        if (TP + FP) != 0 and (TP + FN) != 0:
            print(TP / (TP + FN))

        print('\n')
        # print(max_pred_cagr)
        # print(min_pred_cagr)






        return sum_loss/total, pred_list, label_list

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
        l2=tensor.cpu().numpy().T.tolist()
        for i in range(len(l1)):
            l1[i].extend(l2[i])
        return l1