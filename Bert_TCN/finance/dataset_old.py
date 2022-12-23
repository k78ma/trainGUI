# hardcoded is used to identify lines which are quick fix

import pandas as pd
import numpy as np
from graphics.vision import Vision
import sys
import copy
import math

np.set_printoptions(threshold=sys.maxsize)
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader
from Bert_TCN.finance.io.bert_processor import BertProcessor
from Bert_TCN.finance.configs.basic_config import config

logger = logging.getLogger(__name__)


class AbstractDataPreProcessor():

    def __init__(self, batch_size, data_base_path, Text_label, first_label, first_pred, last_pred, useless_labels, ordinal_labels, duplicate_labels,
                 date_time_labels, bool_labels,
                 test_ratio,
                 device, seed,
                 transform=None):
        assert isinstance(useless_labels, list) and isinstance(date_time_labels, list) and isinstance(bool_labels,
                                                                                                      list) and isinstance(
            ordinal_labels, list) and isinstance(duplicate_labels, list)
        self.batch_size = batch_size
        self.seed = seed
        self.device = device
        self.test_ratio = test_ratio
        self._data_base = pd.read_csv(data_base_path, encoding="ISO-8859-1")

    def modify_x_labels(self, *args):
        raise NotImplementedError

    def modify_date_time(self, *args):
        raise NotImplementedError


class DataPreProcessor(AbstractDataPreProcessor):

    def __init__(self, batch_size, data_base_path, Text_label, first_label, first_pred, last_pred, useless_labels, ordinal_labels,
                 duplicate_labels,
                 date_time_labels, bool_labels,
                 test_ratio, device, seed, visualize):
        """
            data_base_path: str
            y_labels:list of strs
            useless_labels: list of strs
            date_time_labels: list of strs
            bool_labels: list of strs
            valid_ratio: float
            test_ratio: float
        """
        super(DataPreProcessor, self).__init__(batch_size, data_base_path, Text_label, first_label, first_pred, last_pred, useless_labels, ordinal_labels,
                                               duplicate_labels, date_time_labels, bool_labels,
                                               test_ratio, device, seed, visualize)
        self.original_csv = self._data_base
        # print("Size of dataset is: {}".format(self._data_base.shape[0]))
        self.remove_nonEnglishText_and_Testdata(Text_label, first_label)
        # print("Size of dataset is: {}".format(self._data_base.shape[0]))
        # print(self._data_base)



        self.find_train_date(date_time_labels)
        # print(self._data_base)

        self.remove_af2012('ProjectStartDateYear')
        # print(self._data_base)






        self.modify_x_labels(useless_labels, date_time_labels, bool_labels, duplicate_labels, ordinal_labels)
        # print(self._data_base)








        input_ids_dataset, input_mask_dataset, segment_ids_dataset = self.creat_Textfeature(Text_label)
        # print(self._data_base.columns)
        self._data_base = self._data_base.drop('ProjectStartDateYear', axis=1)

        # print(self._data_base.columns)

        # print('ddd')
        # self._data_base.to_csv('input_training.csv', index=False)

        if visualize:
            vision = Vision(self._data_base)
            vision.run()
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        #     logger.info(self._data_base.describe(include = 'all').transpose())
        # print("Size of dataset is: {}".format(self._data_base.shape[0]))
        # print(self._data_base)


        # self._data_base=self._data_base.sample(frac=1)
        columns = self._data_base.columns.tolist()
        first_index = self._data_base.columns.get_loc(first_pred)  # p0
        last_index = self._data_base.columns.get_loc(last_pred)  # p5
        columns = columns[first_index:last_index+1] + columns[:first_index] + columns[last_index+1:]
        self._data_base = self._data_base[columns].values

        self._data_base = np.hstack((self._data_base, input_ids_dataset, input_mask_dataset, segment_ids_dataset))
        #########################################################
        self._data_base_copy = copy.deepcopy(self._data_base)
        # print(self._data_base[:,:last_index-first_index+1])
        # print('\n')
        # self._data_base[:, 0] = self._data_base_copy[:, 0] / self._data_base_copy[:, 1]
        for i in range(1, last_index - first_index + 1):
            # self._data_base[:,i] = (self._data_base_copy[:,i]/self._data_base_copy[:,0])**(1/i)-0.99999999 # cagr
            self._data_base[:, i] = np.exp((self._data_base_copy[:, i] / self._data_base_copy[:, 0]) ** (1 / i) - 0.99999999) # e^x


        # self._data_base[:, 0] = 1        # cagr
        self._data_base[:, 0] = np.exp(1)  # e^x


        d_list = []
        for i in range(self._data_base.shape[0]):
            for j in range(1, last_index - first_index + 1):

                if self._data_base[i, j] > 20 : #or self._data_base[i, j] < 0.1  # e^x
                # if self._data_base[i, j] > 20 or self._data_base[i, j] < 1:  # e^x + no neg
                # if self._data_base[i, j] > 3:  # or self._data_base[i, j] < 0.1 # cagr
                    d_list.append(i)
                    # print(self._data_base_copy[i,j-1])
                    # print(self._data_base_copy[i, j])
                    # print('\n')
                    break
        # print(d_list)
        self._data_base = np.delete(self._data_base, d_list, axis=0)
        # print(self._data_base[:,1])

        # for i in range(self._data_base.shape[0]):
        #     for j in range(1, last_index - first_index + 1):
        #         if self._data_base[i, j] >=1:
        #
        #             self._data_base[i, j] = 1
        #         if self._data_base[i, j] < 1:
        #
        #             self._data_base[i, j] = -1

        # self._data_base[467, 2] = 3
        # self._data_base[496, 2] = 3
        ################################################################
        # print(self._data_base.shape[0])
        test_size = int(test_ratio * self._data_base.shape[0])
        # print(test_size)
        np.random.shuffle(self._data_base)
        X_train,X_test= self._data_base[test_size:],self._data_base[:test_size]

        # m1_min = min(X_train[:,0])
        # m1_max = max(X_train[:,0])
        # p0_min = min(X_train[:,1])
        # p0_max = max(X_train[:,1])
        # p1_min = min(X_train[:,2])
        # p1_max = max(X_train[:,2])
        # p2_min = min(X_train[:,3])
        # p2_max = max(X_train[:,3])
        # p3_min = min(X_train[:,4])
        # p3_max = max(X_train[:,4])
        # p4_min = min(X_train[:,5])
        # p4_max = max(X_train[:,5])

        # print(m1_min)
        # print(m1_max)

        # print(X_train[:, 0])
        # x_train, x_test = self.scale(X_train, X_test, last_index-first_index)
        # print(x_train[:, 0])

        x_train, x_test = X_train, X_test

        # x_test[:, 2]=x_test[:, 1]
        # x_test[:, 3] = x_test[:, 1]
        # x_test[:, 4] = x_test[:, 1]
        # x_test[:, 5] = x_test[:, 1]
        # x_test[:, 6] = x_test[:, 1]



        # x_train, x_test = X_train, X_test
        ############################################Training set balance by duplication and shuffle############################################
        # all_pos_count = 0
        # all_neg_count = 0
        # for i in range(x_train.shape[0]):
        #
        #     if x_train[i, 3] >= 1.2:
        #         all_pos_count += 1
        #     else:
        #         all_neg_count += 1
        #
        #
        # print('ddddddddddddddddddddd')
        # print(all_neg_count)
        # print(all_pos_count)
        #
        # new_x_train = copy.deepcopy(x_train)
        #
        # while (all_pos_count < 1*all_neg_count):
        #     for i in range(x_train.shape[0]):
        #
        #
        #         if x_train[i, 3] >= 1.2:
        #             new_x_train = np.vstack((new_x_train, x_train[i]))
        #             all_pos_count += 1
        #
        #
        #
        #         if all_pos_count >= 1*all_neg_count:
        #             break
        #
        # print(new_x_train.shape[0])
        # print(all_neg_count)
        # print(all_pos_count)
        #
        # x_train = copy.deepcopy(new_x_train)
        # print(x_train.shape[0])
        ##########################################################################################################################

        # all_pos_count = 0
        # all_neg_count = 0
        # for i in range(x_train.shape[0]):
        #
        #     if x_train[i, 3] >= 0.2:
        #         all_pos_count += 1
        #     else:
        #         all_neg_count += 1
        #
        #
        # print('ddddddddddddddddddddd')
        # print(all_neg_count)
        # print(all_pos_count)
        #
        # new_x_train = copy.deepcopy(x_train)
        #
        # while (all_pos_count < all_neg_count):
        #     for i in range(x_train.shape[0]):
        #
        #
        #         if x_train[i, 3] >= 0.2:
        #             new_x_train = np.vstack((new_x_train, x_train[i]))
        #             all_pos_count += 1
        #
        #
        #
        #         if all_pos_count >= all_neg_count:
        #             break
        #
        # print(new_x_train.shape[0])
        # print(all_neg_count)
        # print(all_pos_count)
        #
        # x_train = copy.deepcopy(new_x_train)
        # print(x_train.shape[0])

        # ############################################Training set balance by duplication and shuffle############################################
        # all_pos_count = 0
        # all_neg_count = 0
        # for i in range(x_train.shape[0]):
        #     pos_count = 0
        #     neg_count = 0
        #     for j in range(1, last_index - first_index + 1):
        #         if x_train[i, j] >= 0:
        #             pos_count += 1
        #         else:
        #             neg_count += 1
        #     if neg_count > pos_count:
        #         all_neg_count += 1
        #     else:
        #         all_pos_count += 1
        # print('ddddddddddddddddddddd')
        # print(all_neg_count)
        # print(all_pos_count)
        #
        # new_x_train = copy.deepcopy(x_train)
        #
        # while (all_neg_count < all_pos_count):
        #     for i in range(x_train.shape[0]):
        #         pos_count = 0
        #         neg_count = 0
        #         for j in range(1, last_index - first_index + 1):
        #             if x_train[i, j] >= 0:
        #                 pos_count += 1
        #             else:
        #                 neg_count += 1
        #         if neg_count > pos_count:
        #             new_x_train = np.vstack((new_x_train, x_train[i]))
        #             all_neg_count += 1
        #
        #         if all_neg_count >= all_pos_count:
        #             break
        #
        # print(new_x_train.shape[0])
        # print(all_neg_count)
        # print(all_pos_count)
        #
        # x_train = copy.deepcopy(new_x_train)
        # print(x_train.shape[0])

        #############################################################################################################


        #############################################################################################################

        self.x_train,self.y_train= self.create_record(x_train,0,last_index-first_index)
        self.x_test, self.y_test = self.create_record(x_test, 0, last_index-first_index)
        self.original_input_size = self.x_train.shape[1] - input_ids_dataset.shape[1] - input_mask_dataset.shape[1] - segment_ids_dataset.shape[1]
        self.input_ids_size = input_ids_dataset.shape[1]
        self.input_mask_size = input_mask_dataset.shape[1]
        self.segment_ids_size = segment_ids_dataset.shape[1]
        self.output_size = self.y_train.shape[1]


    def create_record(self, dataset,start, end):
        x_records=[]
        for k in range(dataset.shape[0]):
            x_labels = []
            for i in range(start, end):
                x_labels.append(dataset[k][np.r_[i, end + 1:dataset.shape[1]]]) # add one revenue value each time.
            x_records.append(np.array(x_labels).transpose())
        y_labels = dataset[:,1:end + 1]
        return np.array(x_records), y_labels

    def scale(self, X_train, X_test, end):
        scaler = MinMaxScaler()  # normalization along column, i.e. feature channel

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)  # scaling based on X_train

        # X_train[:,end+1:] = scaler.fit_transform(X_train[:,end+1:])
        # X_test[:,end+1:] = scaler.transform(X_test[:,end+1:]) # scaling based on X_train
        return X_train, X_test

    def remove_nonEnglishText_and_Testdata(self, Text_label, first_label):

        # print(self._data_base)
        french_letters = ['à', 'â', 'è', 'é', 'ê', 'î', 'ï', 'ô', 'ö', 'ù', 'û', 'ü', 'ç', 'œ', 'æ', '€',
                          'À', 'Â', 'Ã', 'È', 'É', 'Ê', 'Ë', 'Î', 'Ï', 'Ô', 'Ö', 'Ù', 'Û', 'Ç', 'Œ', 'Æ', '€']

        # #[ 'é',
        #               'à', 'è', 'ù',
        #               'â', 'ê', 'î', 'ô', 'û',
        #               'ç',
        #               'ë', 'ï', 'ü']

        # À Â È É Ê Ë Î Ï Ô Ö Ù Û Ç Œ Æ €

        # à â è é ê î ï ô ö ù û ü ç œ æ €

        English_letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                           's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                           'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

        for i in range(self._data_base.shape[0]):
            is_french_exist = True if set(french_letters).intersection(self._data_base[Text_label][i]) else False
            no_english = False if set(English_letters).intersection(self._data_base[Text_label][i]) else True
            if is_french_exist:
                self._data_base.at[i,Text_label] = 'NaN'
            elif no_english:
                self._data_base.at[i,Text_label] = 'NaN'

            # if self._data_base['NoteType'][i] == 'TechnicalAspectsOfTheProject':  #BusinessCaseSummary
            #     self._data_base.at[i,Text_label] = 'NaN'

            if self._data_base['CAAmt'][i] < 50000:
                self._data_base.at[i,Text_label] = 'NaN'

        self._data_base = self._data_base.drop(self._data_base[self._data_base[Text_label] == 'NaN'].index)

    def find_train_date(self, date_time_labels):
        """
            dataset: DataFrame
            date_time_labels: list of str
        """
        if (date_time_labels[0] == ""):
            return
        for item in date_time_labels:
            self._data_base[str(item) + 'Year'] = pd.DatetimeIndex(self._data_base[item]).year
            self._data_base = self._data_base.drop(item, axis=1)

    def remove_af2012(self, projectstartyear_label):
        """
            dataset: DataFrame
            date_time_labels: list of str
        """
        self._data_base = self._data_base.drop(self._data_base[self._data_base[projectstartyear_label]>2012].index)
        self._data_base = self._data_base.drop(self._data_base[self._data_base[projectstartyear_label]<2001].index)

    def creat_revenue_7(self, year_label_prefix):
        """
            dataset: DataFrame
            date_time_labels: list of str
        """
        # revenue_m1_list = []
        revenue_p0_list = []
        revenue_p1_list = []
        revenue_p2_list = []
        revenue_p3_list = []
        revenue_p4_list = []
        revenue_p5_list = []
        for i in range(len(self._data_base.index)):
            k = self._data_base.index[i]
            year = self._data_base.at[k, 'ProjectStartDateYear']
            # year_lable_m1 = year_label_prefix + str(year - 1)
            year_lable_p0 = year_label_prefix + str(year)
            year_lable_p1 = year_label_prefix + str(year + 1)
            year_lable_p2 = year_label_prefix + str(year + 2)
            year_lable_p3 = year_label_prefix + str(year + 3)
            year_lable_p4 = year_label_prefix + str(year + 4)
            year_lable_p5 = year_label_prefix + str(year + 5)

            # revenue_m1_list.append(self._data_base.at[k, year_lable_m1])
            revenue_p0_list.append(self._data_base.at[k, year_lable_p0])
            revenue_p1_list.append(self._data_base.at[k, year_lable_p1])
            revenue_p2_list.append(self._data_base.at[k, year_lable_p2])
            revenue_p3_list.append(self._data_base.at[k, year_lable_p3])
            revenue_p4_list.append(self._data_base.at[k, year_lable_p4])
            revenue_p5_list.append(self._data_base.at[k, year_lable_p5])

        # self._data_base['revenue_m1'] = revenue_m1_list
        self._data_base['revenue_p0'] = revenue_p0_list
        self._data_base['revenue_p1'] = revenue_p1_list
        self._data_base['revenue_p2'] = revenue_p2_list
        self._data_base['revenue_p3'] = revenue_p3_list
        self._data_base['revenue_p4'] = revenue_p4_list
        self._data_base['revenue_p5'] = revenue_p5_list


    def modify_x_labels(self, useless_labels, date_time_labels, bool_labels, duplicate_labels, ordinal_labels,
                        null=True):
        """
            useless_labels: list of str
            date_time_labels: list of str
            bool_labels: list of str
        """
        self.remove_useless_labels(useless_labels)
        self.remove_duplicates(duplicate_labels)

        self.creat_revenue_7('Revenue')

        useless_labels_new = ['2020', 'Revenue2000', 'Revenue2001', 'Revenue2002', 'Revenue2003', 'Revenue2004', 'Revenue2005',
                              'Revenue2006', 'Revenue2007', 'Revenue2008', 'Revenue2009', 'Revenue2010', 'Revenue2011',
                              'Revenue2012', 'Revenue2013', 'Revenue2014', 'Revenue2015', 'Revenue2016', 'Revenue2017',
                              'Revenue2018', 'Revenue2019', 'Revenue2020', 'Revenue2021', 'Revenue2022',
                              'RevenueGrowthPct2000', 'RevenueGrowthPct2001', 'RevenueGrowthPct2002',
                              'RevenueGrowthPct2003', 'RevenueGrowthPct2004', 'RevenueGrowthPct2005',
                              'RevenueGrowthPct2006', 'RevenueGrowthPct2007', 'RevenueGrowthPct2008',
                              'RevenueGrowthPct2009', 'RevenueGrowthPct2010', 'RevenueGrowthPct2011',
                              'RevenueGrowthPct2012', 'RevenueGrowthPct2013', 'RevenueGrowthPct2014',
                              'RevenueGrowthPct2015', 'RevenueGrowthPct2016', 'RevenueGrowthPct2017',
                              'RevenueGrowthPct2018', 'RevenueGrowthPct2019', 'RevenueGrowthPct2020',
                              'RevenueGrowthPct2021', 'RevenueGrowthPct2022']

        # self.remove_useless_labels(useless_labels_new)
        #
        # self.modify_ordinals(ordinal_labels)
        # # self.modify_date_time(date_time_labels)
        # # self.modify_bool_labels(bool_labels)
        # self._data_base = self._data_base.dropna()
#############################################################################
        self.remove_useless_labels(useless_labels_new)
        # print(self._data_base.columns)
        self._data_base = self._data_base.dropna()

        self.modify_ordinals(ordinal_labels)
        # self.modify_date_time(date_time_labels)
        # self.modify_bool_labels(bool_labels)

    # def modify_ordinals(self, ordinal_labels):
    #     self._data_base = pd.get_dummies(self._data_base, columns=ordinal_labels)
    def modify_ordinals(self, ordinal_labels): # FinancialMonitoringRating Province NoteType
        # self._data_base = pd.get_dummies(self._data_base, columns=ordinal_labels)

        self._data_base['FinancialMonitoringRating_High'] = 0
        self._data_base['FinancialMonitoringRating_Low'] = 0
        self._data_base['FinancialMonitoringRating_Medium'] = 0
        self._data_base['FinancialMonitoringRating_PPVNotRequired'] = 0
        self._data_base['Province_AB'] = 0
        self._data_base['Province_BC'] = 0
        self._data_base['Province_MB'] = 0
        self._data_base['Province_NB'] = 0
        self._data_base['Province_NL'] = 0
        self._data_base['Province_NS'] = 0
        self._data_base['Province_NT'] = 0
        self._data_base['Province_ON'] = 0
        self._data_base['Province_PE'] = 0
        self._data_base['Province_QC'] = 0
        self._data_base['Province_SK'] = 0
        self._data_base['Province_YT'] = 0
        self._data_base['NoteType_BusinessCaseSummary'] = 0
        self._data_base['NoteType_TechnicalAspectsOfTheProject'] = 0

        for i in range(len(self._data_base.index)):
            k = self._data_base.index[i]
            # print(self._data_base.at[k, 'FinancialMonitoringRating'])

            if self._data_base.at[k, 'FinancialMonitoringRating'] == 'High':
                self._data_base.at[k, 'FinancialMonitoringRating_High'] = 1
            elif self._data_base.at[k, 'FinancialMonitoringRating'] == 'Low':
                self._data_base.at[k, 'FinancialMonitoringRating_Low'] = 1
            elif self._data_base.at[k, 'FinancialMonitoringRating'] == 'Medium':
                self._data_base.at[k, 'FinancialMonitoringRating_Medium'] = 1
            elif self._data_base.at[k, 'FinancialMonitoringRating'] == 'PPVNotRequired':
                self._data_base.at[k, 'FinancialMonitoringRating_PPVNotRequired'] = 1
            else:
                print('FMR: wrong')

            if self._data_base.at[k, 'Province'] == 'AB':
                self._data_base.at[k, 'Province_AB'] = 1
            elif self._data_base.at[k, 'Province'] == 'BC':
                self._data_base.at[k, 'Province_BC'] = 1
            elif self._data_base.at[k, 'Province'] == 'MB':
                self._data_base.at[k, 'Province_MB'] = 1
            elif self._data_base.at[k, 'Province'] == 'NB':
                self._data_base.at[k, 'Province_NB'] = 1
            elif self._data_base.at[k, 'Province'] == 'NL':
                self._data_base.at[k, 'Province_NL'] = 1
            elif self._data_base.at[k, 'Province'] == 'NS':
                self._data_base.at[k, 'Province_NS'] = 1
            elif self._data_base.at[k, 'Province'] == 'NT':
                self._data_base.at[k, 'Province_NT'] = 1
            elif self._data_base.at[k, 'Province'] == 'ON':
                self._data_base.at[k, 'Province_ON'] = 1
            elif self._data_base.at[k, 'Province'] == 'PE':
                self._data_base.at[k, 'Province_PE'] = 1
            elif self._data_base.at[k, 'Province'] == 'QC':
                self._data_base.at[k, 'Province_QC'] = 1
            elif self._data_base.at[k, 'Province'] == 'SK':
                self._data_base.at[k, 'Province_SK'] = 1
            elif self._data_base.at[k, 'Province'] == 'YT':
                self._data_base.at[k, 'Province_YT'] = 1
            else:
                print('Province: wrong')

            if self._data_base.at[k, 'NoteType'] == 'BusinessCaseSummary':
                self._data_base.at[k, 'NoteType_BusinessCaseSummary'] = 1
            elif self._data_base.at[k, 'NoteType'] == 'TechnicalAspectsOfTheProject':
                self._data_base.at[k, 'NoteType_TechnicalAspectsOfTheProject'] = 1
            else:
                print('NoteType: wrong')

        self._data_base = self._data_base.drop('FinancialMonitoringRating', axis=1)
        self._data_base = self._data_base.drop('Province', axis=1)
        self._data_base = self._data_base.drop('NoteType', axis=1)

    def modify_bool_labels(self, bool_labels):
        """
            dataset: DataFrame
            bool_labels: list of str
        """
        # 0 refers to False and 1 refers to True
        if (bool_labels[0] == ""):
            return
        for item in bool_labels:
            self._data_base[item] = (self._data_base[item] == True).astype(int)





    def modify_date_time(self, date_time_labels):
        """
            dataset: DataFrame
            date_time_labels: list of str
        """
        if (date_time_labels[0] == ""):
            return
        for item in date_time_labels:
            self._data_base[str(item) + 'Year'] = pd.DatetimeIndex(self._data_base[item]).year
            self._data_base[str(item) + 'Month'] = pd.DatetimeIndex(self._data_base[item]).month
            self._data_base[str(item) + 'Day'] = pd.DatetimeIndex(self._data_base[item]).day
            self._data_base = self._data_base.drop(item, axis=1)

    def remove_duplicates(self, duplicate_labels):
        self._data_base = self._data_base.drop_duplicates(subset=duplicate_labels)
        # hardcoded
        # self._data_base["OrgName"] = list(range(1, len(self._data_base) + 1)) #reset the value of the OrgName after drop duplicates

    def remove_useless_labels(self, useless_labels):
        """
            dataset: DataFrame
            useless_labels: list of str
        """
        if (useless_labels[0] == ""):
            return
        if len(useless_labels) == 0:
            return

        self._data_base = self._data_base.drop(useless_labels, axis=1)
        # hardcoded
        self._data_base = self._data_base[self._data_base.columns.drop(list(self._data_base.filter(regex="Employee")))]

    def creat_Textfeature(self, Text_label):
        text_data = self._data_base[Text_label].tolist()

        # for i in range(len(text_data)):
        #     text_data[i] = 'None'

        processor = BertProcessor(vocab_path='./Bert_TCN/finance/pretrain/bert/base-uncased/bert_vocab.txt', do_lower_case=True)

        text_examples = processor.create_examples(lines=text_data,
                                                   example_type='test',
                                                   cached_examples_file=config[
                                                'data_dir'] / f"cached_text_examples")



        text_features = processor.create_features(examples=text_examples,
                                                   max_seq_len=256,
                                                   cached_features_file=config[
                                                'data_dir'] / f"cached_text_features")


        # print(len(text_features))

        input_ids_set = []
        input_mask_set = []
        segment_ids_set = []
        for f in text_features:
            input_ids_set.append(f.input_ids)
            input_mask_set.append(f.input_mask)
            segment_ids_set.append(f.segment_ids)

        self._data_base = self._data_base.drop(Text_label, axis=1)

        return np.array(input_ids_set), np.array(input_mask_set), np.array(segment_ids_set)










