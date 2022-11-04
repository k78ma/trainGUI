# hardcoded is used to identify lines which are quick fix

import pandas as pd
import numpy as np
from graphics.vision import Vision
import sys

np.set_printoptions(threshold=sys.maxsize)
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class AbstractDataPreProcessor():

    def __init__(self, batch_size, data_base_path, useless_labels, ordinal_labels, duplicate_labels,
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

    def __init__(self, batch_size, data_base_path, first_pred, last_pred, useless_labels, ordinal_labels,
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
        super(DataPreProcessor, self).__init__(batch_size, data_base_path, useless_labels, ordinal_labels,
                                               duplicate_labels, date_time_labels, bool_labels,
                                               test_ratio, device, seed)

        self.modify_x_labels(useless_labels, date_time_labels, bool_labels, duplicate_labels, ordinal_labels)
        if visualize:
            vision = Vision(self._data_base)
            vision.run()
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        #     logger.info(self._data_base.describe(include = 'all').transpose())
        print("Size of dataset is: {}".format(self._data_base.shape[0]))
        # print(self._data_base)
        # print(list(self._data_base.columns.values))
        self._data_base=self._data_base.sample(frac=1)
        columns = self._data_base.columns.tolist()
        first_index = self._data_base.columns.get_loc(first_pred)  # Revenue2011
        last_index = self._data_base.columns.get_loc(last_pred)  # Revenue2017
        columns = columns[first_index:last_index+1] + columns[:first_index] + columns[last_index+1:]
        self._data_base = self._data_base[columns].values
        test_size = int(test_ratio * self._data_base.shape[0])
        X_train,X_test= self._data_base[test_size:],self._data_base[:test_size]
        x_train, x_test = self.scale(X_train, X_test)
        # x_train, x_test = X_train, X_test

        self.x_train,self.y_train= self.create_record(x_train,0,last_index-first_index)
        # print(x_train)
        self.x_test, self.y_test = self.create_record(x_test, 0, last_index-first_index)
        self.input_size = self.x_train.shape[1]
        self.output_size = self.y_train.shape[1]

    def create_record(self, dataset,start, end):
        x_records=[]
        for k in range(dataset.shape[0]):
            x_labels = []
            for i in range(start, end):
                x_labels.append(dataset[k][np.r_[i, end + 1:dataset.shape[1]]])
            x_records.append(np.array(x_labels).transpose())
        y_labels = dataset[:,1:end + 1]
        return np.array(x_records), y_labels

    def scale(self, X_train, X_test):
        scaler = MinMaxScaler()  # normalization along column, i.e. feature channel
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test

    def modify_x_labels(self, useless_labels, date_time_labels, bool_labels, duplicate_labels, ordinal_labels,
                        null=True):
        """
            useless_labels: list of str
            date_time_labels: list of str
            bool_labels: list of str
        """
        self.remove_useless_labels(useless_labels)
        self.remove_duplicates(duplicate_labels)
        self.modify_ordinals(ordinal_labels)
        self.modify_date_time(date_time_labels)
        self.modify_bool_labels(bool_labels)
        self._data_base = self._data_base.dropna()

    def modify_ordinals(self, ordinal_labels):
        self._data_base = pd.get_dummies(self._data_base, columns=ordinal_labels)

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
        self._data_base["OrgName"] = list(range(1, len(self._data_base) + 1)) #reset the value of the OrgName after drop duplicates

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

