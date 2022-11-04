# coding: utf-8

# Random forest classification for categorical features
import pdb
from sklearn.model_selection import cross_val_score, train_test_split,GridSearchCV, KFold
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
from sklearn.preprocessing import  OneHotEncoder #, StandardScaler

import logging
logger = logging.getLogger(__name__)

from models.abstract_model import AbstractModel
import numpy as np

# TODO: Error importing AbstractModel when run 'python rf.py' for unit test

    # Maybe moving abstract_model.py to the 'classification' folder


class rf(AbstractModel):

    def __init__(self, n_estimators, max_features, seed):

        super().__init__()

        self.classifier = RandomForestClassifier(class_weight="balanced", oob_score=True, n_jobs=-1)
        
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.seed = seed
       
        self.param_grid = {'n_estimators': self.n_estimators,'max_features': self.max_features}
        
    def train(self, x_train, y_train, x_val, y_val):

        inner_cv = KFold(n_splits=4, shuffle=True, random_state=self.seed)
        score1 = 'f1_micro'
        grid_rf = GridSearchCV(self.classifier, self.param_grid, cv = inner_cv, scoring=score1, n_jobs = -1, verbose=1)
        grid_rf.fit(x_train,y_train)

        self.best_estimator = grid_rf.best_estimator_
        self.best_params = grid_rf.best_params_

        print("The best hyper-parameters to get these accuracy are :-\n", grid_rf.best_params_)
        print("The best SVM classifier is :-\n", grid_rf.best_estimator_)
    
        logger.info("The best hyper-parameters to get these accuracy are :-\n", grid_rf.best_params_)
        logger.info("The best SVM classifier is :-\n", grid_rf.best_estimator_)

        y_val = y_val.ravel()
        y_pred = self.best_estimator.predict(x_val)
        
        print("Confusion Matrix: - \n",confusion_matrix(y_val, y_pred))
        print("Classification Report: - \n",classification_report(y_val, y_pred))

        logger.info("Confusion Matrix on validation set: - \n",confusion_matrix(y_val, y_pred))
        logger.info("Classification Report for validation: - \n",classification_report(y_val, y_pred))



    def test(self, x_test, y_test):
        #assert self.best_estimator is not Null
        y_test = y_test.ravel()
        y_pred = self.best_estimator.predict(x_test)
        
        print("Confusion Matrix on test set: - \n",confusion_matrix(y_test, y_pred))
        print("Classification Report for test: - \n",classification_report(y_test, y_pred))

        logger.info("Confusion Matrix on test set: - \n",confusion_matrix(y_test, y_pred))
        logger.info("Classification Report for test: - \n",classification_report(y_test, y_pred))


if __name__ == '__main__':
    
    ######## 1. load data #
    data = pd.read_csv('../../data/car_evaluation/x_y.data', header = None)
    n_samples, n_features = data.shape
    print ('The dimensions of the data set are', n_samples, 'samples by', n_features,'features.')
    # Assigning names to the columns in the dataset
    data.columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'y']
    #data.sample(10)
    #['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'acceptability']


    ######## 2. preprocessing of the dataset to transform into dummy variables
    enc = OneHotEncoder(categories='auto')
    enc.fit(data.iloc[:,:6])
    enc.categories_
    x_c = pd.DataFrame(enc.transform(data.iloc[:,:6]).toarray())
    x_c.head()
    data['y'].replace(('unacc', 'acc', 'good', 'vgood'), (0, 1, 2, 3), inplace = True)
    y_c = data['y']

    ######## 3. setting 80% for training, 20% for testing
    seed = 45
    x_train, x_test, y_train, y_test = train_test_split(x_c, y_c, test_size = 0.20, random_state = seed, stratify = y_c)
    
    x_train_t, x_train_v, y_train_t, y_train_v = train_test_split(x_train, y_train, test_size = 0.2, random_state = seed, stratify = y_train)

    ######## 4. train model and tune hyperparameters
    Cs = [0.01, 0.1, 1, 10, 100]
    gammas = [0.001, 0.01, 0.1, 1]
    kernels = ['linear','rbf']
    dfs = ['ovr','ovo']
    val_prop = 0.2

    my_rf = rf(Cs, gammas, kernels, dfs, seed, val_prop=val_prop)

    #dataset_train = (x_train, y_train)

    my_rf.train(x_train_t, y_train_t, x_train_v, y_train_v)

    ######## 5. test model
    #dataset_test = (x_test, y_test)
    
    my_rf.test(x_test, y_test)



