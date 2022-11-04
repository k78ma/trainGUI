from xgboost import XGBRegressor
import time
import numpy as np
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error,mean_absolute_error
from math import sqrt


from models.abstract_model import AbstractModel

logger = logging.getLogger(__name__)


class XGB_reg(AbstractModel):

    def __init__(self, reg_lambda, max_depth, n_estimators, lr_tree):
        super().__init__()
        self.regressor = XGBRegressor(objective="reg:squarederror", max_depth=max_depth, n_estimators=n_estimators, learning_rate=lr_tree,
                                      reg_lambda=reg_lambda)

    def train(self, X_train,y_train,X_valid, y_valid):
        current_time = time.time()
        eval_set = [(X_valid, y_valid)]
        self.regressor.fit(X_train, y_train, eval_set=eval_set, verbose=True)
        logger.info("passed time: {}".format(time.time() - current_time))

    def test(self, X_test,y_test):
        y_test=np.delete(y_test, 42)
        X_test = np.delete(X_test,(42),axis=0)
        prediction=self.regressor.predict(X_test).reshape(-1,1)
        error=mean_absolute_error(y_test, prediction)
        error2 = mean_absolute_percentage_error(y_test, prediction)
        logger.info("RMSE error of XGBoost is {} ".format(error))
        logger.info("mean absolute percentage error of XGBoost is {} ".format(error2))
