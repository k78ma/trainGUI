from xgboost import XGBClassifier
import time
import logging

from models.abstract_model import AbstractModel

logger = logging.getLogger(__name__)


class XGB_class(AbstractModel):

    def __init__(self, reg_lambda, max_depth, n_estimators, lr_tree):
        super().__init__()
        self.classifier = XGBClassifier(objective="binary:logistic", max_depth=max_depth, n_estimators=n_estimators, learning_rate=lr_tree,
                                        reg_lambda=reg_lambda)

    def train(self, X_train,y_train,X_valid, y_valid):
        current_time = time.time()
        eval_set = [(X_valid, y_valid)]
        self.classifier.fit(X_train, y_train,eval_set=eval_set, verbose=True)
        logger.info("passed time: {}".format(time.time() - current_time))

    def test(self, X_test,y_test):
        total = len(X_test)
        prediction = (self.classifier.predict(X_test) == y_test).astype(int)
        n_correct = prediction.sum()
        logger.info("Accuracy of XGBoost is {} ".format(n_correct / total))
