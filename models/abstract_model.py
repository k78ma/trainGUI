
class AbstractModel:

    def train(self,X_train,y_train,X_valid, y_valid):
        raise NotImplementedError

    def test(self,X_test,y_test):
        raise NotImplementedError
