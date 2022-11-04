from models.classification.xgb_class import XGB_class
from models.regression.xgb_reg import XGB_reg
from models.classification.svm import svm
from models.classification.rf import rf


def get_model(args):

    if(args.algo=="XGB_class"):
        return XGB_class(args.reg_lambda, args.max_depth, args.n_estimators, args.lr_tree)
    elif(args.algo=="XGB_reg"):
        return XGB_reg(args.reg_lambda, args.max_depth, args.n_estimators, args.lr_tree)
    elif(args.algo=="svm"):
        return svm(args.Cs, args.gammas, args.kernels, args.dfs, args.seed, val_ratio=args.val_ratio)
    elif(args.algo=="rf"):
        return rf(args.n_estimators_rf, args.max_features, args.seed)
    else:
        raise Exception("Not implemented error")
