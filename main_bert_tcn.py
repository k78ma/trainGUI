# The program runs from here
# hardcoded is used to identify lines which are quick fix

import argparse
import torch
import numpy as np
import random
import logging
import datetime

from torch import nn
from torch.optim import Adam
from Bert_TCN.finance.tcn_model import TCN
from torch.utils.data import DataLoader
import os

from Bert_TCN.finance.dataset import DataPreProcessor
from Bert_TCN.finance.tcn_train import TCNTrainer
from Bert_TCN.finance.torch_dataset import Financial_dataset
from Bert_TCN.finance.io.bert_processor import BertProcessor
from models.setting import get_model

def NRC_Revenue_Prediction_Training (load_path,test_ratio,load,epochs,data_path,batch_size,lr, root):

    random_seed = 42

    logging.getLogger("matplotlib.font_manager").disabled = True
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description='deep regression model')

    # ************************************** Environment Setting *****************************************
    parser.add_argument('--output_path', default=os.path.join(os.path.expanduser('~'), 'results'),
                        help='output path for files produced by the agent')

    # parser.add_argument('--load_path', default=r".\TCN\saved_model\weight.pth",
    #                     help='output path for files produced by the agent')

    # parser.add_argument('--load', action="store_true",
    #                     help='visulaize the dataset or not')

    # parser.add_argument('-v', '--visualize', action="store_true",
    #                     help='visulaize the dataset or not')

    # parser.add_argument('--val_ratio', type=int, default=0.1,
    #                     help='Validation set ratio')

    # parser.add_argument('--test_ratio', type=int, default=0.32,
    #                     help='Test set ratio')

    # parser.add_argument('--num_workers', type=int, default=2,
    #                     help='number of workers for DataLoader')

    # parser.add_argument('--seed', type=int, default=42, metavar='N',
    #                     help='random seed (default: 42)')

    # parser.add_argument('--epochs', type=int, default=300,
    #                     help='number of training epoch')

    # ************************************** Database Setting *****************************************

    # parser.add_argument('-p', '--data_path', type=str,
    #                     default=r"D:\Pycharm Projects\NRC-master\data\NRC\Tech-2020-02-12.csv",
    #                     help='Validation set ratio')

    # parser.add_argument('-p', '--data_path', type=str,
    #                     default=r"D:\Dataset\TCN-Training-Data-2000-2017-Tech-Bizz-Proj+5.csv",
    #                     help='Validation set ratio')


    parser.add_argument("--first_pred", type=str, default='revenue_p0',
                        help='first year of prediction')

    parser.add_argument("--last_pred", type=str, default='revenue_p5',
                        help='last year of prediction')

    # parser.add_argument('-t', '--date_time_labels', nargs='*', help='time labels list',
    #                     default="ProjectCompleteDate ProjectStartDate")
    parser.add_argument('-t', '--date_time_labels', nargs='*', help='time labels list',
                        default="ProjectStartDate")

    parser.add_argument('-b', '--bool_labels', nargs='*', help='boolean labels list', default="")

    # parser.add_argument('-o', '--ordinal_labels', nargs='*', help='ordinal labels list',
    #                     default="FinancialMonitoringRating ProjectType Province")
    parser.add_argument('-o', '--ordinal_labels', nargs='*', help='ordinal labels list',
                        default="FinancialMonitoringRating Province NoteType")

    parser.add_argument('--duplicate_labels', nargs='*', help='ordinal labels list', default="NoteText")


    # parser.add_argument('-u', '--useless_labels', nargs='*', help='useless labels list',
    #                     default="id CityName docid NoteType ProjectNo NoteGroupType Revenue2018 Revenue2009 Revenue2010")
    parser.add_argument('-u', '--useless_labels', nargs='*', help='useless labels list',
                        default="CAAmt ProjectNo IncorporationYear AnimalEthicsProject Employee2000 Employee2001 Employee2002 Employee2003 Employee2004 Employee2005 Employee2006 Employee2007 Employee2008 Employee2009 Employee2010 Employee2011 Employee2012 Employee2013 Employee2014 Employee2015 Employee2016 Employee2017 Employee2018 Employee2019 Employee2020 Employee2021 Employee2022 EmployeeGrowthPct2000 EmployeeGrowthPct2001 EmployeeGrowthPct2002 EmployeeGrowthPct2003 EmployeeGrowthPct2004 EmployeeGrowthPct2005 EmployeeGrowthPct2006 EmployeeGrowthPct2007 EmployeeGrowthPct2008 EmployeeGrowthPct2009 EmployeeGrowthPct2010 EmployeeGrowthPct2011 EmployeeGrowthPct2012 EmployeeGrowthPct2013 EmployeeGrowthPct2014 EmployeeGrowthPct2015 EmployeeGrowthPct2016 EmployeeGrowthPct2017 EmployeeGrowthPct2018 EmployeeGrowthPct2019 EmployeeGrowthPct2020 EmployeeGrowthPct2021 EmployeeGrowthPct2022 HumanEthicsProject InternationalProject OrgID OrgName OrganizationID ProjectCompleteDate ProjectType")
    # Province FinancialMonitoringRating CAAmt IncorporationYear
    # Revenue2017
    # Revenue2016
    # Revenue2015
    # Revenue2014
    # Revenue2013
    # Revenue2012
    # Revenue2011
    # parser.add_argument('-u', '--useless_labels', nargs='*', help='useless labels list',
    #                     default="id CAAmt CityName Employee2009 Employee2010 Employee2011 Employee2012 Employee2013 Employee2014 Employee2015 Employee2016 Employee2017 Employee2018 FinancialMonitoringRating IncorporationYear ProjectCompleteDate ProjectStartDate ProjectType Province docid NoteText NoteType ProjectNo NoteGroupType Revenue2018 Revenue2009 Revenue2010 Text_Feature0 Text_Feature1 Text_Feature2 Text_Feature3 Text_Feature4 Text_Feature5")

    # ************************************** TCN Setting *****************************************

    # parser.add_argument('--batch_size', type=int, default=8,
    #                     help='batch size for network model')

    # parser.add_argument('--lr', type=float, default=0.001,
    #                     help='learning rate for optimizer SGD')

    parser.add_argument('--dropout', type=float, default=0.45,
                        help='dropout applied to layers (default: 0.45)')

    parser.add_argument('--clip', type=float, default=0.35,
                        help='gradient clip, -1 means no clip (default: 0.35)')

    parser.add_argument('--k_size', type=int, default=3,
                        help='kernel size (default: 3)')

    parser.add_argument('--levels', type=int, default=4,
                        help='# of levels (default: 4)')

    parser.add_argument('--nhid', type=int, default=600,
                        help='number of hidden units per layer (default: 600)')

    parser.add_argument('--bert_output', type=int, default=32,
                        help='output size of bert')

    args = parser.parse_args()

    # sets the seed for making it comparable with other implementations
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    # *********************************** Logging Config ********************************************
    file_path_results = os.path.join(args.output_path,
                                     (str(str(datetime.datetime.now()).split(".")[0]).replace(" ", "_")).replace(":", "!"))
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    os.mkdir(file_path_results)
    logging.basicConfig(level=logging.DEBUG, filename=os.path.join(file_path_results, "log.txt"))
    logging.getLogger().addHandler(logging.StreamHandler())

    header = "===================== Experiment configuration ========================"
    logger.info(header)
    args_keys = list(vars(args).keys())
    args_keys.sort()
    max_k = len(max(args_keys, key=lambda x: len(x)))
    for k in args_keys:
        s = k + '.' * (max_k - len(k)) + ': %s' % repr(getattr(args, k))
        logger.info(s + ' ' * max((len(header) - len(s), 0)))
    logger.info("=" * len(header))

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('cuda')
    else:
        device = torch.device("cpu")
    logger.info("device is set for {}".format(device))

    # preprocessing dataset
    logger.info("Starting data preprocessing ...")

    data_processed = DataPreProcessor(batch_size=batch_size, data_base_path=data_path, Text_label = 'NoteText', first_label = '2020',
                                      first_pred=args.first_pred,
                                      last_pred=args.last_pred,

                                      useless_labels=args.useless_labels.split(" "),
                                      ordinal_labels=args.ordinal_labels.split(" "),
                                      duplicate_labels=args.duplicate_labels.split(" "),
                                      date_time_labels=args.date_time_labels.split(" "),
                                      bool_labels=args.bool_labels.split(" "),test_ratio=test_ratio, device=device, seed=random_seed,
                                      visualize=False)

    # print(data_processed.x_train.shape)

    train_set = Financial_dataset(data_processed.x_train, data_processed.y_train)
    test_set = Financial_dataset(data_processed.x_test, data_processed.y_test)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    logger.info("Dataset has been extraced and preprocessed")

    tcn_input_size = data_processed.original_input_size + args.bert_output
    # print(args.bert_output)

    original_input_size = data_processed.original_input_size
    input_ids_size = data_processed.input_ids_size
    input_mask_size = data_processed.input_mask_size
    segment_ids_size = data_processed.segment_ids_size

    output_size = data_processed.output_size
    num_chans = [args.nhid] * (args.levels - 1) + [tcn_input_size]


    model = TCN(original_input_size, input_ids_size, input_mask_size, segment_ids_size, tcn_input_size, args.bert_output, 1, num_chans, dropout=args.dropout, kernel_size=args.k_size)
    model = model.cuda()

    if (load):
        print('using pretrained model...')
        model.load_state_dict(torch.load(load_path))
    criterian = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)

    # Initilizing the trainer of network
    # logger.info("Training has started")
    trainer = TCNTrainer(trainloader=train_loader, test_loader=test_loader, model=model, optimizer=optimizer,
                         criterion=criterian, device=device,load_path=load_path,output_size=output_size)
    trainer.run(epochs, load, root)
