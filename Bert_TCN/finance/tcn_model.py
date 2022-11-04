import torch
from torch import nn
# import sys
# sys.path.append("../../")
from TCN.tcn import TemporalConvNet
from Bert_TCN.finance.configs.basic_config import config
# from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from Bert_TCN.finance.model.bert_for_multi_label import BertForMultiLable

class TCN(nn.Module):

    def __init__(self, original_input_size, input_ids_size, input_mask_size, segment_ids_size, tcn_input_size, output_size_bert, output_size, num_channels,
                 kernel_size=2, dropout=0.3,  tied_weights=False):
        super(TCN, self).__init__()

        self.original_input_size = original_input_size

        self.input_ids_size = input_ids_size
        self.input_mask_size = input_mask_size
        self.segment_ids_size = segment_ids_size

        self.tcn_input_size = tcn_input_size

        self.bert_model = BertForMultiLable.from_pretrained(config['bert_model_dir'], output_size_bert)


        self.tcn = TemporalConvNet(tcn_input_size, num_channels, kernel_size, dropout=dropout)

        self.decoder = nn.Linear(num_channels[-1], output_size)
        if tied_weights:
            if num_channels[-1] != tcn_input_size:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
            print("Weight tied")
        self.init_weights()

    def init_weights(self):
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, input):
        """Input ought to have dimension (N, C_in, L_in), where L_in is the seq_len; here the input is (N, L, C)"""
        # print(input.shape)
        tcn_original_input = input[:,:self.original_input_size,:]
        bert_input_ids_input = input[:,self.original_input_size:self.original_input_size+self.input_ids_size,:][:,:,0].long()
        bert_input_mask_input = input[:,self.original_input_size+self.input_ids_size:self.original_input_size+self.input_ids_size+self.input_mask_size,:][:,:,0].long()
        bert_segment_ids_input = input[:, self.original_input_size+self.input_ids_size+self.input_mask_size:, :][:,:,0].long()
        #
        # bert_input_ids_input = torch.LongTensor(bert_input_ids_input)
        # bert_input_mask_input = torch.LongTensor(bert_input_mask_input)
        # bert_segment_ids_input = torch.LongTensor(bert_segment_ids_input)


        #################################Bert feature learning#########################################################
        output = self.bert_model(bert_input_ids_input, bert_segment_ids_input, bert_input_mask_input)
        output = output.unsqueeze(-1).repeat(1,1,5)




        ############################################################################################

        tcn_input = torch.cat([tcn_original_input,output],1)
        y = self.tcn(tcn_input)
        y = self.decoder(y.transpose(1,2))
        return y.contiguous().squeeze(-1)

