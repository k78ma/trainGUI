#encoding:utf-8
import torch
import numpy as np
from ..common.tools import model_device
from ..callback.progressbar import ProgressBar

class Predictor(object):
    def __init__(self,model,logger,n_gpu):
        self.model = model
        self.logger = logger
        self.model, self.device = model_device(n_gpu= n_gpu, model=self.model)

    def predict(self,data):
        pbar = ProgressBar(n_total=len(data),desc='Testing')
        all_logits = None
        all_last_layer_feature = None
        for step, batch in enumerate(data):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                input_ids, input_mask, segment_ids, label_ids = batch
                logits,last_layer_feature = self.model(input_ids, segment_ids, input_mask)
                logits = logits.sigmoid()
            if all_logits is None:
                all_logits = logits.detach().cpu().numpy()
            else:
                all_logits = np.concatenate([all_logits,logits.detach().cpu().numpy()],axis = 0)
            if all_last_layer_feature is None:
                all_last_layer_feature = last_layer_feature.detach().cpu().numpy()
            else:
                all_last_layer_feature = np.concatenate([all_last_layer_feature,last_layer_feature.detach().cpu().numpy()],axis = 0)
            pbar(step=step)
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        return all_logits,all_last_layer_feature






