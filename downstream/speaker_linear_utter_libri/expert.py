# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ expert.py ]
#   Synopsis     [ the speaker linear downstream wrapper ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import math
import random
from collections import defaultdict
#-------------#
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
#-------------#
from .model import Model
from .dataset import SpeakerDataset
import numpy as np

class DownstreamExpert(nn.Module):
    

    #predict = torch.tensor(a)
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, downstream_expert, expdir, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']
        print(downstream_expert['datarc'])
        self.train_dataset = SpeakerDataset('train', self.datarc['train_batch_size'], **self.datarc)
        self.dev_dataset = SpeakerDataset('dev', self.datarc['eval_batch_size'], **self.datarc)
        self.test_dataset = SpeakerDataset('test', self.datarc['eval_batch_size'], **self.datarc)

        self.model = Model(input_dim=self.upstream_dim, output_class_num=self.train_dataset.class_num, **self.modelrc)
        self.objective = nn.CrossEntropyLoss()

        self.logging = os.path.join(expdir, 'log.log')
        self.logging_predict = os.path.join(expdir, 'log_predict.log')
        self.best = defaultdict(lambda: 0)
        #self.a=[]
        #self.predict = torch.tensor(self.a)

    def _get_train_dataloader(self, dataset):
        return DataLoader(
            dataset, batch_size=1, # for bucketing
            shuffle=True, num_workers=self.datarc['num_workers'],
            drop_last=False, pin_memory=True, collate_fn=dataset.collate_fn
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset, batch_size=1, # for bucketing
            shuffle=False, num_workers=self.datarc['num_workers'],
            drop_last=False, pin_memory=True, collate_fn=dataset.collate_fn
        )

    """
    Datalaoder Specs:
        Each dataloader should output in the following format:

        [[wav1, wav2, ...], your_other_contents1, your_other_contents2, ...]

        where wav1, wav2 ... are in variable length
        each wav is torch.FloatTensor in cpu with dim()==1 and sample_rate==16000
    """

    # Interface
    def get_train_dataloader(self):
        return self._get_train_dataloader(self.train_dataset)

    # Interface
    def get_dev_dataloader(self):
        return self._get_eval_dataloader(self.dev_dataset)

    # Interface
    def get_test_dataloader(self):
        return self._get_eval_dataloader(self.test_dataset)


    # Interface
    def get_dataloader(self, mode):
        return eval(f'self.get_{mode}_dataloader')()

    # Interface
    def forward(self, mode, features, labels, records, **kwargs):
        """
        Args:
            features:
                list of unpadded features [feat1, feat2, ...]
                each feat is in torch.FloatTensor and already
                put in the device assigned by command-line args

            labels:
                the utterance-wise spekaer labels

            records:
                defaultdict(list), by appending contents into records,
                these contents can be averaged and logged on Tensorboard
                later by self.log_records every log_step

        Return:
            loss:
                the loss to be optimized, should not be detached
        """
        global a
        a=[]
        predict= np.array(a)


        features = torch.stack([f.mean(dim=0) for f in features], dim=0) # (batch_size, seq_len, feature_dim) -> (batch_size, feature_dim)
        #labels = torch.randn_like(labels, dtype=torch.short)#確認用　
        #labels = torch.randn_like(labels, dtype=torch.int64)#確認用
        #labels = torch.zeros(16, dtype=torch.int64)#確認用
        #labels = torch.zeros(labels.shape, dtype=torch.int64)#これを使う
        #labels=torch.randint(60, labels.shape)#これを使う
        labels = labels.to(features.device)
        #print(labels)
        
        #labels = torch.rand(labels.shape).to(features.device)
        #labels = torch.randn_like(labels, dtype=torch.int64)
        #print(labels.dtype)

        predicted = self.model(features)
        loss = self.objective(predicted, labels)
        # print('predicte',predicted)
        #predicts=torch.cat((self.predict,predicted))
        #print('predict',predicts)
        _predicted=predicted.to('cpu').detach().numpy().copy()
        #torch.save(predicts, 'predicted.pt')
        #np.append(predict, _predicted)
        #print('predict',ipredict)
        '''
        for i in range(1000):
            pathroot='predict'
            if os.path.isfile(pathroot+'/'+str(i)):
                pass
            else:
                #path=pathroot+'/'+str(i)
                #os.makedirs(path)
                torch.save(predicted, pathroot+'/'+str(i))
                torch.save(labels, pathroot+'/'+'labels'+'-'+str(i))
                break
        '''
        #np.savetxt('predict.csv', _predicted)
        predicted_classid = predicted.max(dim=-1).indices
        records['acc'] += (predicted_classid == labels).view(-1).cpu().float().tolist()
        
        for i in range(1000):
            # pathroot='_'
            #pathroot='/home/negiryosuke/refine/s3prl/predict/predict_Su_CommonAve_RegionLabel_MeanSpectrogram'
            #pathroot='/home/negiryosuke/refine/s3prl/predict/predict_M4_RegionLabel'
            pathroot='/$share/predict/sample'
            #pathroot ='predict_Chibi_CommonAve'
            #pathroot='predict_George_Common_Ave'
            if os.path.isdir(pathroot+'/'+str(i)):
                #print('a')
                continue
            
            #path=pathroot+'/'+str(i)
        
             #print('pathroot+/+str(i)',pathroot+'/'+str(i))
             #print('pathroot/labels-+str(i)',pathroot+'/'+'labels'+'-'+str(i))
             #print('pathroot+/+predicted_classid+-+str(i)',pathroot+'/'+'predicted_classid'+'-'+str(i))
            print('i',i)
            try:

                os.makedirs(pathroot+'/'+str(i))
            except FileExistsError:
                pass
            torch.save(predicted, pathroot+'/'+str(i)+'/'+'vector.pt')
            torch.save(labels, pathroot+'/'+str(i)+'/'+'labels.pt')
            torch.save(predicted_classid, pathroot+'/'+str(i)+'/'+'predicted_classid.pt')
             #print('labels',labels)
             #print('predicted_classid',predicted_classid)
            break
            
        return loss

    # interface
    def log_records(self, mode, records, logger, global_step, **kwargs):
        """
        Args:
            records:
                defaultdict(list), contents already appended

            logger:
                Tensorboard SummaryWriter
                please use f'{prefix}your_content_name' as key name
                to log your customized contents

            global_step:
                global_step in runner, which is helpful for Tensorboard logging
        """
        prefix = f'libri_speaker/{mode}-'
        average = torch.FloatTensor(records['acc']).mean().item()
        
        logger.add_scalar(
            f'{prefix}acc',
            average,
            global_step=global_step
        )
        message = f'{prefix}|step:{global_step}|acc:{average}\n'
        save_ckpt = []
        if average > self.best[prefix]:
            self.best[prefix] = average
            message = f'best|{message}'
            name = prefix.split('/')[-1].split('-')[0]
            save_ckpt.append(f'best-states-{name}.ckpt')
        with open(self.logging, 'a') as f:
            f.write(message)
        print(message)

        return save_ckpt
