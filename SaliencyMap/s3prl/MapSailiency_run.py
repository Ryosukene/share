import copy
import math
import random
import sys
from distutils.util import strtobool
from functools import lru_cache
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import yaml
from collections import OrderedDict
from typing import List, Tuple
from torch import Tensor

#from ..interfaces import UpstreamBase
from upstream.interfaces import UpstreamBase
from upstream.mockingjay.builder import PretrainedTransformer
from torch.nn.utils.rnn import pad_sequence
import optimizers
#from ..baseline.extracter import get_extracter
#from ..baseline.preprocessor import get_preprocessor
from upstream.baseline.preprocessor import get_preprocessor
#from .model import TransformerConfig, TransformerModel, TransformerSpecPredictionHead
from upstream.interfaces import Featurizer


sys.path.append('/home/negiryosuke/refine/s3prl')
#print(load_weights['Transformer'])
"""
options:{'load_pretrain': 'True', 'no_grad': 'False', 'dropout': 'default', 'spec_aug': 'False',
 'spec_aug_prev': 'True', 'output_hidden_states': 'True', 'permute_input': 'False',
 'ckpt_file': '/home/negiryosuke/refine/s3prl/result/pretrain/TERA_2/states-450000.ckpt', 'select_layer': -1}
load_weights['Config'] {'runner': {'n_epochs': -1, 'total_steps': 700000, 'gradient_clipping': 5.0, 'gradient_accumulate_steps': 4, 'log_step': 100000, 'save_step': 100000, 'max_keep': 10, 'fp16': False}, 'optimizer': {'name': 'AdamW_with_schedule', 'lr': 0.0002, 'warmup_proportion': 0.07}, 'pretrain_expert': {'datarc': {'num_workers': 8, 'train_batch_size': 8, 'max_timestep': -25, 'libri_root': '/home/negiryosuke/refine/s3prl/ECoG_data/george_20120724_session1_CommonAve_ReginLabel/', 'file_path': '/home/negiryosuke/refine/s3prl/ECoG_data/george_20120724_session1_CommonAve_ReginLabel/E_data_len_for_bucket', 'sets': ['train']}}}
load_weights['Upstream_Config'] {'transformer': {'input_dim': -1, 'hidden_size': 768, 'num_hidden_layers': 3, 'num_attention_heads': 12, 'intermediate_size': 3072, 'hidden_act': 'gelu', 'hidden_dropout_prob': 0.1, 'attention_probs_dropout_prob': 0.1, 'initializer_range': 0.02, 'layer_norm_eps': 1e-12, 'share_layer': False, 'pre_layer_norm': False}, 'task': {'loss': 'L1', 'sequence_length': 1500, 'position_encoding_size': 768, 'mask_proportion': 0.15, 'mask_consecutive_min': 7, 'mask_consecutive_max': 7, 'mask_allow_overlap': True, 'mask_bucket_ratio': 1.5, 'mask_frequency': 0.2, 'noise_proportion': 0.0}, 'audio': {'target_level': -25, 'win_ms': 25, 'hop_ms': 10, 'n_freq': 201, 'n_mels': 80, 'n_mfcc': 13, 'input': {'feat_type': 'mel', 'channel': 0, 'log': True, 'delta': 0, 'cmvn': True}, 'target': {'feat_type': 'mel', 'channel': 0, 'log': True, 'delta': 0, 'cmvn': True}}}
self_config['audio'] {'target_level': -25, 'win_ms': 25, 'hop_ms': 10, 'n_freq': 201, 'n_mels': 80, 'n_mfcc': 13, 'input': {'feat_type': 'mel', 'channel': 0, 'log': True, 'delta': 0, 'cmvn': True}, 'target': {'feat_type': 'mel', 'channel': 0, 'log': True, 'delta': 0, 'cmvn': True}}
"""

#-----------extracter
path='/home/negiryosuke/refine/s3prl/result/pretrain/George_0724_session1_30s_split_CommonAve_RegionLabel/states-300000.ckpt'
load_weights = torch.load(path)
self_config=load_weights['Upstream_Config']
target_level=25
#------------PretrainedTransformer
options={'load_pretrain': 'True', 'no_grad': 'False', 'dropout': 'default', 'spec_aug': 'False',
 'spec_aug_prev': 'True', 'output_hidden_states': 'True', 'permute_input': 'False',
 'ckpt_file': '/home/negiryosuke/refine/s3prl/result/pretrain/George_0724_session1_30s_split_CommonAve_RegionLabel/states-300000.ckpt', 'select_layer': -1}
#-------------
args={'mode':'evaluate', 'evaluate_split':'test', 'override':None, 'backend':'nccl', 'local_rank':None,
      'past_exp':'/home/negiryosuke/refine/s3prl/result/downstream/Kin2_20110524_session1_30s_split_CommonAve_RegionLabel_MeanSpectrogram/cpt300000/best-states-dev.ckpt',
      'init_ckpt':'/home/negiryosuke/refine/s3prl/result/downstream/Kin2_20110524_session1_30s_split_CommonAve_RegionLabel_MeanSpectrogram/cpt300000/best-states-dev.ckpt',
      'config':'./downstream/speaker_linear_utter_libri/config.yaml', 'downstream':'speaker_linear_utter_libri', 'downstream_variant':None, 'hub':'torch', 'upstream':'tera_local',
      'upstream_ckpt':'/home/negiryosuke/refine/s3prl/result/pretrain/Kin2_20110524_session1_CommonAve_RegionLabel_MeanSpectrogram/states-300000.ckpt',
      'upstream_model_config':None, 'upstream_refresh':False, 'upstream_trainable':False, 'upstream_feature_selection':'hidden_states', 'upstream_layer_selection':None,
      'upstream_feature_normalize':False, 'upstream_model_name':'model.pt', 'upstream_revision':None, 'expname':'Kin2_20110524_session1_30s_split_CommonAve_RegionLabel_MeanSpectrogram/cpt300000',
      'expdir':'result/downstream/Kin2_20110524_session1_30s_split_CommonAve_RegionLabel_MeanSpectrogram/cpt300000', 'auto_resume':'False', 'push_to_hf_hub':'False',
      'hf_hub_org':'None', 'seed':1337, 'device':'cuda', 'cache_dir':None, 'verbose':False, 'disable_cudnn':False}

config= {'runner': {'total_steps': 1000000, 'gradient_clipping': 1, 'gradient_accumulate_steps': 1,
                    'log_step': 5000, 'eval_step': 10000, 'save_step': 10000, 'max_keep': 1,
                    'eval_dataloaders': ['dev', 'test']}, 'optimizer': {'name': 'AdamW', 'lr': 0.0002},
         'downstream_expert': {'datarc': {'num_workers': 8, 'train_batch_size': 32, 'eval_batch_size': 32,
         'libri_root': '/home/negiryosuke/refine/s3prl/ECoG_data/Kin2_20110524_session1_CommonAve_ReginLabel/',
         'split_file': '/home/negiryosuke/refine/s3prl/downstream/speaker_linear_utter_libri/data/',
         'bucket_file': '/home/negiryosuke/refine/s3prl/ECoG_data/Kin2_20110524_session1_CommonAve_ReginLabel/E_data_len_for_bucket/',
         'sample_rate': 16000, 'train_dev_seed': 1337}, 'modelrc': {'none': 'None'}}}
init_ckpt=torch.load("/home/negiryosuke/refine/s3prl/result/downstream/George_0724_session1_30s_split_CommonAve_RegionLabel_MeanSpectrogram/cpt300000/best-states-dev.ckpt")

#-------------
def normalize_wav_decibel(wav):
    rms=wav.pow(2).mean().pow(0.5)
    scalar=(10**(target_level/20))/(rms+1e-10)
    wav=wav*scalar
    return wav

def preprocess(x):
    x=[normalize_wav_decibel(x_i) for x_i in x]
    x_lens=[len(x_) for x_ in x]
    x=pad_sequence(x,batch_first = True)
    x=x.unsqueeze(1)
    extracter,_,_ = get_preprocessor(self_config['audio'])
    x=extracter(x,wavs_len=x_lens)[0]
    return x
#------------------------------------------------------------------
def _load_weight(model, name):
    init_weight = init_ckpt.get(name)
    if init_weight:
        show(f'[Runner] - Loading {name} weights from the previous experiment')
        model.load_state_dict(init_weight)
        
def _init_model(model, name, trainable, interfaces=None):
    for interface in interfaces or []:
        assert hasattr(model, interface), interface

    _load_weight(model, name)
    if is_initialized() and trainable and any((p.requires_grad for p in model.parameters())):
        model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=True)
        for interface in interfaces or []:
            setattr(model, interface, getattr(model.module, interface))

    return model#ModelEntry(model, name, trainable, interfaces)


#x=[torch.randn(30000) for wav in range(16)]
x=[torch.randn(16000)]
x=preprocess(x)
# tensor([0.0918, 0.5290, 0.0390, 0.8799])
Pretrainedtransformer = PretrainedTransformer(options, inp_dim=-1)
class UpstreamExpert(UpstreamBase):
    def __init__(self):
        super().__init__()
        self.transtormer=Pretrainedtransformer
    def forward(self,wavs):
        last_hidden_state,hidden_state=self.transtormer(wavs)
        return{
                "last_hidden_state":last_hidden_state,
                "hidden_state":hidden_state.unbind(dim=0),
                 }
upstreamexpert=UpstreamExpert()
#print('upstreamexpert(x)',upstreamexpert(x))
def _get_featurizer():
    model = Featurizer(
        upstream = upstreamexpert,
        feature_selection = args['upstream_feature_selection'],
        layer_selection = args['upstream_layer_selection'],
        upstream_device = args['device'],
        normalize = args['upstream_feature_normalize'],
    ).to(args['device'])

    return _init_model(
        model = model,
        name = 'Featurizer',
        trainable = True,
        interfaces = ['output_dim', 'downsample_rate']
        )
'''
def _get_downstream():
    expert = importlib.import_module(f"s3prl.downstream.{args.downstream}.expert")
    Downstream = getattr(expert, "DownstreamExpert")

    model = Downstream(
        upstream_dim = featurizer.model.output_dim,
        upstream_rate = self.featurizer.model.downsample_rate,
            **self.config,
            **vars(self.args)
    ).to(self.args.device)

    return self._init_model(
            model = model,
            name = 'Downstream',
            trainable = True,
            interfaces = ['get_dataloader', 'log_records']
    )
'''
#print(x)

#feature=transformer(x)

#print(feature)
Featurizer=_get_featurizer()
feature=Featurizer(feature)
#print(Featurizer(x,feature))
