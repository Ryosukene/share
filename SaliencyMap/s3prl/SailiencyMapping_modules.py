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

import os
import sys
import math
import glob
import uuid
import shutil
import random
import tempfile
import importlib
from pathlib import Path

import torchaudio
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import is_initialized, get_rank, get_world_size

from s3prl import hub
from s3prl.optimizers import get_optimizer
from s3prl.schedulers import get_scheduler
from s3prl.upstream.interfaces import Featurizer
from s3prl.utility.helper import is_leader_process, get_model_state, show, defaultdict

from huggingface_hub import HfApi, HfFolder, Repository
from s3prl import downstream
import expert_refine



import matplotlib.pyplot as plt
from torchsummary import summary
import requests
from PIL import Image
#from ..interfaces import UpstreamBase
from upstream.interfaces import UpstreamBase
from upstream.mockingjay.builder_refine import PretrainedTransformer_refine
from upstream.mockingjay.builder import PretrainedTransformer
#from upstream.mockingjay.expert import UpStreamExpert

from torch.nn.utils.rnn import pad_sequence
import optimizers
#from ..baseline.extracter import get_extracter
#from ..baseline.preprocessor import get_preprocessor
from upstream.baseline.preprocessor import get_preprocessor
#from .model import TransformerConfig, TransformerModel, TransformerSpecPredictionHead
from upstream.interfaces import Featurizer

SAMPLE_RATE = 16000
#-----------extracter
target_level=25
#-------------
def normalize_wav_decibel(wav):
    rms=wav.pow(2).mean().pow(0.5)
    scalar=(10**(target_level/20))/(rms+1e-10)
    wav=wav*scalar
    return wav

def preprocess(x,self_config):
    x=[normalize_wav_decibel(x_i) for x_i in x]
    x_lens=[len(x_) for x_ in x]
    x=pad_sequence(x,batch_first = True)
    x=x.unsqueeze(1)
    extracter,_,_ = get_preprocessor(self_config['audio'])
    x=extracter(x,wavs_len=x_lens)[0]
    return x
#------------------------------------------------------------------
def _load_weight(model,init_ckpt, name):
    init_weight = init_ckpt.get(name)
    if init_weight:
        show(f'[Runner] - Loading {name} weights from the previous experiment')
        model.load_state_dict(init_weight)
        
def _init_model(model,init_ckpt, name, trainable, interfaces=None):
    for interface in interfaces or []:
        assert hasattr(model, interface), interface

    _load_weight(model,init_ckpt, name)
    if is_initialized() and trainable and any((p.requires_grad for p in model.parameters())):
        model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=True)
        for interface in interfaces or []:
            setattr(model, interface, getattr(model.module, interface))

    return model#ModelEntry(model, name, trainable, interfaces)

with open('options_arg_config.yaml', 'r') as yml:
    options_arg_config = yaml.safe_load(yml)
options=options_arg_config['options']
args=options_arg_config['args']
config=options_arg_config['config']
init_ckpt=torch.load(options_arg_config['init_ckpt'])

Pretrainedtransformer = PretrainedTransformer(options, inp_dim=-1)

class UpstreamExpert(UpstreamBase):
    def __init__(self):
        super().__init__()
        self.transformer=Pretrainedtransformer
    def forward(self,wavs):
        last_hidden_state,hidden_state=self.transformer(wavs)
        return{
                "last_hidden_state":last_hidden_state,
                "hidden_states":hidden_state.unbind(dim=0),
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
        init_ckpt=init_ckpt,
        name = 'Featurizer',
        trainable = True,
        interfaces = ['output_dim', 'downsample_rate']
        )
featurizer=_get_featurizer()
def _get_downstream():
    expert = importlib.import_module(f"expert_refine")
    Downstream = getattr(expert, "DownstreamExpert")    
    model = Downstream(
        upstream_dim = featurizer.output_dim,
        upstream_rate = featurizer.downsample_rate,
        
            **config,
            #**vars(args)
    ).to(args['device'])

    return _init_model(
            model = model,
            init_ckpt=init_ckpt,
            name = 'Downstream',
            trainable = True,
            interfaces = []#['get_dataloader', 'log_records']
    )


