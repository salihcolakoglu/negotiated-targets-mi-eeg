"""Controlled EEGNet/FBCNet feature extractors and raw CE/Walsh outputs.

FBCNet adapted from Ravikiran Mane et al., MIT; the unmodified author source,
license and pinned commit are in sources/author and source bindings.
Adaptations: 128Hz/four1s windows, explicit sample variance, pure eval,
post-Adam constraints, raw logits for either output instead of LogSoftmax.
EEGNet core follows the immutable preceding local SHU implementation.
"""
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def set_seed(seed):
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(seed))


def setup_torch(threads=2):
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _renorm_(weight, maximum):
    with torch.no_grad(): weight.copy_(torch.renorm(weight, p=2, dim=0, maxnorm=maximum))


class EEGNet(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.conv1 = nn.Conv2d(1,8,(1,64),padding=(0,32),bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.dw = nn.Conv2d(8,16,(29,1),groups=8,bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.AvgPool2d((1,4)); self.drop1 = nn.Dropout(.5)
        self.sep_dw = nn.Conv2d(16,16,(1,16),padding=(0,8),groups=16,bias=False)
        self.sep_pw = nn.Conv2d(16,16,1,bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d((1,8)); self.drop2 = nn.Dropout(.5)
        # Same dummy construction/BN behavior as the preceding local EEGNet.
        with torch.no_grad(): self.n_features = self.features(torch.zeros(2,29,512)).shape[1]
        assert self.n_features == 256
        self.fc = nn.Linear(self.n_features,n_out)

    def features(self,x):
        if x.ndim!=3 or tuple(x.shape[1:])!=(29,512): raise ValueError('EEGNet input must be N29x512')
        x = self.bn1(self.conv1(x.unsqueeze(1)))
        x = F.elu(self.bn2(self.dw(x)))
        x = self.drop1(self.pool1(x))
        x = F.elu(self.bn3(self.sep_pw(self.sep_dw(x))))
        return self.drop2(self.pool2(x)).flatten(1)

    def forward(self,x): return self.fc(self.features(x))
    def constrain(self): _renorm_(self.dw.weight,1.); _renorm_(self.fc.weight,.25)


class LogVarLayer(nn.Module):
    def forward(self,x):
        if x.shape[-1]!=128: raise ValueError('FBC variance requires128samples/window')
        return torch.log(torch.clamp(torch.var(x,dim=-1,keepdim=True,correction=1),1e-6,1e6))


class FBCNet(nn.Module):
    def __init__(self,n_out):
        super().__init__()
        self.spatial = nn.Conv2d(9,288,(29,1),groups=9,bias=True)
        self.bn = nn.BatchNorm2d(288)
        self.logvar = LogVarLayer()
        self.n_features = 1152
        # Head is constructed last, preserving common backbone initialization.
        self.fc = nn.Linear(self.n_features,n_out)

    def features(self,x):
        if x.ndim!=4 or tuple(x.shape[1:])!=(9,29,512): raise ValueError('FBCNet input must be N9x29x512')
        x = self.bn(self.spatial(x))
        x = x*torch.sigmoid(x)
        x = x.reshape(len(x),288,4,128)
        return self.logvar(x).flatten(1)

    def forward(self,x): return self.fc(self.features(x))
    def constrain(self): _renorm_(self.spatial.weight,2.); _renorm_(self.fc.weight,.5)


def build_model(architecture,output):
    if output not in ('CE','WALSH'): raise ValueError('Output must be CE or WALSH')
    if architecture not in ('EEGNet','FBCNet'): raise ValueError('Unknown architecture')
    model = {'EEGNet':EEGNet,'FBCNet':FBCNet}[architecture](2 if output=='CE' else 16)
    model.architecture=architecture; model.output=output
    return model


def _batches(n,batch,gen):
    chunks=list(torch.randperm(n,generator=gen).split(batch))
    if len(chunks)>1 and len(chunks[-1])<2:
        chunks[-2]=torch.cat([chunks[-2],chunks[-1]]); chunks.pop()
    return chunks


@torch.no_grad()
def predict_outputs(model,X_t,bs=64):
    model.eval()
    logits=torch.cat([model(X_t[i:i+bs]) for i in range(0,len(X_t),bs)]).float()
    probabilities=torch.sigmoid(logits) if model.output=='WALSH' else torch.softmax(logits,dim=1)
    return logits.cpu().numpy(),probabilities.cpu().numpy()
