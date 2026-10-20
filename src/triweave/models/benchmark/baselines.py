"""Latest CMGM Deep V3 architectures, adapted to the frozen Research23 masks."""
import math
import copy
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .common import NeutralAdapter, last_valid, masked_adjacency, sanitize, masked_softmax, zero_invalid
from ..slbr_core import AdaptiveGraphLearner

DEEP_V3 = ('RNN', 'GRU', 'LSTM', 'VanillaTransformer', 'GraphWaveNet', 'MTGNN', 'MSGNet', 'CrossGNN')


class Base(nn.Module):
    def __init__(self, assets, targets):
        super().__init__();self.assets=assets;self.targets=targets
        self.config=SimpleNamespace(horizons=(1,5,10,20))
        self.adapter=NeutralAdapter(assets, targets);self.n_targets=len(targets)

    def output(self, p):return p.reshape(len(p), 4, self.n_targets)

    def switch_loss(self):return next(self.parameters()).new_zeros(())

    def set_epoch(self, epoch):return 0.


class Recurrent(Base):
    def __init__(self, assets, targets, cell):
        super().__init__(assets, targets)
        self.cell=cell
        self.rnn={'RNN':nn.RNN, 'GRU':nn.GRU, 'LSTM':nn.LSTM}[cell](
            self.adapter.n_tokens*26, 128, num_layers=2, dropout=.1, batch_first=True)
        self.head=nn.Linear(128, 4*self.n_targets)

    def temporal_states(self, X, input_valid_mask, feature_valid_mask, feature_applicable_mask):
        x,valid,_=self.adapter(X,input_valid_mask,feature_valid_mask,feature_applicable_mask)
        sequence=x.flatten(2);time=valid.any(-1);lengths=time.sum(1)
        compact=torch.zeros_like(sequence);positions=[]
        for i in range(len(x)):
            idx=torch.where(time[i])[0];positions.append(idx);compact[i,:len(idx)]=sequence[i,idx]
        packed=pack_padded_sequence(compact,lengths.clamp_min(1).cpu(),batch_first=True,enforce_sorted=False)
        encoded,_=self.rnn(packed)
        encoded,_=pad_packed_sequence(encoded,batch_first=True,total_length=20)
        states=sequence.new_zeros(len(x),20,128)
        for i,idx in enumerate(positions):states[i,idx]=encoded[i,:len(idx)]
        return states,time

    def forward(self, X, input_valid_mask, feature_valid_mask, feature_applicable_mask, return_diagnostics=False):
        h,valid=self.temporal_states(X,input_valid_mask,feature_valid_mask,feature_applicable_mask)
        p=self.output(self.head(last_valid(h,valid)))
        return (p,dict(temporal=h.detach(),temporal_valid=valid)) if return_diagnostics else p


class VanillaLayer(nn.Module):
    """TransformerEncoderLayer post-norm equations with safe masked softmax."""
    def __init__(self):
        super().__init__();self.attention=nn.MultiheadAttention(128,4,dropout=.1,batch_first=True)
        self.norm1=nn.LayerNorm(128);self.norm2=nn.LayerNorm(128)
        self.fc1=nn.Linear(128,256);self.fc2=nn.Linear(256,128);self.dropout=nn.Dropout(.1)

    def forward(self,x,valid):
        b,t,_=x.shape
        q,k,v=F.linear(x,self.attention.in_proj_weight,self.attention.in_proj_bias).chunk(3,-1)
        q,k,v=[a.reshape(b,t,4,32).transpose(1,2) for a in (q,k,v)]
        causal=torch.ones(t,t,device=x.device,dtype=torch.bool).tril()
        support=valid[:,None,:,None]&valid[:,None,None,:]&causal
        alpha=masked_softmax(q@k.transpose(-1,-2)/math.sqrt(32),support)
        a=(self.dropout(alpha)@v).transpose(1,2).reshape(b,t,128)
        x=zero_invalid(self.norm1(x+self.dropout(self.attention.out_proj(a))),valid)
        ff=self.fc2(self.dropout(F.relu(self.fc1(x))))
        return zero_invalid(self.norm2(x+self.dropout(ff)),valid)


class VanillaTransformer(Base):
    def __init__(self, assets, targets):
        super().__init__(assets, targets);self.projection=nn.Linear(self.adapter.n_tokens*26,128)
        pos=torch.arange(20)[:,None];freq=torch.exp(torch.arange(0,128,2)*(-math.log(10000.)/128))
        pe=torch.zeros(20,128);pe[:,0::2]=torch.sin(pos*freq);pe[:,1::2]=torch.cos(pos*freq)
        self.register_buffer('position',pe);layer=VanillaLayer();self.layers=nn.ModuleList([layer,copy.deepcopy(layer)])
        self.head=nn.Linear(128,4*self.n_targets)

    def temporal_states(self,X,input_valid_mask,feature_valid_mask,feature_applicable_mask):
        x,valid,_=self.adapter(X,input_valid_mask,feature_valid_mask,feature_applicable_mask)
        time=valid.any(-1);h=zero_invalid(self.projection(x.flatten(2))+self.position,time)
        for layer in self.layers:h=layer(h,time)
        return h,time

    def forward(self,X,input_valid_mask,feature_valid_mask,feature_applicable_mask,return_diagnostics=False):
        h,valid=self.temporal_states(X,input_valid_mask,feature_valid_mask,feature_applicable_mask)
        p=self.output(self.head(last_valid(h,valid)))
        return (p,dict(temporal=h.detach(),temporal_valid=valid)) if return_diagnostics else p


class DiffusionGraph(nn.Module):
    def __init__(self):
        super().__init__();self.projection=nn.Conv2d(96,32,1);self.dropout=nn.Dropout(.1)

    def forward(self,x,a,valid):
        first=torch.einsum('bcnt,btnm->bcmt',x,a)
        second=torch.einsum('bcnt,btnm->bcmt',first,a)
        return torch.where(valid[:,None].transpose(2,3),self.dropout(self.projection(torch.cat([x,first,second],1))),0.)


class WaveLayer(nn.Module):
    def __init__(self,dilation):
        super().__init__();self.dilation=dilation
        self.filter=nn.Conv2d(32,32,(1,2),dilation=(1,dilation));self.gate=nn.Conv2d(32,32,(1,2),dilation=(1,dilation))
        self.skip=nn.Conv2d(32,256,1);self.graph=DiffusionGraph()

    def forward(self,x,a,valid):
        m=valid.transpose(1,2)[:,None];padded=F.pad(x,(self.dilation,0,0,0))
        z=torch.where(m,self.filter(padded).tanh()*self.gate(padded).sigmoid(),0.)
        return torch.where(m,x+self.graph(z,a,valid),0.),torch.where(m,self.skip(z),0.)


class GraphWaveNet(Base):
    def __init__(self,assets,targets):
        super().__init__(assets,targets);n=self.adapter.n_tokens
        self.nodevec1=nn.Parameter(torch.randn(n,10));self.nodevec2=nn.Parameter(torch.randn(10,n))
        self.start=nn.Conv2d(26,32,1);self.layers=nn.ModuleList([WaveLayer(d) for _ in range(4) for d in (1,2)])
        self.end=nn.Sequential(nn.ReLU(),nn.Conv2d(256,512,1),nn.ReLU(),nn.Conv2d(512,4,1))

    def temporal_states(self,X,input_valid_mask,feature_valid_mask,feature_applicable_mask):
        x,valid,_=self.adapter(X,input_valid_mask,feature_valid_mask,feature_applicable_mask)
        a=masked_adjacency(F.relu(self.nodevec1@self.nodevec2).softmax(-1),valid)
        mask=valid.transpose(1,2)[:,None]
        h=torch.where(mask,self.start(x.permute(0,3,2,1)),0.);skip=None
        for layer in self.layers:
            h,s=layer(h,a,valid);skip=s if skip is None else skip+s
        result=torch.where(mask,self.end(skip),0.).permute(0,3,2,1)
        return result,valid

    def forward(self,X,input_valid_mask,feature_valid_mask,feature_applicable_mask,return_diagnostics=False):
        h,valid=self.temporal_states(X,input_valid_mask,feature_valid_mask,feature_applicable_mask)
        p=h[:,-1,4:].transpose(1,2)
        return (p,dict(temporal=h.detach(),node_valid=valid)) if return_diagnostics else p


class MaskedMixHop(nn.Module):
    def __init__(self):
        super().__init__();self.Ws=nn.ModuleList([nn.Linear(64,64) for _ in range(3)])

    def forward(self,x,a,valid):
        a=masked_adjacency(a,valid,self_loops=True);h=x;out=self.Ws[0](h)
        for i in range(1,3):
            h=.05*x+.95*torch.einsum('btnm,btmd->btnd',a,h)
            out=out+self.Ws[i](h)
        return zero_invalid(out,valid)


class MTBlock(nn.Module):
    def __init__(self,dilation):
        super().__init__();self.padding=2*dilation
        self.filter=nn.Conv2d(64,64,(1,3),dilation=(1,dilation));self.gate=nn.Conv2d(64,64,(1,3),dilation=(1,dilation))
        self.forward_graph=MaskedMixHop();self.reverse_graph=MaskedMixHop()

    def forward(self,x,a,valid):
        padded=F.pad(x.permute(0,3,2,1),(self.padding,0,0,0))
        h=zero_invalid((self.filter(padded).tanh()*self.gate(padded).sigmoid()).permute(0,3,2,1),valid)
        return zero_invalid(F.relu(x+self.forward_graph(h,a,valid)+self.reverse_graph(h,a.T,valid)),valid)


class MTGNN(Base):
    """CMGM MTGNN-style architecture, explicitly not an official reproduction."""
    def __init__(self,assets,targets):
        super().__init__(assets,targets);self.projection=nn.Linear(26,64)
        self.graph=AdaptiveGraphLearner(self.adapter.n_tokens,10,.5,10)
        self.blocks=nn.ModuleList([MTBlock(1),MTBlock(2)]);self.head=nn.Linear(64,4)

    def temporal_states(self,X,input_valid_mask,feature_valid_mask,feature_applicable_mask):
        x,valid,_=self.adapter(X,input_valid_mask,feature_valid_mask,feature_applicable_mask)
        h=zero_invalid(self.projection(x),valid);a=self.graph()
        for block in self.blocks:h=block(h,a,valid)
        return h,valid

    def forward(self,X,input_valid_mask,feature_valid_mask,feature_applicable_mask,return_diagnostics=False):
        h,valid=self.temporal_states(X,input_valid_mask,feature_valid_mask,feature_applicable_mask)
        p=self.head(h[:,-1,4:]).transpose(1,2)
        return (p,dict(temporal=h.detach(),node_valid=valid)) if return_diagnostics else p


class ScalarGraph(Base):
    def __init__(self,assets,targets,kind):
        super().__init__(assets,targets)
        from .msgnet import MSGNetCore
        from .crossgnn import CrossGNNCore
        self.projection=nn.Linear(26,1)
        self.core={'MSGNet':MSGNetCore,'CrossGNN':CrossGNNCore}[kind](self.adapter.n_tokens)

    def forward(self,X,input_valid_mask,feature_valid_mask,feature_applicable_mask,return_diagnostics=False):
        x,valid,_=self.adapter(X,input_valid_mask,feature_valid_mask,feature_applicable_mask)
        scalar=torch.where(valid,self.projection(x).squeeze(-1),0.)
        p=self.core(scalar,valid)[:,:,4:]
        return (p,dict(core=self.core.last,node_valid=valid)) if return_diagnostics else p


class TCN(Base):
    """Supplement: latest repaired WeightNorm+Chomp TCN, full311-node input."""
    def __init__(self,assets,targets):
        super().__init__(assets,targets)
        from .tcn_source import TemporalConvNet
        self.tcn=TemporalConvNet(len(assets)*26,[128,128,128],kernel_size=3,dropout=.1)
        self.head=nn.Linear(128,4*self.n_targets)

    def temporal_states(self,X,input_valid_mask,feature_valid_mask,feature_applicable_mask):
        x,m=sanitize(X,input_valid_mask,feature_valid_mask,feature_applicable_mask)
        valid=m.any((2,3));h=x.flatten(2).transpose(1,2);mask=valid[:,None]
        for block in self.tcn.network:
            first=torch.where(mask,block.dropout1(block.relu1(block.chomp1(block.conv1(h)))),0.)
            second=torch.where(mask,block.dropout2(block.relu2(block.chomp2(block.conv2(first)))),0.)
            residual=h if block.downsample is None else block.downsample(h)
            h=torch.where(mask,block.relu(second+residual),0.)
        return h.transpose(1,2),valid

    def forward(self,X,input_valid_mask,feature_valid_mask,feature_applicable_mask,return_diagnostics=False):
        h,valid=self.temporal_states(X,input_valid_mask,feature_valid_mask,feature_applicable_mask)
        p=self.output(self.head(last_valid(h,valid)))
        return (p,dict(temporal=h.detach(),temporal_valid=valid)) if return_diagnostics else p
