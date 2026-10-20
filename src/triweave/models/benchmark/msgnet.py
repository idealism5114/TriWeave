"""MSGNet core adapted from YoZhibo/MSGNet. No external repository required at runtime."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .common import masked_softmax, masked_adjacency, zero_invalid


def sample_fft(x,k):
    """Select non-DC frequencies per sample, never by batch-average amplitude."""
    amplitude=torch.fft.rfft(x,dim=1).abs().mean(-1)
    if k>amplitude.shape[1]-1:raise ValueError('Not enough nonzero FFT bins')
    # Stable tie breaking makes zero/constant windows well-defined without selecting DC.
    frequencies=amplitude[:,1:].argsort(dim=-1,descending=True,stable=True)[:,:k]+1
    return frequencies,amplitude.gather(1,frequencies)


class ScaleGraph(nn.Module):
    """Official latent-to-node graph mapping, MixHop and residual graph refinement."""
    def __init__(self, nodes):
        super().__init__()
        if not 1 <= nodes <= 32:raise ValueError('MSGNet latent-to-node map requires at most32 neutral nodes')
        self.nodes=nodes
        self.nodevec1=nn.Parameter(torch.randn(nodes,10));self.nodevec2=nn.Parameter(torch.randn(10,nodes))
        self.start=nn.Conv2d(1,32,(32-nodes+1,1))
        self.mix=nn.Conv2d(3*32,32,1)
        self.end=nn.Conv2d(32,20,(1,20));self.linear=nn.Linear(nodes,32);self.norm=nn.LayerNorm(32)

    def adjacency(self):return F.relu(self.nodevec1@self.nodevec2).softmax(-1)

    def forward(self,x,valid):
        a=self.adjacency();self.last_adjacency=a.detach()
        a=masked_adjacency(a,valid,self_loops=True)
        mask=valid.transpose(1,2)[:,None]
        origin=torch.where(mask,self.start(x.transpose(1,2).unsqueeze(1)),0.);h=origin;hops=[h]
        for _ in range(2):
            h=.3*origin+.7*torch.einsum('bcnt,btmn->bcmt',h,a);hops.append(h)
        h=torch.where(mask,F.gelu(self.mix(torch.cat(hops,1))),0.)
        h=self.end(h).squeeze(-1)  # Keep B axis even for a single sample.
        h=torch.where(valid,h,0.)  # End-convolution bias must not revive absent graph nodes.
        return zero_invalid(self.norm(x+self.linear(h)),valid.any(-1))


class ScaleAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention=nn.MultiheadAttention(32,8,dropout=.1,batch_first=True)
        self.ffn=nn.Sequential(nn.Linear(32,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,32),nn.Dropout(.1))
        self.dropout=nn.Dropout(.1);self.norm1=nn.LayerNorm(32);self.norm2=nn.LayerNorm(32)
        self.outnorm=nn.LayerNorm(32)

    def forward(self,x,valid):
        # Canonical within-period causal attention; full-window FFT itself is not prefix causal.
        b,t,_=x.shape
        q,k,v=F.linear(x,self.attention.in_proj_weight,self.attention.in_proj_bias).chunk(3,-1)
        q,k,v=[z.reshape(b,t,8,4).transpose(1,2) for z in (q,k,v)]
        causal=torch.ones(t,t,dtype=torch.bool,device=x.device).tril()
        support=valid[:,None,:,None]&valid[:,None,None,:]&causal
        alpha=masked_softmax(q@k.transpose(-1,-2)/2.,support)
        alpha=F.dropout(alpha,p=.1,training=self.training)
        attended=self.attention.out_proj((alpha@v).transpose(1,2).reshape(b,t,32))
        x=zero_invalid(self.norm1(x+self.dropout(attended)),valid)
        x=zero_invalid(self.norm2(x+self.ffn(x)),valid)
        return zero_invalid(F.gelu(self.outnorm(x)),valid)


class MSGNetCore(nn.Module):
    def __init__(self,nodes=27):
        super().__init__();self.top_k=3;self.nodes=nodes
        # Canonical value embedding and absolute positions, no timestamp covariates.
        self.embedding=nn.Conv1d(nodes,32,3,padding=1,padding_mode='circular',bias=False)
        nn.init.kaiming_normal_(self.embedding.weight,mode='fan_in',nonlinearity='leaky_relu')
        pos=torch.arange(20).float()[:,None];freq=torch.exp(torch.arange(0,32,2)*(-math.log(10000.)/32))
        pe=torch.zeros(20,32);pe[:,0::2]=torch.sin(pos*freq);pe[:,1::2]=torch.cos(pos*freq)
        self.register_buffer('position',pe);self.dropout=nn.Dropout(.1)
        self.graphs=nn.ModuleList([ScaleGraph(nodes) for _ in range(3)])
        self.attentions=nn.ModuleList([ScaleAttention() for _ in range(3)])
        self.norm=nn.LayerNorm(32);self.output_projection=nn.Linear(32,nodes)
        self.horizon_head=nn.Linear(20,4);self.output_dropout=nn.Dropout(.1)

    def forward(self,x,valid):
        if x.shape[1:]!=(20,self.nodes):raise ValueError('MSGNet input shape mismatch')
        time=valid.any(-1)
        h=zero_invalid(self.dropout(self.embedding(x.transpose(1,2)).transpose(1,2)+self.position),time)
        frequencies,amplitudes=sample_fft(h,self.top_k);periods=torch.div(20,frequencies,rounding_mode='floor')
        paths=[]
        for rank,(graph,attention) in enumerate(zip(self.graphs,self.attentions)):
            g=graph(h,valid);result=torch.zeros_like(g)
            # Batch grouping is solely an execution optimization; each sample selects its own periods.
            for period in torch.unique(periods[:,rank]).tolist():
                indices=torch.where(periods[:,rank]==period)[0];group=g.index_select(0,indices)
                length=math.ceil(20/period)*period;group=F.pad(group,(0,0,0,length-20))
                key_valid=F.pad(time.index_select(0,indices),(0,length-20),value=False)
                v=attention(group.reshape(-1,period,32),key_valid.reshape(-1,period)).reshape(len(indices),length,32)[:,:20]
                result=result.index_copy(0,indices,v)
            paths.append(result)
        weights=amplitudes.softmax(-1)
        encoded=zero_invalid(self.norm(h+(torch.stack(paths,-1)*weights[:,None,None,:]).sum(-1)),time)
        nodes=torch.where(valid,self.output_projection(encoded),0.).transpose(1,2)
        prediction=self.output_dropout(self.horizon_head(nodes)).transpose(1,2)
        self.last=dict(frequencies=frequencies.detach(),periods=periods.detach(),scale_weights=weights.detach(),fft_amplitudes=amplitudes.detach())
        return prediction  # Slots are 1/5/10/20d, without input-level de-normalization.
