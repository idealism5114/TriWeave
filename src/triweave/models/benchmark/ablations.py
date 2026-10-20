"""Source-mapped SLBR controls; all start from the shared-bond full model."""
import math
import torch
from torch import nn
from torch.nn import functional as F

from ..shared_bond import TriWeaveSLBR
from ..slbr_triweave import masked_softmax, zero_invalid, _detach
from .fusion_operators import ResidualFusion, BranchPreservingFusion

FORMAL = {
    'full':'FullD0B-Control', 'commodity_only':'CommodityOnly', 'no_stock':'w/o Stock',
    'no_bond':'w/o Bond', 'no_spatial':'w/o Spatial Branch', 'no_temporal':'w/o Temporal Branch',
    'no_graph':'w/o Graph Propagation', 'no_tempweighted':'w/o TempWeighted',
    'no_markov_switching':'w/o Markov Switching', 'no_microstate':'w/o Microstate',
    'no_switch_kl':'w/o Switch KL', 'no_balanced_readout':'w/o Balanced Readout',
    'no_adaptive_gate':'w/o Adaptive Fusion Gate',
}
PHASE4 = {
    'fixed_mean_fusion':'switching_latent_fixed_mean_fusion',
    'no_graph_prior':'switching_latent_no_graph_prior',
    'uniform_router':'switching_latent_uniform_router',
    'simplified_all':'switching_latent_simplified_all',
}
FUSION = {
    'global_concat_fusion':'switching_latent_balanced_concat_fusion',
    'concat_linear64':'ConcatLinear64', 'gated_concat128':'GatedConcat128', 'mean_concat128':'MeanConcat128',
    'residual_fusion':'residual_cross_branch_fusion', 'branch_preserving_fusion':'branch_preserving_fusion',
}
NODE_HISTORY = {'node_history_mean':'switching_latent_balanced_node_history_mean',
                'node_history_attention':'switching_latent_balanced_node_history_attention'}
CONTROLS = FORMAL | PHASE4 | FUSION | NODE_HISTORY


class NodeHistoryReadout(nn.Module):
    def __init__(self,attention):
        super().__init__();self.attention=attention
        self.W_v=nn.Linear(128,32,bias=False);self.W_o=nn.Linear(32,64,bias=False)
        self.fusion_projection=nn.Linear(256,64)
        if attention:self.W_q=nn.Linear(64,32,bias=False);self.W_k=nn.Linear(128,32,bias=False)

    def forward(self,nodes,history,spatial,temporal,valid):
        values=self.W_v(history)
        if self.attention:
            scores=self.W_q(nodes)@self.W_k(history).transpose(-1,-2)/math.sqrt(32)
            alpha=masked_softmax(scores,valid[:,None,:])
            micro=self.W_o(alpha@values)
        else:
            alpha=valid[:,None,:].to(values.dtype)/valid.sum(-1)[:,None,None].clamp_min(1)
            micro=self.W_o((values*valid[...,None]).sum(1)/valid.sum(1,keepdim=True).clamp_min(1))[:,None].expand_as(nodes)
        count=nodes.size(1)
        features=torch.cat([nodes,micro,spatial[:,None].expand(-1,count,-1),temporal[:,None].expand(-1,count,-1)],-1)
        result=F.gelu(self.fusion_projection(features))
        return zero_invalid(result,valid.any(-1)[:,None].expand(-1,count)),alpha


class SLBRControl(TriWeaveSLBR):
    def __init__(self,assets,targets,variant='full'):
        if variant not in CONTROLS:raise ValueError('Unknown SLBR control: '+variant)
        super().__init__(assets,targets);self.variant=variant;self.removed_modules=[]
        self.uniform=variant in ('no_markov_switching','uniform_router','simplified_all')
        self.fixed_gate=variant in ('no_adaptive_gate','fixed_mean_fusion','simplified_all')
        if variant in ('fixed_mean_fusion','simplified_all'):
            del self.fusion_gate.gate_fc;self.removed_modules.append('fusion_gate.gate_fc')
        if variant in ('no_graph_prior','simplified_all'):
            del self.spatial_branch.graph_learner;self.removed_modules.append('spatial_branch.graph_learner')
            self.register_buffer('static_A',torch.zeros(len(assets),len(assets)))
            self.spatial_branch.attn_mixhop1.prior_scale=0.;self.spatial_branch.attn_mixhop2.prior_scale=0.
        if variant in ('uniform_router','simplified_all'):
            filt=self.switching_regime.regime_filter
            del filt.regime_evidence;del filt.transition_logits
            self.removed_modules.extend(['switching_regime.regime_filter.regime_evidence',
                                         'switching_regime.regime_filter.transition_logits'])
        if variant in FUSION or variant in NODE_HISTORY:
            if variant!='gated_concat128':
                del self.fusion_gate.gate_fc;self.removed_modules.append('fusion_gate.gate_fc')
            if variant in ('global_concat_fusion','concat_linear64'):self.fusion=nn.Linear(128,64)
            elif variant=='residual_fusion':self.fusion=ResidualFusion()
            elif variant=='branch_preserving_fusion':self.fusion=BranchPreservingFusion()
            elif variant in NODE_HISTORY:self.node_readout=NodeHistoryReadout(variant=='node_history_attention')
            if variant in ('gated_concat128','mean_concat128'):self.prediction_head[0]=nn.Linear(128,64)

    def _spatial(self,X,iv,fv,fa,diagnostics):
        branch=self.spatial_branch
        sequence,valid=branch.encoder(X,iv,fv,fa)
        if self.variant=='no_tempweighted':alpha=valid.to(X.dtype)/valid.sum(1,keepdim=True).clamp_min(1)
        else:alpha=masked_softmax(branch.temporal_score(sequence).squeeze(-1),valid,dim=1)
        nodes=(sequence*alpha[...,None]).sum(1);node_valid=valid.any(1)
        attention=[]
        if self.variant=='no_graph':h=nodes
        else:
            a=self.static_A if hasattr(self,'static_A') else branch.graph_learner()
            h,a1=branch.attn_mixhop1(nodes,a,node_valid,diagnostics)
            h,a2=branch.attn_mixhop2(F.relu(h),a,node_valid,diagnostics);attention=a1+a2
        h=zero_invalid(branch.gcn_norm(h),node_valid);pools=[]
        for k in range(3):
            idx=branch.market_indices==k;mask=node_valid[:,idx]
            pools.append(h[:,idx].sum(1)/mask.sum(1,keepdim=True).clamp_min(1))
        hs=zero_invalid(branch.type_agg(torch.cat(pools,-1)),node_valid.any(-1))
        return hs,h,node_valid,dict(spatial_time_attention=alpha,graph_attention=attention)

    def _uniform_states(self,memory,valid):
        b,t,_=memory.shape;z=memory.new_zeros(b,64);p=memory.new_full((b,3),1/3);states=[]
        for i in range(t):
            proposed,_=self.switching_regime.latent_transition(memory[:,i],z,p)
            z=torch.where(valid[:,i,None],proposed,z);states.append(z)
        return torch.stack(states,1),dict(regime_probabilities=p[:,None].expand(-1,t,-1),
            regime_priors=p[:,None].expand(-1,t,-1),regime_evidence=memory.new_zeros(b,t,3))

    def switch_loss(self):
        if self.variant in ('no_switch_kl','no_temporal') or self.uniform:
            return next(self.prediction_head.parameters()).new_zeros(())
        return super().switch_loss()

    def forward(self,X,input_valid_mask,feature_valid_mask,feature_applicable_mask,return_diagnostics=False):
        if self.variant=='full':
            return super().forward(X,input_valid_mask,feature_valid_mask,feature_applicable_mask,return_diagnostics)
        iv=input_valid_mask
        excluded={'commodity_only':(0,1),'no_stock':(0,),'no_bond':(1,)}.get(self.variant,())
        if excluded:
            iv=iv.clone()
            for k in excluded:iv[:,:,self.spatial_branch.market_indices==k]=False
        fm=feature_valid_mask
        app=feature_applicable_mask
        if app.ndim==2:app=app[None,None]
        elif app.ndim==3:app=app[:,None]
        has=(iv[...,None]&fm&app).any((1,2,3))
        diag={};hs=ht=None
        if self.variant!='no_spatial':
            hs,nodes,node_valid,sd=self._spatial(X,iv,fm,feature_applicable_mask,return_diagnostics)
            diag.update(sd,spatial=hs,graph_nodes=nodes,graph_node_valid=node_valid)
        if self.variant!='no_temporal':
            memory,valid,td=self.temporal_branch(X,iv,fm,feature_applicable_mask,return_diagnostics)
            states,rd=self._uniform_states(memory,valid) if self.uniform else self.switching_regime(memory,valid)
            ht,long,micro=self.balanced_readout(memory,states,valid)
            if self.variant=='no_microstate':
                ht=zero_invalid(self.balanced_readout.state_readout(torch.cat([long,torch.zeros_like(micro)],-1)),has)
                micro=torch.zeros_like(micro)
            elif self.variant=='no_balanced_readout':
                idx=torch.arange(20,device=X.device)[None].expand_as(valid).masked_fill(~valid,-1).max(1).values.clamp_min(0)
                row=torch.arange(len(X),device=X.device)
                long=zero_invalid(self.balanced_readout.long_memory_readout(memory[row,idx]),has)
                micro=zero_invalid(self.balanced_readout.micro_state_readout(states[row,idx]),has)
                ht=zero_invalid(self.balanced_readout.state_readout(torch.cat([long,micro],-1)),has)
            diag.update(td,**rd,temporal=ht,balanced_long=long,balanced_micro=micro)
        if hs is None:fused=zero_invalid(self.fusion_gate.lstm_proj(ht),has)
        elif ht is None:fused=zero_invalid(self.fusion_gate.gcn_proj(hs),has)
        else:
            s=zero_invalid(self.fusion_gate.gcn_proj(hs),has);t=zero_invalid(self.fusion_gate.lstm_proj(ht),has)
            if self.variant in NODE_HISTORY:
                target_idx=torch.tensor([self.target_to_node[c] for c in self.targets],device=X.device)
                sn=zero_invalid(self.fusion_gate.gcn_proj(nodes.index_select(1,target_idx)),node_valid.index_select(1,target_idx))
                fused,alpha=self.node_readout(sn,memory,s,t,valid);diag['node_history_attention']=alpha
                z=self.prediction_head[2](self.prediction_head[1](self.prediction_head[0](fused)))
                w=self.prediction_head[3].weight.view(4,len(self.targets),64)
                bias=self.prediction_head[3].bias.view(4,len(self.targets))
                prediction=torch.einsum('bcd,hcd->bhc',z,w)+bias[None]
                diag['fused']=fused
                return (prediction,_detach(diag)) if return_diagnostics else prediction
            if self.variant in ('global_concat_fusion','concat_linear64'):
                fused=self.fusion(torch.cat([s,t],-1))
                if self.variant=='global_concat_fusion':fused=F.gelu(fused)
            elif self.variant in ('residual_fusion','branch_preserving_fusion'):fused=self.fusion(s,t)
            elif self.variant=='mean_concat128':fused=torch.cat([.5*t,.5*s],-1)
            else:
                gate=hs.new_full(hs.shape,.5) if self.fixed_gate else torch.sigmoid(self.fusion_gate.gate_fc(torch.cat([hs,ht],-1)))
                fused=torch.cat([gate*t,(1-gate)*s],-1) if self.variant=='gated_concat128' else gate*t+(1-gate)*s
                diag['fusion_gate']=gate
            fused=zero_invalid(fused,has)
        prediction=self.prediction_head(fused).view(len(X),4,len(self.targets))
        diag['fused']=fused
        return (prediction,_detach(diag)) if return_diagnostics else prediction
