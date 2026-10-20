"""Mechanism, numerical and causal tests; never load HOLD/FWD3/TEST data."""
import inspect
import ast
import math
from pathlib import Path
import numpy as np
import pytest
import torch
from triweave.models.slbr_triweave import (
    TriWeaveSLBR, SLBRConfig, KINDS, masked_huber, masked_softmax, MaskedEdgeAttnMixHop, CausalAttention,
)
from triweave.slbr_data import manifests, SLBRDataset, collate, INPUT_KEYS

torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parents[1]


def synthetic(n=311, batch=2):
    kinds = [KINDS[0]] * 263 + [KINDS[1]] * 18 + [KINDS[2]] * 7 + [KINDS[3]] * (n - 288)
    assets = [dict(asset_id=f'a{i}', economic_type=k, market='stock' if i<263 else 'bond' if i<288 else 'commodity') for i,k in enumerate(kinds)]
    target = [a['asset_id'] for a in assets[-(n-288):]]
    model = TriWeaveSLBR(assets, target)
    b = dict(X=torch.randn(batch,20,n,26), input_valid_mask=torch.rand(batch,20,n)>.2,
             feature_valid_mask=torch.rand(batch,20,n,26)>.3,
             feature_applicable_mask=torch.ones(batch,n,26,dtype=torch.bool))
    return model, b


@pytest.mark.parametrize('n', [311,313])
def test_configurable_shape_backward(n):
    model, b = synthetic(n)
    y, d = model(**b, return_diagnostics=True)
    assert y.shape == (2,4,n-288)
    for name, shape in [('spatial',(2,64)),('temporal',(2,64)),('long_memory',(2,20,128)),
                        ('regime_probabilities',(2,20,3)),('fusion_gate',(2,64)),('balanced_long',(2,64))]:
        assert d[name].shape == shape and torch.isfinite(d[name]).all()
    model.set_epoch(20)
    (y.square().mean()+model.switch_loss()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert sum(model.parameter_counts()[k] for k in model.parameter_counts() if k!='total_params') == model.parameter_counts()['total_params']


@pytest.mark.parametrize('scenario', ['stock_missing','future_partial','bond_inapplicable','empty_market_step','left_padding','all_invalid'])
def test_masks_arbitrary_invalid_perturbation(scenario):
    model, b = synthetic(batch=1); model.eval()
    if scenario=='stock_missing': b['input_valid_mask'][:,:,0]=False
    if scenario=='future_partial': b['input_valid_mask'][:,:10,-1]=False
    if scenario=='bond_inapplicable': b['feature_applicable_mask'][:,263:281,:20]=False
    if scenario=='empty_market_step': b['input_valid_mask'][:,10,:263]=False
    if scenario=='left_padding': b['input_valid_mask'][:,:19]=False
    if scenario=='all_invalid': b['input_valid_mask'][:]=False
    valid = b['feature_valid_mask'] & b['feature_applicable_mask'][:,None] & b['input_valid_mask'][...,None]
    changed={k:v.clone() for k,v in b.items()}
    changed['X'][~valid]=float('nan')
    with torch.no_grad():
        y,d=model(**b,return_diagnostics=True); y2,d2=model(**changed,return_diagnostics=True)
    torch.testing.assert_close(y,y2,atol=0,rtol=0)
    assert torch.isfinite(y).all()
    if scenario=='all_invalid':
        for key in ('spatial','temporal','graph_nodes','long_memory','balanced_long','balanced_micro'):
            assert torch.count_nonzero(d[key])==0
        assert all(torch.count_nonzero(a)==0 for a in d['graph_attention']+d['temporal_attention'])


def test_invalid_gradients_and_no_neighbors():
    model,b=synthetic(batch=1)
    b['input_valid_mask'][:]=False
    b['input_valid_mask'][:,:,-1]=True
    b['X'].requires_grad_()
    y=model(**b);y.sum().backward()
    assert torch.count_nonzero(b['X'].grad[:,:,:-1])==0
    assert torch.isfinite(b['X'].grad).all()
    logits=torch.randn(2,4,requires_grad=True)
    empty=masked_softmax(logits,torch.zeros(2,4,dtype=torch.bool))
    assert torch.count_nonzero(empty)==0
    empty.sum().backward();assert torch.count_nonzero(logits.grad)==0


def test_masked_labels_and_horizon_denominators():
    p=torch.randn(2,4,23,requires_grad=True); y=torch.randn_like(p);mask=torch.rand_like(p)>.4
    mask[:,2]=False;mask[:,:,-1]=False
    y[~mask]=float('nan')
    loss,counts=masked_huber(p,y,mask)
    expected=torch.stack([torch.nn.functional.huber_loss(p[:,h][mask[:,h]],y[:,h][mask[:,h]],delta=.02) for h in (0,1,3)]).mean()
    torch.testing.assert_close(loss,expected)
    assert counts[2]==0
    loss.backward();assert torch.count_nonzero(p.grad[~mask])==0 and torch.isfinite(p.grad).all()
    assert masked_huber(p,y,torch.zeros_like(mask))==(None,[0,0,0,0])


@pytest.mark.parametrize('training', [False, True])
def test_diagnostic_flag_rng_neutral(training):
    model,b=synthetic(batch=1);model.train(training)
    torch.manual_seed(42);p=model(**b);rng=torch.get_rng_state()
    torch.manual_seed(42);q,_=model(**b,return_diagnostics=True);rng2=torch.get_rng_state()
    torch.testing.assert_close(p,q,rtol=0,atol=0);assert torch.equal(rng,rng2)


def test_temporal_prefix_and_regime_no_future():
    model,b=synthetic(batch=1);model.eval()
    with torch.no_grad():
        memory,valid,_=model.temporal_branch(b['X'],b['input_valid_mask'],b['feature_valid_mask'],b['feature_applicable_mask'])
        states,diag=model.switching_regime(memory,valid)
        b['X'][:,10:]*=100
        changed,vm,_=model.temporal_branch(b['X'],b['input_valid_mask'],b['feature_valid_mask'],b['feature_applicable_mask'])
        states2,diag2=model.switching_regime(changed,vm)
    torch.testing.assert_close(memory[:,:10],changed[:,:10],atol=0,rtol=0)
    torch.testing.assert_close(states[:,:10],states2[:,:10],atol=0,rtol=0)
    torch.testing.assert_close(diag['regime_probabilities'][:,:10],diag2['regime_probabilities'][:,:10],atol=0,rtol=0)
    assert list(inspect.signature(model.forward).parameters)==['X','input_valid_mask','feature_valid_mask','feature_applicable_mask','return_diagnostics']


def test_gate_direction_balanced_readout_mechanisms():
    model,_=synthetic(batch=1);f=model.fusion_gate
    s,t=torch.randn(2,64),torch.randn(2,64);valid=torch.tensor([True,True])
    fused,g=f(s,t,valid)
    torch.testing.assert_close(fused,g*f.lstm_proj(t)+(1-g)*f.gcn_proj(s))
    br=model.balanced_readout;memory=torch.randn(2,20,128);states=torch.randn(2,20,64);mask=torch.ones(2,20,dtype=torch.bool)
    output,long,micro=br(memory,states,mask)
    torch.testing.assert_close(long,br.long_memory_norm(br.long_memory_readout(memory[:,-1])))
    torch.testing.assert_close(micro,br.micro_state_norm(br.micro_state_readout(states[:,-1])))
    torch.testing.assert_close(output,br.state_readout(torch.cat([long,micro],-1)))


def original_class(file, name):
    """Compile only a frozen class for audit; no CMGM runtime/project import."""
    path=ROOT/'experiments/phase_f_slbr_migration/source_snapshots/45879f5'/file
    tree=ast.parse(path.read_text());node=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==name)
    env=dict(torch=torch,nn=torch.nn,F=torch.nn.functional,math=math,LSTM_HIDDEN_DIM=64,FEATURE_DIM=21)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),env)
    return env[name]


def test_original_graph_and_causal_attention_equivalence():
    cfg=SLBRConfig();graph=MaskedEdgeAttnMixHop(cfg).eval()
    OldGraph=original_class('cmgm/models/hetero_mixhop_model.py','EdgeAttnMixHop')
    old=OldGraph(in_dim=64,out_dim=64,K=2,beta=.05,n_heads=8,dropout=.1,prior_scale=.5).eval()
    old.load_state_dict(graph.state_dict(),strict=True)
    x=torch.randn(2,7,64);adj=torch.rand(7,7)
    with torch.no_grad():
        expected=old(x,adj);actual,_=graph(x,adj,torch.ones(2,7,dtype=torch.bool))
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=2e-6)
    attention=CausalAttention(cfg).eval()
    OldAttention=original_class('cmgm/models/switching_latent_transformer.py','CausalBaseRPEAttention')
    old_attn=OldAttention(128,4).eval();old_attn.load_state_dict(attention.state_dict(),strict=True)
    tokens=torch.randn(2,20,128);bias=torch.randn(2,4,20,20)
    with torch.no_grad():
        expected=old_attn(tokens,bias);actual,alpha=attention(tokens,bias,torch.ones(2,20,dtype=torch.bool),True)
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=2e-6)
    assert torch.count_nonzero(alpha.triu(1))==0


def test_masked_markov_recurrence_matches_original_valid_history():
    model,b=synthetic(batch=2);branch=model.switching_regime
    memory=torch.randn(2,20,128);valid=torch.ones(2,20,dtype=torch.bool)
    states,diag=branch(memory,valid)
    p=torch.full((2,3),1/3);z=torch.zeros(2,64);transition=branch.regime_filter.transition_matrix()
    for i in range(20):
        prior,e,p,kl=branch.regime_filter.step(memory[:,i],p,transition)
        z,_=branch.latent_transition(memory[:,i],z,p)
        torch.testing.assert_close(states[:,i],z,rtol=0,atol=0)
        torch.testing.assert_close(diag['regime_probabilities'][:,i],p,rtol=0,atol=0)


def test_real_data_earliest_extreme_and_sparse():
    assets,targets=manifests(ROOT);ds=SLBRDataset(ROOT,'TRAIN');model=TriWeaveSLBR(assets,targets)
    assert len(ds)==2726 and len(SLBRDataset(ROOT,'VAL'))==483
    first=ds.base[0];assert (first['reference_indices']<0).sum()==19
    extreme_day=int(np.unravel_index(np.abs(ds.base.a['features']).argmax(),ds.base.a['features'].shape)[0])
    origins=ds.base.a['train_origin_indices'];extidx=int(np.argmin(abs(origins-extreme_day)))
    sparse=int(np.argmin(ds.base.a['train_label_valid_mask'].sum((1,2))))
    b=collate([ds[i] for i in (0,extidx,sparse,1000)])
    p=model(**{k:b[k] for k in INPUT_KEYS});loss,c=masked_huber(p,b['Y'],b['label_valid_mask'])
    assert p.shape==(4,4,23) and torch.isfinite(p).all() and torch.isfinite(loss)
    loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    with pytest.raises(ValueError):SLBRDataset(ROOT,'TEST')
