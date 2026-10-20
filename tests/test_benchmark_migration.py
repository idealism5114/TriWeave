"""CMGM suite migration: mechanisms/masks/causality; no real training runs."""
import ast
import copy
from pathlib import Path
import math
import numpy as np
import pytest
import torch

from triweave.models.benchmark import make_model, MODEL_IDS, BASELINES, registry
from triweave.models.benchmark.ablations import PHASE4, SLBRControl
from triweave.models.benchmark.baselines import WaveLayer, DiffusionGraph, MaskedMixHop, VanillaLayer
from triweave.models.benchmark.common import NeutralAdapter
from triweave.models.benchmark.msgnet import ScaleGraph, ScaleAttention
from triweave.models.shared_bond import TriWeaveSLBR
from triweave.models.slbr_triweave import masked_huber
from triweave.slbr_data import manifests, SLBRDataset, collate, INPUT_KEYS

ROOT=Path(__file__).resolve().parents[1]
NEURAL=tuple(n for n in MODEL_IDS if not n.endswith(('/Ridge','/RandomForest','/XGBoost')))
torch.set_num_threads(4)


def real_batch():
    ds=SLBRDataset(ROOT,'TRAIN');dates=ds.base.a['reference_dates'][ds.base.a['train_origin_indices']]
    return collate([ds[0],ds[int(np.where(dates=='2016-12-23')[0][0])]])


@pytest.mark.parametrize('model_id',NEURAL)
def test_real_shapes_backward_and_invalid_input_invariance(model_id):
    torch.manual_seed(42);model=make_model(model_id,*manifests(ROOT));b=real_batch()
    inputs={k:b[k] for k in INPUT_KEYS}
    model.set_epoch(20);prediction=model(**inputs)
    loss,_=masked_huber(prediction,b['Y'],b['label_valid_mask'])
    (loss+.25*model.switch_loss()).backward()
    assert prediction.shape==(2,4,23) and torch.isfinite(prediction).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in model.parameters())
    model.eval();valid=b['feature_valid_mask']&b['input_valid_mask'][...,None]&b['feature_applicable_mask'][:,None]
    changed={k:v.clone() for k,v in inputs.items()};changed['X'][~valid]=float('nan')
    with torch.no_grad():
        expected=model(**inputs);actual=model(**changed)
    torch.testing.assert_close(expected,actual,rtol=0,atol=0)


@pytest.mark.parametrize('model_id',NEURAL)
def test_all_missing_and_diagnostic_rng_parity(model_id):
    model=make_model(model_id,*manifests(ROOT));b=real_batch();b={k:v[:1] for k,v in b.items() if isinstance(v,torch.Tensor)}
    b['input_valid_mask'][:]=False;b['X'][:]=float('nan')
    args={k:b[k] for k in INPUT_KEYS}
    model.train();torch.manual_seed(42);p=model(**args);rng=torch.get_rng_state()
    torch.manual_seed(42);q,_=model(**args,return_diagnostics=True)
    assert torch.isfinite(q).all();torch.testing.assert_close(p,q,rtol=0,atol=0)
    assert torch.equal(rng,torch.get_rng_state())


@pytest.mark.parametrize('name',['RNN','GRU','LSTM','VanillaTransformer','GraphWaveNet','MTGNN','TCN'])
def test_prefix_causality_and_single_sample_consistency(name):
    model=make_model('baseline/'+name,*manifests(ROOT)).eval();b=real_batch()
    args={k:b[k] for k in INPUT_KEYS};changed={k:v.clone() for k,v in args.items()};changed['X'][:,10:]*=-31
    with torch.no_grad():
        h,_=model.temporal_states(**args);g,_=model.temporal_states(**changed)
        p=model(**args);q=torch.cat([model(**{k:v[i:i+1] for k,v in args.items()}) for i in range(2)])
    torch.testing.assert_close(h[:,:10],g[:,:10],rtol=0,atol=0)
    torch.testing.assert_close(p,q,rtol=1e-5,atol=2e-6)


def test_neutral_adapter_exact_valid_denominators_and_joint_bond_group():
    assets,targets=manifests(ROOT);a=NeutralAdapter(assets,targets)
    assert sum(p.numel() for p in a.parameters())==0
    b=real_batch();x,valid,fm=a(**{k:b[k] for k in INPUT_KEYS})
    assert x.shape==(2,20,27,26)
    f=25;t=19;mask=b['feature_valid_mask'][1,t,263:288,f]&b['input_valid_mask'][1,t,263:288]&b['feature_applicable_mask'][1,263:288,f]
    if mask.any():
        expected=b['X'][1,t,263:288,f][mask].mean()
        torch.testing.assert_close(x[1,t,2,f],expected)
    assert torch.equal(fm.any(-1),valid)


def test_full_control_identical_to_current_shared_bond_model():
    assets,targets=manifests(ROOT)
    torch.manual_seed(42);a=TriWeaveSLBR(assets,targets).eval()
    torch.manual_seed(42);b=SLBRControl(assets,targets,'full').eval()
    assert a.parameter_counts()['total_params']==b.parameter_counts()['total_params']==529757
    for name,p in a.state_dict().items():torch.testing.assert_close(p,b.state_dict()[name],rtol=0,atol=0)
    x=real_batch()
    with torch.no_grad():torch.testing.assert_close(a(**{k:x[k] for k in INPUT_KEYS}),b(**{k:x[k] for k in INPUT_KEYS}),rtol=0,atol=0)


@pytest.mark.parametrize('variant',list(PHASE4))
def test_phase4_removes_actual_parameters_and_preserves_shared_initialization(variant):
    a,t=manifests(ROOT);torch.manual_seed(42);full=TriWeaveSLBR(a,t)
    torch.manual_seed(42);control=SLBRControl(a,t,variant)
    fp=dict(full.named_parameters());cp=dict(control.named_parameters())
    assert sum(p.numel() for p in cp.values())<529757
    for n,p in cp.items():torch.testing.assert_close(p,fp[n],rtol=0,atol=0)
    for prefix in control.removed_modules:assert not any(n==prefix or n.startswith(prefix+'.') for n in cp)


def test_market_removal_is_absence_and_no_microstate_is_post_normalization():
    a,t=manifests(ROOT);b=real_batch();args={k:b[k] for k in INPUT_KEYS}
    for variant,nodes in [('no_stock',slice(0,263)),('no_bond',slice(263,288)),('commodity_only',slice(0,288))]:
        model=SLBRControl(a,t,variant).eval();changed={k:v.clone() for k,v in args.items()}
        changed['X'][:,:,nodes]=float('nan')
        with torch.no_grad():torch.testing.assert_close(model(**args),model(**changed),rtol=0,atol=0)
    m=SLBRControl(a,t,'no_microstate').eval()
    with torch.no_grad():_,diag=m(**args,return_diagnostics=True)
    assert torch.count_nonzero(diag['balanced_micro'])==0


def old_classes(file):
    path=ROOT/'experiments/phase_f_benchmark_migration/source_snapshots'/file
    tree=ast.parse(path.read_text());classes=[n for n in tree.body if isinstance(n,ast.ClassDef)]
    env=dict(torch=torch,nn=torch.nn,F=torch.nn.functional,math=math)
    exec(compile(ast.Module(body=classes,type_ignores=[]),str(path),'exec'),env)
    return env


def test_original_wave_diffusion_and_msg_graph_attention_equations():
    old=old_classes('cmgm/models/baselines/graph_wavenet.py')
    teacher=old['WaveLayer'](1).eval();student=WaveLayer(1).eval();student.load_state_dict(teacher.state_dict())
    x=torch.randn(2,32,7,20);a=torch.rand(7,7).softmax(-1);valid=torch.ones(2,20,7,dtype=torch.bool)
    with torch.no_grad():
        p,s=teacher(x,a);q,t=student(x,a[None,None].expand(2,20,7,7),valid)
    torch.testing.assert_close(p,q,rtol=1e-5,atol=2e-6);torch.testing.assert_close(s,t,rtol=0,atol=0)
    old=old_classes('cmgm/models/baselines/msgnet.py')
    teacher=old['ScaleGraph']().eval();student=ScaleGraph(28).eval();student.load_state_dict(teacher.state_dict())
    x=torch.randn(2,20,32)
    with torch.no_grad():p=teacher(x);q=student(x,torch.ones(2,20,28,dtype=torch.bool))
    torch.testing.assert_close(p,q,rtol=1e-5,atol=2e-6)
    teacher=old['ScaleAttention']().eval();student=ScaleAttention().eval();student.load_state_dict(teacher.state_dict())
    x=torch.randn(2,5,32)
    with torch.no_grad():p=teacher(x);q=student(x,torch.ones(2,5,dtype=torch.bool))
    torch.testing.assert_close(p,q,rtol=1e-5,atol=2e-6)


def test_vanilla_transformer_matches_original_pytorch_block_and_clone_initialization():
    original=torch.nn.TransformerEncoderLayer(128,4,256,.1,batch_first=True).eval()
    adapted=VanillaLayer().eval();adapted.attention.load_state_dict(original.self_attn.state_dict())
    for target,source in [('fc1','linear1'),('fc2','linear2'),('norm1','norm1'),('norm2','norm2')]:
        getattr(adapted,target).load_state_dict(getattr(original,source).state_dict())
    x=torch.randn(2,20,128);causal=torch.ones(20,20,dtype=torch.bool).triu(1)
    with torch.no_grad():
        p=original(x,src_mask=causal);q=adapted(x,torch.ones(2,20,dtype=torch.bool))
    torch.testing.assert_close(p,q,rtol=1e-5,atol=2e-6)
    model=make_model('baseline/VanillaTransformer',*manifests(ROOT))
    for k,v in model.layers[0].state_dict().items():torch.testing.assert_close(v,model.layers[1].state_dict()[k],rtol=0,atol=0)


def test_msg_graph_output_bias_cannot_reintroduce_missing_node_messages():
    graph=ScaleGraph(27).eval();valid=torch.zeros(1,20,27,dtype=torch.bool);valid[:,:,0]=True
    with torch.no_grad():
        graph.linear.weight.zero_();graph.linear.weight[:,-1]=1.
        graph.end.bias.fill_(10.);a=graph(torch.ones(1,20,32),valid)
        graph.end.bias.fill_(100.);b=graph(torch.ones(1,20,32),valid)
    torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_classical_training_masks_and_split_barrier():
    a,t=manifests(ROOT);model=make_model('baseline/Ridge',a,t)
    calls=[]
    class Spy:
        def fit(self,x,y):
            assert np.isfinite(x).all() and np.isfinite(y).all();calls.append((x.copy(),y.copy()));self.outputs=y.shape[1];return self
        def predict(self,x):return np.zeros((len(x),self.outputs))
    model.make_estimator=lambda:Spy()
    b=real_batch();y=b['Y'].numpy().copy();mask=b['label_valid_mask'].numpy().copy();mask[:,:,-1]=False;y[~mask]=np.nan
    args={k:b[k].numpy() for k in INPUT_KEYS};args['feature_applicable_mask']=b['feature_applicable_mask'].numpy()
    with pytest.raises(ValueError):model.fit(**args,Y=y,label_valid_mask=mask,split='VAL')
    model.fit(**args,Y=y,label_valid_mask=mask,split='TRAIN');assert calls
    p,valid=model.predict(**args,return_mask=True);assert p.shape==(2,4,23) and not valid[:,:,-1].any()
    with pytest.raises(RuntimeError):model.predict(**args)
    for name in ('Ridge','RandomForest','XGBoost'):
        estimator=make_model('baseline/'+name,a,t).make_estimator();assert estimator is not None


def test_registry_scopes_and_checkpoint_roundtrip():
    assert len(BASELINES)==12 and len(MODEL_IDS)==37
    rows=registry();assert len(rows)==len(MODEL_IDS) and all(not r['trained_on_TriWeave'] for r in rows)
    assert not next(r for r in rows if r['model_id']=='baseline/MSGNet')['prefix_causal_encoder']
    for name in ('baseline/MSGNet','baseline/CrossGNN','ablation/node_history_attention'):
        m=make_model(name,*manifests(ROOT)).eval();state=copy.deepcopy(m.state_dict());other=make_model(name,*manifests(ROOT)).eval()
        other.load_state_dict(state,strict=True);b=real_batch()
        with torch.no_grad():torch.testing.assert_close(m(**{k:b[k] for k in INPUT_KEYS}),other(**{k:b[k] for k in INPUT_KEYS}),rtol=0,atol=0)
