"""Phase F: source-backed SLBR audits, one seed42 fit, and sealed VAL reports.

Run from project root: python -m triweave.slbr_phase --preflight / --train / --finalize.
TRAIN/VAL only. Frozen artifacts are never overwritten by a second training run.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .models.slbr_triweave import TriWeaveSLBR, SLBRConfig, masked_huber
from .slbr_data import DATA, OUT, INPUT_KEYS, SLBRDataset, collate, to_device, manifests


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path,index=False)


def seed42():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(4)


def source_hashes(root):
    return {str(p.relative_to(root)):sha(p) for p in sorted((root/'src/triweave').glob('slbr*.py'))} | {
        str(p.relative_to(root)):sha(p) for p in sorted((root/'src/triweave/models').glob('*.py'))} | {
        'tests/test_slbr_triweave.py':sha(root/'tests/test_slbr_triweave.py')}


def environment():
    return dict(python=sys.version,pytorch=str(torch.__version__),numpy=np.__version__,pandas=pd.__version__,
        platform=platform.platform(),cuda_compiled=torch.version.cuda,cuda_available=torch.cuda.is_available(),
        cuda_device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        deterministic=True,float_dtype='float32',AMP=False,seed=42)


def frozen_training_config():
    return dict(seed=42,optimizer='Adam',learning_rate=1e-4,weight_decay=1e-5,batch_size=32,max_epochs=200,
        patience=10,scheduler='ReduceLROnPlateau',scheduler_factor=.5,scheduler_patience=5,gradient_clip_norm=1.,
        loss='mean of per-horizon valid-label Huber means; skip/renormalize empty horizons',huber_delta=.02,
        original_switch_KL_multiplier=.25,original_switch_beta_max=5e-4,switch_warmup_epochs=20,
        checkpoint_selection='equal mean of 4 per-horizon commodity-macro VAL MAE',
        train_shuffle=False,drop_last=False,num_workers=0,float32=True,AMP=False,
        split_usage=dict(TRAIN='fit',VAL='selection and descriptive diagnostics',TEST='PROHIBITED'),
        rationale='Retain original Adam/lr/decay/epochs/patience/scheduler. Batch32 bounds graph activation memory on shared GPU. Keep tail and masks. Clip1 for untrimmed z tails. Divide native KL by4 with horizon-mean loss, preserving full-label original relative scale.')


def loaders(root, batch_size):
    return {s:DataLoader(SLBRDataset(root,s),batch_size=batch_size,shuffle=False,drop_last=False,num_workers=0,
                         collate_fn=collate) for s in ('TRAIN','VAL')}


def fresh_model(root):
    assets,targets=manifests(root)
    cfg=json.loads((root/OUT/'model_config.json').read_text())
    return TriWeaveSLBR(assets,targets,SLBRConfig(**cfg))


def gate(root):
    result=subprocess.run([sys.executable,'-m','triweave.research23','--verify'],cwd=root,text=True,capture_output=True)
    (root/OUT/'dataset_gate.txt').write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError('Frozen dataset verification failed')
    value=json.loads(result.stdout.strip().splitlines()[-1])
    assert value['EXPLORATORY_RESEARCH_READY'] and value['DATASET_TECHNICALLY_VALID']
    return value


def future_source_test(root,model):
    """Actual quote perturbation, rebuild legal X at a fixed VAL origin, no refit."""
    from . import research23 as r23, after_close as ac
    from .research23_audit import perturb_future_sources
    protocol,assets=r23.definitions(root)
    frames,sources,logs,replay=r23.load_sources(root,assets)
    calendar=pd.read_csv(root/'data/manifests/baseline15y/reference_trading_calendar.csv',dtype=str).date.tolist()
    rules=pd.read_csv(root/r23.OUT/'source_quarantine_manifest.csv',keep_default_na=False).to_dict('records')
    params=json.loads((root/DATA/'scaler_params.json').read_text())
    cutoff='2023-06-30';fixed=SLBRDataset(root,'VAL').base
    ix=int(np.where(fixed.a['reference_dates'][fixed.a['val_origin_indices']]==cutoff)[0][0])
    sample=fixed[ix]
    rebuilt=[]
    for fs in (frames,perturb_future_sources(frames,cutoff)):
        aligned,_=ac.align_sources(fs,assets,calendar,protocol)
        aligned,_=r23.apply_quarantine(aligned,assets,rules)
        x,m,app=ac.feature_panel(aligned,assets)
        z,fm=ac.transform(x,m,assets,params) # frozen55 params; no fitting
        ids=sample['reference_indices']
        batch=dict(X=torch.from_numpy(z[ids].copy())[None],input_valid_mask=torch.from_numpy(aligned['valid'][ids].copy())[None],
                   feature_valid_mask=torch.from_numpy(fm[ids].copy())[None],feature_applicable_mask=torch.from_numpy(app.copy())[None])
        rebuilt.append(batch)
    for key in INPUT_KEYS:assert torch.equal(rebuilt[0][key],rebuilt[1][key]),key
    original=collate([{k:sample[k] for k in INPUT_KEYS+('Y','label_valid_mask','origin_date')}])
    for key in INPUT_KEYS:assert torch.equal(rebuilt[0][key],original[key]),key
    model.eval()
    with torch.no_grad():
        p,d=model(**rebuilt[0],return_diagnostics=True);q,e=model(**rebuilt[1],return_diagnostics=True)
    assert torch.equal(p,q) and torch.equal(d['adjacency'],e['adjacency'])
    return dict(actual_raw_future_perturbation=True,cutoff=cutoff,source_count=len(sources),source_quote_read_log=logs,
        all_rebuilt_inputs_equal=True,rebuilt_matches_frozen_dataset=True,prediction_max_difference=float((p-q).abs().max()),
        graph_max_difference=float((d['adjacency']-e['adjacency']).abs().max()),scaler_refitted=False,
        TEST_numeric_rows_parsed=sum(x.get('TEST_numeric_rows_parsed',0) for x in logs),
        graph='trainable identity adjacency + content from legal window only; no correlation fit',
        labels_not_forward_arguments=True,software_causality_pass=True,historical_PIT_verified=False)


def preflight(root):
    out=root/OUT
    if (out/'preflight_gate.json').exists():raise FileExistsError('Preflight already frozen')
    g=gate(root);seed42()
    write_json(out/'model_config.json',asdict(SLBRConfig()))
    write_json(out/'training_config.json',frozen_training_config())
    model=fresh_model(root)
    write_json(out/'parameter_count.json',model.parameter_counts()|dict(original_CMGM=520549,
        delta=model.parameter_counts()['total_params']-520549))
    assets,targets=manifests(root)
    write_json(out/'prediction_identity_mapping.json',dict(assets=assets,target_to_prediction_index=model.target_to_index,
                                                          target_to_input_node=model.target_to_node))
    run=subprocess.run([sys.executable,'-m','pytest','-q'],cwd=root,text=True,capture_output=True)
    (out/'tests.txt').write_text(run.stdout+run.stderr)
    if run.returncode:raise RuntimeError('Tests failed, training prohibited')
    seed42();model=fresh_model(root).eval();ds=SLBRDataset(root,'TRAIN')
    a=ds.base.a;origin=a['train_origin_indices']
    extreme=int(np.unravel_index(np.abs(a['features'][:2733]).argmax(),a['features'][:2733].shape)[0])
    indices=dict(earliest_left_padding=[0],real_single=[1000],real_batch=list(range(500,504)),
         largest_missing=[int(np.argmin(a['input_valid_mask'][origin].sum(1)))],
         sparse_labels=[int(np.argmin(a['train_label_valid_mask'].sum((1,2))))],
         extreme_train=[int(np.argmin(abs(origin-extreme)))])
    shapes=[];numerics=[]
    for name,idx in indices.items():
        b=collate([ds[i] for i in idx]);model.zero_grad(set_to_none=True)
        p,d=model(**{k:b[k] for k in INPUT_KEYS},return_diagnostics=True)
        loss,counts=masked_huber(p,b['Y'],b['label_valid_mask']);loss.backward()
        finite=all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
        assert finite and torch.isfinite(p).all()
        shapes.append(dict(case=name,origins=b['origin_date'],X=list(b['X'].shape),Y=list(p.shape),
            spatial=list(d['spatial'].shape),temporal=list(d['temporal'].shape),regime=list(d['regime_probabilities'].shape),
            gate=list(d['fusion_gate'].shape),balanced_readout=list(d['balanced_long'].shape),horizon_label_counts=counts,backward_finite=finite))
        numerics.append(dict(stage='preflight',case=name,input_abs_max=float(b['X'].abs().max()),prediction_abs_max=float(p.abs().max()),
            gradient_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'))),finite=True))
    write_json(out/'shape_tests.json',dict(real_cases=shapes,synthetic_311_313='pytest configurable shape/backward tests passed'))
    write_json(out/'mask_behavior_tests.json',dict(status='PASS',evidence='tests/test_slbr_triweave.py',
        scenarios=['stock missing','future partial','bond inapplicable','empty market step','left padding','all graph neighbors invalid',
                   'all attention keys invalid','commodity labels absent','horizon absent','batch absent','NaN invalid perturbation',
                   'zero invalid gradients','diagnostic RNG parity train/eval'],tests_sha256=sha(root/'tests/test_slbr_triweave.py')))
    write_csv(out/'preflight_numerical_stability.csv',numerics)
    causal=future_source_test(root,model)
    write_json(out/'causality_tests.json',causal|dict(temporal_prefix_test='PASS',regime_prefix_test='PASS',
               evidence='test_temporal_prefix_and_regime_no_future'))
    write_json(out/'preflight_environment.json',environment())
    write_json(out/'preflight_gate.json',dict(PASS=True,dataset_gate=g,adaptation_hashes=source_hashes(root),
            model_config_sha256=sha(out/'model_config.json'),training_config_sha256=sha(out/'training_config.json'),
            timestamp_utc=pd.Timestamp.now(tz='UTC').isoformat()))
    print('PREFLIGHT PASS: all tests and actual-source causal reconstruction; real fit not started',flush=True)


@torch.no_grad()
def evaluate(model,loader,device,diagnostics=False):
    model.eval();preds=[];ys=[];ms=[];origins=[];diagnostic_rows=[];graph_rows=[]
    for batch in loader:
        b=to_device(batch,device)
        result=model(**{k:b[k] for k in INPUT_KEYS},return_diagnostics=diagnostics)
        if diagnostics:
            p,d=result
            regime=d['regime_probabilities'][:,-1].cpu().numpy()
            gates=d['fusion_gate'].cpu().numpy()
            for i,date in enumerate(b['origin_date']):
                pr=regime[i];gg=gates[i]
                row=dict(origin_date=date,regime_state=int(pr.argmax()),regime_entropy=float(-(pr*np.log(pr+1e-12)).sum()),
                    regime_max_probability=float(pr.max()),gate_mean=float(gg.mean()),gate_std=float(gg.std()),
                    gate_min=float(gg.min()),gate_max=float(gg.max()))
                for k in range(3):row[f'regime_p{k}']=float(pr[k]);row[f'regime_prior{k}']=float(d['regime_priors'][i,-1,k]);row[f'regime_evidence{k}']=float(d['regime_evidence'][i,-1,k])
                diagnostic_rows.append(row)
            # Attention is descriptive, no frozen-model interventions here.
            av=d['graph_node_valid'];adj=d['adjacency'];alpha=d['graph_attention'][-1]
            entropy=-(alpha.clamp_min(1e-12).log()*alpha).sum(2).mean(-1)
            maximum=alpha.max(2).values.mean(-1)
            ta=d['temporal_attention'][-1]
            for i,date in enumerate(b['origin_date']):
                active=av[i];count=int(active.sum())
                graph_rows.append(dict(origin_date=date,graph_num_nodes=len(model.assets),valid_nodes=count,
                    positive_prior_edges=int((adj>0).sum()),prior_edge_density=float((adj>0).float().mean()),
                    effective_attention_support_pairs=count*count, # soft log prior: nonedges remain possible
                    attention_entropy=float(entropy[i,active].mean()) if count else 0.,
                    attention_max_weight=float(maximum[i,active].mean()) if count else 0.,
                    temporal_attention_future_mass=float(ta[i].triu(1).sum()),
                    spatial_norm=float(d['spatial'][i].norm()),temporal_norm=float(d['temporal'][i].norm()),
                    balanced_long_norm=float(d['balanced_long'][i].norm()),balanced_micro_norm=float(d['balanced_micro'][i].norm())))
        else:p=result
        if not torch.isfinite(p).all():raise FloatingPointError('Nonfinite VAL prediction')
        preds.append(p.cpu().numpy());ys.append(b['Y'].cpu().numpy());ms.append(b['label_valid_mask'].cpu().numpy());origins+=b['origin_date']
    p,y,m=map(np.concatenate,(preds,ys,ms))
    target_rows=[]
    for hi,h in enumerate(model.config.horizons):
        for ci,target in enumerate(model.targets):
            mask=m[:,hi,ci];err=(p[:,hi,ci][mask]-y[:,hi,ci][mask]).astype(np.float64)
            n=len(err);absolute=np.abs(err)
            target_rows.append(dict(horizon=h,asset_id=target,valid_label_count=n,MAE=float(absolute.mean()) if n else None,
                RMSE=float(np.sqrt(np.mean(err**2))) if n else None,
                Huber=float(np.mean(np.where(absolute<=.02,.5*err**2,.02*(absolute-.01)))) if n else None,
                direction_hit=float(np.mean(np.sign(p[:,hi,ci][mask])==np.sign(y[:,hi,ci][mask]))) if n else None))
    horizon_rows=[]
    for hi,h in enumerate(model.config.horizons):
        selected=[r for r in target_rows if r['horizon']==h and r['valid_label_count']]
        mask=m[:,hi];err=(p[:,hi][mask]-y[:,hi][mask]).astype(np.float64);ab=np.abs(err)
        horizon_rows.append(dict(horizon=h,valid_label_count=len(err),evaluable_targets=len(selected),
             macro_MAE=float(np.mean([r['MAE'] for r in selected])),micro_MAE=float(ab.mean()),
             macro_RMSE=float(np.mean([r['RMSE'] for r in selected])),micro_RMSE=float(np.sqrt(np.mean(err**2))),
             macro_Huber=float(np.mean([r['Huber'] for r in selected])),micro_Huber=float(np.mean(np.where(ab<=.02,.5*err**2,.02*(ab-.01)))),
             macro_direction_hit=float(np.mean([r['direction_hit'] for r in selected])),
             micro_direction_hit=float(np.mean(np.sign(p[:,hi][mask])==np.sign(y[:,hi][mask])))))
    all_err=(p[m]-y[m]).astype(np.float64)
    metrics=dict(VAL_SELECTION_SCORE=float(np.mean([r['macro_MAE'] for r in horizon_rows])),
        macro_MAE=float(np.mean([r['MAE'] for r in target_rows if r['valid_label_count']])),
        micro_MAE=float(np.abs(all_err).mean()),micro_RMSE=float(np.sqrt(np.mean(all_err**2))),
        valid_labels=int(m.sum()),origin_count=len(p),target_count=len(model.targets))
    return dict(metrics=metrics,horizons=horizon_rows,targets=target_rows,prediction=p,Y=y,mask=m,origin_dates=origins,
                diagnostics=diagnostic_rows,graph_diagnostics=graph_rows)


def failure_dump(out,epoch,batch,model,reason):
    write_json(out/f'nonfinite_epoch{epoch}.json',dict(reason=reason,origins=batch['origin_date'],
        asset_ids=[a['asset_id'] for a in model.assets],feature_schema='frozen feature_registry.json',
        input_abs_max=float(batch['X'].abs().max()),gradient_nonfinite=[n for n,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]))
    torch.save(dict(batch={k:v.cpu() if isinstance(v,torch.Tensor) else v for k,v in batch.items()},
              model_state_dict=model.state_dict()),out/f'nonfinite_epoch{epoch}.pt')


def train(root,device):
    out=root/OUT;frozen=json.loads((out/'preflight_gate.json').read_text())
    if not frozen['PASS'] or frozen['adaptation_hashes']!=source_hashes(root):raise RuntimeError('Preflight code mismatch')
    assert frozen['model_config_sha256']==sha(out/'model_config.json') and frozen['training_config_sha256']==sha(out/'training_config.json')
    gate(root)
    if (out/'training_started.json').exists():raise FileExistsError('seed42 has already started; refusing an implicit rerun')
    seed42();cfg=json.loads((out/'training_config.json').read_text())
    if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent fallback')
    dev=torch.device(device);model=fresh_model(root).to(dev);ls=loaders(root,cfg['batch_size'])
    write_json(out/'environment.json',environment()|dict(device=str(dev)))
    write_json(out/'training_started.json',dict(timestamp_utc=pd.Timestamp.now(tz='UTC').isoformat(),seed=42,device=str(dev),fresh_initialization=True))
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=5)
    best=float('inf');best_epoch=0;stale=0;rows=[];begin=time.perf_counter()
    checkpoints=out/'checkpoints';checkpoints.mkdir(exist_ok=True)
    metadata=dict(data_protocol_id='TRIWEAVE_23_AFTER_CLOSE_RESEARCH_V1',dataset_manifest_sha256=sha(root/DATA/'dataset_manifest.json'),
        data_fingerprints={str(p.relative_to(root)):sha(p) for p in sorted((root/DATA).iterdir()) if p.is_file()},
        source_manifest_sha256=sha(out/'source_code_manifest.json'),adaptation_hashes=source_hashes(root),
        model_config=asdict(model.config),training_config=cfg,environment=environment(),assets=model.assets,targets=model.targets,
        target_to_prediction_index=model.target_to_index,feature_schema=json.loads((root/DATA/'feature_registry.json').read_text()),seed=42)
    for epoch in range(1,cfg['max_epochs']+1):
        t=time.perf_counter();model.train();model.set_epoch(epoch)
        total=pred_total=kl_total=0.;steps=empty=clips=0;gradmax=gradtotal=actmax=predmax=0.;counts=np.zeros(4,np.int64);target_counts=np.zeros(23,np.int64)
        if dev.type=='cuda':torch.cuda.reset_peak_memory_stats(dev)
        for batch in ls['TRAIN']:
            b=to_device(batch,dev);optimizer.zero_grad(set_to_none=True)
            try:
                p,d=model(**{k:b[k] for k in INPUT_KEYS},return_diagnostics=True)
                loss,den=masked_huber(p,b['Y'],b['label_valid_mask'],cfg['huber_delta'])
                counts+=den;target_counts+=b['label_valid_mask'].sum((0,1)).cpu().numpy()
                if loss is None:empty+=1;continue
                kl=cfg['original_switch_KL_multiplier']*model.switch_loss();objective=loss+kl
                if not torch.isfinite(objective) or not torch.isfinite(p).all():raise FloatingPointError('Nonfinite prediction or loss')
                objective.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip_norm'],error_if_nonfinite=True)
                if not all(v.grad is None or torch.isfinite(v.grad).all() for v in model.parameters()):raise FloatingPointError('Nonfinite gradient')
                optimizer.step()
            except (FloatingPointError,RuntimeError) as err:
                failure_dump(out,epoch,b,model,str(err));raise
            gn=float(norm);gradmax=max(gradmax,gn);gradtotal+=gn;clips+=gn>cfg['gradient_clip_norm'];steps+=1
            total+=float(objective.detach());pred_total+=float(loss.detach());kl_total+=float(kl.detach());predmax=max(predmax,float(p.detach().abs().max()))
            actmax=max(actmax,max(float(d[k].abs().max()) for k in ('graph_nodes','long_memory','latent_states','fused')))
            del p,d,objective,loss,kl
        result=evaluate(model,ls['VAL'],dev);score=result['metrics']['VAL_SELECTION_SCORE'];scheduler.step(score)
        row=dict(epoch=epoch,train_loss=total/max(steps,1),train_prediction_huber=pred_total/max(steps,1),train_switch_KL=kl_total/max(steps,1),
            VAL_SELECTION_SCORE=score,gradient_norm_mean=gradtotal/max(steps,1),gradient_norm_max=gradmax,gradient_clip_count=clips,
            activation_abs_max=actmax,prediction_abs_max=predmax,nonfinite_count=0,empty_label_batches=empty,optimizer_steps=steps,
            learning_rate=optimizer.param_groups[0]['lr'],epoch_seconds=time.perf_counter()-t,
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(dev) if dev.type=='cuda' else 0)
        for i,h in enumerate(model.config.horizons):row[f'train_valid_labels_{h}d']=int(counts[i]);row[f'val_macro_MAE_{h}d']=result['horizons'][i]['macro_MAE']
        rows.append(row);write_csv(out/'training_history.csv',rows)
        print(json.dumps(row),flush=True)
        if score<best:
            best=score;best_epoch=epoch;stale=0
            torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),scheduler_state_dict=scheduler.state_dict(),
                best_epoch=epoch,best_val_selection_score=best,metadata=metadata,rng_state=torch.get_rng_state(),
                cuda_rng_states=torch.cuda.get_rng_state_all() if dev.type=='cuda' else []),checkpoints/'best.tmp.pt')
            (checkpoints/'best.tmp.pt').replace(checkpoints/'best_seed42.pt')
        else:stale+=1
        if stale>=cfg['patience']:break
    write_json(out/'training_complete.json',dict(seed=42,epochs=len(rows),best_epoch=best_epoch,best_score=best,
        total_training_seconds=time.perf_counter()-begin,early_stopped=stale>=cfg['patience'],TRAIN_label_counts_by_target_last_epoch=dict(zip(model.targets,map(int,target_counts))),
        checkpoint_sha256=sha(checkpoints/'best_seed42.pt'),other_seeds_run=False))
    print('SEED42 TRAINING COMPLETE; run --finalize for independent reload and audits',flush=True)


def finalize(root,device):
    out=root/OUT
    if not (out/'training_complete.json').exists():raise RuntimeError('Training incomplete')
    seed42();dev=torch.device(device);model=fresh_model(root).to(dev)
    path=out/'checkpoints/best_seed42.pt';cp=torch.load(path,map_location=dev,weights_only=False)
    assert cp['metadata']['adaptation_hashes']==source_hashes(root)
    model.load_state_dict(cp['model_state_dict'],strict=True);model.eval()
    ls=loaders(root,32);result=evaluate(model,ls['VAL'],dev,diagnostics=True)
    history=pd.read_csv(out/'training_history.csv');idx=history.VAL_SELECTION_SCORE.idxmin()
    assert int(history.loc[idx,'epoch'])==cp['best_epoch']
    assert abs(result['metrics']['VAL_SELECTION_SCORE']-cp['best_val_selection_score'])<1e-8
    b=to_device(next(iter(ls['VAL'])),dev)
    with torch.no_grad():
        p=model(**{k:b[k] for k in INPUT_KEYS})
        state=torch.get_rng_state();cuda_state=torch.cuda.get_rng_state_all() if dev.type=='cuda' else []
        q,d=model(**{k:b[k] for k in INPUT_KEYS},return_diagnostics=True)
        assert torch.equal(state,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(cuda_state,torch.cuda.get_rng_state_all() if dev.type=='cuda' else []))
        torch.testing.assert_close(p,q,atol=0,rtol=0)
        other=fresh_model(root).to(dev);other.load_state_dict(cp['model_state_dict'],strict=True);other.eval()
        restored=other(**{k:b[k] for k in INPUT_KEYS});torch.testing.assert_close(p,restored,atol=0,rtol=0)
    write_json(out/'checkpoint_audit.json',dict(PASS=True,checkpoint=str(path.relative_to(root)),sha256=sha(path),best_epoch=cp['best_epoch'],
        history_argmin_epoch=int(history.loc[idx,'epoch']),best_score=cp['best_val_selection_score'],recomputed_score=result['metrics']['VAL_SELECTION_SCORE'],
        independent_recovery_max_diff=float((p-restored).abs().max()),diagnostics_prediction_max_diff=float((p-q).abs().max()),
        diagnostics_rng_unchanged=True,optimizer_state_saved=True,model_metadata_keys=list(cp['metadata'])))
    write_csv(out/'validation_metrics.csv',[result['metrics']]);write_csv(out/'per_horizon_mae.csv',result['horizons'])
    write_csv(out/'per_target_horizon_metrics.csv',result['targets'])
    target_rows=[]
    for target in model.targets:
        rr=[r for r in result['targets'] if r['asset_id']==target]
        target_rows.append(dict(asset_id=target,macro_MAE=float(np.mean([r['MAE'] for r in rr])),valid_label_count=sum(r['valid_label_count'] for r in rr)))
    write_csv(out/'per_commodity_mae.csv',target_rows)
    write_csv(out/'regime_diagnostics.csv',[{k:v for k,v in row.items() if k.startswith('regime') or k=='origin_date'} for row in result['diagnostics']])
    write_csv(out/'fusion_gate_diagnostics.csv',[{k:v for k,v in row.items() if k.startswith('gate') or k in ('origin_date','regime_state')} for row in result['diagnostics']])
    write_csv(out/'graph_market_diagnostics.csv',result['graph_diagnostics'])
    write_csv(out/'numerical_stability_report.csv',history.to_dict('records'))
    np.savez_compressed(out/'validation_predictions.npz',prediction=result['prediction'],target=result['Y'],label_valid_mask=result['mask'],
                        origin_dates=np.array(result['origin_dates']),target_ids=np.array(model.targets),horizons=np.array(model.config.horizons))
    rg=pd.DataFrame(result['diagnostics']);gg=pd.DataFrame(result['graph_diagnostics'])
    state_counts={str(k):int((rg.regime_state==k).sum()) for k in range(3)}
    write_json(out/'diagnostic_summary.json',dict(regime_argmax_occupancy=state_counts,
        regime_mean_probabilities=[float(rg[f'regime_p{k}'].mean()) for k in range(3)],regime_entropy_mean=float(rg.regime_entropy.mean()),
        regime_max_probability_mean=float(rg.regime_max_probability.mean()),gate_mean=float(rg.gate_mean.mean()),
        gate_origin_mean_std=float(rg.gate_mean.std(ddof=0)),gate_min=float(rg.gate_min.min()),gate_max=float(rg.gate_max.max()),
        gate_within_origin_channel_std_mean=float(rg.gate_std.mean()),prediction_std_per_target_horizon=result['prediction'].std(0).tolist(),
        graph_prior_positive_edges=int(gg.positive_prior_edges.iloc[0]),graph_prior_density=float(gg.prior_edge_density.iloc[0]),
        graph_valid_nodes_min=int(gg.valid_nodes.min()),graph_valid_nodes_max=int(gg.valid_nodes.max()),temporal_future_attention_mass=float(gg.temporal_attention_future_mass.max()),
        descriptive_only=True,causal_interpretation=False))
    print(json.dumps(dict(best_epoch=cp['best_epoch'],**result['metrics'])),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    modes=p.add_mutually_exclusive_group(required=True)
    for name in ('preflight','train','finalize'):modes.add_argument('--'+name,action='store_true')
    p.add_argument('--device',default='cpu');args=p.parse_args();root=Path.cwd()
    if args.preflight:preflight(root)
    elif args.train:train(root,args.device)
    else:finalize(root,args.device)


if __name__=='__main__':main()
