"""Migrated CMGM model suite. Default actions never fit or read TEST.

python -m triweave.benchmark --list
python -m triweave.benchmark --check
Training requires explicit --train --model <one registered ID>.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import torch

from . import slbr_phase as phase
from .slbr_data import DATA, INPUT_KEYS, SLBRDataset, collate, manifests, to_device
from .models.benchmark import MODEL_IDS, make_model, registry
from .models.benchmark.classical import MaskedClassical
from .models.slbr_triweave import masked_huber

OUT=Path('experiments/phase_f_benchmark_migration')


def hashes(root):
    paths=list((root/'src/triweave/models/benchmark').glob('*.py'))
    paths += [root/'src/triweave/benchmark.py',root/'src/triweave/models/shared_bond/__init__.py',
              root/'src/triweave/models/slbr_core.py',root/'src/triweave/models/slbr_triweave.py',
              root/'src/triweave/slbr_data.py',root/'tests/test_benchmark_migration.py']
    return {str(p.relative_to(root)):phase.sha(p) for p in sorted(paths)}


def dataset_gate(root,out):
    r=subprocess.run([sys.executable,'-B','-m','triweave.research23','--verify'],cwd=root,text=True,capture_output=True)
    (out/'dataset_gate.txt').write_text(r.stdout+r.stderr)
    if r.returncode:raise RuntimeError('Frozen dataset gate failed')
    gate=json.loads(r.stdout.strip().splitlines()[-1]);assert gate['EXPLORATORY_RESEARCH_READY']
    return gate


def check(root):
    phase.seed42();out=root/OUT
    if (out/'preflight.json').exists():raise FileExistsError('Preflight already exists; use --verify')
    gate=dataset_gate(root,out);assets,targets=manifests(root);ds=SLBRDataset(root,'TRAIN')
    dates=ds.base.a['reference_dates'][ds.base.a['train_origin_indices']]
    extreme=int(np.where(dates=='2016-12-23')[0][0]);b=collate([ds[0],ds[extreme]])
    inputs={k:b[k] for k in INPUT_KEYS};rows=[]
    for model_id in MODEL_IDS:
        phase.seed42();model=make_model(model_id,assets,targets)
        if isinstance(model,MaskedClassical):
            estimator=model.make_estimator()
            design=model.design(**inputs)
            assert design.shape==(2,161720) and np.isfinite(design).all()
            rows.append(dict(model_id=model_id,status='PASS',parameters=None,
                validation_scope='Constructor/full-input layout; masked fitting covered by fixture tests. No real fit.',
                fixed_configuration=model.configuration()))
            continue
        model.set_epoch(20);p=model(**inputs);loss,counts=masked_huber(p,b['Y'],b['label_valid_mask'])
        (loss+.25*model.switch_loss()).backward()
        assert p.shape==(2,4,23) and torch.isfinite(p).all()
        assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
        active=sum(v.numel() for v in model.parameters() if v.grad is not None)
        model.eval()
        valid=b['feature_valid_mask']&b['input_valid_mask'][...,None]&b['feature_applicable_mask'][:,None]
        changed={k:v.clone() for k,v in inputs.items()};changed['X'][~valid]=float('nan')
        with torch.no_grad():
            before=model(**inputs);after=model(**changed)
            torch.testing.assert_close(before,after,rtol=0,atol=0)
            single=torch.cat([model(**{k:v[i:i+1] for k,v in inputs.items()}) for i in range(2)])
            torch.testing.assert_close(before,single,rtol=1e-5,atol=2e-6)
        rows.append(dict(model_id=model_id,status='PASS',parameters=sum(v.numel() for v in model.parameters()),
            parameters_with_gradient_in_real_fixture=active,output_shape=list(p.shape),prediction_abs_max=float(p.detach().abs().max()),
            loss=float(loss.detach()),label_counts=counts,invalid_perturbation_difference=float((before-after).abs().max()),
            batch_single_max_difference=float((before-single).abs().max()),real_train_samples=[str(dates[0]),str(dates[extreme])]))
        print('PASS '+model_id,flush=True)
    phase.write_json(out/'preflight.json',dict(PASS=True,dataset_gate=gate,models=rows,source_hashes=hashes(root),
        real_training_runs=0,real_classical_fits=0,TEST_numeric_rows_parsed=0,MAE_computed=0))
    phase.write_csv(out/'parameter_counts.csv',[{k:r.get(k) for k in ('model_id','parameters','parameters_with_gradient_in_real_fixture')} for r in rows])
    print('All37 registered models checked; no real fitting or MAE evaluation.',flush=True)


def save_validation(out,result):
    phase.write_csv(out/'validation_metrics.csv',[result['metrics']])
    phase.write_csv(out/'per_horizon_metrics.csv',result['horizons']);phase.write_csv(out/'per_target_horizon_metrics.csv',result['targets'])
    np.savez_compressed(out/'validation_predictions.npz',prediction=result['prediction'],target=result['Y'],label_valid_mask=result['mask'],origin_dates=np.array(result['origin_dates']))


def training_protocol():
    return dict(seed=42,optimizer='Adam',learning_rate=1e-4,weight_decay=1e-5,batch_size=32,max_epochs=200,patience=10,
        gradient_clip_norm=1.,AMP=False,shuffle=False,drop_last=False,loss='equal mean of nonempty masked horizon Huber .02',
        switch_KL_multiplier=.25,scheduler='ReduceLROnPlateau factor.5 patience5',
        selection='equal four-horizon commodity-macro VAL MAE',TEST='PROHIBITED',hyperparameter_search=False,
        comparator='shared-bond SLBR, fresh initialization; old four-encoder checkpoint is not a reference fit')


def train(root,model_id,device,run_id):
    gate=json.loads((root/OUT/'preflight.json').read_text())
    if not gate['PASS'] or hashes(root)!=gate['source_hashes']:raise RuntimeError('Model preflight source mismatch')
    phase.seed42();dev=torch.device(device)
    if dev.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,99}',run_id):raise ValueError('Invalid run-id')
    out=root/OUT/'runs'/run_id;out.mkdir(parents=True,exist_ok=False)
    dataset_gate(root,out);assets,targets=manifests(root);model=make_model(model_id,assets,targets)
    cfg=training_protocol()
    metadata=dict(model_id=model_id,model_registry_entry=next(r for r in registry() if r['model_id']==model_id),
        protocol=cfg,source_hashes=hashes(root),dataset_manifest_sha256=phase.sha(root/DATA/'dataset_manifest.json'),
        assets=assets,targets=targets,environment=phase.environment(),seed=42)
    phase.write_json(out/'config.json',metadata)
    if isinstance(model,MaskedClassical):return train_classical(root,out,model,metadata)
    model.to(dev);loaders=phase.loaders(root,32)
    opt=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=5)
    best=float('inf');stale=0;history=[]
    for epoch in range(1,201):
        start=time.perf_counter();model.train();model.set_epoch(epoch)
        total=0.;steps=empty=0;gradmax=0.;den=np.zeros(4,np.int64)
        for batch in loaders['TRAIN']:
            b=to_device(batch,dev);opt.zero_grad(set_to_none=True)
            try:
                prediction=model(**{k:b[k] for k in INPUT_KEYS})
                loss,counts=masked_huber(prediction,b['Y'],b['label_valid_mask']);den+=counts
                if loss is None:empty+=1;continue
                objective=loss+.25*model.switch_loss()
                if not torch.isfinite(objective):raise FloatingPointError('Nonfinite loss; stop run')
                objective.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                opt.step();steps+=1;total+=float(objective.detach());gradmax=max(gradmax,float(norm))
            except (FloatingPointError,RuntimeError) as error:
                phase.failure_dump(out,epoch,b,model,str(error));raise
        result=phase.evaluate(model,loaders['VAL'],dev);score=result['metrics']['VAL_SELECTION_SCORE'];scheduler.step(score)
        row=dict(epoch=epoch,train_loss=total/max(steps,1),VAL_SELECTION_SCORE=score,gradient_norm_max=gradmax,
            empty_batches=empty,optimizer_steps=steps,epoch_seconds=time.perf_counter()-start,
            valid_label_denominators=den.tolist(),learning_rate=opt.param_groups[0]['lr'])
        history.append(row);phase.write_csv(out/'training_history.csv',history)
        if score<best:
            best=score;stale=0
            torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),best_epoch=epoch,best_score=score,metadata=metadata),out/'best.tmp.pt')
            (out/'best.tmp.pt').replace(out/'best.pt')
        else:stale+=1
        print(json.dumps(row),flush=True)
        if stale>=10:break
    cp=torch.load(out/'best.pt',map_location=dev,weights_only=False);model.load_state_dict(cp['model_state_dict'],strict=True)
    result=phase.evaluate(model,loaders['VAL'],dev);save_validation(out,result)
    assert cp['best_epoch']==min(history,key=lambda row:row['VAL_SELECTION_SCORE'])['epoch']
    assert abs(result['metrics']['VAL_SELECTION_SCORE']-cp['best_score'])<1e-8
    restored=make_model(model_id,assets,targets).to(dev);restored.load_state_dict(cp['model_state_dict'],strict=True);restored.eval()
    batch=to_device(next(iter(loaders['VAL'])),dev)
    with torch.no_grad():
        p=model(**{k:batch[k] for k in INPUT_KEYS});q=restored(**{k:batch[k] for k in INPUT_KEYS})
    torch.testing.assert_close(p,q,rtol=0,atol=0)
    phase.write_json(out/'checkpoint_audit.json',dict(PASS=True,best_epoch=cp['best_epoch'],
        independent_reload_max_difference=float((p-q).abs().max()),history_argmin_matches=True))
    phase.write_json(out/'training_complete.json',dict(model_id=model_id,best_epoch=cp['best_epoch'],epochs=len(history),
        best_score=best,checkpoint_sha256=phase.sha(out/'best.pt'),TEST_numeric_rows_parsed=0))


def train_classical(root,out,model,metadata):
    """Explicit future fit only; matrix written batchwise, without complete-case deletion."""
    import joblib
    splits={s:SLBRDataset(root,s) for s in ('TRAIN','VAL')}
    matrices={}
    for split,ds in splits.items():
        path=out/(split.lower()+'_design.npy')
        matrix=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(ds),161720))
        for start in range(0,len(ds),32):
            batch=collate([ds[i] for i in range(start,min(start+32,len(ds)))])
            matrix[start:start+len(batch['X'])]=model.design(**{k:batch[k] for k in INPUT_KEYS})
        matrix.flush();matrices[split]=matrix
    train=splits['TRAIN'].base
    model.fit_design(matrices['TRAIN'],train.a['train_targets'],train.a['train_label_valid_mask'],split='TRAIN')
    joblib.dump(dict(estimator=model,metadata=metadata),out/'estimator.joblib',compress=3)
    # Adapt the fixed fitted estimator to the same masked evaluator, never to fit.
    class Predictor(torch.nn.Module):
        def __init__(self,estimator):
            super().__init__();self.estimator=estimator;self.targets=estimator.targets
            from types import SimpleNamespace
            self.config=SimpleNamespace(horizons=(1,5,10,20))
        def forward(self,**inputs):return torch.from_numpy(self.estimator.predict(**inputs).astype(np.float32))
    result=phase.evaluate(Predictor(model),phase.loaders(root,32)['VAL'],torch.device('cpu'))
    save_validation(out,result)
    phase.write_json(out/'training_complete.json',dict(model_id=metadata['model_id'],TRAIN_only=True,TEST_numeric_rows_parsed=0,
        fitted_outputs=model.fitted_outputs.tolist(),estimator_groups=len(model.groups),estimator_sha256=phase.sha(out/'estimator.joblib')))


def main():
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group(required=True)
    for name in ('list','check','verify','train'):mode.add_argument('--'+name,action='store_true')
    p.add_argument('--model',choices=MODEL_IDS);p.add_argument('--device',default='cpu');p.add_argument('--run-id')
    args=p.parse_args();root=Path.cwd()
    if args.list:print(json.dumps(registry(),indent=2))
    elif args.check:check(root)
    elif args.verify:
        value=json.loads((root/OUT/'preflight.json').read_text())
        assert value['PASS'] and value['source_hashes']==hashes(root)
        print('Benchmark preflight source verification PASS')
    else:
        if not args.model or not args.run_id:p.error('--train requires one --model and a new --run-id')
        train(root,args.model,args.device,args.run_id)


if __name__=='__main__':main()
