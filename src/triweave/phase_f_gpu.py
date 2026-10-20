"""Explicit seed42 checkpoint continuation; preserve the interrupted CPU run.

This adapter never edits frozen SLBR model/trainer sources or old artifacts.
CPU -> CUDA continuation preserves weights/Adam/scheduler, not bitwise dropout.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import shutil
import time

import numpy as np
import pandas as pd
import torch

from . import slbr_phase as phase
from .models.slbr_triweave import masked_huber
from .slbr_data import DATA, INPUT_KEYS, OUT, to_device


def verify_parent(root):
    parent = root / OUT
    frozen = json.loads((parent / 'preflight_gate.json').read_text())
    if not frozen['PASS'] or frozen['adaptation_hashes'] != phase.source_hashes(root):
        raise RuntimeError('Frozen model/preflight source mismatch')
    for name in ('model_config', 'training_config'):
        if phase.sha(parent / (name + '.json')) != frozen[name + '_sha256']:
            raise RuntimeError('Frozen configuration mismatch: ' + name)
    cp_path = parent / 'checkpoints/best_seed42.pt'
    cp = torch.load(cp_path, map_location='cpu', weights_only=False)
    if cp['metadata']['adaptation_hashes'] != frozen['adaptation_hashes'] or cp['metadata']['seed'] != 42:
        raise RuntimeError('Checkpoint lineage mismatch')
    for name, digest in cp['metadata']['data_fingerprints'].items():
        if phase.sha(root / name) != digest:
            raise RuntimeError('Dataset changed: ' + name)
    if phase.sha(parent / 'source_code_manifest.json') != cp['metadata']['source_manifest_sha256']:
        raise RuntimeError('CMGM source manifest changed')
    history = pd.read_csv(parent / 'training_history.csv')
    prefix = history_prefix(history, cp['best_epoch'])
    return parent, cp_path, cp, prefix, int(history.epoch.max())


def history_prefix(history, checkpoint_epoch):
    """The old best checkpoint cannot restore unsaved later optimizer steps."""
    prefix = history.loc[history.epoch <= checkpoint_epoch].copy()
    if prefix.epoch.astype(int).tolist() != list(range(1, checkpoint_epoch + 1)):
        raise RuntimeError('Checkpoint/history prefix has gaps')
    return prefix


def restore_training_state(model, optimizer, scheduler, payload, device):
    model.load_state_dict(payload['model_state_dict'], strict=True)
    optimizer.load_state_dict(payload['optimizer_state_dict'])
    scheduler.load_state_dict(payload['scheduler_state_dict'])
    torch.set_rng_state(payload['rng_state'].cpu())
    if device.type == 'cuda' and payload.get('cuda_rng_states'):
        torch.cuda.set_rng_state_all([s.cpu() for s in payload['cuda_rng_states']])


def require_device(device):
    dev = torch.device(device)
    if dev.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable in this shell. Run in the GPU server terminal; no CPU fallback.')
        # Allocate and execute, rather than trusting an availability flag alone.
        x = torch.ones(2, 2, device=dev)
        assert float((x @ x).sum()) == 8.
    return dev


def run(root, device, run_id, check_only=False):
    phase.seed42()
    dev = require_device(device)
    parent, cp_path, cp, prefix, observed_last_epoch = verify_parent(root)
    print(json.dumps(dict(device=str(dev),environment=phase.environment(),frozen_source_gate=True,
        dataset_fingerprints_pass=True,resume_checkpoint_epoch=cp['best_epoch'],
        last_CPU_history_epoch=observed_last_epoch,seed=42)), flush=True)
    if check_only:
        return
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', run_id):
        raise ValueError('run-id must be a simple new directory name')
    output = parent / 'gpu_runs' / run_id
    output.mkdir(parents=True, exist_ok=False)
    for name in ('model_config.json', 'training_config.json', 'source_code_manifest.json', 'preflight_gate.json'):
        shutil.copy2(parent / name, output / name)
    # Reuse the frozen trainer's loading/evaluation/finalization in this NEW output.
    # Runtime output routing does not change any frozen source or configuration.
    phase.OUT = output.relative_to(root)
    phase.gate(root)
    model = phase.fresh_model(root).to(dev)
    cfg = json.loads((output / 'training_config.json').read_text())
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.5, patience=5)
    restore_training_state(model, optimizer, scheduler, cp, dev)
    first_epoch = int(cp['best_epoch']) + 1
    best, best_epoch, stale = float(cp['best_val_selection_score']), int(cp['best_epoch']), 0
    rows = prefix.to_dict('records')
    loaders = phase.loaders(root, cfg['batch_size'])
    checkpoints = output / 'checkpoints'
    checkpoints.mkdir()
    shutil.copy2(cp_path, checkpoints / 'best_seed42.pt')
    metadata = copy.deepcopy(cp['metadata'])
    metadata.update(environment=phase.environment(), resume_adapter_sha256=phase.sha(Path(__file__)),
        resume_parent_checkpoint_sha256=phase.sha(cp_path), resume_parent_epoch=best_epoch,
        CPU_to_CUDA_bitwise_equivalence_claimed=False)
    phase.write_json(output / 'resume_provenance.json', dict(parent_checkpoint=str(cp_path),
        parent_checkpoint_sha256=phase.sha(cp_path),parent_history_sha256=phase.sha(parent/'training_history.csv'),
        resume_epoch=first_epoch,parent_last_recorded_epoch=observed_last_epoch,
        unsaved_tail_epochs_replayed_from_best=list(range(first_epoch,observed_last_epoch+1)),
        device=str(dev),adapter_sha256=phase.sha(Path(__file__)),seed=42,
        CPU_to_CUDA_bitwise_equivalence_claimed=False,old_artifacts_unchanged=True))
    phase.write_json(output / 'environment.json', phase.environment() | dict(device=str(dev)))
    begin = time.perf_counter()
    for epoch in range(first_epoch, cfg['max_epochs'] + 1):
        t = time.perf_counter()
        model.train(); model.set_epoch(epoch)
        total = pred_total = kl_total = grad_total = grad_max = act_max = pred_max = 0.
        steps = empty = clips = 0
        counts = np.zeros(4,np.int64);target_counts=np.zeros(23,np.int64)
        if dev.type == 'cuda':torch.cuda.reset_peak_memory_stats(dev)
        for batch in loaders['TRAIN']:
            b=to_device(batch,dev);optimizer.zero_grad(set_to_none=True)
            try:
                prediction,diag=model(**{k:b[k] for k in INPUT_KEYS},return_diagnostics=True)
                loss,den=masked_huber(prediction,b['Y'],b['label_valid_mask'],cfg['huber_delta'])
                counts+=den;target_counts+=b['label_valid_mask'].sum((0,1)).cpu().numpy()
                if loss is None:empty+=1;continue
                kl=cfg['original_switch_KL_multiplier']*model.switch_loss();objective=loss+kl
                if not torch.isfinite(objective) or not torch.isfinite(prediction).all():
                    raise FloatingPointError('Nonfinite loss/prediction')
                objective.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip_norm'],error_if_nonfinite=True)
                optimizer.step()
            except (FloatingPointError,RuntimeError) as err:
                phase.failure_dump(output,epoch,b,model,str(err));raise
            gn=float(norm);grad_total+=gn;grad_max=max(grad_max,gn);clips+=gn>cfg['gradient_clip_norm'];steps+=1
            total+=float(objective.detach());pred_total+=float(loss.detach());kl_total+=float(kl.detach())
            pred_max=max(pred_max,float(prediction.detach().abs().max()))
            act_max=max(act_max,max(float(diag[k].abs().max()) for k in ('graph_nodes','long_memory','latent_states','fused')))
            del prediction,diag,loss,kl,objective
        result=phase.evaluate(model,loaders['VAL'],dev);score=result['metrics']['VAL_SELECTION_SCORE'];scheduler.step(score)
        row=dict(epoch=epoch,train_loss=total/max(steps,1),train_prediction_huber=pred_total/max(steps,1),
            train_switch_KL=kl_total/max(steps,1),VAL_SELECTION_SCORE=score,gradient_norm_mean=grad_total/max(steps,1),
            gradient_norm_max=grad_max,gradient_clip_count=clips,activation_abs_max=act_max,prediction_abs_max=pred_max,
            nonfinite_count=0,empty_label_batches=empty,optimizer_steps=steps,learning_rate=optimizer.param_groups[0]['lr'],
            epoch_seconds=time.perf_counter()-t,gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(dev) if dev.type=='cuda' else 0)
        for i,h in enumerate(model.config.horizons):
            row[f'train_valid_labels_{h}d']=int(counts[i]);row[f'val_macro_MAE_{h}d']=result['horizons'][i]['macro_MAE']
        rows.append(row);phase.write_csv(output/'training_history.csv',rows)
        improved=score<best
        if improved:best=score;best_epoch=epoch;stale=0
        else:stale+=1
        payload=dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=scheduler.state_dict(),best_epoch=best_epoch,best_val_selection_score=best,
            last_epoch=epoch,stale_epochs=stale,metadata=metadata,rng_state=torch.get_rng_state(),
            cuda_rng_states=torch.cuda.get_rng_state_all() if dev.type=='cuda' else [])
        torch.save(payload,checkpoints/'last.tmp.pt');(checkpoints/'last.tmp.pt').replace(checkpoints/'last_seed42.pt')
        if improved:
            shutil.copy2(checkpoints/'last_seed42.pt',checkpoints/'best.tmp.pt')
            (checkpoints/'best.tmp.pt').replace(checkpoints/'best_seed42.pt')
        print(json.dumps(row),flush=True)
        if stale>=cfg['patience']:break
    phase.write_json(output/'training_complete.json',dict(seed=42,epochs=len(rows),best_epoch=best_epoch,best_score=best,
        continuation_seconds=time.perf_counter()-begin,early_stopped=stale>=cfg['patience'],other_seeds_run=False))
    phase.finalize(root,str(dev))
    phase.write_json(output/'resume_validation_complete.json',dict(seed42_training_complete=True,
        checkpoint_recovery_pass=True,validation_metrics_saved=True,TEST_numeric_rows_parsed=0,
        full_Phase_F_research_report_complete=False,CONFIRMATORY_RESEARCH_READY=False))
    print('Seed42 continuation and VAL export complete: '+str(output),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--check',action='store_true',help='Check GPU, frozen sources and checkpoint/data hashes; no training')
    action.add_argument('--run',action='store_true',help='Continue seed42 in a NEW output directory and export VAL metrics')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--run-id',default='seed42_gpu_resume')
    args=parser.parse_args()
    run(Path.cwd(),args.device,args.run_id,args.check)


if __name__=='__main__':main()
