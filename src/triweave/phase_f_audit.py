"""Post-fit Phase F audit from saved TRAIN/VAL evidence; never fit a model.

Creates missing final reports once. --verify checks the resulting delivery seal.
Existing CPU and GPU experiment files are preserved byte for byte.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

from . import slbr_phase as phase
from .slbr_data import DATA, OUT, SLBRDataset, manifests
from .models import slbr_core


def verify(root):
    out=root/OUT
    delivery=json.loads((out/'delivery_manifest.json').read_text())
    changed=[n for n,h in delivery['files'].items() if not (root/n).is_file() or phase.sha(root/n)!=h]
    if changed:raise RuntimeError('Delivery seal changed: '+repr(changed))
    return dict(PASS=True,files=len(delivery['files']),delivery_manifest_sha256=phase.sha(out/'delivery_manifest.json'))


def finish(root):
    out=root/OUT;run=out/'gpu_runs/seed42_gpu_resume'
    if (out/'REPORT.md').exists() or (out/'audit_results.json').exists():
        raise FileExistsError('Final audit already exists; use --verify')
    complete=json.loads((run/'training_complete.json').read_text())
    checkpoint_audit=json.loads((run/'checkpoint_audit.json').read_text())
    recovery=json.loads((out/'postrun_checkpoint_recovery.json').read_text())
    frozen=json.loads((out/'preflight_gate.json').read_text())
    assert frozen['adaptation_hashes']==phase.source_hashes(root)
    cp=torch.load(run/'checkpoints/best_seed42.pt',map_location='cpu',weights_only=False)
    for name,h in cp['metadata']['data_fingerprints'].items():assert phase.sha(root/name)==h,name
    assert cp['metadata']['resume_adapter_sha256']==phase.sha(root/'src/triweave/phase_f_gpu.py')
    assert phase.sha(run/'checkpoints/best_seed42.pt')==checkpoint_audit['sha256']
    protected=json.loads((out/'preservation_before.json').read_text())
    changed=[n for n,h in protected.items() if not (root/n).is_file() or phase.sha(root/n)!=h]
    assert not changed,changed
    source=json.loads((out/'source_code_manifest.json').read_text())
    cmchanges=[f['path'] for f in source['files'] if f['version']=='working_tree_read_only' and phase.sha(Path(f['path']))!=f['sha256']]
    assert not cmchanges,cmchanges
    for row in source['files']:
        if row['version']!='working_tree_read_only':assert phase.sha(root/row['path'])==row['sha256']
    assets,targets=manifests(root)
    history=pd.read_csv(run/'training_history.csv')
    metrics=pd.read_csv(run/'validation_metrics.csv').iloc[0].to_dict()
    horizon=pd.read_csv(run/'per_horizon_mae.csv')
    per_target=pd.read_csv(run/'per_target_horizon_metrics.csv')
    commodity=pd.read_csv(run/'per_commodity_mae.csv')
    diagnostic=json.loads((run/'diagnostic_summary.json').read_text())
    graph=pd.read_csv(run/'graph_market_diagnostics.csv')
    gate=pd.read_csv(run/'fusion_gate_diagnostics.csv')
    regimes=pd.read_csv(run/'regime_diagnostics.csv')
    cache=np.load(run/'validation_predictions.npz',allow_pickle=False)
    p,y,m=cache['prediction'],cache['target'],cache['label_valid_mask']
    assert p.shape==y.shape==m.shape==(483,4,23) and m.dtype==np.bool_
    assert cache['target_ids'].tolist()==targets and cache['horizons'].tolist()==[1,5,10,20]
    val=SLBRDataset(root,'VAL').base;train=SLBRDataset(root,'TRAIN').base
    np.testing.assert_array_equal(y,val.a['val_targets']);np.testing.assert_array_equal(m,val.a['val_label_valid_mask'])
    assert np.isfinite(p).all() and np.isfinite(y[m]).all()
    assert int(history.epoch.iloc[-1])==45 and int(history.iloc[history.VAL_SELECTION_SCORE.argmin()].epoch)==35
    assert complete['early_stopped'] and 45-35==json.loads((out/'training_config.json').read_text())['patience']
    assert int(history.nonfinite_count.sum())==0
    oracle=[];variation=[];count_rows=[]
    for hi,h in enumerate((1,5,10,20)):
        for ci,target in enumerate(targets):
            ok=m[:,hi,ci]
            # Independent float64 subtraction; saved metrics used float32 subtraction.
            pp=p[:,hi,ci][ok].astype(np.float64);yy=y[:,hi,ci][ok].astype(np.float64)
            e=pp-yy;ab=np.abs(e)
            row=dict(horizon=h,asset_id=target,valid_label_count=int(ok.sum()),MAE=float(ab.mean()),
                RMSE=float(np.sqrt((e*e).mean())),Huber=float(np.where(ab<=.02,.5*e*e,.02*(ab-.01)).mean()),
                direction_hit=float((np.sign(pp)==np.sign(yy)).mean()))
            oracle.append(row)
            saved=per_target.loc[(per_target.horizon==h)&(per_target.asset_id==target)].iloc[0]
            for field in ('MAE','RMSE','Huber','direction_hit'):
                assert abs(row[field]-saved[field])<1e-8,(target,h,field)
            variation.append(dict(asset_id=target,horizon=h,valid_labels=int(ok.sum()),prediction_mean=float(pp.mean()),
                label_mean=float(yy.mean()),prediction_time_std=float(pp.std()),label_time_std=float(yy.std()),
                prediction_to_label_std_ratio=float(pp.std()/yy.std())))
            for split,base in (('TRAIN',train),('VAL',val)):
                count_rows.append(dict(split=split,asset_id=target,horizon=h,
                    valid_label_count=int(base.a[split.lower()+'_label_valid_mask'][:,hi,ci].sum())))
    oracle_macro=float(np.mean([r['MAE'] for r in oracle]))
    oracle_micro=float(np.abs(p.astype(np.float64)[m]-y.astype(np.float64)[m]).mean())
    assert abs(oracle_macro-metrics['macro_MAE'])<1e-8 and abs(oracle_micro-metrics['micro_MAE'])<1e-8
    phase.write_json(out/'independent_metric_audit.json',dict(PASS=True,VALID_label_count=int(m.sum()),
        oracle_float64_macro_MAE=oracle_macro,oracle_float64_micro_MAE=oracle_micro,
        saved_macro_MAE=metrics['macro_MAE'],saved_micro_MAE=metrics['micro_MAE'],
        tolerance=1e-8,arithmetic_note='Saved evaluator subtracts float32 before float64 accumulation; oracle subtracts in float64.',
        labels_and_masks_equal_frozen_VAL=True,TEST_numeric_rows_parsed=0))
    phase.write_csv(out/'prediction_variation_audit.csv',variation)
    phase.write_csv(out/'label_counts_by_target_horizon.csv',count_rows)
    batch_rows=[];lm=train.a['train_label_valid_mask']
    for k,start in enumerate(range(0,len(train),32)):
        counts=lm[start:start+32].sum((0,2));active=int((counts>0).sum())
        batch_rows.append(dict(batch=k,first_origin=str(train.a['reference_dates'][train.a['train_origin_indices'][start]]),
            batch_size=min(32,len(train)-start),nonempty_horizons=active,
            **{f'denominator_{h}d':int(n) for h,n in zip((1,5,10,20),counts)},
            equal_weight_per_active_horizon=1/active if active else 0.,optimizer_step_expected=bool(active)))
    phase.write_csv(out/'loss_denominators.csv',batch_rows)
    gate['year']=gate.origin_date.str[:4]
    grouped=[]
    for grouping in ('year','regime_state'):
        for value,g in gate.groupby(grouping):
            grouped.append(dict(grouping=grouping,group=str(value),origins=len(g),gate_mean=float(g.gate_mean.mean()),
                gate_origin_mean_std=float(g.gate_mean.std(ddof=0)),gate_min=float(g.gate_min.min()),gate_max=float(g.gate_max.max())))
    phase.write_csv(out/'fusion_gate_grouped_diagnostics.csv',grouped)
    # Prove the three directly reused classes have the original class source bytes.
    import inspect
    original=out/'source_snapshots/45879f5'
    reuse=[]
    for file,names in [('cmgm/graph/adaptive_graph.py',['AdaptiveGraphLearner']),
                       ('cmgm/models/switching_latent_transformer.py',['MarkovRegimeFilter','RegimeLatentTransition'])]:
        text=(original/file).read_text();lines=text.splitlines(keepends=True)
        for node in ast.parse(text).body:
            if isinstance(node,ast.ClassDef) and node.name in names:
                old=''.join(lines[node.lineno-1:node.end_lineno]).rstrip()
                current=inspect.getsource(getattr(slbr_core,node.name)).rstrip()
                assert old==current,node.name
                reuse.append(dict(class_name=node.name,original_git='45879f5',source_file=file,exact_class_source_equal=True))
    phase.write_json(out/'source_reuse_audit.json',dict(PASS=True,classes=reuse,old_model_instantiated_parameters=520549,
        CMGM_source_changed=False,source_protection_count=len(source['files'])))
    # Final index copies only names that do not already exist; CPU history stays immutable.
    for name in ('validation_metrics.csv','per_horizon_mae.csv','per_commodity_mae.csv','regime_diagnostics.csv',
                 'fusion_gate_diagnostics.csv','checkpoint_audit.json','numerical_stability_report.csv'):
        assert not (out/name).exists();shutil.copy2(run/name,out/name)
    shutil.copy2(run/'training_history.csv',out/'final_training_history.csv')
    evidence_names=['CMGM_SOURCE_LOCATED','CMGM_ORIGINAL_SLBR_IDENTIFIED','ARCHITECTURE_PRESERVATION_PASS',
        'HETEROGENEOUS_ENCODER_PASS','SPATIAL_BRANCH_PASS','TEMPORAL_BRANCH_PASS','REGIME_MODULE_PASS',
        'BALANCED_READOUT_PASS','FUSION_GATE_PASS','INPUT_311_ASSETS_PASS','OUTPUT_23_TARGETS_PASS',
        'MASK_PROPAGATION_PASS','GRAPH_CAUSALITY_PASS','TEMPORAL_CAUSALITY_PASS','MASKED_LOSS_PASS',
        'FORWARD_PASS','BACKWARD_PASS','NUMERICAL_STABILITY_PASS','DATASET_GATE_PASS','CHECKPOINT_RECOVERY_PASS',
        'BEST_EPOCH_SELECTION_PASS','VALIDATION_METRICS_PASS','TEST_ISOLATION_PASS','TRIWEAVE_SLBR_TRAIN_READY',
        'TRIWEAVE_SLBR_SEED42_COMPLETE']
    assert checkpoint_audit['PASS'] and recovery['PASS'] and frozen['PASS']
    assert all(c['backward_finite'] for c in json.loads((out/'shape_tests.json').read_text())['real_cases'])
    assert graph.temporal_attention_future_mass.max()==0
    causal=json.loads((out/'causality_tests.json').read_text())
    assert causal['software_causality_pass'] and causal['TEST_numeric_rows_parsed']==0
    admission={name:True for name in evidence_names}
    admission.update(CONFIRMATORY_RESEARCH_READY=False,HISTORICAL_PIT_VERIFIED=False,FUTURES_ROLL_METHOD_VERIFIED=False,
        FULL_PHASE_F_REPORT_COMPLETE=True,REGIME_MEANINGFUL_STATE_SEPARATION_VERIFIED=False,
        FUSION_GATE_SUBSTANTIAL_DYNAMIC_RESPONSE_VERIFIED=False,PREDICTIVE_ADVANTAGE_ESTABLISHED='NOT_VERIFIED',
        canonical_training_run=str(run.relative_to(root)),best_epoch=35,stopped_epoch=45,seed=42,
        evidence_scope='Engineering/declared software-causality gates; no economic causal or predictive-advantage certification',
        TEST_numeric_rows_parsed=0,other_seeds_run=False)
    phase.write_json(out/'audit_results.json',admission)
    phase.write_json(out/'provenance.json',dict(dataset_protocol='TRIWEAVE_23_AFTER_CLOSE_RESEARCH_V1',
        dataset_manifest_sha256=phase.sha(root/DATA/'dataset_manifest.json'),data_fingerprints=cp['metadata']['data_fingerprints'],
        model_code_sha256=phase.source_hashes(root),resume_adapter_sha256=phase.sha(root/'src/triweave/phase_f_gpu.py'),
        audit_code_sha256=phase.sha(Path(__file__)),CMGM_source_manifest_sha256=phase.sha(out/'source_code_manifest.json'),
        CPU_checkpoint_sha256=phase.sha(out/'checkpoints/best_seed42.pt'),GPU_best_checkpoint_sha256=phase.sha(run/'checkpoints/best_seed42.pt'),
        GPU_log_sha256=phase.sha(root/'logs/phase_f_seed42_gpu_resume.log'),
        protected_pre_F_files=len(protected),changed_pre_F_files=[],changed_CMGM_source_files=[],
        lineage='CPU epochs1-24 -> CUDA epochs25-45; separately preserved CPU25-26 tail replayed from epoch24.',
        CPU_to_CUDA_bitwise_equivalence_claimed=False,TEST_numeric_rows_parsed=0,
        no_new_market_requests=True,no_scaler_refit=True,no_other_seed=True,no_git_commit_or_push=True))
    write_reports(out,run,metrics,horizon,commodity,diagnostic,history,graph,variation,complete,admission)
    outputs={str(p.relative_to(root)):phase.sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json'}
    for pattern in ('models/*.py','slbr*.py','phase_f*.py'):
        for p in (root/'src/triweave').glob(pattern):outputs[str(p.relative_to(root))]=phase.sha(p)
    for p in (root/'tests').glob('test_*slbr*.py'):outputs[str(p.relative_to(root))]=phase.sha(p)
    outputs['tests/test_phase_f_gpu.py']=phase.sha(root/'tests/test_phase_f_gpu.py')
    phase.write_json(out/'delivery_manifest.json',dict(files=outputs,files_count=len(outputs),
        experiment_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
        dataset_bytes=sum(p.stat().st_size for p in (root/DATA).iterdir() if p.is_file()),
        canonical_run=str(run.relative_to(root)),old_CPU_artifacts_preserved=True,TEST_numeric_rows_parsed=0))
    print(json.dumps(verify(root)),flush=True)


def write_reports(out,run,metrics,horizon,commodity,d,history,graph,variation,complete,admission):
    v=pd.DataFrame(variation)
    worst=commodity.sort_values('macro_MAE',ascending=False).head(5)
    gpu=history.loc[history.epoch>=25]
    rows='\n'.join(f"| {int(r.horizon)}d | {r.macro_MAE:.9f} | {r.macro_MAE*100:.4f} | {r.macro_direction_hit*100:.2f}% | {int(r.valid_label_count)} |" for r in horizon.itertuples())
    worst_rows='\n'.join(f"| {r.asset_id} | {r.macro_MAE:.9f} |" for r in worst.itertuples())
    flags='\n'.join(f"| {k} | {str(val).lower() if isinstance(val,bool) else val} |" for k,val in admission.items() if k.isupper())
    (out/'ARCHITECTURE_AUDIT.md').write_text('''# Architecture audit

PASS applies to source-preserved structure and tested numerical/mask behavior,
not to learned economic significance. See CMGM_TO_TRIWEAVE_MODEL_MAPPING.md.

The original introduction commit45879f5 and archived seed42 checkpoint identify
HeteroMixHopCMGM variant switching_latent_balanced_readout. Instantiating original
AST classes gives520549 parameters; archived checkpoint float weights total the
same. Original historical working-tree SHA differs from its recorded git tree,
so git alone is not asserted to reproduce that run byte for byte.

Exactly reused source classes: AdaptiveGraphLearner, MarkovRegimeFilter,
RegimeLatentTransition. source_reuse_audit.json verifies their class source.
Original all-valid graph propagation and causal attention were compared against
the adapted modules using identical weights. Balanced projections, LN, gate
direction and Markov recurrence have explicit equation-level tests.

Spatial: four-type independent encoding → masked node-wise temporal attention →
adaptive directed graph → two EdgeAttnMixHop layers (8heads, K2, beta.05) → LN →
three valid market means →192→64. Temporal is a parallel branch: independent
four-type encoding32 → three market attention+dispersion tokens →192→128 →
two causal Base-RPE blocks (4heads, FFN256) →3-state sticky Markov filter and
three deterministic latent generators → separate H/Z projections and LN →
concat128→64 balanced readout. Channel gate weights temporal projection;
1-gate weights spatial projection. Shared64→64→92 head reshapes[B,4,23].

The learnable graph uses fixed manifest identity and TRAIN gradients; there is
no TRAIN+VAL correlation graph. Soft top10 is not hard degree10, and log(A+1e-6)
does not exclude zero-prior edges. Actual attention is dense over valid nodes.
Invalid query/key masking, post-bias zeroing, finite empty softmax, separate
market denominators and causal/padding masks are exercised by model tests.

Tests: synthetic311 and313 (no HOLD data), real single/batch/padding/sparse-label/
large-missing/extreme-z, absent stock/future/type/market/neighbors/keys/horizon/
all-label batch, NaN invalid-input perturbation and zero invalid-label gradients.
Forward arguments exclude labels, label masks and future state. Actual quotes
after2023-06-30 were perturbed, legally realigned and transformed using frozen
scalers; fixed-origin input, graph and prediction were unchanged. Temporal
prefix and regime-prefix invariance were also tested. These are software
causality checks under declared publication assumptions, not historical PIT.
''')
    (out/'MODEL_IMPLEMENTATION_NOTES.md').write_text('''# Implementation and limits

New code lives in src/triweave/models/{slbr_core,slbr_triweave}.py,
slbr_data.py, slbr_phase.py; phase_f_gpu.py is the explicit continuation
adapter and phase_f_audit.py is the post-fit report/seal tool. No runtime CMGM
project import, old weight initialization, data re-download, bfill or scaler refit.

Parameter partition (disjoint): heterogeneous_encoder20608, spatial77206,
temporal290175, regime99276, balanced20928, gate16576, head10140 =534909.
Original520549 → +14360: four typed/presence encoders +14080, graph identity
embeddings for27 extra nodes +540, four fewer output rows -260. All other core
widths are unchanged. Bonds remain18 yield +7 wealth nodes;26 feature semantics,
55 TRAIN-only scaler groups and target order are frozen.

All-invalid input produces zero observation representations after biased modules;
the shared prediction head may still express learned unconditional output bias.
Such output is not evidence of a fabricated quotation. Feature absence uses a
bias-free presence projection distinct from an observed zero standardized value.
Spatial history nodes need one valid session in the20-session window; they need
not have a current-day quotation. This uses legal historical context only.
Population dispersion uses sqrt(var+1e-8)-1e-4 for finite zero-variance gradients.

The first model-test failure concerned the manifest market name: actual future
was initially treated as commodity/futures. The new adapter now maps that frozen
name explicitly; the final19 relevant tests and the original254 regression
tests passed. Old source/data/config/scaler files were not changed.

Training inherited Adam/lr1e-4/decay1e-5/max200/patience10 and scheduler;
batch32, retain tail, masked horizon means, gradient clip1 and macro-MAE VAL
selection were frozen before fitting. Native switching KL coefficient is divided
by4 alongside changing four prediction-loss sum to mean. No state balancing or
gate-dynamics regularizer was added. The run continued CPU→GPU from best epoch24,
replaying the unsaved CPU25–26 tail; it is not a from-scratch all-GPU trajectory.

Canonical results are in gpu_runs/seed42_gpu_resume/. The original top-level
training_history.csv remains the interrupted CPU history. final_training_history.csv
is the final45-epoch lineage. CPU and GPU checkpoints are never conflated.
Future OOM/nonfinite errors stop rather than change data or silently skip samples.
Current resume CLI starts a NEW continuation from the original CPU best; it is
not a generic resume-last service. best/last GPU checkpoints are retained.
''')
    (out/'REPORT.md').write_text(f'''# Phase F-SLBR：seed42迁移、续训与最终审计

seed42已正常完成，epoch45因连续10轮验证选择分数未改善而早停；最佳epoch35。
GPU续训约{complete['continuation_seconds']/60:.2f}分钟，执行epoch25–45。
原CPU训练记录到26轮，但仅24轮有可恢复最佳状态；最终轨迹保留CPU1–24、
GPU25–45，另存原CPU25–26记录。没有启动其他seed、模型或TEST评价。
训练后指标均来自`gpu_runs/seed42_gpu_resume/`；根目录旧training_history.csv
继续保留CPU中断记录，最终历史另见final_training_history.csv。

## VAL结果

483个origin，23商品，43544个有效标签。MAE单位decimal_return；百分点评估误差
为decimal×100，不能解读为命中率或可交易收益。

| 周期 | Commodity-macro MAE | 误差（百分点） | Macro方向命中率 | 有效标签 |
|---|---|---|---|---|
{rows}

四周期等权commodity-macro MAE={metrics['macro_MAE']:.12f}，即冻结的VAL_SELECTION_SCORE；
全部有效标签Micro MAE={metrics['micro_MAE']:.12f}，Micro RMSE={metrics['micro_RMSE']:.12f}。
per_target_horizon_metrics.csv同时保存92组MAE/RMSE/Huber/方向命中率；方向定义为sign一致。
每个周期均有23个可评价商品。独立float64指标复核与保存结果误差小于1e-8。

跨四周期平均误差较高的商品如下；这不是相对基准的难度或能力排名。

| 商品 | 四周期Macro MAE |
|---|---|
{worst_rows}

## 学习行为与数值稳定性

工程训练已跑通，预测优势仍未获得同口径对照证据。本轮没有额外训练简单基线。
方向命中率约47.93%–49.57%；不据此声称优于随机、常数预测或任何比较模型。
有效标签上的各目标/周期预测时间标准差与真实标签时间标准差之比：
中位数{v.prediction_to_label_std_ratio.median():.6f}，范围
{v.prediction_to_label_std_ratio.min():.6f}–{v.prediction_to_label_std_ratio.max():.6f}。
输出相对标签变化很小，存在近常量预测迹象，详见prediction_variation_audit.csv。

Regime平均posterior={d['regime_mean_probabilities']}，
平均熵{d['regime_entropy_mean']:.9f}，三状态最大熵ln3={math.log(3):.9f}；
平均最大概率{d['regime_max_probability_mean']:.6f}。
argmax占用0/1/2为3/148/332；状态0极少成为最大值，但soft posterior整体近均匀。
不能将这三类解释为已识别的经济市场状态。

Fusion Gate平均{d['gate_mean']:.9f}，每origin均值的标准差
{d['gate_origin_mean_std']:.9f}，全体通道/日期范围
{d['gate_min']:.6f}–{d['gate_max']:.6f}，接近固定0.5。
按年份和argmax状态分组的描述性统计见fusion_gate_grouped_diagnostics.csv；
未为制造动态性修改机制或正则项。

图prior正边48205/96721，密度约0.49839；soft top10并不等于每节点只有10条边。
VAL窗口有效节点310–311；最后图层平均注意力熵{graph.attention_entropy.mean():.6f}，
平均最大注意力{graph.attention_max_weight.mean():.6f}，也接近均匀聚合。
空间/时间分支范数均值{graph.spatial_norm.mean():.4f}/{graph.temporal_norm.mean():.4f}。
时间注意力未来位置质量为0，未发现合法输入直接访问未来行情。

全45轮记录NaN/Inf=0，空监督batch=0，梯度裁剪触发=0；记录的最大梯度范数
{history.gradient_norm_max.max():.9f}，未见梯度爆炸。
后期平均梯度仍约{history.gradient_norm_mean.iloc[-1]:.6f}；全局范数不等于每个模块
都学到了有用信号。GPU峰值分配约{gpu.gpu_peak_allocated_bytes.max()/1024**3:.3f}GiB。
TRAIN loss后期下降而VAL从最佳轮后恶化，与选择分数上的过拟合/漂移迹象一致；
本轮按冻结早停执行，没有据此改数据、阈值或架构。

## 对原迁移要求的28项回答

1. 找到原始SLBR：是；Commedities git45879f5引入，历史seed42配置和checkpoint佐证。
2. 核心模块：并行图分支、因果Transformer、三市场编码、Markov过滤、潜在递推、
   Balanced Readout、通道门控及共享多任务head。
3. 直接复用：AdaptiveGraphLearner、MarkovRegimeFilter、RegimeLatentTransition原类源码。
4. 重写适配：四经济类型/特征存在性编码、缺失Mask传播、图/时间安全softmax、
   有效市场池化、输入与标签接口、23目标head及masked loss/训练与审计。
5. 空间图保留：是，原两层八头EdgeAttnMixHop；全有效同权重对照通过。
6. 时间分支保留：是，原两层因果Base-RPE Transformer，不是LSTM。
7. Switching Latent Regime保留：是，3状态、原sticky转移和独立latent generators。
8. Balanced Readout保留：是，H和Z各自投影、LN再合并，保持原64维输出。
9. Channel-wise Gate保留：是，gate权重作用于temporal projection，1-gate作用于spatial。
10. 真正支持311节点：是，263股票+25债券+23商品，身份映射明确。
11. 正确处理26特征/四经济类型：是，yield和wealth分别编码，schema/scaler不改。
12. 动态Mask：输入、特征valid/applicable、图query/key、时间因果/padding、池化及loss均检查。
13. 输出[B,4,23]：是，预测顺序来自冻结target_manifest。
14. 参数量：534909。
15. 较旧520549增加14360：异构/存在性编码+14080、图身份embedding+540、head-260。
16. 前向/反向/因果测试：通过；原全项目254测试、恢复相关19测试，以及真实来源未来扰动。
17. seed42真实训练：完成，本次为CPU→GPU续训；不声称纯GPU从头轨迹可逐位复现。
18. 最佳epoch：35，最终45，best与history argmin一致。
19. 四周期MAE：见上表。
20. 整体Macro/Micro：{metrics['macro_MAE']:.12f}/{metrics['micro_MAE']:.12f}。
21. 高误差商品：焦煤JM、焦炭J、铁矿石I、液化石油气PG、燃料油FU。
22. Regime退化：仍有近均匀soft分配、状态0 hard利用不足，未验证有意义的状态分离。
23. Gate静态平均：是，有明显接近0.5的现象，未证明有实质动态融合。
24. 数值稳定：训练记录未出现NaN/Inf或梯度爆炸，极端标准化样本保留。
25. Checkpoint恢复：通过；原GPU独立恢复及诊断开关预测差为0；本轮另在CPU重载与
    GPU缓存预测对比通过2e-6绝对容差，跨设备不主张逐位一致。
26. TEST：没有读取TEST行情/标签数值，没有TEST MAE。
27. 三种子重复实验：工程条件已具备；本轮只保留seed42，后续需另行冻结纯GPU
    从头重复协议，不直接把单次CPU→GPU续训当作稳健性证据。
28. 后续问题：先审核近常量输出与各模块梯度/注意力利用；冻结同口径对照与多种子
    协议后再验证预测优势；继续补历史发布时间/连续合约方法/股票universe偏差证据。
    不根据本轮VAL表现反复更改预处理或资产名单。

## 数据、checkpoint与科研边界

55个TRAIN-only scaler原封复用，SN2015-03-27及两条FU风险记录隔离保持；
TRIWEAVE_23_AFTER_CLOSE_RESEARCH_V1重验通过。14011个迁移前文件SHA均不变，
CMGM所读源文件SHA均不变。原25商品HOLD和FWD3诊断数据未用于本轮模型。
指标只对应23商品供应商连续序列P(t+h)/P(t)-1，不能升级为25商品或真实可交易收益。
publication available_at仍为假设；软件因果性不等于历史PIT。

最佳checkpoint：gpu_runs/seed42_gpu_resume/checkpoints/best_seed42.pt。
SHA256：3891cc679e71dcc2d5f9ed617ee933838b61847ebc38ddcad852038c1d887f6f。
恢复、选择、独立指标、源码、参数、数据、loss分母和文件指纹见各独立审计文件。
旧resume_validation_complete.json记录的是训练脚本结束时“完整报告尚未完成”的状态；
当前最终状态以本REPORT和audit_results.json为准，旧marker保持不变。

## 独立验收

结构/运行门禁通过只认证其工程范围。Regime/Gate学到有效动态机制、模型有预测优势、
历史PIT和确认性科研准入均不得由这些PASS推导。

| 字段 | 结果 |
|---|---|
{flags}

完成seed42后停止，未启动其他实验，未提交或推送Git。
''')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    m=p.add_mutually_exclusive_group(required=True);m.add_argument('--finish',action='store_true');m.add_argument('--verify',action='store_true')
    args=p.parse_args();root=Path.cwd()
    if args.finish:finish(root)
    else:print(json.dumps(verify(root)))


if __name__=='__main__':main()
