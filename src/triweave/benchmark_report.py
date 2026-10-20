"""Seal the model-suite migration without fitting models or reading TEST."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import pandas as pd
from . import benchmark as suite
from . import slbr_phase as phase
from .models.benchmark import registry


def verify(root):
    out=root/suite.OUT;manifest=json.loads((out/'delivery_manifest.json').read_text())
    changed=[name for name,h in manifest['files'].items() if not (root/name).is_file() or phase.sha(root/name)!=h]
    if changed:raise RuntimeError('Migrated suite changed: '+repr(changed))
    return dict(PASS=True,files=len(manifest['files']),delivery_sha256=phase.sha(out/'delivery_manifest.json'))


def finish(root):
    out=root/suite.OUT
    if (out/'REPORT.md').exists():raise FileExistsError('Report exists; verify instead')
    pre=json.loads((out/'preflight.json').read_text())
    assert pre['PASS'] and pre['source_hashes']==suite.hashes(root)
    before=json.loads((out/'preservation_before.json').read_text())
    changed=[n for n,h in before.items() if not (root/n).is_file() or phase.sha(root/n)!=h]
    assert not changed,changed
    source=json.loads((out/'source_manifest.json').read_text())
    for item in source['sources']:
        assert phase.sha(Path(item['source_path']))==item['source_sha256']
        assert phase.sha(root/item['snapshot'])==item['source_sha256']
    tests=(out/'tests.txt').read_text();assert '352 passed' in tests
    rows=registry();assert len(rows)==37 and len(pre['models'])==37
    phase.write_json(out/'model_registry.json',rows)
    phase.write_json(out/'environment.json',{p:importlib.metadata.version(p) for p in ('torch','numpy','pandas','scikit-learn','xgboost','joblib')})
    flags=dict(LATEST_DEEP_V3_8_MIGRATION_PASS=True,SUPPLEMENTAL_4_BASELINES_MIGRATION_PASS=True,
        SLBR_25_CONTROL_CONFIGURATIONS_PASS=True,INPUT_311_FEATURE26_TARGET23_PASS=True,
        BOND_SHARED_ENCODER_PASS=True,DYNAMIC_MASK_PASS=True,FORWARD_BACKWARD_PASS=True,
        DECLARED_PREFIX_CAUSAL_TEST_PASS=True,ORIGIN_WINDOW_INPUT_ONLY_PASS=True,
        DATASET_GATE_PASS=True,OLD_FILES_UNCHANGED=True,CMGM_SOURCE_UNCHANGED=True,
        TEST_ISOLATION_PASS=True,TRAINING_STARTED=False,REAL_CLASSICAL_ESTIMATOR_FITS=0,
        REAL_NEURAL_TRAINING_RUNS=0,MAE_EVALUATION_RUNS=0,TEST_NUMERIC_ROWS_PARSED=0,
        GPU_EXECUTION_VERIFIED='NOT_VERIFIED',TRIWEAVE_CLASSICAL_REAL_FIT_VERIFIED='NOT_VERIFIED',
        HISTORICAL_PIT_VERIFIED=False,FUTURES_ROLL_METHOD_VERIFIED=False,CONFIRMATORY_RESEARCH_READY=False,
        tests_passed=352,neural_preflight_models=34,classical_interface_preflight_models=3,
        protected_old_files=len(before),limitations=['MSGNet/CrossGNN are observed-window, not prefix-causal encoders',
        'DeepV3 uses neutral aggregation; supplemental TCN/classical use full input',
        'Original MoE-only references/ablations and stale historical development registries are excluded',
        'Training loop selection was fixture-tested with optimizer updates disabled; no production run executed'])
    phase.write_json(out/'audit_results.json',flags)
    phase.write_json(out/'development_issues.json',dict(issues=[
        dict(issue='CrossGNN per-variable temporal support broadcasting',resolution='Expand temporal logits to B,N,40,40 before masked softmax; real-mask and all-missing tests pass'),
        dict(issue='MSGNet graph end-convolution bias can revive an absent node',resolution='Re-mask B,20,N after end convolution; dedicated bias-invariance regression passes'),
        dict(issue='Vanilla Transformer initialization fidelity',resolution='Use native MultiheadAttention initialization and deepcopy encoder blocks, preserving original cloned initialization; matched-weight block comparison passes'),
        dict(issue='Temporary new test function placement caused NameError',resolution='Moved source-attention comparison into its owning test; final complete suite passed')],
        initial_preflight='development_preflight_initial.json',old_files_changed=False))
    table='\n'.join(f"| {r['model_id']} | {r['group']} | {r['source_variant']} |" for r in rows)
    count=pd.read_csv(out/'parameter_counts.csv')
    paramtable='\n'.join(f"| {r.model_id} | {int(r.parameters) if pd.notna(r.parameters) else '估计器拟合后确定'} |" for r in count.itertuples())
    (out/'REPORT.md').write_text(f'''# CMGM baseline与消融迁移完成

迁移37个可选配置：12个baseline，25个SLBR配置（含1个Full及24个对照）。
原CMGM只读，TriWeave原始数据、标准化数据、标签、Scaler和旧实验共{len(before)}个文件SHA不变。
本轮没有真实训练、网格搜索、多种子实验、VAL MAE计算或TEST数值读取。
完成352项测试；34个神经模型均通过真实TRAIN样本的前向、反向、
有效数值、无效位置NaN扰动、batch/single一致性检查。
三个传统模型检查构造、全输入布局、TRAIN-only入口和逐标签mask分组接口；
其拟合接口由fixture测试覆盖，不宣称已在本数据上完成估计器拟合。

## 最新来源的确定与范围

主baseline套件取当前cmgm/models/comparison_baselines.py的DeepV3八个注册模型，
对照20260922_143029实验定义。补充迁入当前修复后的TCN（WeightNorm+Chomp，
不使用已废弃的简化TCN）和fixed_baseline_protocol.py的三个固定传统估计器。
源码工作树SHA、路径最近Git记录和完整快照见source_manifest.json及source_snapshots/；
未根据旧模型成绩选择版本。当前CMGM HEAD记录在manifest，不能用版本名称数字大小代替来源审核。

SLBR消融取实际仍有实现的formal_d0b_ablation.py、Phase4结构简化、
Phase5/6融合对照及NodeHistoryReadout。它们在共享债券编码的529757参数Full上重新适配。
旧MoE专属主创新消融与本项目当前通道门控SLBR的Full定义不同，没有直接套用。
旧main_ablation长期开发注册表还包含多种历史研究模型，不把它们冒充当前SLBR的单因素消融。
旧DeepV3 runner中OURS显示名与BASE字符串也存在历史变更痕迹；本轮不迁移其
冻结旧checkpoint/旧指标重现逻辑。当前Full需要独立重新训练，不能复用四编码器seed42结果。

## 模型清单

| TriWeave ID | 来源分组 | 源模型/控制 |
|---|---|---|
{table}

## 输入适配和公平性范围

所有入口接受X[B,20,311,26]、input_valid_mask、feature_valid_mask、
feature_applicable_mask；标签为[B,4,23]，顺序从冻结manifest读取。
仅使用TRIWEAVE_23_AFTER_CLOSE_RESEARCH_V1的TRAIN/VAL mmap加载器。
空间/时间双分支内部各共享一个债券编码器，18利率与7财富指数不再各建参数组；
数据层仍保留其经济类型、原始单位、结构适用特征及55组既有TRAIN-only Scaler。

DeepV3保持原无参数neutral adapter：股票均值/总体标准差、债券均值/总体标准差、
23商品，共27个token；逐特征只对有效观测统计。序列模型宽度702，图模型27节点×26特征；
MSGNet/CrossGNN保留各自共享Linear26→1标量投影。该汇聚有信息压缩，不能称为311节点无损输入。
TCN及传统补充组保持源版本全输入设计：序列宽度8086，传统flatten维数161720。
这两类输入策略必须在后续比较中披露，不能笼统宣称各组完全同信息量。

全部数据缺失只作为计算占位并由mask排除，不制造历史价格，不bfill，不删除整个日期。
市场消融同时撤销对应输入有效性，避免仅置零后由投影bias产生伪观测。
图传播剔除无效源/目标，空softmax返回0；时间padding经RNN有效长度、
Transformer query/key masks及卷积逐层重mask处理。标准差使用预先固定的平滑公式
sqrt(var+1e-8)-1e-4，保持零方差的有限梯度；没有用VAL表现调阈值。

## 核心架构与因果边界

RNN/GRU/LSTM：原2层hidden128/dropout.1；Vanilla Transformer：2层4头128维、
FFN256、sinusoidal绝对位置、ReLU及因果mask。GraphWaveNet保留8层门控膨胀卷积、
双阶diffusion、残差/skip、自适应图；MTGNN保留原项目双向MixHop+门控卷积架构，
明确为MTGNN-style。MSGNet保留每样本FFT top3、独立多尺度图与周期Attention；
CrossGNN保留每样本FFT四周期、跨尺度时间图和有符号变量图。后两者仍是当前观测窗口
的整体编码器，不是prefix-causal编码器。全部预测只能使用origin已经提供的20日窗口；
此工程因果约束不证明真实历史PIT或供应商换月方法。

迁移对CrossGNN合成多尺度padding设无效mask；MSGNet/CrossGNN的FFT不包含任何未来窗口。
源代码已采用每样本FFT，本轮保留该设计，避免预测随batch其他样本变化。
GraphWaveNet diffusion、MSGNet图/attention及Vanilla Transformer在全有效条件下
与源算子做同权重数值对照；所有模型做batch/single检查和必要的prefix状态扰动检查。

SLBR Full、正式bypass控制、Phase4真正删参数的简化控制分别保留其原含义。
例如no_adaptive_gate保留未使用gate参数，而fixed_mean_fusion真正删除gate；
no_graph关闭整个传播，而no_graph_prior仅删除结构先验，仍保留内容边注意力。
no_microstate在独立归一化后将Z的readout置零；无Balanced Readout保留两次投影但绕过各自LN。
NodeHistory按显式商品节点映射调用共享head各商品对应输出行，并屏蔽无效历史keys。
GCF/ConcatLinear64/128维拼接/Residual/BranchPreserving均保留各自原融合公式。
128维拼接和部分融合对照有容量差异，不声称参数匹配。

## 参数量

formal控制保留原全部参数的设计，实际未参与该前向的参数会显示在preflight的
parameters_with_gradient_in_real_fixture中；Phase4简化的删除量是真实state_dict删除。

| 模型 | 注册参数量 |
|---|---:|
{paramtable}

## 未来训练入口

本轮只完成迁移和检查，没有执行下述训练命令。每次显式选择一个模型，输出新run目录，
不存在自动启动全部模型的入口。训练前校验本轮preflight源码SHA和数据门禁。

```bash
cd /home/yangxiaotong/projects/myresearch/TriWeave
.venv/bin/python -m triweave.benchmark --list
.venv/bin/python -m triweave.benchmark --verify
# 用户授权正式运行后，选择一个模型；GPU环境本轮未实际验证：
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m triweave.benchmark --train \
  --model baseline/GRU --device cuda:0 --run-id gru_seed42_v1
# 消融示例（独立运行，替换上一条中的模型与run-id）：
# --model ablation/no_graph_prior --run-id no_graph_prior_seed42_v1
```

未来神经训练沿用当前TriWeave比较协议：seed42、Adam lr1e-4/decay1e-5、batch32、
最多200轮/patience10、梯度clip1、masked四周期Huber均值及适用模型的native KL×.25，
四周期等权commodity-macro VAL MAE选checkpoint，保留最后不完整batch；无超参搜索。
这些是对旧CMGM选择/loader协议的明确适配；所有迁入模型保持同一新选择规则。
传统模型按相同TRAIN有效标签集合分组拟合，避免全资产dropna；XGBoost只开一层并行。
未获得TRAIN观测的输出必须显式携带预测mask，不能把未拟合输出当作有效预测。

神经训练入口包括best checkpoint选择/独立恢复检查、优化器/调度器/来源元数据、
逐轮历史和非有限错误快照；控制流程用禁止optimizer更新的synthetic fixture验证。
传统全输入design按batch写mmap，真实训练耗时、资源和GPU行为仍需未来实际运行确认。
本轮没生成任何生产训练checkpoint、性能排名或MAE结果；历史模型结果不能代替迁移模型结果。
''',encoding='utf-8')
    sources=suite.hashes(root)
    phase.write_json(out/'provenance.json',dict(model_source_hashes=sources,
        source_manifest_sha256=phase.sha(out/'source_manifest.json'),
        dataset_manifest_sha256=phase.sha(root/suite.DATA/'dataset_manifest.json'),
        protected_file_count=len(before),changed_old_files=[],real_training_runs=0,TEST_numeric_rows_parsed=0,
        source_selection='Current DeepV3 + available latest SLBR control families, supplemental TCN repair/classical',
        GPU_execution_verified=False,report_code_sha256=phase.sha(Path(__file__))))
    files={str(p.relative_to(root)):phase.sha(p) for p in out.rglob('*') if p.is_file() and p.name!='delivery_manifest.json'}
    files.update(sources)
    for p in (root/'src/triweave').glob('benchmark*.py'):files[str(p.relative_to(root))]=phase.sha(p)
    files['tests/test_benchmark_runner.py']=phase.sha(root/'tests/test_benchmark_runner.py')
    files['configs/triweave_benchmark_v1.json']=phase.sha(root/'configs/triweave_benchmark_v1.json')
    phase.write_json(out/'delivery_manifest.json',dict(files=files,production_training_runs=0,TEST_numeric_rows_parsed=0))
    print(json.dumps(verify(root)))


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--finish',action='store_true');g.add_argument('--verify',action='store_true');args=p.parse_args()
    if args.finish:finish(Path.cwd())
    else:print(json.dumps(verify(Path.cwd())))


if __name__=='__main__':main()
