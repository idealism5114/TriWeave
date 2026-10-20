# 并行 baseline / 消融实验及最终 TEST 评估

入口：`python -m triweave.experiment_suite`。这是独立的新工作流；原模型迁移、四编码器 GPU 实验、Research23 的 TRAIN/VAL 数组、Scaler、raw 和历史报告保持原样。

运行范围为 **311 输入、23 商品目标、26 特征、20 日窗口、seed42**，共享债券编码器。默认包含12个 baseline 和25个消融/对照配置，其中包括 Full。不是原25商品协议的正式确认性实验。

## 启动全部37项

在服务器终端运行：

```bash
cd /home/yangxiaotong/projects/myresearch/TriWeave
mkdir -p logs
nohup .venv/bin/python -B -u -m triweave.experiment_suite \
  --run --with-test --run-id bench_all37_seed42_v1 \
  --gpus 0,1 --min-free-gib 6 \
  >> logs/bench_all37_seed42_v1.launcher.log 2>&1 &
```

两张卡各最多一个神经网络任务。GPU0优先baseline，GPU1优先消融；某队列完成后会接另一队列。空闲显存低于6GiB时等待，每30秒报告状态，不占用其他用户已有进程。该检查是资源准入估计，无法保证其他进程不在检查后申请显存；OOM会明确失败并阻断TEST。

Ridge、RandomForest、XGBoost保留迁移时固定参数，在CPU依次运行，与GPU任务并行。三者共享一份约1.64GiB的TRAIN设计矩阵，避免复制三份缓存。此数据有80组不同的TRAIN标签mask，传统模型可能耗时较长，日志会记录每组开始时间。不会因某个目标缺失删除整个日期。

本机启动前检查发现两张V100 32GB都在使用中、各约4GB空闲，所以命令启动后可能先等待GPU资源。磁盘约11GiB空闲；启动器会检查空间，绝不自动清理历史文件。

如只用一张卡，改为`--gpus 0`，神经网络会顺序排队。不要为同一run-id启动两个launcher。

## 流程与结果

1. 校验封存模型源码和TRAIN/VAL数据。所有神经网络使用原冻结训练循环：Adam1e-4、batch32、最多200epoch、patience10，只按VAL四周期商品macro-MAE等权均值选择best。
2. 各模型训练完成后校验checkpoint SHA与VAL回读结果。全部37项完成才写`models_frozen.json`，记录所有模型、配置、来源、best epoch和SHA；任一失败都不开放TEST。
3. 根据用户本次明确授权，从既有归档读取2025-01-01至2026-09-30数据，不联网、不重采。使用冻结的TRAIN-only Scaler、相同特征/隔离规则，重建全历史并验证原TRAIN/VAL输入和标签完全一致。仅保存最后19个VAL上下文session和TEST。
4. 每个固定模型分别做一次成功的TEST评估；不更新参数，不按TEST挑checkpoint或修改阈值。不完整评估恢复会保留旧attempt，并对同一冻结模型重新评估；已成功的结果只校验与复用。
5. 汇总每模型1/5/10/20日macro-MAE、micro-MAE、样本/标签数量和逐商品指标。收益率单位为小数，例如MAE0.01表示1个百分点。汇总沿预先登记顺序展示，不以TEST排名选择模型。

训练器旧配置里的`TEST=PROHIBITED`仍表示拟合及选模阶段禁止TEST。新的外层`suite_config.json`单独保存本次最终评估授权，不改旧协议与封存记录。TEST已开放后，这段数据不能继续声称为未查看的确认集；历史PIT、连续换月方法和正式25商品研究的限制继续存在。

```bash
tail -f logs/bench_all37_seed42_v1.launcher.log
.venv/bin/python -B -m triweave.experiment_suite --status --run-id bench_all37_seed42_v1
.venv/bin/python -B -m triweave.experiment_suite --results --run-id bench_all37_seed42_v1
```

最终结果目录：`experiments/phase_g_benchmark_test/bench_all37_seed42_v1/`

- `test_summary.csv`：每模型四周期TEST macro-MAE、总体指标与有效标签数。
- `test_per_target_horizon.csv`：各模型、各目标、各周期的详细TEST指标。
- `runs/<model>__attempt001/`：checkpoint、训练历史、VAL指标、TEST完成记录。
- `runs/.../test_attempt_001/`：TEST预测、标签、mask、逐周期与逐商品指标。
- `test_dataset/`：独立TEST数组、数据来源、冻结Scaler副本、历史前缀一致性审计。
- `models_frozen.json`、`suite_config.json`、`tasks.json`、`test_tasks.json`、`logs/`：固定方案、授权、状态和失败证据。

`--results`只在全部结果完成后返回成功；缺少结果会逐项列出，绝不把部分结果称为完成。

## 中断恢复

```bash
nohup .venv/bin/python -B -u -m triweave.experiment_suite \
  --run --with-test --resume --run-id bench_all37_seed42_v1 \
  >> logs/bench_all37_seed42_v1.launcher.log 2>&1 &
```

已成功训练与评估的产物按SHA复用。如果存在中断或失败的训练attempt，先检查对应日志、解决资源问题，再显式加入`--retry-failed`。重试会创建新attempt并从头训练，**不宣称优化器级断点续训**，不覆盖旧attempt。全部模型冻结并开放TEST后不能在同一套实验里重新训练模型。未完成的TEST构建会另存为`test_dataset_incomplete_<timestamp>`后重建；不会删除失败证据。

## 本轮实现验证范围

实现测试使用合成报价验证TEST流程，不读取真实TEST行情、不计算真实TEST MAE。另以真实归档、数值日期门禁截至2024-12-31重建历史输入和标签，与封存TRAIN/VAL逐项比较。详细证据保存在`experiments/phase_g_benchmark_test/implementation_audit/`。提供启动命令不等于已经完成37项正式训练；实际结果需等待上述命令结束。
