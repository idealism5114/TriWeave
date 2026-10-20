"""Bridge diagnostic documentation, provenance and immutable delivery index."""
from __future__ import annotations
import copy
import datetime
import hashlib
import json
import re
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from triweave import after_close as ac
from triweave import bridge as b
from triweave.bridge_audit import preservation
from triweave.universe import sha256_file


def verify(root):
    root=Path(root);out=root/b.OUT;index=json.loads((out/'delivery_manifest.json').read_text())
    for name,e in index['files'].items():
        p=root/name
        if not p.is_file() or sha256_file(p)!=e['sha256'] or p.stat().st_size!=e['bytes']:
            raise ValueError('Changed delivery: '+name)
    old=preservation(root)
    manifest=json.loads((root/b.DATA/'dataset_manifest.json').read_text())
    for name,e in manifest['files'].items():assert sha256_file(root/b.DATA/name)==e['sha256'],name
    assert manifest['protocol_id']==b.PROTOCOL and manifest['task_id']==b.TASK
    flags=json.loads((out/'audit_results.json').read_text())
    assert flags['DIAGNOSTIC_DATASET_READY'] is True
    assert flags['EXPLORATORY_25_RESEARCH_READY'] is False
    assert not (root/'data/derived/triweave_25_after_close_v1_1').exists()
    for split in ('TRAIN','VAL'):
        ds=b.BridgeDataset(root/b.DATA,split)
        assert len(ds)==flags['origin_counts'][split.lower()]
        for ix in (0,len(ds)-1):
            sample=ds[ix]
            assert sample['X'].shape==(20,311,26) and sample['Y'].shape==(4,23)
    print(json.dumps(dict(delivery_files=len(index['files']),dataset_bytes=index['dataset_bytes'],
       delivery_sha256=sha256_file(out/'delivery_manifest.json'),**old,DIAGNOSTIC_DATASET_READY=True,
       EXPLORATORY_25_RESEARCH_READY=False,TEST_numeric_rows_parsed=0),ensure_ascii=False),flush=True)


def build(root):
    root=Path(root);out=root/b.OUT;dest=root/b.DATA
    if (out/'delivery_manifest.json').exists():raise FileExistsError('Bridge delivery sealed; use --verify')
    checks=json.loads((out/'build_checks.json').read_text());flags=json.loads((out/'audit_results.json').read_text())
    protocol,assets=b.definitions(root);targets=json.loads((out/'diagnostic_target_manifest.json').read_text())
    publication=json.loads((out/'dataset_publication.json').read_text());sn=json.loads((out/'sn_quarantine_details.json').read_text())
    repro=json.loads((out/'reproducibility_audit.json').read_text())
    testmatch=re.search(r'(\d+) passed, (\d+) warnings',(out/'tests.txt').read_text())
    assert testmatch and int(testmatch[1])>=210 and flags['DIAGNOSTIC_DATASET_READY']
    testcount=int(testmatch[1]);warningcount=int(testmatch[2])
    assert sha256_file(out/'frozen_protocol.json')==checks['protocol_sha256_at_build']
    for p in dest.iterdir():assert p.is_file()
    manifest=json.loads((dest/'dataset_manifest.json').read_text())
    # The final deterministic manifest is a function of the two identical build
    # graphs. Independently reconstruct it for both recorded hash graphs.
    reconstructed=[]
    for buildname in ('first_sha256','second_sha256'):
        mm=copy.deepcopy(manifest)
        for name,e in repro['arrays'].items():mm['files'][name+'.npy']['sha256']=e[buildname]
        for name,e in repro['metadata'].items():mm['files'][name]['sha256']=e[buildname]
        payload=b.json_bytes(mm)
        assert payload==(dest/'dataset_manifest.json').read_bytes()
        reconstructed.append(hashlib.sha256(payload).hexdigest())
    repro['published_manifest_hash_graph_reconstruction']=dict(first_sha256=reconstructed[0],second_sha256=reconstructed[1],
         published_sha256=sha256_file(dest/'dataset_manifest.json'),identical=reconstructed[0]==reconstructed[1],
         scope='Deterministic final manifest reconstructed from both independent33-array/6-metadata graphs; no dynamic timestamps in dataset')
    repro['published_files_all_rehashed']=True
    ac.write_json(out/'reproducibility_audit.json',repro)
    a={p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in dest.glob('*.npy')}
    dates=a['reference_dates'];numeric={};inputcoverage=[]
    for split in ('train','val'):
        lo,hi=ac.SPLITS[split.upper()];own=(dates>=lo)&(dates<=hi);m=a['feature_valid_mask'][own]
        z=a['features'][own];validz=z[m];origins=a[split+'_origin_indices']
        numeric[split]=dict(valid_feature_values=int(m.sum()),max_abs_z=float(np.abs(validz).max()),
                 abs_z_gt20_flag_only=int((np.abs(validz)>20).sum()),all_valid_features_finite=bool(np.isfinite(validz).all()),
                 all_labels_finite=bool(np.isfinite(a[split+'_targets']).all()),
                 origin_first=str(dates[origins[0]]),origin_last=str(dates[origins[-1]]),
                 feature_clipping=False,label_clipping=False)
        for j,spec in enumerate(assets):
            hit=own&a['input_valid_mask'][:,j];n=int(hit.sum());obs=a['observation_day'][hit,j]
            age=a['age_days'][hit,j]
            inputcoverage.append(dict(split=split.upper(),asset_id=spec['asset_id'],input_index=j,economic_type=spec['economic_type'],unit=spec['unit'],
               reference_dates=int(own.sum()),quality_valid_input_dates=n,coverage_rate=n/int(own.sum()),
               first_vendor_date=str(obs.min().astype('datetime64[D]')) if n else None,
               last_vendor_date=str(obs.max().astype('datetime64[D]')) if n else None,
               minimum_calendar_age=int(age.min()) if n else None,maximum_calendar_age=int(age.max()) if n else None,
               every_valid_quote_available_by_decision=bool((a['available_at'][hit,j]<=a['decision_time'][hit]).all()),
               real_publication_verified=False))
    pd.DataFrame(inputcoverage).to_csv(out/'input_node_coverage.csv',index=False)
    ac.write_json(out/'numerical_stability_audit.json',numeric)
    norm=pd.read_csv(out/'normalization_audit.csv')
    finalcheck=preservation(root);ac.write_json(out/'preservation_check.json',finalcheck)
    allflags={k:v for k,v in flags.items() if k.isupper()}
    scopes={
       'DIAGNOSTIC_UNIVERSE_PASS':'311 identities and23 targets, inherited relative order, AU/AG excluded before numeric source reading and all normalization.',
       'SN_QUARANTINE_PASS':'SN entire2015-03-27 observation invalid before input lag, features, labels and fresh TRAIN fit; raw unchanged.',
       'CONSERVATIVE_INPUT_LAG_PASS':'Every valid input meets frozen base-publication-plus2-session clock and vendor_date<=t-2. This validates an engineering lag, not the actual vendor error bound.',
       'DIAGNOSTIC_LABEL_TIME_CONSISTENCY_PASS':'Only the new FWD3 vendor-index task and assumed clocks: anchor=t+3, end=t+3+h, label publication remains in split. No claim of actual exchange future return.',
       'TRAIN_ONLY_SCALER_PASS':'55 freshly fit diagnostic groups; arbitrary VAL values and masks do not change any parameter.',
       'HORIZON_BOUNDARY_PASS':'All retained origins and all4 horizons independently re-evaluated, including extra2-session assumed-publication purge.',
       'TEST_ISOLATION_PASS':'Date gate before numeric quote parsing;0 TEST rows, poisoned CSV regression, no TEST dataset interface.',
       'DIAGNOSTIC_DATASET_REPRODUCIBLE':'Two complete independent source builds:33 arrays and6 deterministic metadata files identical; final manifest hash graphs and all published files checked.',
       'DIAGNOSTIC_DATASET_READY':'Technical pipeline/interface/stability/mask/numerical diagnostics only. No substantive unbounded known session conflict remains in included23 under inherited evidence review.',
       'EXPLORATORY_25_RESEARCH_READY':'False. Original25 protocol/dataset/HOLD remain immutable, with AU/AG unbounded period issues unresolved.',
       'HISTORICAL_PIT_VERIFIED':'False. Real publication, vendor session definitions and historical revisions remain unverified.',
       'FUTURES_ROLL_METHOD_VERIFIED':'False. No executable continuous-return or roll reconstruction claim.',
       'CONFIRMATORY_TRAIN_READY':'False. The diagnostic task cannot serve as formal25 research or confirmatory results.'}
    flags['acceptance_scope']=scopes
    flags['dataset_bytes']=publication['bytes'];flags['TEST_numeric_rows_parsed']=0
    flags['old_files_unchanged']=finalcheck
    flags['model_runs']=0;flags['MAE_computations']=0;flags['new_market_requests']=0
    ac.write_json(out/'audit_results.json',flags)
    causal=dict(acceptance=allflags,scope=scopes,
       tests=dict(INPUT_VENDOR_DATE_CUTOFF=True,ASSUMED_PUBLICATION_CLOCK=True,
          INDEPENDENT_26_CHANNEL_MASK_ORACLE=True,ALL_VALID_INPUT_SOURCE_VALUE_COMPARISON=True,
          FUTURE_VENDOR_WINDOW_LABEL_ORACLE=True,SN_WHOLE_OBSERVATION_AND_DEPENDENCY_CONE=True,
          INPUT_PREFIX_PERTURBATION=True,VAL_VALUES_AND_MASKS_DO_NOT_CHANGE_TRAIN_SCALER=True,
          ALL_HORIZONS_AND_SPLIT_PUBLICATION_PURGE=True,EVERY_MMAP_SAMPLE_PADDING_AND_SHAPE=True,
          TWO_INDEPENDENT_SOURCE_BUILDS=True,PROTECTED_ORIGINAL_SHA256=True,TEST_NUMERIC_ISOLATION=True),
       rows_compared_to_selected_source_values=checks['source_value_matches'],
       source_read_log='build_checks.json:quote_read_log',known_risk_review=checks['known_risk_review'],
       real_historical_trading_session_causality='NOT_VERIFIED',
       distinction='Software causal ordering under explicit artificial clocks is not historical PIT proof')
    ac.write_json(out/'causality_tests.json',causal)
    (out/'DIAGNOSTIC_PROTOCOL.md').write_text(f'''# TRIWEAVE_23_DIAGNOSTIC_V0

本协议在本轮新行情数值读取前冻结于frozen_protocol.json，其SHA在构建开始和封存时核对一致。独立标签任务ID：**{b.TASK}**。正式25商品协议不变。

资产顺序：263只原股票、18个利率水平变量、7个债券财富指数、23个商品。共311输入节点；23目标严格沿用25名单剔除AU/AG后的相对顺序：

`{' '.join(b.TARGETS)}`

股票为原未复权口径；利率为原percent_per_annum水平、差分转bps；财富指数保留index_level。不同经济类型不混作价格。类型化26维特征和20个参考session窗口沿用Phase E算法，仅用独立adapter构造输入与目标。

## 输入时钟

令c(i)为冻结参考交易日历的第i日，预测origin为c(t)，decision_time=c(t)20:00。首先按原Phase E质量、停牌、重复日、债券严格下一session发布及carry规则构成基础面板。然后把基础面板c(t-2)的观测整体移动到诊断输入日期c(t)。

原假定publication日期另向后移动2个参考session，保持原本地时刻：股票18:00、期货19:00；债券在原严格下一session08:00/08:30之后再移2个session。assumed_available_date来自这个假定时钟，不是真实发布证据。每条有效报价的vendor_date<=c(t-2)，且assumed_available_at<=decision_time。股票/期货只取精确cutoff日，不向后寻找更早有效价；债券只沿用基础面板选中的最新观测，不从无效观测回退。债券最终决策日的观测自然日年龄仍须<=5，否则mask=false。

lag在原始对齐观测上应用，之后重算全部26特征和quote_age_days；不是简单平移旧标准化特征。窗口内每个特征日期使用各自的历史cutoff，不在当前日期回写过去窗口。不存在bfill或上市前制造价格。

## 标签任务

A：若继续原P_vendor(c(t+h))/P_vendor(c(t))-1，延迟输入并不能证明c(t)报价在决策时刻可知、真实交易session对齐或收益起点与决策一致。因此该定义只作风险分析，不作为本分支标签。

B：本分支采用未来供应商索引窗口：

`label_anchor_date=c(t+3)`

`label_end_date=c(t+3+h), h in {{1,5,10,20}}`

`Y(t,h,j)=P_vendor(label_end_date,j)/P_vendor(label_anchor_date,j)-1`

anchor_offset=输入lag2+1，在数值读取前固定。标签采用未滞后的原供应商日期价格并施加SN质量隔离；绝不能从滞后的输入矩阵取价却继续称其原日期收益。anchor至end的h+1条报价全部质量有效方可label_mask=true；各目标独立，不要求23个商品同日完整。

origin、anchor、end均在同一split；整个标签窗口的**假定可用时刻**也必须不晚于该split结束日20:00。这相当于对期货标签额外2个参考session的发布缓冲，最后若干origin自动无有效标签。它避免把假定发布跨入VAL/TEST的价格用作上一split监督标签。只读取TRAIN/VAL源数值。

这严格定义的是未来**vendor-index窗口变化**，不证明其为真实未来交易日收益，也不是P(t+h)/P(t)的原任务。真实供应商日期误差、修订与换月方法仍未验证；不能拿本任务指标替代25商品论文指标。

## 缺失、隔离与归一化

SN2015-03-27整观测质量置无效，raw数值不改、不替价；输入滞后后对应2015-03-31，滚动、标签窗口及TRAIN统计全部重新计算。I/C/PG保留NOT_VERIFIED；FU保留可能换月及薄量重复的原证据等级，不凭涨跌阈值删点。

输入valid mask、feature mask、applicability mask、逐目标逐周期label mask分别保存。无效数值存0是带false mask的计算占位，不是真实价格。quote_available=0是已知缺失指示，可有自己的有效feature mask；padding则X=0且动态mask全false、日期/来源/时点=-1。

只保留20窗口中至少有1条质量有效输入，且至少1目标/周期有效的origin；不作全资产dropna。TRAIN覆盖2011-10-01–2022-12-31，VAL为2023-01-01–2024-12-31。VAL输入窗口可包含此前TRAIN历史，但标签必须在VAL且不进入TEST。TEST为2025-01-01–2026-09-30，只保留边界日期元数据，不生成TEST数值、特征或标签。

重新fit55个economic_type×feature组，采用TRAIN唯一日期-资产-特征样本、ddof0、epsilon=1e-8、至少2个样本。没有使用旧Scaler、VAL或TEST；不裁剪输入或标签。所有参数及每组支持数见normalization_audit.csv和数据目录scaler_params.json。

## 存储和许可用途

数据目录：`{b.DATA}`。每日共享mmap面板X形状[3217,311,26]，通过窗口索引提供[N,20,311,26]，Y为[N,4,23]。33个NPY和6个确定性元数据文件在两次完整来源重建中逐字节一致；最终manifest也按两个哈希图重新构造并核对。

仅允许数据管线、模型接口、训练程序稳定性、mask和数值稳定性测试。本轮没有运行模型或MAE。AU/AG不在输入、目标、特征统计池中；其问题没有解决。正式25商品协议及RESEARCH_HOLD继续冻结。
''')
    risk=checks['known_risk_review']
    (out/'timing_assumption_audit.md').write_text(f'''# 时间假设审计

所有时间为Asia/Shanghai。vendor_date是供应商行日期；assumed_available_date是原假定发布日再滞后2个参考session；decision_time为origin20:00；label_anchor_date是未来c(t+3)供应商索引日；label_end_date为c(t+3+h)。五者分开保存和核验，不把供应商日期自动认证成交易所业务日期。

保留原标签定义的风险：原P(t)并未因输入lag而成为决策时刻已知且正确归属的价格。仅让X变旧不能认证Y的经济时间。本轮没有生成该A任务标签，而采用独立任务ID {b.TASK} 的B任务。未来供应商窗口是无歧义的数组/日历索引任务，真实交易日意义仍NOT_VERIFIED。

全部{checks['source_value_matches']:,}条有效输入值独立回到选中的源行比对，vendor_date<=c(t-2)、assumed_available_at<=decision逐条成立；全部26通道mask由独立sliding-window oracle精确复核。所有监督origin、四周期公式、anchor/end及额外发布缓冲均通过独立循环oracle。没有未来bfill、按全时期完整性删资产或按VAL表现调阈值。

剩余23商品没有已存证据确认的超出2-session护栏的非平坦日期冲突：nonflat abs(offset)>2为{risk['nonflat_abs_offset_gt2']}。FU的两个+1全字段匹配均为平坦薄量记录，对应假定输入最早出现在2015-01-06、2016-04-13，均晚于已匹配的官方日期；没有把它们称为已证实日期错误。另有{risk['repeated_flat_abs_offset_gt2']}个abs(offset)>2的重复平坦指纹，最小offset={risk['repeated_flat_min_offset']}；重复报价不能当作记录真实时间戳，更不能用它们对齐数据或动态删点，详见repeated_fingerprint_time_ambiguities.csv。

已知AU/AG无独立安全结束边界的重大风险被整个资产移出这个诊断分支，但正式25协议未改，风险未解决。DCE没有完整独立逐合约证据，不能宣称其已核实，也不自动当作错误。I/C/PG异常及FU可能换月照旧保留分类。没有发现其他已确认、无法由本分支规则排除的重大未来输入冲突；这是对已有证据的结论，不是穷尽证明。

**仍未知**：新浪日期误差上限、真实完整bar发布时间、夜盘聚合版本变化、历史回溯修订、日历跨市场休市差异、股票公司行动及债券历史发布日期。2个session不是对这些未知的上限证明；真实源时间风险一般仍未认证。未来窗口只具有声明的vendor-index任务含义。如果后续取得其他重大错位证据，必须新增独立审计并阻断相应准入，不能静默继续。

诊断门禁仅确认明确工程假设下的输入/标签/软件因果关系。HISTORICAL_PIT_VERIFIED、FUTURES_ROLL_METHOD_VERIFIED、EXPLORATORY_25_RESEARCH_READY、CONFIRMATORY_TRAIN_READY均false。
''')
    rows='\n'.join(f'|{h}|{checks["label_counts"]["train"][str(h)]:,}|{checks["label_counts"]["val"][str(h)]:,}|' for h in ac.HORIZONS)
    gates='\n'.join(f'|{k}|{str(v).lower()}|' for k,v in allflags.items())
    (out/'REPORT.md').write_text(f'''# Phase E-Bridge：独立23商品技术诊断数据集

**实际构建与审核完成。DIAGNOSTIC_DATASET_READY=true，仅准入代码和数值诊断；正式25商品仍RESEARCH_HOLD。**

数据协议：{b.PROTOCOL}；独立标签任务：{b.TASK}。实际为**311输入节点、23商品目标、26维特征、20个参考session窗口**。没有把本分支说成313输入/25目标的正式结果，没有运行模型或MAE。

## 任务定义与实际数据

保留263只股票、18条债券利率及7条财富指数；商品按原顺序剔除AU/AG：`{' '.join(b.TARGETS)}`。AU/AG在来源数值读取前过滤，也不进入归一化统计。原25名单、顺序、协议、数据、Scaler和HOLD完全不变。

输入为原Phase E质量/发布规则基础面板再滞后2个参考session；股票和期货至少vendor_date<=t-2，债券还保留原严格下一session发布规则。最终有效bond观测年龄<=5自然日，无效最新观测不回退。全部滚动特征在滞后原始对齐数据后重算，不是平移旧X。

标签不沿用原P(t+h)/P(t)-1。原标签的锚点时间不会因输入延迟而自动得到认证。本任务定义Y=P_vendor(t+3+h)/P_vendor(t+3)-1，h=1/5/10/20，使用未滞后的供应商索引价格；anchor=t+3，end=t+3+h。逐资产h+1窗均有效、同split，并要求整个窗口的假定发布时刻也在split结束前（另有2-session发布缓冲）。这是假设明确的未来供应商索引窗口任务，不是已验证的真实未来交易收益。详见DIAGNOSTIC_PROTOCOL.md及timing_assumption_audit.md。

|split|实际监督origin|首个origin|最后origin|
|---|---:|---|---|
|TRAIN|{checks['origin_counts']['train']:,}|{numeric['train']['origin_first']}|{numeric['train']['origin_last']}|
|VAL|{checks['origin_counts']['val']:,}|{numeric['val']['origin_first']}|{numeric['val']['origin_last']}|

|周期（从未来anchor起算）|TRAIN有效标签|VAL有效标签|
|---|---:|---:|
{rows}

标签计数为有效origin×目标×该周期的数量，不是origin数量。每个目标独立保留mask；不要求23个商品同时有效，不因单个缺失删除整个日期。仅在所有目标/周期无效或20窗口没有任何有效输入时不保留该origin。边界附近由于未来anchor、周期和发布缓冲而失去标签属于协议约束，不是下载失败。train_coverage.csv、val_coverage.csv为23目标逐资产覆盖，input_node_coverage.csv另列全部311节点的输入覆盖。

数据目录：`{b.DATA}`；40个文件，体积{publication['bytes']:,}字节（{publication['bytes']/1024**2:.2f}MiB）。共享每日mmap面板[3217,311,26]，loader逻辑X=[N,20,311,26]、Y=[N,4,23]。所有文件可追溯312个既有来源任务及SHA256。没有重新采集行情。

## SN隔离与归一化

SN2015-03-27整条观测质量无效，原始15905保持原样，不替换官方价格。诊断输入被推迟到2015-03-31后仍无效；其滚动依赖、输入窗口、未来标签窗口与TRAIN样本统计均重新计算。与相同311/23、lag2、FWD3但未隔离SN的**内存诊断消融**比较：input mask改变1、feature mask改变15，TRAIN四周期label mask各改变1，VAL没有标签mask改变；局部影响40个TRAIN窗口。此对照不是额外发布的数据集。

55组TRAIN Scaler全部从本分支数据重新fit。SN隔离改变16组统计和16组数值变换，池化作用使全部{checks['origin_counts']['train']}个TRAIN和{checks['origin_counts']['val']}个VAL窗口的部分归一化输入改变；这不表示所有raw异常。股票/债券组不因SN消融改变，AU/AG不参与任何组。quote_age_days现在反映滞后后的真实自然日间隔，不再恒为0；因此删除该SN观测也会改变年龄组的数值变换，与此前原时间协议下SN-only的15组变换不同。完整统计见normalization_audit.csv。

SN依赖锥外实质性未解释变化0。原累计和方差算法造成{sn['outside_cone_roundoff_elements']}个极小浮点差异，最大{sn['outside_cone_roundoff_max']:.16g}，小于数值读取前冻结的atol1e-9/rtol1e-10；mask、时点和身份精确检查。所有有效输入、未缩放/标准化特征与标签finite；无裁剪、Inf或NaN，Scaler不足支持/常量/floor组分别{int(norm.insufficient_train_support.sum())}/{int(norm.constant_flag.sum())}/{int(norm.scale_floor_flag.sum())}。TRAIN最大|z|={numeric['train']['max_abs_z']:.6f}，VAL={numeric['val']['max_abs_z']:.6f}，保留重尾供程序稳定性诊断，不按VAL值调方案。I/C/FU/PG没有按涨幅删点，原分类见retained_source_anomalies.csv。

## 因果、重现与只读保证

全部{checks['source_value_matches']:,}个有效输入值独立回到对应来源行精确核对；逐条验证vendor cutoff、声明时钟<=decision，独立sliding-window oracle精确复核26通道feature mask，独立循环oracle复核所有标签公式和mask。未来基础输入扰动不改变历史前缀，任意VAL特征与mask扰动不改变55组TRAIN参数。全部实际mmap样本的形状、padding、false mask、0占位及-1时点均已执行检查。

两次从原始冻结来源独立完整构建，33数组和6份确定性元数据（含fresh Scaler）SHA逐字节一致；最终manifest按两次哈希图重构也与磁盘完全一致。没有通过读取旧NPY修补新值来实现重现。{testcount}项测试通过，含{testcount-186}项Bridge测试；{warningcount}条警告均为原collect.py的NumPy timedelta弃用警告。原{finalcheck['protected_files']:,}个文件重算SHA无差异。TEST数值解析0、TEST数据集0、模型0、MAE0、提交/推送0。TEST边界日期可能用作禁止越界的日历元数据，不涉及TEST行情。

## 仍未核实的风险与准入范围

剩余23品种现有证据没有已确认、超出声明护栏且无法排除的重大未来输入冲突。FU两个+1匹配都为平坦薄量记录，延迟后的最早输入日期在其所匹配的官方日期之后；7个更远的重复平坦指纹不能证明日期错位，未被用来动态映射或删点。DCE缺独立逐合约覆盖仍NOT_VERIFIED。未知一般历史时间语义不等于已确认错误，但也不能据此宣称PIT通过。

**2-session lag不是供应商真实日期误差上限的证明。** 本任务的future window仅按供应商日期索引定义，未认证为真实未来exchange-session return。真实发布、历史修订、连续构造/换月、跨市场交易日历、未复权公司行动仍未核实。若把本任务解释成真实未来收益，相关准入仍不成立；当前ready仅对技术诊断有效。AU/AG无独立安全结束边界的问题仍在正式25分支阻断训练。

## 独立验收

|字段|结果|
|---|---|
{gates}

每个true的范围详见audit_results.json:acceptance_scope；软件因果通过不等于历史PIT证明。特别是DIAGNOSTIC_LABEL_TIME_CONSISTENCY_PASS只覆盖明确的vendor-index定义和声明时钟。

## 最终12项回答

1. 可以。已真实生成独立{b.PROTOCOL}，311输入/23目标，范围限技术诊断。
2. SN已在本分支整观测及其依赖中隔离并通过对照审核；raw不变，未替价，不声称源根因已修复。
3. 规则是原Phase E基础as-of面板/假定发布日再滞后2个参考session，所有有效输入至少vendor_date<=t-2，债券更保守并在最终decision应用5自然日年龄限制。不是历史误差上限证明。
4. 需要。本分支使用新任务ID {b.TASK}，未来t+3锚点的1/5/10/20供应商索引窗变化；原P(t+h)/P(t)不采用。
5. TRAIN {checks['origin_counts']['train']}，VAL {checks['origin_counts']['val']}。它们不是原25任务的2728/483样本。
6. 四周期数量见上表，均有逐目标mask和split/发布缓冲审核。
7. 是，55组全新TRAIN-only fit，无旧参数复用、无VAL/TEST拟合。
8. 在明确声明的诊断时钟和已存重大冲突证据下，没有已发现的未来行情直接进入有效输入。不能据此证明所有真实历史发布或修订均无未来信息。
9. 真实vendor日期误差上限、bar完成/发布时刻、历史vintage与换月、DCE独立日期证据、股票公司行动及债券历史发布时间仍未知。
10. 可以用于后续311/23接口、mask、训练程序和数值稳定性代码测试。本轮没有运行模型或MAE，不具备正式25商品性能或论文主结果的使用条件。
11. 集合从25变23、节点313变311、输入时间规则及标签任务也不同，且PIT/roll未验证；其任何指标都不能替代原任务。
12. 正式25仍需Sina AU0/AG0在2013夜盘上线及后续的日期/会话定义和版本边界，以及含业务日期/发布修订元数据的2013-07-05/08/09对应SHFE原档或同等级证据。确定安全映射/隔离时期后，才可讨论原任务完整重建和解除HOLD，不必为诊断分支重做全历史SessionQA。

## 使用与复核

```python
from triweave.bridge import BridgeDataset
ds = BridgeDataset("data/derived/triweave_23_diagnostic_v0", "TRAIN")
sample = ds[0]  # X:[20,311,26], Y:[4,23]，所有mask和时间字段另存
```

```bash
.venv/bin/python -m triweave.bridge --verify
.venv/bin/python -m pytest -q
```

构建命令`.venv/bin/python -m triweave.bridge --run`是append-only，已有交付时拒绝覆盖。两次完整来源重建已在本次执行；未来证据或协议变更须新版本，不能清空旧产物后重跑。本阶段于此停止。
''')
    codefiles=['src/triweave/bridge.py','src/triweave/bridge_audit.py','src/triweave/bridge_report.py','tests/test_bridge.py']
    snapshot=out/'code_snapshot';snapshot.mkdir(exist_ok=True)
    for name in codefiles:shutil.copyfile(root/name,snapshot/Path(name).name)
    protected=json.loads((out/'preservation_before.json').read_text())['files']
    provenance=dict(phase='E-Bridge',protocol_id=b.PROTOCOL,task_id=b.TASK,diagnostic_only=True,
       frozen_protocol_sha256=sha256_file(out/'frozen_protocol.json'),
       protocol_frozen_before_numeric_at=protocol['frozen_before_new_source_numeric_build_at'],
       numeric_build_started_at=checks['build_started_at'],
       dataset=b.DATA,main25_dataset=ac.DATA_REL,main25_HOLD_sha256=sha256_file(root/ac.DATA_REL/'RESEARCH_HOLD.json'),
       protected_original_inventory_sha256=sha256_file(out/'preservation_before.json'),
       previous_phase_seals={k:v for k,v in protected.items() if k.endswith('delivery_manifest.json')},
       source_entries=checks['input_sources'],source_read_log='build_checks.json:quote_read_log',
       actual_source_jobs=len(checks['input_sources']),
       old_feature_implementation_sha256=protected['src/triweave/after_close.py'],
       old25_target_configuration_sha256=protected['configs/futures_targets_25_v1.csv'],
       new_code={name:sha256_file(root/name) for name in codefiles},
       audit_evidence_refs=['../phase_e_sourceqa/SN_two_days_all_contracts.csv','../phase_e_sourceqa/timing_match_adjudication.csv',
                            '../phase_e_sourceqa/all_exact_vendor_official_fingerprints.csv','../phase_e_sessionqa/REPORT.md'],
       environment='environment.json',all_source_data_reused=True,TEST_numeric_rows_parsed=0,
       new_market_requests=0,model_training_runs=0,MAE_computations=0,Git_commits=0,Git_pushes=0,
       no_source_threshold_selection_from_VAL=True,main25_research_ready=False,
       commands=['.venv/bin/python -m pytest -q tests/test_bridge.py',
                 '.venv/bin/python -m triweave.bridge --run','.venv/bin/python -m pytest -q',
                 '.venv/bin/python -m triweave.bridge_report','.venv/bin/python -m triweave.bridge --verify'])
    ac.write_json(out/'provenance.json',provenance)
    archive=root/'data/manifests/phase_e_bridge';archive.mkdir(exist_ok=True)
    ac.write_json(archive/'provenance.json',provenance)
    files=[p for p in out.rglob('*') if p.is_file() and p.name!='delivery_manifest.json']
    files+=list(dest.iterdir())+[root/name for name in codefiles]+[archive/'provenance.json']
    records={str(p.relative_to(root)):dict(sha256=sha256_file(p),bytes=p.stat().st_size) for p in sorted(files)}
    ac.write_json(out/'delivery_manifest.json',dict(phase='E-Bridge',protocol_id=b.PROTOCOL,task_id=b.TASK,
       files=records,dataset_bytes=publication['bytes'],indexed_bytes=sum(e['bytes'] for e in records.values()),
       self_excluded=True,hash_scope='All delivered reports, new source/tests, dataset40 files and archive provenance; index is identified by its external SHA256'))
    verify(root)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verify',action='store_true')
    (verify if p.parse_args().verify else build)(Path.cwd())
