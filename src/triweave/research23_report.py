"""Seal the scientific research23 evidence without changing earlier phases."""
from __future__ import annotations
import copy
import hashlib
import json
import re
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from triweave import after_close as ac
from triweave import research23 as r
from triweave.research23_audit import preservation
from triweave.universe import sha256_file


def verify(root):
    root=Path(root);out=root/r.OUT;index=json.loads((out/'delivery_manifest.json').read_text())
    for name,e in index['files'].items():
        p=root/name
        if not p.is_file() or sha256_file(p)!=e['sha256'] or p.stat().st_size!=e['bytes']:
            raise ValueError('Delivery fingerprint mismatch: '+name)
    old=preservation(root);m=json.loads((root/r.DATA/'dataset_manifest.json').read_text())
    for name,e in m['files'].items():assert sha256_file(root/r.DATA/name)==e['sha256']
    assert m['label_anchor_offset']==0 and m['label_formula']=='P(t+h)/P(t)-1' and not m['FWD3_used']
    for split in ('TRAIN','VAL'):
        ds=r.Research23Dataset(root/r.DATA,split)
        assert len(ds)==m['split_summary'][split]['origin_count']
        for i in (0,len(ds)-1):assert ds[i]['label_anchor_date']==ds[i]['origin_date']
    print(json.dumps(dict(delivery_files=len(index['files']),delivery_sha256=sha256_file(out/'delivery_manifest.json'),
      dataset_bytes=index['dataset_bytes'],**old,DATASET_TECHNICALLY_VALID=m['DATASET_TECHNICALLY_VALID'],
      EXPLORATORY_RESEARCH_READY=m['EXPLORATORY_RESEARCH_READY'],CONFIRMATORY_RESEARCH_READY=False,TEST_numeric_rows_parsed=0),ensure_ascii=False),flush=True)


def build(root):
    root=Path(root);out=root/r.OUT;dest=root/r.DATA
    if (out/'delivery_manifest.json').exists():raise FileExistsError('Sealed research23; use --verify')
    results=json.loads((out/'build_results.json').read_text());flags=json.loads((out/'audit_results.json').read_text())
    protocol,assets=r.definitions(root);coverage=results['coverage'];norm=results['normalization'];impact=results['quarantine']
    timing=json.loads((out/'timing_evidence.json').read_text());publication=json.loads((out/'publication.json').read_text())
    post=json.loads((out/'post_fix_validation.json').read_text());assert all(post['published_arrays_still_identical'].values())
    assert sha256_file(out/'frozen_protocol.json')==results['protocol_sha256']
    tests=re.search(r'(\d+) passed, (\d+) warnings',(out/'tests.txt').read_text());assert tests and int(tests[1])>=237
    testcount=int(tests[1]);warnings=int(tests[2])
    frame=pd.read_csv(out/'per_asset_coverage.csv');markets=pd.read_csv(out/'per_market_coverage.csv')
    labels=pd.read_csv(out/'per_target_label_coverage.csv');extreme=pd.read_csv(out/'extreme_feature_audit.csv')
    source_status=pd.read_csv(root/'experiments/phase_e_sourceqa/early_futures_anomaly_audit.csv')
    source_status=source_status.loc[source_status.asset_id.isin(['FUT_'+c for c in r.TARGETS])]
    source_status.to_csv(out/'inherited_source_anomaly_classification.csv',index=False)
    extreme.groupby(['split','market','feature_name']).size().rename('count').to_csv(out/'extreme_attribution_summary.csv')
    feature_registry=ac.registry();ac.write_json(out/'feature_registry.json',feature_registry)
    # Final deterministic manifest is reproduced using each independent hash graph.
    repro=json.loads((out/'reproducibility_audit.json').read_text());manifest=json.loads((dest/'dataset_manifest.json').read_text())
    hashes=[]
    for key in ('first_sha256','second_sha256'):
        mm=copy.deepcopy(manifest)
        for name,e in repro['arrays'].items():mm['files'][name+'.npy']['sha256']=e[key]
        for name,e in repro['metadata'].items():mm['files'][name]['sha256']=e[key]
        payload=r.json_bytes(mm);assert payload==(dest/'dataset_manifest.json').read_bytes();hashes.append(hashlib.sha256(payload).hexdigest())
    repro['final_manifest_hash_graph_reconstruction']=dict(first_sha256=hashes[0],second_sha256=hashes[1],published_sha256=sha256_file(dest/'dataset_manifest.json'),identical=hashes[0]==hashes[1])
    repro['independent_third_raw_build_after_causal_test_dtype_fix_identical']=all(post['published_arrays_still_identical'].values())
    ac.write_json(out/'reproducibility_audit.json',repro)
    old=preservation(root);ac.write_json(out/'preservation_check.json',old)
    allflags={k:v for k,v in flags.items() if k.isupper()}
    scopes={
      'AFTER_CLOSE_TIMING_AUDITED':'Actual same-day anchor, declared stock18/futures19, bond next-session08/08:30; remaining23 re-matched against existing official archive;2 FU forward-risk dates masked. Not historical PIT certification.',
      'CROSS_MARKET_ALIGNMENT_PASS':'Frozen reference calendar, no common-date intersection, no stock/futures carry, bond5-day age with original observation dates retained.',
      'SN_QUARANTINE_PASS':'SN identity mismatch entire observation and all input/label dependencies invalid; no price replacement.3 total rules include2 precautionaryFU observations.',
      'SOURCE_ANOMALY_AUDIT_PASS':'Raw replay fidelity and evidence-based triage completed. I/C/PG and roll/vintage doubts remain explicitly NOT_VERIFIED rather than falsely certified.',
      'FEATURE_TYPE_SEMANTICS_PASS':'26 typed channels and independent boolean oracle; yields percent/bps, wealth levels distinct from prices.',
      'TRAIN_ONLY_SCALER_PASS':'55 freshly fit unique TRAIN date-asset-feature groups; no old25/Bridge parameters; arbitraryVAL values/masks do not change fit.',
      'NORMALIZATION_NUMERIC_PASS':'All valid numbers finite, no extreme floor amplification,0 clipping. Heavy tails remain a training stability risk, not a guarantee of neural optimizer stability.',
      'PREFIX_CAUSALITY_PASS':'Actual source quotes after2023-06-30 perturbed, realigned, feature-rebuilt and TRAIN refitted; prefix identical.',
      'DATASET_TECHNICALLY_VALID':'Masks, decimal-return labels, clocks under declared assumptions, boundaries, mmap padding and independent reconstructions verified.',
      'EXPLORATORY_RESEARCH_READY':'Conditional exploratory research on vendor continuous mathematical returns with known risk records excluded and all listed limitations disclosed; no real PIT/tradability/confirmatory claim.',
      'HISTORICAL_PIT_VERIFIED':'False: real publication times, vendor vintages and cohort construction are not historically certified.',
      'FUTURES_ROLL_METHOD_VERIFIED':'False: source-index mathematical returns are not verified executable futures holding returns.',
      'CONFIRMATORY_RESEARCH_READY':'False: historical timing, source construction, retrospective universe and priorVAL design lack independent confirmation evidence.'}
    flags.update(acceptance_scope=scopes,dataset_bytes=publication['bytes'],old_files_unchanged=old,
       TEST_numeric_rows_parsed=0,models_run=0,MAE_computed=0,new_market_requests=0)
    ac.write_json(out/'audit_results.json',flags)
    ac.write_json(out/'causality_tests.json',dict(checks=results['checks'],post_fix_checks=post['checks'],
      source_quarantine=impact,known_time_conflicts=results['timing']['unhandled_forward_conflicts'],
      all_sample_padding_and_anchor_pass=True,TEST_numeric_rows_parsed=0,
      TEST_test_method='Isolated raw date-first/Chinese-date-column poisoned CSVs with different TEST suffixes; parser guards and identical permitted artifacts, never actualTEST quotes',
      scope='Source-derived software causality and frozen exploratory availability assumptions, not real historical PIT'))
    counts='\n'.join(f'|{h}|{coverage["TRAIN"]["label_counts"][str(h)]:,}|{coverage["VAL"]["label_counts"][str(h)]:,}|' for h in ac.HORIZONS)
    targets_table=[]
    for c in r.TARGETS:
        x=labels.loc[labels.asset_id.eq('FUT_'+c)];tr=x.loc[x['split'].eq('TRAIN')].iloc[0];va=x.loc[x['split'].eq('VAL')].iloc[0]
        info=frame.loc[frame.asset_id.eq('FUT_'+c)&frame['split'].eq('TRAIN')].iloc[0]
        targets_table.append('|'+c+'|'+str(info.first_valid_date)+'|'+str(int(info.valid_input_dates))+'|'+
           '|'.join(str(int(v[f'{h}d_valid_labels'])) for v in (tr,va) for h in ac.HORIZONS)+'|')
    target_table='\n'.join(targets_table)
    market_table='\n'.join(f'|{x["split"]}|{x["economic_type"]}|{x["nodes"]}|{x["mean_valid_nodes"]:.3f}|{x["missing_ratio"]:.3%}|' for x in markets.to_dict('records'))
    gates='\n'.join(f'|{k}|{str(v).lower() if isinstance(v,bool) else v}|' for k,v in allflags.items())
    docs={}
    docs['TARGET_SELECTION.md']='''# 目标选择及设计历史披露

冻结23顺序：LU SC CF PG HC J JM FU C MA SR OI RM RB M CU AL ZN SN NI A I Y。每个都是输入和目标，没有额外辅助期货。263股票与25债券身份和相对顺序不变，实际311节点。

AU/AG因已记录的重大源时间语义疑点及未知影响边界移出本协议，不代表问题已解决。没有按VAL MAE、TEST或模型表现选择。本次修改发生于原25协议的数据质量审计之后，且原商品池确曾用VAL覆盖设计，必须披露为事后协议变更。23结果不能冒充原25同任务全目标比较。

新配置configs/futures_targets_23_research_v1.csv，固定规则docs/TARGET_23_RESEARCH_PROTOCOL.md。原25与旧Bridge配置、来源、Scaler、数组和报告均只读保留。
'''
    docs['DATA_PROTOCOL.md']=f'''# {r.PROTOCOL}

TRAIN2011-10-01–2022-12-31；VAL2023-01-01–2024-12-31；TEST2025-01-01–2026-09-30。只构建TRAIN/VAL。参考日历SHA见provenance.json，统一3217日，不取全资产共同报价交集。

20:00北京时间AFTER_CLOSE_PREDICTION。股票18:00、期货19:00同vendor日，债券严格下一参考日08:00/08:30、最多5自然日沿用，均AVAILABLE_AT_ASSUMED。股票/期货不carry，不从无效最新债券观测向更旧报价fallback。不使用Bridge统一lag2。

R(t,h,c)=P(t+h,c)/P(t,c)-1，h=1/5/10/20参考session，decimal_return。anchor=t，endpoint=t+h，无FWD3、无未来锚点及Bridge发布缓冲。所有h+1报价质量有效、所有日期在同split；各目标各周期独立mask。对连续供应商收盘价的数学变化，不是已认证持仓收益。

311节点=263股票+18利率+7财富指数+23商品；26经济类型特征，20历史窗口。先施加source_quarantine_manifest.csv整观测规则，再重算特征、标签及55组TRAIN-only Scaler。不替价、不bfill、不按20%或|z|>20删点/剪裁。

无效X/Y存0只是false mask的计算占位，不能作为报价或0收益纳入统计/损失。有效的真实0收益由true mask区分。availability指示通道可在报价缺失时有效且值为0；它不是价格。applicability是静态经济结构，padding动态input/feature mask全false、数据0、来源/时点-1。最早19个TRAIN样本有padding，没有复制未来首价。

新数据路径{r.DATA}；每日共享mmap面板[3217,311,26]；TRAIN逻辑X=[2726,20,311,26]、Y=[2726,4,23]，VAL X=[483,20,311,26]、Y=[483,4,23]。完整来源raw重标准化，两个独立构建均不读取旧NPY/Scaler。所有参数在新数值读取前冻结于frozen_protocol.json。
'''
    docs['DATA_ALIGNMENT_REPORT.md']=f'''# 三市场对齐

统一参考日历经排序/唯一性检查，不压缩缺报日期。股票按证券日，期货保留供应商日但不认证真实session；非参考日原记录保留源档、未映射为当前价格。债券估值日在信息假定可用后才允许进入输入，观测日和年龄保留，周末观测严格落到下一参考日。晚上市/晚起历史保留无效，不用未来价格补齐。

|split|经济类型|节点|平均有效节点/日|缺失比例|
|---|---|---:|---:|---:|
{market_table}

263只股票全部有TRAIN/VAL有效历史，但不是全期完整：TRAIN平均覆盖93.334%，最小53.019%，最大内部缺口sh.600153为234个参考session。VAL sh.600297最后有效日2024-07-17，尾部112日缺失；不能无公告证据归为下载失败或确定原因。55只股票TRAIN开头无可用观测，原始状态/上市前缺失不填补。

18收益率各有2713/2733个TRAIN有效输入日，VAL各481/484。7财富指数中1–3Y只有2410/2733日，首个可用输入2013-01-07，领先缺失304日，其余6条各2713日；不以插值伪造早年。利率是百分数水平及bps变化，财富指数是水平及相对变化，绝不混为价格。

10个商品在TRAIN开头尚无当前源可用报价；first_valid_date只表示本快照可用起点，不能全部解释为已核实上市日期，尤其MA源历史起点2014-12-24不能据此声称该日才上市。具体23逐项表见REPORT及per_asset_coverage.csv。期货TRAIN最长内部缺口FU13日，VAL最长内部缺口1日。没有根据缺失率删除任何资产。
'''
    docs['FEATURE_ENGINEERING_REPORT.md']='''# 26维特征及经济类型

复用src/triweave/after_close.py的registry与feature_panel，独立mask oracle使用sliding_window_view重新验证全部26通道；未为凑维度新增指标。

股票：1/5/10/20历史价格变化、5/10/20日MA偏离和波动、5–20动量、成交量/额变化、换手、可用性与年龄。未复权，不代表总收益；公司行动可能影响价格特征。

18利率：百分数水平，1/5/10/20日差分×100转bps，国债10Y相对1Y/3Y期限利差，AAA相对同期限国债信用利差；利差必须双方同一观测日。不能对利率水平当价格计算收益率。

7财富指数：相对变化、趋势/MA、波动/动量、可用性与年龄。23商品：历史价格变化、波动/MA/动量、量/OI变化、OHLC区间、可用性与年龄。原价格尺度没有跨市场直接混合成单一统计组。

returns_h需要h+1连续参考日报价；MA_w需要w条；vol_w需要w+1条；momentum需要21条；活动变化需要相邻两条、前值>0。所有滚动向后，只用当前及历史有效报价，无压缩时间跳过缺失。SN和FU隔离先于这些运算，不能通过历史收益/MA/vol/momentum重新进入有效输入。

feature_applicable_mask标记结构性经济不适用；feature_valid_mask表示该日期具体依赖有效；input_valid_mask表示报价有效；label_valid_mask独立到商品和周期。不存在全资产dropna(any)。详见feature_registry.json与per_feature_coverage.csv。
'''
    docs['NORMALIZATION_REPORT.md']=f'''# TRAIN-only标准化与全部极端值

全部55组economic_type×feature由TRAIN唯一日期-资产-特征重fit，ddof0、epsilon1e-8、至少2样本。不按20窗口重叠重复计数；无效样本不贡献。原25和Bridge Scaler未复用。任意VAL值及mask扰动不改变TRAIN参数；每组count/mean/std独立复算精确一致。

TRAIN有效特征值{norm['TRAIN']['valid_feature_values']:,}个，|z|>20有355个；VAL有效{norm['VAL']['valid_feature_values']:,}个，|z|>20有16个。全部371条均在extreme_feature_audit.csv，按市场归因见extreme_attribution_summary.csv；不是只展示几个最大值。2个常量/floor组是股票与期货quote_age_days=0，无极端值由floor放大；不足支持组0，全部有效值finite，剪裁/自动删点0。

TRAIN最大sz.002831 2016-12-23 amount_change_1d：原成交额2492755→1343509376元，比例特征537.9656729，TRAIN均值0.1295671、std1.9130254，z281.1442566。原单位RMB没有检测到切换，量级及原始行已核验；未独立证明是公司行动或真实市场事件。

VAL最大sh.600837 2024-10-14 amount_change_1d：56377523.16→8256687790.92元，特征145.4535391，同一TRAIN参数得z75.9655228。没有按此VAL极端值选择方案。

商品Y2015-06-24 OI2→539366，特征269682，z176.416611；MA同日OI2→427984；I2015-10-09成交量6→2463276，反映相对变化对历史分母敏感以及可能的合约统计/拼接风险，不是单位错误已确认或真实行情已认证。I2015-10-08价格变化+50.41%及其后5日波动尾部仍保留NOT_VERIFIED。

债券6条极端均来自2013-06-20估值、2013-06-21假定可用的短端曲线。国债3M从3.3932%到5.0132%，是162bps而非价格收益；其他80.65–90.93bps。同日多个曲线出现跳变提供共同源事件线索，但不证明独立真实市场冲击，未按阈值更正。

371条中27条有原preclose与前参考收盘不同的公司行动参考提示；这不是公告核实。源错误、历史单位口径、拆分分红、换月与真实变化仍保留证据等级，不把数值大直接当错。它们构成神经网络梯度/优化数值风险，NORMALIZATION_NUMERIC_PASS仅表示变换正确且有限，不保证训练稳定。本轮没有模型可验证优化器。

后续如研究log1p活动量、分母保护或稳健缩放，应先提出独立TRAIN证据方案和新协议，与当前版本分开；不能为降低VAL MAE直接改当前特征或裁剪。本轮没有实施这些变更。
'''
    docs['LABEL_CONSTRUCTION_REPORT.md']=f'''# 原收益率锚点已恢复

R(t,h,c)=P(t+h,c)/P(t,c)-1；anchor就是origin t，endpoint为t+h参考session，decimal_return。0.02为2%。原25与新23公式、horizon和收盘后定义相同；旧Bridge FWD3无复用，无anchor+3、无2session输入或标签发布缓冲。

独立逐目标/逐周期oracle复算全部true和false mask、origin、日期和数值，所有有效值finite。h+1报价窗要求完整质量有效；SN以及两FU隔离点不能作为锚点、终点或中间质量依赖。每个商品独立mask，某个商品无效不删其他标签。无效0存储占位不能当0收益或参与统计。

|h|TRAIN有效标签|VAL有效标签|
|---|---:|---:|
{counts}

TRAIN有2726个origin，VAL483个；daily面板仍保留2733/484个参考日。TRAIN7个无监督日为2011-12-01/02、2013-11-19/20、2019-04-19/22、2022-12-30；最后一天没有未来同split终点，其余涉及源缺失。没有为凑origin填报价。相比25任务的2728，目标变更及质量规则改变可评价日期；相比Bridge2719，任务时钟和anchor不同，不能直接拿计数做模型结果比较。

各周期TRAIN末h个origin窗和VAL末h个窗不能跨split；全部窗口逐一检查，最长20步边界不是只抽查一个样本。VAL输入可读取此前TRAIN历史，这是历史上下文，标签不跨界。有效标签min/max、分位数见label_distribution.csv，不是模型误差或TEST指标。
'''
    docs['SOURCE_QUALITY_REPORT.md']=f'''# 来源真实性、隔离与覆盖

312个既有来源任务（263 BaoStock股票、23 Sina期货、19收益率日期分块、7财富指数任务）均重验raw/std/manifest SHA。先在raw的date或中文日期列门禁丢弃TEST行，再使用原source normalizer从raw重建。312个任务与封存standardized的身份/日期/单位/相关数值比较，差异0；这不是把旧NPY删两列。完整来源、参数、时间和SHA见provenance.json；官档另有400份既有月任务可追溯。

SN2015-03-27原close15905与同日官方普通合约联合价/结算/活动证据不兼容，整条隔离，不替成119270。FU2014-12-31和2016-04-11与下一参考日官方薄量平坦bar全7指纹相同；前者同日也能匹配，不能证明date错误。为保护当前t锚点，本协议在新数值读取前明确预防性隔离这两条，未扩大到全部FU或按幅度删点。

3条报价仍保留raw，input mask改变3，feature mask改变295；TRAIN四周期label mask分别改变5/13/23/43，VAL0。16个期货组统计改变、15组变换改变，年龄恒0组仅支持数变化。依赖锥外无实质性未解释变动；{impact['outside_cone_roundoff_elements']}个方差累计和舍入变化最大{impact['max_outside_cone_roundoff']:.16g}，在既有1e-9/1e-10冻结容差内。所有mask及身份时点精确相等检查。见quarantine_impact.json及依赖表。

I/C/PG没有完整独立DCE逐合约对照，继续NOT_VERIFIED；FU可能合约切换及薄量/旧价风险不被“原始文件一致”自动消除。历史连续方法未验证，不将所有大涨跌视为错误或真实市场变化。AU/AG已移出本新研究协议，但其旧任务问题没有修复。

263股票均有TRAIN和VAL历史；25债券及23期货均有真实观测。并不等于每个日子有报价：各资产缺失、最长连续gap、上市前/起始前缺失、内部缺口、尾缺失见per_asset_coverage.csv。first_valid_date不等于已核实上市日；尤其商品代码历史衔接可能造成早年源缺失，未拼接或插值。预处理只屏蔽已声明坏OHLC/停牌/重复和3条明确规则，不替换原价、不改raw。
'''
    docs['TIMING_CAUSALITY_REPORT.md']=f'''# 收盘后时间与真实前缀检查

observation_date是原估值/报价日期；vendor_date是Sina字段；trading_session在未核实处UNKNOWN；assumed_available_at是冻结工程时钟；decision_time为当日20:00。不能由vendor_date<=t推导真实PIT。当前源19:00假设保留为AVAILABLE_AT_ASSUMED，而非供应商已认证发布时间。

本轮针对新raw重放23商品重新与既有普通官方合约日档做±2参考session及不限制日期的7字段联合指纹比较，15品种有独立对比覆盖，8个DCE无该覆盖。所有正向全7匹配只有FU两条，均已经整条隔离；未处理正向冲突0。CU/AL/ZN/RB/HC/SN等同日匹配只证明匹配子集，不认证整段历史。非匹配也不自动判错。SDK官方raw的date来自请求档案日期，并非原HTTP业务日期，历史vintage仍未证实。

每条有效输入在声明时钟下available_at<=decision；每个有效商品标签的P(t)对应当日vendor日且未被隔离，声明available_at<=t20。终点t+h声明晚于当前决策且同split。对FU两条可能未来日期记录，没有使用错误t锚点而是保守排除；没有偷偷改未来anchor。一般历史发布未知在用户允许的探索性假设下披露，不自动否决全部探索性研究。

真实PREFIX_CAUSALITY_TEST取2023-06-30，修改此后所有真实来源的价格/量/额/OI/利率等数值，重新对齐、隔离、算特征并全量重新fit TRAIN。此前输入及所有mask逐字节不变，TRAIN Scaler完全相同。另任意替换全部VAL特征及其mask，TRAIN55组不变。初版测试扰动整数列产生pandas类型兼容警告，修正为显式浮点副本后，额外第三次独立raw重建/完整因果审核通过，发布28数组SHA仍一致；没有修改数据本身。

TEST实验是隔离合成CSV的日期门禁毒化：更改2025后数值尾部、置为不可解析字段，实际raw parser不能看到这些尾部，允许的TRAIN/VAL抽取工件相同。真实封存TEST报价没有被解析。旧文件做不透明字节SHA不属于行情数值读取。因果软件通过不等于历史PIT、未来修订/股票cohort都已被认证。
'''
    docs['SCIENTIFIC_DATA_QUALITY_REVIEW.md']='''# 科研质量分级：A–H

**A真实性**：312来源raw重标准化且与封存值差异0，所有原始文件SHA可追溯；3条有证据/预防性规则被隔离。不把源档一致当独立市场真值；I/C/PG/连续构造仍有未证实风险。

**B完整性**：263/25/23身份均有数据，但早期起点、停牌、源缺口和较晚历史均保留mask。库存日期全保留，不按未来完整性删资产；最长gap/内部分段及逐日市场有效节点、逐商品标签均出表。

**C时间**：20:00原锚点恢复，三市场声明available_at逐条符合规则；当前23重查官方指纹后所有正向匹配记录均隔离。真正交易session/发布时间未全证实，PIT=false。

**D因果**：无bfill、无跨split标签、无全期拟合；实际未来原始数值扰动后的对齐/特征/重fit前缀完全一致，TEST毒化隔离，无真实TEST开发。

**E特征**：26经济类型schema、单位和独立mask oracle通过，利率不当价格；371个|z|>20全部回溯原字段和分母。重尾与小分母、未复权公司行动和可能拼接仍有数值/经济解释风险，未因VAL表现改特征。

**F标签**：原P(t+h)/P(t)-1、decimal_return，h+1窗逐目标有效；SN/FU隔离点不参与；所有valid标签finite，覆盖和数值范围出表。它是固定供应商连续快照的数学变化，不是认证可执行收益。

**G复现**：两次独立raw重标准化构建28数组/7确定性元数据SHA一致，final manifest按两个哈希图重构一致；改进测试dtype后第三次raw重建也与发布数组完全相同。资产顺序、feature registry、全部55参数、源码/配置/源档皆归档。

**H限制**：固定263股票cohort可能含历史选择/幸存偏差；未复权公司行动；连续换月/历史回溯修订未知；真实发布时间及交易所档案日期不完全核实；原VAL覆盖参与选池且本23协议为质量审计后的变更；日频重叠累计标签相关、不能把N×23×4当独立样本。未来统计评价须尊重时间依赖、边界及独立确认。

技术有效=true：文件/mask/数值/原锚点/复现通过。探索性科研准入=true：在明确的供应商数学目标、时钟假设和上述限制下，已存重大未来时间风险记录不进入有效训练。不是由缺PIT就一票否决，也不是由pytest通过就放过具体时间冲突。确认性科研准入=false：缺少更强PIT、换月/源版本、历史cohort、独立研究设计及评价证据。

当前没有阻断23探索性训练的特定未处理已知未来记录。后续确认优先取得DCE I/C/PG争议日期独立逐合约证据、实际发布时间/vintage和连续构造定义；若要恢复AU/AG，须先补其2013会话日期/版本边界及原官方业务日期证据。无需因本轮缺这些证据而伪造数据或无限重试受阻接口。本轮不运行任何模型或MAE。
'''
    docs['REPORT.md']=f'''# Phase E-23：23商品研究数据集构建与科研审核

**本轮真实构建及完整TRAIN/VAL审核完成。DATASET_TECHNICALLY_VALID=true，EXPLORATORY_RESEARCH_READY=true，CONFIRMATORY_RESEARCH_READY=false。** 探索性准入只针对明确的供应商连续数学收益及AVAILABLE_AT_ASSUMED协议，不表示历史PIT或可交易持仓收益已验证。

新协议{r.PROTOCOL}；311输入=263股票+18利率+7财富指数+23商品，26特征、20session窗口。原25任务和旧23诊断Bridge均只读保留。本轮恢复R(t,h,c)=P(t+h,c)/P(t,c)-1，anchor=t，h=1/5/10/20，decimal_return，20:00收盘后预测；**无FWD3、无统一lag2、无Bridge未来锚点**。

## 实际规模

|split|参考日|origin|有padding的origin|逻辑X|Y|
|---|---:|---:|---:|---|---|
|TRAIN|2733|2726|19|[2726,20,311,26]|[2726,4,23]|
|VAL|484|483|0|[483,20,311,26]|[483,4,23]|

|h|TRAIN有效标签|VAL有效标签|
|---|---:|---:|
{counts}

新数据目录`{r.DATA}`，36文件、{publication['bytes']:,}字节（{publication['bytes']/1024**2:.2f}MiB）。28个NPY和7份确定性元数据在两个独立raw重建中SHA完全一致，最终manifest亦由两个哈希图重构一致；第三次独立raw复核发布数组也相同。不是从旧25删两列或复用旧Bridge标签。

## 科研质量结论及已处理问题

312个来源任务从raw经日期门禁及原normalizer重新执行，身份/日期/单位/相关数值与旧standardized比较差异0；400份现有官方月档重新用于23商品时间核查。15品种有官方可比较子集，8个DCE没有；不将无对照当错误或已核实。正向7字段匹配只有两条FU，均预防性隔离；没有剩余未处理正向匹配记录。真实历史发布时间仍ASSUMED，SDK日期不是原HTTP业务日期证明。

SN2015-03-27整报价隔离，原15905不改、不替119270。另隔离FU2014-12-31、2016-04-11两条薄量平坦的正向指纹风险记录，这是t锚点保护，不声称已证明FU日期错位。I/C/FU/PG其他既有异常维持证据分类，不因20%或|z|>20删点。3条输入mask、295个feature mask被改变；四周期TRAIN标签mask改变5/13/23/43，VAL不变。16组期货统计重算，其中15组变换改变；依赖锥外无实质性未解释变化，6998个浮点累计舍入最大7.55e-13在事前容差内。

全部55组TRAIN-only Scaler新fit，不复用旧25或Bridge。股票/期货年龄恒0的2常量/floor组未放大极端，全部有效X/Y finite。但仍有TRAIN355、VAL16个|z|>20，最大281.1443/75.9655。股票活动比率重尾、历史小分母、未复权公司行动、期货量/OI或拼接及短端利率冲击仍是科研与训练数值风险。全部371条原值、历史分母、mean/std/z、源文件SHA在extreme_feature_audit.csv，未裁剪、未以VAL MAE改方案。

263股票均有数据而非全期完整；TRAIN平均有效股票245.47/263，VAL262.57。最长股票内部gap234个参考session，VAL sh.600297尾缺112日。18利率各TRAIN2713/2733日，1–3Y财富指数只有2410日且前304日缺失；不伪造早期。10个商品当前源历史起点晚于首参考日；起点不是全都已核实的上市日。全资产日期仍保留，缺失只通过mask。详见各专门报告及coverage表。

## 23目标逐项覆盖

first_valid为处理后的有效报价起点，SN首日隔离后起点为03-30；MA历史不足不自动解释为上市晚。完整VAL日期/缺失与内部gap在per_asset_coverage.csv。

|商品|TRAIN first valid|TRAIN有效报价|TR1|TR5|TR10|TR20|VA1|VA5|VA10|VA20|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{target_table}

## 独立验收

|字段|结果|
|---|---|
{gates}

技术、探索性、确认性分别判断。NORMALIZATION_NUMERIC_PASS仅表示数值变换正确有限，不保证神经网络优化稳定；SOURCE_ANOMALY_AUDIT_PASS表示已审查/隔离/分类，不是每条市场报价都获独立认证；时间相关true仅在声明假设下成立。源方法、历史PIT、确认性结论没有提升为true。

{testcount}项测试通过，{warnings}条旧collect.py弃用警告。新适配器初始JSON列表/元组split比较修正、因果测试整数列转浮点兼容修复均有回归，原测试夹具包含边界及浮点表示断言修正，失败日志留存。dtype修复后完整第三次raw验证与发布数组SHA一致，没有改任何价格/参数/标签。原{old['protected_file_count']:,}文件SHA无差异。TEST行情解析0，未生成TEST标签，模型/MAE/下载/提交/推送均0。

## 对最终22项问题的回答

1. 23目标正式冻结，顺序按用户列表，独立配置及TARGET_SELECTION.md，原25配置仍保留。
2. 是，恢复原1/5/10/20的P(t+h)/P(t)-1、decimal_return和收盘后20:00定义。
3. 完全不使用FWD3。anchor_offset=0，原Bridge及其数据未覆盖。
4. 311身份正确，263+25+23，商品输入/目标同集合，无额外辅助期货。
5. 在冻结参考日历和时钟假设下对齐正确；不取共同报价交集、债券旧观测有限沿用、无bfill。
6. 声明available_at逐有效记录<=decision；重跑现有官方跨日期指纹并屏蔽两FU风险；有效P(t)确为当前原供应商日且非隔离点。真实发布时间/PIT仍未验证。
7. 263只均有真实历史，不能说全期完整。停牌、缺报、55只TRAIN起始不可用及较晚历史按mask保留，逐只覆盖和最长gap出表。
8. 25债券正确区分18利率和7指数，百分数/bps及指数相对变化不混淆。1–3Y指数早期缺口304日如实保留。
9. 23逐项见上表和coverage：LU/PG2020起、SC2018起等；MA当前源2014起不能伪称上市起点，SN首日隔离。FU最长内部13日，VAL商品内部缺口最多1日。
10. SN整观测及滚动/标签/Scaler传播均已隔离，原始值不改，不替官方价。
11. 重新检查没有发现新未处理重大正向匹配；原FU两条已保守隔离。其他非匹配、负偏移或薄量重复不直接等于时间错误；DCE未知独立核验状态保留。
12. 26维类型和mask正确，由独立sliding-window oracle精确复核。利率不计算价格收益，结构性不适用不计为真实缺失。
13. 55组独立TRAIN-only Z-score合理且复核通过，唯一日资产特征拟合，无窗口重叠计数，无旧Scaler/VAL/TEST复用。
14. 极端值构成潜在优化/数值风险；finite和无floor放大不等于训练稳定。保留371条原字段/分母证据，未裁剪、未认证所有极端都是真实市场动作。
15. TRAIN2726个origin。
16. VAL483个origin。
17. 四周期有效标签TRAIN52567/51939/51209/49752，VAL11101/10993/10858/10592，全部按mask统计。
18. 软件前缀、split及TRAIN-fit检查未发现允许规则之外的未来数值泄漏；真实source-vintage/PIT仍未知。没有把vendor_date单独当时间证明。
19. 两次完整raw构建28数组/7元数据一致，manifest哈希图一致；第三次独立raw复核也一致；旧产物原样。
20. 满足明确来源/时钟假设和限制下的探索性科研训练准入；当前没有未处理的具体已知实质性未来记录阻断23分支。本轮没有训练。
21. 不满足确认性论文准入，PIT/roll及独立设计/评价证据不足。
22. 23探索性当前无额外硬阻断。确认性需更强发布/历史vintage、连续构造、cohort/公司行动、独立确认方案；若恢复AU/AG，仍须2013会话/版本边界和官方原业务日期证据。I/C/PG争议点的独立DCE证据是局部质量优先项，不无限扩张为全历史下载要求。

## 复核与使用

```python
from triweave.research23 import Research23Dataset
ds = Research23Dataset("{r.DATA}", "TRAIN")
sample = ds[0]  # X:[20,311,26], Y:[4,23]，必须保留并使用所有mask
```

`.venv/bin/python -m triweave.research23 --verify`只读核对来源/旧文件/交付指纹和接口。构建`--run`为append-only，已有版本拒绝覆盖。新证据或协议修改必须另冻版本。本阶段停止于数据构建和审核，不运行模型或MAE。
'''
    for name,text in docs.items():(out/name).write_text(text)
    codefiles=['src/triweave/research23.py','src/triweave/research23_audit.py','src/triweave/research23_quality.py','src/triweave/research23_report.py','tests/test_research23.py']
    snap=out/'code_snapshot';snap.mkdir(exist_ok=True)
    for name in codefiles:shutil.copyfile(root/name,snap/Path(name).name)
    protected=json.loads((out/'preservation_before.json').read_text())['files']
    provenance=dict(phase='E-23',protocol_id=r.PROTOCOL,target_order=list(r.TARGETS),actual_input_nodes=311,actual_targets=23,
       user_task_file='/home/yangxiaotong/.codex/attachments/6a930a20-c63f-4b66-b2d4-a7ae329dcd23/已粘贴的文本.txt',
       user_task_sha256=sha256_file('/home/yangxiaotong/.codex/attachments/6a930a20-c63f-4b66-b2d4-a7ae329dcd23/已粘贴的文本.txt'),
       frozen_protocol_sha256=sha256_file(out/'frozen_protocol.json'),frozen_at=protocol['frozen_before_numeric_build_at'],build_started_at=results['started_at'],
       raw_replay_count=312,raw_to_standardized_numeric_differences=results['raw_replay_numeric_differences'],sources=results['sources'],
       official_sources=timing['official_sources'],all_source_quote_logs='build_results.json:quote_read_log',
       source_quarantine_sha256=sha256_file(out/'source_quarantine_manifest.csv'),
       reference_calendar_sha256=sha256_file(root/'data/manifests/baseline15y/reference_trading_calendar.csv'),
       environment='environment.json',source_code={name:sha256_file(root/name) for name in codefiles},
       shared_code={k:protected[k] for k in ('src/triweave/after_close.py','src/triweave/normalize.py','src/triweave/rescue_providers.py','src/triweave/bridge_audit.py','src/triweave/sessionqa_match.py')},
       config_sha256=sha256_file(root/'configs/futures_targets_23_research_v1.csv'),target_doc_sha256=sha256_file(root/'docs/TARGET_23_RESEARCH_PROTOCOL.md'),
       previous_protocols_preserved=['TRIWEAVE_25_AFTER_CLOSE_V1_EXPLORATORY','TRIWEAVE_23_DIAGNOSTIC_V0'],
       before_inventory_sha256=sha256_file(out/'preservation_before.json'),old_artifact_check=old,
       previous_seals={name:h for name,h in protected.items() if name.endswith('delivery_manifest.json')},
       prior_VAL_design_disclosure=True,normalization_or_feature_changes_from_VAL_MAE=False,
       TEST_numeric_rows_parsed=0,models_run=0,MAE_computed=0,new_market_requests=0,Git_commits=0,Git_pushes=0,
       implementation_corrections=['New JSON list/tuple split validation corrected; regression covers saved JSON semantics.',
         'Prefix test explicitly casts integer source-activity copies tofloat; future warnings avoided and independent third raw rebuild unchanged.',
         'Synthetic test inclusive endpoint and float representation expectations corrected; failed preflight logs retained, production data unchanged.'],
       commands=['.venv/bin/python -m triweave.research23 --run','.venv/bin/python -m pytest -q',
                 '.venv/bin/python -m triweave.research23_report','.venv/bin/python -m triweave.research23 --verify'])
    ac.write_json(out/'provenance.json',provenance)
    archive=root/'data/manifests/phase_e_23_research';archive.mkdir(exist_ok=True);ac.write_json(archive/'provenance.json',provenance)
    files=[p for p in out.rglob('*') if p.is_file() and p.name!='delivery_manifest.json']+list(dest.iterdir())
    files += [root/name for name in codefiles]+[archive/'provenance.json',root/'configs/futures_targets_23_research_v1.csv',root/'docs/TARGET_23_RESEARCH_PROTOCOL.md']
    indexed={str(p.relative_to(root)):dict(sha256=sha256_file(p),bytes=p.stat().st_size) for p in sorted(files)}
    ac.write_json(out/'delivery_manifest.json',dict(phase='E-23',protocol_id=r.PROTOCOL,files=indexed,dataset_bytes=publication['bytes'],
       total_indexed_bytes=sum(e['bytes'] for e in indexed.values()),self_excluded=True,
       fingerprint_scope='All delivered reports, source/tests/config/doc, dataset36files and archive provenance; this self-referential index has external SHA256'))
    verify(root)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verify',action='store_true')
    (verify if p.parse_args().verify else build)(Path.cwd())
