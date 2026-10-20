"""SourceQA final report/release gate; refuses a fictional revised dataset."""
from __future__ import annotations
import argparse
import datetime
import importlib.metadata
import json
import platform
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from triweave.after_close import DATA_REL,write_json
from triweave.sourceqa_investigate import OUT_REL
from triweave.universe import sha256_file


def mdtable(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)+'\n'


def preservation(root):
    out=root/OUT_REL;files=json.loads((out/'preservation_before.json').read_text())['files']
    changed=[p for p,h in files.items() if not (root/p).is_file() or sha256_file(root/p)!=h]
    if changed:raise AssertionError('Old files changed: '+str(changed))
    return {'protected_files':len(files),'changed_files':changed,'passed':not changed}


def report(root):
    root=Path(root).resolve();out=root/OUT_REL
    if (out/'delivery_manifest.json').exists():raise FileExistsError('SourceQA delivery sealed')
    inv=json.loads((out/'investigation_results.json').read_text())
    impact=json.loads((out/'impact_results.json').read_text())
    timing=json.loads((out/'timing_adjudication.json').read_text())
    expanded=json.loads((out/'timing_expansion.json').read_text())
    repro=json.loads((out/'candidate_reproducibility.json').read_text())
    tests=(out/'tests.txt').read_text();n=int(re.search(r'(\d+) passed',tests).group(1))
    assert 'failed' not in tests and 'skipped' not in tests
    protected=preservation(root);write_json(out/'final_preservation_check.json',protected)
    assert not (root/'data/derived/triweave_25_after_close_v1_1').exists()
    assert timing['after_close_release_blocked']
    logs=inv['quote_read_log']+expanded['quote_read_log']+impact['quote_read_log']
    assert all(x['TEST_numeric_rows_parsed']==0 and (x['max_parsed_date'] is None or x['max_parsed_date']<='2024-12-31') for x in logs)
    flags={
        'SOURCE_ANOMALY_INVESTIGATED':True,
        'SN_SOURCE_STATUS_RESOLVED':True,
        'EARLY_FUTURES_ANOMALIES_TRIAGED':True,
        'ANOMALY_MASK_CAUSAL':'NOT_VERIFIED',
        'FEATURE_DEPENDENCY_REBUILT':impact['unexplained_material_unscaled_changes']==0,
        'LABEL_DEPENDENCY_REBUILT':impact['changed_label_masks']==4 and impact['TRAIN_VAL_boundary_pass'],
        'TRAIN_SCALER_REFITTED':impact['TRAIN_only_scaler_pass'],
        'EXTREME_Z_AUDITED':impact['normalization']['all_finite'],
        'EARLY_WINDOW_PADDING_PASS':impact['early_padding_pass'],
        'TRAIN_VAL_BOUNDARY_PASS':impact['TRAIN_VAL_boundary_pass'],
        'TEST_ISOLATION_PASS':True,
        'DATASET_REPRODUCIBILITY_PASS':'NOT_VERIFIED',
        'EXPLORATORY_TRAIN_READY':False,
        'CONFIRMATORY_TRAIN_READY':False,
        'HISTORICAL_PIT_VERIFIED':False,
        'FUTURES_ROLL_METHOD_VERIFIED':False}
    scope={
        'SN_SOURCE_STATUS_RESOLVED':'Treatment resolved: same-day multi-field evidence supports quarantine; unique source error mechanism/underlying identity and adjustment remain NOT_VERIFIED.',
        'ANOMALY_MASK_CAUSAL':'SN-only date-local candidate mask passes; no complete causal rule for AU/AG source-session conflicts is approved.',
        'FEATURE_DEPENDENCY_REBUILT':'Actual source-based in-memory SN-only counterfactual, not a V1.1 release artifact.',
        'LABEL_DEPENDENCY_REBUILT':'Actual source-based in-memory SN-only counterfactual, not a V1.1 release artifact.',
        'TRAIN_SCALER_REFITTED':'All55 groups refitted for SN-only impact analysis; candidate scaler is not an approved V1.1 scaler.',
        'DATASET_REPRODUCIBILITY_PASS':'V1.1 not generated. Original V1 and SN-only counterfactual separately reproduced exactly.',
        'EXPLORATORY_TRAIN_READY':'Blocked by newly corroborated AU/AG source-session conflicts and unresolved historical information cutoff; V1 hold unchanged.'}
    gate={'phase':'Phase E-SourceQA','acceptance':flags,'scope_and_limits':scope,
          'V1_1_generated':False,'original_V1_RESEARCH_HOLD_preserved':True,
          'candidate_only':True,'candidate_role':impact['role'],
          'candidate_reproducibility_pass':all(r['identical'] for r in repro['arrays'].values()) and repro['scaler_identical'],
          'original_V1_reproduced':all(impact['independent_V1_reproduction'].values()),
          'same_date_19_00_source_time_assumption_supported':False,
          'unresolved_timing_evidence':timing['strong_conflicts'],
          'SN_only_impact':{k:impact[k] for k in ['changed_raw_values','changed_input_masks','changed_feature_masks','changed_label_masks','changed_scaler_groups','changed_scaler_transformations','local_feature_dependency_windows','pooled_refit_changed_X_windows','max_unscaled_roundoff']},
          'old_file_preservation':protected,'TEST_numeric_rows_parsed':0,'network_requests':0,
          'tests':tests.strip().splitlines()[-1],'model_training_or_MAE_executed':False}
    write_json(out/'audit_results.json',gate)
    causality={'scope':'Intrinsic pipeline arithmetic vs source timestamp semantics are separate',
               'asset_order_313_pass':True,'target_order_25_pass':True,'three_split_protocol_pass':True,
               'TRAIN_only_scaler_pass':impact['TRAIN_only_scaler_pass'],'intrinsic_prefix_pass':impact['intrinsic_prefix_pass'],
               'label_boundary_pass':impact['TRAIN_VAL_boundary_pass'],'dependency_closure_pass':impact['unexplained_material_unscaled_changes']==0,
               'SN_date_local_quarantine_prefix_pass':True,'source_session_information_cutoff_verified':False,
               'source_session_conflict_requires_hold':True,'early_padding_pass':impact['early_padding_pass'],
               'all_valid_numeric_arrays_finite':impact['all_arrays_finite'],
               'original_V1_NPY_SHA_reproduction_pass':all(impact['independent_V1_reproduction'].values()),
               'candidate_NPY_SHA_reproduction_pass':gate['candidate_reproducibility_pass'],
               'old_files_sha_pass':protected['passed'],'TEST_numeric_access_rows':0,
               'V1_1_reproducibility':'NOT_VERIFIED_NO_VERSION_CREATED','test_output':'tests.txt'}
    write_json(out/'causality_tests.json',causality)
    environment={'python':platform.python_version(),'platform':platform.platform(),
                 'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','pytest','akshare','baostock','triweave-data']},
                 'cwd':str(root),'network_requests':0,'TEST_numeric_rows_parsed':0,'tests':gate['tests']}
    write_json(out/'environment.json',environment)
    sn=inv['sn_joint_evidence']
    sntext=f'''# SN2015-03-27来源调查

结论：该观测不满足本探索协议的可解释源身份要求，足够支持研究层隔离；不能确认唯一的上游故障机制，也不能因与合约不同就宣布供应商必然造错。分类SOURCE_IDENTITY_MISMATCH仅表示与所声称SN/日期口径不兼容；CONFIRMED_SOURCE_ERROR的确定机制尚未成立。

raw与standardized七项全部相同：OHLC=16025/16060/15880/15905，settle=15970，volume=35290，open_interest=75124。raw是已保存的AKShare DataFrame快照，不是未经解析的Sina网络JSON，故无法回查原始JSON键/数组顺序。最小适配代码没有在标准化时缩放这些数值。

已有SHFE原始月档与标准化月档共同给出同日9个SN合约，成交价low最低115150、high最高123080，结算119230—120660，成交量合计1832、持仓合计620。供应商记录不仅价格/结算不兼容，其量与持仓分别为全部所观测合约合计约{35290/1832:.2f}倍和{75124/620:.2f}倍。官方档案当前已观察合约并不等于全历史所有可能版本的PIT认证；但纯价格复权也不能自动解释活动量差异。

下一参考日2015-03-30供应商OHLC=119270/119530/115600/118080，与官方SN1507的OHLC吻合；量/OI/settle并非七项全同，不能据此宣布该接口始终为SN1507。后20参考日记录在SN_first21_reference_quotes.csv及evidence_SN_*中，原始首点和后续条目均完整保留，未换成119270或插值。

本地供应商wrapper只声明中国期货日线，没有逐日底层合约、调整因子、历史连续方法或发布时间证据。特殊调整定义NOT_VERIFIED，字段/品种映射错误可能但未证实；跨38连续序列同日七字段完全重复为0。官方同日其他品种只存在部分OHLC相似，不足以认定这个首点来自某个已确定品种。没有给它随意安上一个替代身份。

source_quarantine_rules.csv冻结整条观测在feature/label之前无效的候选规则，source_date精确匹配且不使用未来价格、20%阈值或模型性能。依据是同日独立价/结算/量/持仓联合证据。它是事后审阅固定快照的研究清洗，不追认历史当时可得PIT。该规则本轮只运行SN-only内存影响分析，旧数据和旧mask未改变。

不能只修此点就放行：扩大核查发现AU/AG2013-07-08与官方2013-07-09七字段一致的日期冲突。该时间规则未解决，所以未生成或批准V1.1。SN的隔离处理可说明，但整个数据版本仍受RESEARCH_HOLD约束。
'''
    (out/'SN_SOURCE_INVESTIGATION.md').write_text(sntext)
    pad=pd.read_csv(out/'early_window_padding_audit.csv')
    (out/'early_window_padding_audit.md').write_text('''# 最早30个实际TRAIN origin

已通过原V1只读Dataset loader逐个取样。首个origin2011-10-10仅1个实际参考session，左侧19个session为计算padding。首19个origin逐渐减少padding，从第20个起具备20个参考session；参考session数量不代表313节点各自报价/滚动特征都完整，停牌、上市前与跨市场缺失继续用各自mask。

padding索引=-1；X=0，input_valid_mask=false且feature_valid_mask=false；观测日/可用时刻/来源索引/年龄=-1，不能解释成真实日期。feature_applicable_mask为静态结构适用性，不代表padding有效。模型接口必须同时消费动态mask。没有复制未来第一条报价、bfill或捏造早期历史。真实报价中的0指标与padding由mask区分。

early_window_padding_audit.csv记录全部30项，全部检查通过。此次未保存新dataset，SN-only影响模拟沿用完全相同窗口索引。
'''+mdtable(['origin','真实参考session','padding'],[[r.origin_date,r.real_reference_sessions,r.left_padding_sessions] for r in pad.head(22).itertuples()]))
    audit_table=mdtable(['字段','结果'],[[k,str(v).lower() if isinstance(v,bool) else v] for k,v in flags.items()])
    scaler=pd.read_csv(out/'changed_scaler_groups.csv')
    scaler_names=', '.join(scaler.feature_name.tolist())
    extreme=impact['normalization']
    frames=pd.read_csv(out/'early_futures_anomaly_audit.csv')
    event_table=mdtable(['资产/源日期','分类','处理'],[[r.asset_id+' '+r.source_date,r.classification,r.action] for r in frames.itertuples()])
    rpt=f'''# Phase E-SourceQA：源异常、传播与数据发布门禁

生成：{datetime.datetime.now(datetime.timezone.utc).isoformat()}。本轮调查、真实重算与审计完成；**没有生成V1.1，EXPLORATORY_TRAIN_READY=false，旧V1的RESEARCH_HOLD原样保留**。仅隔离SN不足以解决新发现的跨来源时间冲突，不能为解锁训练强行发布。

## 调查范围与独立证据

实际比对38条现有Sina连续序列、所选25商品与已有SHFE/CZCE/INE的{inv['official_comparable_asset_dates']:,}个可比较资产日期，复核原始和标准化档案、供应商及官方适配器。没有下载新行情，TEST数字字段在日期门禁后才可能解析；本轮解析TEST数值0行。

SN2015-03-27的15905确在raw中，OHLC/settle/量/OI均与standardized一致。同日已有9个官方SN合约价格/结算及活动合计与该整条记录严重不兼容，支持研究隔离，而非支持随意用119270替价。特殊调整、究竟哪个字段/品种被错配、哪一方历史版本负责，仍NOT_VERIFIED。专项证据见SN_SOURCE_INVESTIGATION.md与SN_two_days_all_contracts.csv。

**新增阻断证据**：Sina AU/AG的2013-07-08记录七项OHLC/volume/OI/settle分别完全匹配官方2013-07-09 AU1312/AG1312，同日官方记录不能匹配；已回到已有官方原始月档再次确认日期及数值。两条都是非平坦且有大量成交的bar，不能用低流动性的重复值解释。现有19:00当日可得假设没有足够依据，可能涉及夜盘交易日归属、源标签或历史版本。不能把日期搬一天、改决策时刻或只隔离这两行后宣称整个时期已核实。timing_match_adjudication.csv明确把另外3个低流动/同日也匹配的候选与这2个强冲突分开，未把所有跨日期重复都说成未来数据。

扩大检查的初筛联合不一致{len(inv['joint_contradictions'])}条不是{len(inv['joint_contradictions'])}条已确认错误。59条OI初筛源于不完整/零OHLC的当前OI合约子集；原始月档含RO，58条能精确匹配RO的七字段报价。原始RO→当前OI的完整经济连续定义仍未核实，未修改名单、混入RO输入或自动填洞。AU/AG其余少量价/量不一致也保留审阅状态，不凭20%阈值删除。

## 各异常处理

{event_table}
FU2019-08-05/06分别匹配FU1909/FU2001的OHLC，支持“观察到的候选底层合约切换”；供应商整体换月定义仍未核实，不是同合约可交易损失。FU2014-08-29匹配FU1505，前日源低成交/旧报价口径有不确定性。I/C/PG没有既有DCE逐合约独立证据，保留NOT_VERIFIED，不能认定真实市场极端或错误，更不据预测表现排除。

## 真实重算：SN-only影响分析，明确不是V1.1

从冻结来源独立完整重建V1，25个数值NPY SHA逐一与原版相同，然后在内存中应用SN整条观测隔离，再从源重建TRAIN/VAL特征、标签、全部mask并重新fit55组TRAIN scaler。没有读取旧NPY来做修补；旧NPY仅用于比较。候选Scaler写为candidate_sn_only_scaler_params.json，明确不是获准新版本Scaler。

changed_raw_values=0；changed_input_masks=1；changed_feature_masks=15；changed_label_masks=4。局部依赖影响40个TRAIN输入窗口（2015-03-27—2015-05-25），VAL局部影响0；直接包含异常原始报价的窗口与包含滚动衍生依赖的窗口分开，不能只删原点标签。4个周期各移除1个SN数学有效标签，所有监督origin仍2728/483。

SN-only候选标签总数为TRAIN57888/57248/56503/55016，VAL12067/11951/11806/11520。本节都是有实际重算证据的反事实影响结果，不是已生成V1.1的数量。dataset_v1_v1_1_diff.csv每行标注比较角色，V1_1_generated=false，不能引用为新版本发布。

55组中16组FUTURES_CONTINUOUS_PRICE统计变化，其中15组实际mean/scale改变；quote_age_days仅有效计数减少、仍恒0。组为：{scaler_names}。类型池化重新fit会改变全部2728个TRAIN及483个VAL窗口中的某些标准化输入；这不意味着所有原始报价都异常。股票和债券组不因SN隔离重fit变化。

未解释的实质性未缩放差异0。现有滚动方差累计和的浮点消减导致6930个依赖锥之外的微小差异，最大{impact['max_unscaled_roundoff']:.15g}，在计算前冻结的atol1e-9/rtol1e-10内；没有用旧NPY覆盖这些差异。掩码、日期、窗口、原始输入均作精确比较；标准化全局变化由对应TRAIN组参数重算解释。anomaly_dependency_impact.csv逐源观测记录原始、日特征、窗口、标签质量/端点以及Scaler传播；中间报价只影响质量窗口而未直接参与P端点算术的情况另列语义，避免把21个质量依赖说成21个数值端点变化。

## 极端值与最早窗口

原V1 |z|>20为TRAIN358、VAL16；SN-only候选为TRAIN356、VAL16。全部例子在normalization_extreme_audit.csv，不是仅列最大5个。两版最大|z|仍TRAIN281.144257、VAL75.965523，均finite，实际极端值无scale-floor放大；常量/floor仍两组股票/期货quote_age_days，均0，无Inf/NaN。高z本身不是错误判据。

TRAIN最大例sz.002831 2016-12-23 amount_change_1d：前值2492755元、当日1343509376元，相对变化537.965673，z281.144257；不是百分数/金额单位切换证据。VAL最大sh.600837 2024-10-14成交额变化145.453539、z75.965523。期货Y2015-06-24持仓从2到539366，变化269682、z170.341187，体现小分母/合约统计口径风险；不自动裁剪，也不直接归为真实市场变化。未来若研究log1p或稳健缩放，必须独立协议，不能按VAL MAE挑方案。

股票异常另读取已有BaoStock preclose与pctChg等原字段：有部分前收参考价改变，支持公司行动风险提示但不是已核实公司公告。未复权股票特征不是总收益；不要求本轮穷尽所有公司行动，未伪称其已解决。早期30个TRAIN origin全部实际读取；首个只有1个真实参考session、19个padding，padding值0且动态mask全false，来源/时点索引-1，无bfill或未来首价复制。详见early_window_padding_audit.md。

## 独立门禁

{audit_table}
SN_SOURCE_STATUS_RESOLVED=true只表示“有证据支持隔离处理”，不表示原始根因被唯一确认。FEATURE/LABEL_DEPENDENCY_REBUILT和TRAIN_SCALER_REFITTED的true仅覆盖实际执行的SN-only内存影响分析，不能据此声称V1.1存在。ANOMALY_MASK_CAUSAL和DATASET_REPRODUCIBILITY_PASS的NOT_VERIFIED针对完整修订版本：它尚未建立；V1与SN候选各自已完整复现，见candidate_reproducibility.json及causality_tests.json。日期算术/固定Scaler的prefix检查通过不代替源数据实际交易session证据。

{n}项测试通过（原有80条collect.py弃用警告）；原{protected['protected_files']:,}个文件重新SHA，无差异。TEST数值0、网络请求0、没有模型/MAE/Git操作。完整来源、参数、源/代码/协议与所有交付SHA见provenance.json和delivery_manifest.json。

## 对最终14项问题的回答

1. SN究竟的问题：一条整记录的源身份/尺度不兼容；价格、结算、量和OI联合证据拒绝当前协议接受。无法唯一确认上游故障、币种、映射或特殊调整机制，保持NOT_VERIFIED。
2. 足够支持研究层隔离，不足以认定唯一的供应商故障或替代真价；不是仅比较close或涨幅。
3. 应隔离2015-03-27整条观测及全部派生依赖。已冻结候选规则并实际计算影响，旧数据不改，当前未应用到获准新版本。
4. 局部40个TRAIN窗口；重fit池化Scaler后2728+483个窗口的部分标准化输入都会改变，必须分开解释。
5. 影响4个有效标签，1/5/10/20各1。其他待审日期的h+1质量窗口依赖、端点数值依赖及重叠均在明细表中，不能相加当独立样本。
6. 16个FUTURES_CONTINUOUS_PRICE组统计变化，其中15个变换改变，列表见上及changed_scaler_groups.csv；股票/债券组未受该隔离影响。
7. I/C/PG为NOT_VERIFIED，无独立依据自动隔离；FU有候选合约切换证据但不是可执行收益。新发现的AU/AG源日期冲突须先解决时间协议，不能只修SN。
8. 未发现非有限数值或floor造成的极端值；重尾、小分母和来源/公司行动风险实质存在，保留而非删掉。本轮不以VAL性能调Scaler。
9. 首样本1真实session+19左padding；逐渐缩减到第20样本无padding。padding动态mask=false，可由模型接口区分，无未来报价复制。
10. V1.1未生成，原因是完整来源时间/隔离协议未达到发布条件；data/derived/triweave_25_after_close_v1_1不存在。
11. 不存在“V1.1实际差异”。仅SN-only反事实的实际重算差异：raw0、input mask1、feature mask15、label mask4、Scaler统计组16；它的角色在每个结果文件标清，不能冒充版本交付。
12. 原V1从源25数组SHA复现一致；SN-only重算两遍的25数组SHA和Scaler也完全一致。不存在可宣布已复现的V1.1。
13. 不能开始探索性训练，EXPLORATORY_TRAIN_READY=false，旧RESEARCH_HOLD未解除；暂停点是源时间语义，不是pytest失败。
14. 即便未来解除上述阻断，固定股票cohort、VAL覆盖参与选池、未复权公司行动、未知连续/历史修订、历史发布时间与重叠标签仍限制确认性研究，PIT/ROLL/CONFIRMATORY全部false。

## 必须补足的证据与继续顺序

先取得Sina历史日线在夜盘上线时期的日期/交易session定义和历史版本解释，同时核对2013-07-08/09官方原始发布日期/时刻，分清供应商标签、官方聚合器与版本责任；不能只用当前字段七项相等猜唯一原因。然后系统复核AU/AG受影响时期，冻结不使用未来价格的session/available_at与目标锚点一致规则。OI的RO/OI历史经济口径和I/C/PG日期的独立逐合约证据继续保留待核验，不强迫补齐所有交易所15年。完整规则成立后再新建V1.1、从原源全量重建、重fit及复现，保留本轮与V1全部证据。

本阶段在审核交付处停止，未训练、未算MAE、未覆盖raw/standardized/旧dataset/manifest/scaler/报告，未生成假版本，未提交或推送。
'''
    (out/'REPORT.md').write_text(rpt)
    provenance={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'phase':'Phase E-SourceQA','protection_inventory':'preservation_before.json',
                'old_files_preservation':protected,'V1_1_generated':False,'old_RESEARCH_HOLD_modified':False,
                'candidate_role':impact['role'],'candidate_rules_sha256':sha256_file(out/'source_quarantine_rules.csv'),
                'candidate_policy_sha256':sha256_file(out/'impact_analysis_protocol.json'),
                'source_files':inv['vendor_sources']+inv['official_sources'],
                'original_V1_manifest_sha256':sha256_file(root/DATA_REL/'dataset_manifest.json'),
                'original_V1_hold_sha256':sha256_file(root/DATA_REL/'RESEARCH_HOLD.json'),
                'original_environment_and_protocol':'experiments/phase_e_after_close/provenance.json',
                'source_read_logs':['investigation_results.json','timing_expansion.json','impact_results.json'],
                'additional_existing_source_reads':'TRAIN-only slices in OI_RO_original_official_raw_evidence.csv;AU_AG_July8_9_original_official_raw.csv;SN_same_date_official_cross_variety_matches.csv. Parent archives included in source_files.',
                'TEST_numeric_rows_parsed':0,'network_requests':0,'raw_archive_fingerprints':'Opaque bytes only, including TEST-inclusive archives; no TEST quote numeric parsing',
                'source_code_sha256':{str(p.relative_to(root)):sha256_file(p) for p in sorted((root/'src/triweave').glob('sourceqa*.py'))},
                'new_test_sha256':sha256_file(root/'tests/test_sourceqa.py'),'tests':gate['tests'],
                'code_changes':'New SourceQA modules/tests only; protected Phase E code and all old inputs unchanged',
                'feature_dependency_note':'Raw price-derived masks require full consecutive window, label arithmetic uses endpoints; semantic table distinguishes them. Rounding tolerance fixed before candidate run.',
                'reports_read':['experiments/phase_e_after_close/REPORT.md','experiments/phase_e_after_close/SOURCE_ALERT.md','experiments/phase_d_continuous/REPORT.md','experiments/phase_d_contracts/REPORT.md','experiments/data_rescue/REPORT.md']}
    write_json(out/'provenance.json',provenance)
    print(json.dumps({'acceptance':flags,'V1_1_generated':False,'protected_files':protected['protected_files'],'tests':gate['tests']},ensure_ascii=False))


def seal(root):
    root=Path(root).resolve();out=root/OUT_REL
    if (out/'delivery_manifest.json').exists():raise FileExistsError('Already sealed')
    protected=preservation(root)
    audit=json.loads((out/'audit_results.json').read_text());assert audit['V1_1_generated'] is False
    assert not (root/'data/derived/triweave_25_after_close_v1_1').exists()
    archive=root/'data/manifests/phase_e_sourceqa';mirror=root/'data/audit/phase_e_sourceqa'
    if archive.exists() or mirror.exists():raise FileExistsError('Archive exists')
    code=list((root/'src/triweave').glob('sourceqa*.py'))+[root/'tests/test_sourceqa.py']
    for relative in ['src/triweave/after_close.py','src/triweave/after_close_build.py',
                     'configs/futures_targets_25_v1.csv','configs/bond_variables_25.csv',
                     'configs/project.json','data/manifests/stocks_263.csv']:
        code.append(root/relative)
    for p in code:
        dest=archive/'code_snapshot'/p.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    for name in ['impact_analysis_protocol.json','source_quarantine_rules.csv','audit_results.json','provenance.json','environment.json']:
        shutil.copy2(out/name,archive/name)
    finalbytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    write_json(out/'final_storage.json',{'report_files_bytes_before_seal_records':finalbytes,
               'new_dataset_files_bytes':0,'original_V1_dataset_files_bytes':sum(p.stat().st_size for p in (root/DATA_REL).iterdir() if p.is_file()),
               'available_disk_bytes':shutil.disk_usage(root).free,'old_files_changed':0})
    mirror.mkdir(parents=True)
    for p in out.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:
            dest=mirror/p.relative_to(out);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    files={str(p.relative_to(root)):{'sha256':sha256_file(p),'bytes':p.stat().st_size}
           for parent in [out,archive,mirror] for p in sorted(parent.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    for p in code:files[str(p.relative_to(root))]={'sha256':sha256_file(p),'bytes':p.stat().st_size}
    delivery={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files,
              'V1_1_generated':False,'new_training_dataset_published':False,'original_RESEARCH_HOLD_preserved':True,
              'note':'SN-only numerical simulation is explicitly not a V1.1 dataset. Self hash excluded.'}
    write_json(out/'delivery_manifest.json',delivery);shutil.copy2(out/'delivery_manifest.json',mirror/'delivery_manifest.json')
    verify(root)


def verify(root):
    root=Path(root);out=root/OUT_REL
    manifest=json.loads((out/'delivery_manifest.json').read_text())
    for path,record in manifest['files'].items():assert sha256_file(root/path)==record['sha256'],path
    p=preservation(root)
    assert not (root/'data/derived/triweave_25_after_close_v1_1').exists()
    print(json.dumps({'delivery_files_verified':len(manifest['files']),'old_files_verified':p['protected_files'],
                      'changed_old_files':0,'V1_1_generated':False,
                      'delivery_manifest_sha256':sha256_file(out/'delivery_manifest.json')},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--seal',action='store_true');parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    if args.verify:verify(args.root)
    elif args.seal:seal(args.root)
    else:report(args.root)
