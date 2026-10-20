"""Archive SessionQA audit and independently verify all protected/delivery bytes."""
from __future__ import annotations
import argparse
import datetime
import inspect
import json
import re
import shutil
from pathlib import Path
import pandas as pd
from triweave import after_close as ac
from triweave.sessionqa_match import OUT, summarize
from triweave.universe import sha256_file


def protected_check(root):
    prior=json.loads((root/OUT/'preservation_before.json').read_text())
    differences=[]
    for name,expected in prior['files'].items():
        p=root/name
        if not p.is_file() or sha256_file(p)!=expected:differences.append(name)
    if differences:raise RuntimeError(f'Protected files changed: {differences}')
    return dict(protected_file_count=len(prior['files']),changed_files=differences,
                all_original_files_unchanged=True,comparison='OPAQUE_BYTE_SHA256_NOT_TEST_NUMERIC_INGESTION')


def verify(root):
    out=root/OUT
    manifest=json.loads((out/'delivery_manifest.json').read_text())
    for name,e in manifest['files'].items():
        p=root/name
        assert p.is_file() and sha256_file(p)==e['sha256'],name
        assert p.stat().st_size==e['bytes'],name
    check=protected_check(root)
    assert not (root/'data/derived/triweave_25_after_close_v1_1').exists()
    flags=json.loads((out/'audit_results.json').read_text())
    assert flags['EXPLORATORY_TRAIN_READY'] is False
    assert flags['DATASET_V1_1_GENERATED'] is False
    print(json.dumps(dict(delivery_files=len(manifest['files']),delivery_sha256=sha256_file(out/'delivery_manifest.json'),
                          **check,V1_1_generated=False,EXPLORATORY_TRAIN_READY=False),ensure_ascii=False),flush=True)


def build(root):
    out=root/OUT
    if (out/'delivery_manifest.json').exists():raise FileExistsError('Delivery sealed; use --verify')
    check=protected_check(root);ac.write_json(out/'preservation_check.json',check)
    train=pd.read_csv(out/'train_all25_date_matching.csv',low_memory=False)
    val=pd.read_csv(out/'val_all25_date_matching.csv',low_memory=False)
    tr=json.loads((out/'train_matching_results.json').read_text())
    vr=json.loads((out/'val_matching_results.json').read_text())
    replay=json.loads((out/'v1_replay_checks.json').read_text())
    freeze=json.loads((out/'frozen_disposition.json').read_text())
    sn=json.loads((root/'experiments/phase_e_sourceqa/impact_results.json').read_text())
    assert tr['raw_standardized_mismatches']==vr['raw_standardized_mismatches']==0
    assert not freeze['complete_revised_temporal_protocol']
    testmatch=re.search(r'(\d+) passed, (\d+) warnings', (out/'tests.txt').read_text())
    assert testmatch and int(testmatch[1])>=185
    test_count=int(testmatch[1]);new_test_count=test_count-165
    statuses={
      'AU_AG_SESSION_INVESTIGATED':True,'AU_AG_DATE_SHIFT_RESOLVED':False,
      'HISTORICAL_NIGHT_SESSION_RULES_AUDITED':True,'OTHER_FUTURES_SESSION_AUDITED':True,
      'SESSION_QUARANTINE_CAUSAL':'NOT_VERIFIED','SN_QUARANTINE_APPLIED':False,
      'INPUT_LABEL_TIME_CONSISTENCY_PASS':False,'TRAIN_SCALER_REFITTED':'BLOCKED',
      'HORIZON_BOUNDARY_PASS':True,'PREFIX_CAUSALITY_PASS':True,'TEST_ISOLATION_PASS':True,
      'DATASET_V1_1_GENERATED':False,'DATASET_V1_1_REPRODUCIBLE':'BLOCKED',
      'EXPLORATORY_TRAIN_READY':False,'HISTORICAL_PIT_VERIFIED':False,
      'FUTURES_ROLL_METHOD_VERIFIED':False,'CONFIRMATORY_TRAIN_READY':False}
    scopes={
      'AU_AG_SESSION_INVESTIGATED':'All TRAIN years and fixed-rule VAL inspected, including raw SDK dates and contemporary documents.',
      'AU_AG_DATE_SHIFT_RESOLVED':'Localized all7 +1 identities do not identify a complete vendor date mapping; unknown extent blocks release.',
      'HISTORICAL_NIGHT_SESSION_RULES_AUDITED':'Dated2013 contemporary material checked; not a complete signed historical rulebook or vendor/PIT certification.',
      'OTHER_FUTURES_SESSION_AUDITED':'All25 screened; 17 have some ordinary official comparison, 8 DCE have none; audited does not mean verified.',
      'SESSION_QUARANTINE_CAUSAL':'Minimum5 fixed observation rules pass diagnostic mask tests, but full affected extent is not bounded.',
      'SN_QUARANTINE_APPLIED':'False for a released complete revision. Previous SN-only counterfactual and this minimum boolean diagnostic are not a release.',
      'INPUT_LABEL_TIME_CONSISTENCY_PASS':'False for V1 and unavailable for a complete candidate. Declared-clock arithmetic alone passes but4 known risk rows remain in V1.',
      'TRAIN_SCALER_REFITTED':'BLOCKED for a complete new candidate. V1 fresh55-group fits replayed twice; previous SN-only55-group fit retained as separate evidence.',
      'HORIZON_BOUNDARY_PASS':'Existing V1 all TRAIN/VAL origins, all horizons independently recomputed and checked; no new candidate exists.',
      'PREFIX_CAUSALITY_PASS':'Existing V1 source-derived unscaled and normalized prefix invariant with fixed TRAIN scaler; not source-session/PIT verification.',
      'TEST_ISOLATION_PASS':'All new quote reads date-gated before CSV/numeric parsing; TEST numerical rows0, synthetic poisoned TEST row regressions pass.',
      'DATASET_V1_1_REPRODUCIBLE':'No V1.1. Two successful bitwise V1 replays cannot satisfy a nonexistent candidate release gate.'}
    ac.write_json(out/'audit_results.json',dict(statuses,scope=scopes,release_disposition='RESEARCH_HOLD',
      complete_temporal_protocol=False,TRAIN_vendor_rows=tr['vendor_rows'],VAL_vendor_rows=vr['vendor_rows'],
      official_ordinary_rows_TRAIN=tr['official_ordinary_rows'],official_ordinary_rows_VAL=vr['official_ordinary_rows'],
      official_monthly_files=tr['official_source_files']+vr['official_source_files'],
      original_files_unchanged=check,TEST_numeric_rows_parsed=0,new_market_requests=0,
      model_training_runs=0,MAE_computations=0,new_dataset_files=0,raw_values_changed=0))
    align=[];uncertainties=[]
    for c in ac.TARGETS:
        t=train.loc[train.variety.eq(c)];v=val.loc[val.variety.eq(c)]
        n=int(t['comparable_contracts_+0'].gt(0).sum());nv=int(v['comparable_contracts_+0'].gt(0).sum())
        if n+nv==0:status='INSUFFICIENT_OFFICIAL_EVIDENCE'
        elif c=='SN':status='SOURCE_IDENTITY_MISMATCH'
        elif c in ['AU','AG','FU']:status='SESSION_AMBIGUOUS'
        else:status='SESSION_ALIGNED_EVIDENCED'
        local='SESSION_SHIFT_EVIDENCED' if c in ['AU','AG'] else ('SOURCE_IDENTITY_MISMATCH' if c=='SN' else status)
        row=dict(asset_id='FUT_'+c,variety=c,source_id='sina_continuous_exploratory_v1',status=status,
                 evidence_scope='MATCHED_SUBSET_ONLY_NOT_WHOLE_HISTORY_OR_AVAILABLE_AT',local_event_status=local,
                 TRAIN_rows=len(t),TRAIN_first_date=t.date.min(),TRAIN_last_date=t.date.max(),TRAIN_comparable_dates=n,
                 TRAIN_same_date_all7=int(t['full7_contract_matches_+0'].gt(0).sum()),
                 TRAIN_next_date_all7=int(t['full7_contract_matches_+1'].gt(0).sum()),
                 TRAIN_unique_active_shift=int(t.unique_active_shift.sum()),TRAIN_multi_date_ambiguous=int(t.ambiguous_dates.sum()),
                 TRAIN_no_full7_match=int(t.no_full7_match.sum()),TRAIN_zero_volume=int(t.zero_volume.sum()),
                 TRAIN_flat_ohlc=int(t.flat_ohlc.sum()),TRAIN_off_calendar=int(t.off_reference_calendar.sum()),
                 VAL_rows=len(v),VAL_comparable_dates=nv,VAL_same_date_all7=int(v['full7_contract_matches_+0'].gt(0).sum()),
                 VAL_next_date_all7=int(v['full7_contract_matches_+1'].gt(0).sum()),
                 vendor_date_definition_verified=False,original_official_HTTP_business_date_verified=False,
                 continuous_roll_verified=False,asset_auto_dropped=False,approved_mapping_count=0)
        align.append(row)
        uncertainties.append(dict(asset_id='FUT_'+c,review_start='2013-07-05' if c in ['AU','AG'] else t.date.min(),
           review_end=v.date.max(),status=status,
           uncertainty=('MATERIAL_POST_LAUNCH_SESSION_RISK_WITHOUT_INDEPENDENT_SAFE_END_BOUNDARY' if c in ['AU','AG'] else
                        'LOW_ACTIVITY_AND_REPEATED_BARS_CANNOT_ESTABLISH_DATE_SHIFT' if c=='FU' else
                        'FIRST_OBSERVATION_IDENTITY_MISMATCH_PLUS_UNVERIFIED_REMAINING_VENDOR_CONSTRUCTION' if c=='SN' else
                        'NO_EXISTING_DCE_INDEPENDENT_CONTRACT_COMPARISON_NOT_PROOF_OF_ERROR' if n+nv==0 else
                        'MATCHED_SUBSET_ALIGNED_NONMATCHES_AND_HISTORICAL_AVAILABLE_AT_UNVERIFIED'),
           period_semantics='REVIEW_EXTENT_NOT_ASSERTION_EVERY_BAR_IS_WRONG_OR_AN_APPROVED_PERIOD_QUARANTINE',
           automatic_period_quarantine=False,known_minimum_rules=sum(r['asset_id']=='FUT_'+c for r in freeze['minimum_quarantine_rules']),
           release_effect='BLOCKED_BY_UNBOUNDED_KNOWN_SESSION_RISK' if c in ['AU','AG'] else 'RETAIN_EVIDENCE_STATUS_NO_AUTOMATIC_ASSET_REMOVAL',
           evidence='train_year_matching_summary.csv;val_year_matching_summary.csv;HISTORICAL_NIGHT_SESSION_RULES.md'))
    pd.DataFrame(align).to_csv(out/'futures_session_alignment.csv',index=False)
    pd.DataFrame(uncertainties).to_csv(out/'session_uncertainty_register.csv',index=False)
    pd.DataFrame(columns=['vendor_date','exchange_trading_date','source_id','contract_evidence','rule_version','verification_status']).to_csv(out/'session_mapping_records.csv',index=False)
    window=train.loc[train.date.between('2013-05-01','2013-08-31') & train.variety.isin(['AU','AG','CU','AL','ZN','RB'])]
    transition=summarize(window,['variety','period'])
    transition.to_csv(out/'transition_and_control_summary.csv',index=False)
    extended=train.loc[train.variety.isin(['AU','AG','CU','AL','ZN','RB'])].copy()
    extended['audit_period']=extended.date.str[:4]
    in2013=extended.date.str.startswith('2013')
    extended.loc[in2013 & extended.date.lt('2013-07-05'),'audit_period']='2013_PRE_JUL05'
    extended.loc[in2013 & extended.date.between('2013-07-05','2013-08-31'),'audit_period']='2013_JUL05_AUG31'
    extended.loc[in2013 & extended.date.ge('2013-09-01'),'audit_period']='2013_SEP_DEC_REMAINDER'
    summarize(extended,['variety','audit_period']).to_csv(out/'train_extended_period_summary.csv',index=False)
    original=json.loads((root/ac.DATA_REL/'scaler_params.json').read_text())
    snparams=json.loads((root/'experiments/phase_e_sourceqa/candidate_sn_only_scaler_params.json').read_text())['parameters']
    scaler=[]
    for p,s in zip(original,snparams):
        scaler.append(dict(economic_type=p['economic_type'],feature_name=p['feature_name'],feature_index=p['feature_index'],
          V1_valid_train_count=p['valid_train_count'],V1_mean=p['mean_train'],V1_std=p['std_train'],V1_scale=p['effective_scale'],
          V1_fresh_replay_fit_exact=True,SN_only_valid_train_count=s['valid_train_count'],SN_only_mean=s['mean_train'],
          SN_only_std=s['std_train'],SN_only_scale=s['effective_scale'],SN_only_statistics_changed=p!=s,
          SN_only_transform_changed=any(p[k]!=s[k] for k in ['mean_train','effective_scale']),
          SN_only_evidence='PREVIOUS_PHASE_SOURCEQA_COUNTERFACTUAL_SHA_VERIFIED_NOT_RECOMPUTED_AS_A_NEW_VERSION',
          full_candidate_fit_status='BLOCKED_NO_COMPLETE_TEMPORAL_PROTOCOL',full_candidate_parameters_reused=False,
          VAL_used_for_fit=False,TEST_used_for_fit=False))
    pd.DataFrame(scaler).to_csv(out/'scaler_rebuild_audit.csv',index=False)
    versions=[dict(role='ORIGINAL_V1_READ_ONLY_HOLD',artifact=ac.DATA_REL,exists=True,changed_raw_values=0,
                  changed_input_masks=0,changed_feature_masks=0,changed_label_masks=0,changed_scaler_groups=0,
                  changed_origins=0,TRAIN_origins=2728,VAL_origins=483,TRAIN_labels=json.dumps(replay['V1_valid_labels']['train']),
                  VAL_labels=json.dumps(replay['V1_valid_labels']['val']),training_allowed=False),
              dict(role='SN_ONLY_PREVIOUS_MEMORY_COUNTERFACTUAL_NOT_VERSION',artifact='experiments/phase_e_sourceqa/impact_results.json',exists=True,
                  changed_raw_values=0,changed_input_masks=sn['changed_input_masks'],changed_feature_masks=sn['changed_feature_masks'],
                  changed_label_masks=sn['changed_label_masks'],changed_scaler_groups=sn['changed_scaler_groups'],changed_origins=0,
                  TRAIN_origins=2728,VAL_origins=483,TRAIN_labels=json.dumps(sn['valid_label_counts_after_SN_only_simulation']['train']),
                  VAL_labels=json.dumps(sn['valid_label_counts_after_SN_only_simulation']['val']),training_allowed=False),
              dict(role='COMPLETE_V1_1_NOT_CREATED',artifact='NONE',exists=False,changed_raw_values=0,
                  changed_input_masks='NOT_VERIFIED',changed_feature_masks='NOT_VERIFIED',changed_label_masks='NOT_VERIFIED',
                  changed_scaler_groups='NOT_VERIFIED',changed_origins='NOT_VERIFIED',TRAIN_origins='NOT_VERIFIED',VAL_origins='NOT_VERIFIED',
                  TRAIN_labels='NOT_VERIFIED',VAL_labels='NOT_VERIFIED',training_allowed=False)]
    pd.DataFrame(versions).to_csv(out/'dataset_version_diff.csv',index=False)
    checks=dict(TRADING_SESSION_CAUSALITY_TEST=dict(pass_=False,scope='V1 fails conservative historical-session risk test; no complete candidate',
                evidence='v1_replay_checks.json:risk_cases;input_label_temporal_consistency.csv'),
                MINIMUM_FIVE_RULE_MASK_CONTRACT_TEST=dict(pass_=True,scope='Boolean diagnostic only, no new full feature/scaler/candidate'),
                INPUT_PREFIX_CAUSALITY_TEST=dict(pass_=True,scope='Fresh V1 source replay, fixed TRAIN scaler'),
                TRAIN_ONLY_SCALER_TEST=dict(pass_=True,scope='Fresh V1 fit; VAL feature perturbation does not change55 parameters'),
                HORIZON_AND_SPLIT_BOUNDARY_TEST=dict(pass_=True,scope='Every existing TRAIN/VAL origin and1/5/10/20 horizon'),
                PADDING_TEST=dict(pass_=True,scope='Every V1 TRAIN/VAL sample, false masks,0 placeholders,-1 provenance'),
                RAW_FINGERPRINT_TEST=dict(pass_=True,scope=f'All {check["protected_file_count"]} protected original files'),
                V1_TWICE_DETERMINISTIC_REPLAY_TEST=dict(pass_=True,scope='25 arrays bitwise identical to oldV1; not V1.1'),
                TEST_NUMERIC_ISOLATION_TEST=dict(pass_=True,scope='All quote gates before numeric parse; poisoned TEST CSV regressions'),
                COMPLETE_CANDIDATE_MASK_AND_REPRODUCIBILITY_TEST=dict(status='BLOCKED',scope='No full revised protocol; do not fabricate candidate'))
    ac.write_json(out/'causality_tests.json',dict(checks=checks,independent_acceptance=statuses,TEST_numeric_rows_parsed=0,
                V1_replay_evidence='v1_replay_checks.json',no_new_dataset_generated=True))
    (out/'sn_quarantine_integration.md').write_text('''# SN规则的集成边界

SN2015-03-27整条观测的隔离依据沿用已封存的九个同日普通SN合约联合证据；不使用119270或SN1507替价。原始值、标准化、V1及旧Scaler均未修改。

本轮把SN规则与AU/AG最小四条时间风险规则一起记录在session_quarantine_rules.csv，先冻结再读取VAL。minimum boolean mask诊断已验证SN观测不能参与任何1/5/10/20周期的有效性窗口、锚点或端点；修改后续价格不影响该规则。该诊断没有发布新特征、标签或Scaler。

上一轮真实SN-only完整内存重建的作用仍然单列：input mask1、feature mask15、label mask4、Scaler统计16组（15组变换变化），局部40个TRAIN窗口；池化重fit会改变全部2728个TRAIN和483个VAL窗口的部分标准化特征。原始改变0。其25数组两次SHA一致，但它没有处理完整AU/AG时期，因此不能当V1.1。本轮重新SHA验证其证据，不把旧结果写成新的完整重建。

完整新协议尚未形成，所以SN_QUARANTINE_APPLIED=false针对获准发布的数据版本。整条观测隔离必须在未来完整构建时先于滚动特征和标签应用，重新fit全部55组TRAIN参数，再独立复核所有依赖mask；不能直接修改旧NPY或复用旧Scaler。本轮未解除RESEARCH_HOLD。
''')
    section=[]
    for _,r in transition.iterrows():
        section.append(f'|{r.variety}|{r.period}|{r.vendor_rows}|{r["comparable_+0"]}|{r["full7_matches_+0"]}|{r["comparable_+1"]}|{r["full7_matches_+1"]}|{r.unique_match_rows}|{r.ambiguous_date_rows}|{r.no_full7_match_rows}|')
    (out/'au_ag_session_transition_report.md').write_text('''# AU/AG夜盘上线前后及对照

冻结区间2013-05-01–08-31；d±2个冻结参考session，七字段精确联合指纹，无容差、无单位拟合。价格四字段、零成交、平坦OHLC、重复joint bars另列，不推定换月或真实交易收益。计量口径未独立认证时仅报告原生数值指纹，非匹配不判错。

|资产|相对AU/AG上线时期|vendor总数|同日可比|同日all7|+1可比|+1 all7|唯一日期及合约|多日期歧义|无all7|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(section)+'''

AU/AG上线后各41条中同日all7及OHLC4都为0，仅07-08有+1 all7，其余40条无±2 all7。上线前AG42/43、AU38/43有同日all7。不能把这1/41推广成统一提前一天；也不能把40条非匹配均认定造假或未来记录。

扩大至各TRAIN年度已实际执行，详见train_year_matching_summary.csv；2013剩余、2014、2015、后续各年均保留分母。AU2013为同日110/240、+1为1/240，2014同日143/245、+1为0；AG分别92/240、1/240及144/246、0。后续大量同日匹配反驳“整段统一+1”，但不能给最初非匹配片段提供独立可靠的结束边界。CU/AL/ZN于2013年底、RB于2014年底才上线夜盘，按同一7月切点作对照，不按结果选对照。

TRAIN中FU另有6条active_nonflat唯一负偏移指纹；还有平坦、低量及多日期匹配。active_nonflat仅表示volume>0、OI>0且有价差，不等于足够流动性。成交量原值和重复标志保留于逐条文件；不能把它们全算作未来泄漏。正偏移FU的平坦薄量证据不支持日期映射，未扩大隔离范围。

VAL仅按已冻结的五偏移规则独立检查，AU同日353/484，AG347/484，两者+1 all7为0；25资产没有active_nonflat唯一偏移记录。没有用VAL表现、分布或结果修改规则；这些结果也不能回证2013日期或历史PIT。TEST没有数值读取。

官方原始SDK月档与standardized在相同交易所/普通合约/所选25范围逐月唯一行对照，所有日期与七字段一致。SHFE raw响应混有INE部分品种，按原配置交易所分区后比较，不把正常排除项当日期解析错误。SDK原始日期却来自请求参数而非原HTTP业务日期，因此H3上游档案问题仍未排除。没有新请求行情。

裁决：局部跨日期数值身份和首夜收盘价证据支持重大时间风险；历史制度已调查，供应商完整会话定义与版本边界仍未核实。采用最小5条整观测隔离规则契约及未知范围登记，不批准任何日期映射、不执行不完整版本构建，保留训练阻断。
''')
    gates='\n'.join(f'|{k}|{str(v).lower() if isinstance(v,bool) else v}|' for k,v in statuses.items())
    (out/'REPORT.md').write_text(f'''# Phase E-SessionQA：交易session归属与数据发布门禁

**本轮实际审计完成；AU/AG受影响范围仍无法可靠界定，V1.1未生成，EXPLORATORY_TRAIN_READY=false。** 原V1及RESEARCH_HOLD原样保留。不能只删除已发现的两条记录就放行训练。

实际处理TRAIN供应商{tr['vendor_rows']:,}条、VAL {vr['vendor_rows']:,}条；按±2个参考session比对现有官方普通合约TRAIN {tr['official_ordinary_rows']:,}条、VAL {vr['official_ordinary_rows']:,}条，复核400份现有月档的raw SDK和standardized日期/数值。25商品全部逐品种出表，17个有官方比较，8个DCE没有；不把无来源等同于错误。历史规则、匹配规则和最小隔离契约先于VAL冻结，见frozen_matching_policy.json与frozen_disposition.json。本轮没有新行情采集、TEST数值、模型、MAE、提交或推送。

## 关键发现

2013-07-05当晚AU/AG启动夜盘；当时官方站转载报道记载首夜于次日02:30结束，黄金收盘245.45、白银3843，与供应商07-05 close吻合。详见[历史证据说明](HISTORICAL_NIGHT_SESSION_RULES.md)及其中官方原链接。这支持新增07-05 20:00后信息的重大疑点，但不独立认证完整七字段会话或供应商全历史构造。

07-08两条记录all7分别唯一匹配次日AU1312/AG1312，二者非平坦、大量成交。同日不能匹配。但初始上线后每个资产41条中，仅此1条+1 all7，同日all7和OHLC4均为0，其他40条±2无all7匹配。上线前43条中AG42、AU38条同日all7。该结果足以拒绝“整段直接+1”的修法，不足以证明所有非匹配都是未来数据，也不足以确定一个可靠恢复同日可用的历史边界。2014及后续TRAIN统计和CU/AL/ZN/RB对照均已执行，详见au_ag_session_transition_report.md及train_year_matching_summary.csv。

H2当时制度得到支持；H1有局部支持而非全时期证明；H3标准化层日期平移未发现，但AKShare get_shfe_daily把请求日赋给date，本地raw不是原HTTP档案业务日期，所以上游日期仍不确定；H4缺少历史vintage和供应商构造文档，NOT_VERIFIED。没有将2026规则当作2013原版。

所有匹配保留原始精度和单位，没有从数据拟合倍数或容差。七字段是数值指纹，OHLC4只是诊断；计量、历史版本及连续主合约经济身份未由相等数值自动认证。FU低量/平坦/多日重复与高信息量AU/AG证据分开；FU6条活跃非平坦负偏移也不自动算未来污染。OI旧RO问题不做无文档别名合并。其他15个品种的同日匹配只证明匹配子集，不认证整段PIT。

## 时间一致性与隔离边界

固定20:00不变，没有全局±1映射。session_mapping_records.csv只有schema、0条映射。session_quarantine_rules.csv记录SN2015-03-27及AU/AG2013-07-05、07-08，共5个整观测最小规则。必须在任何未来完整版本中先于报价可用性、滚动特征、h+1标签质量窗及TRAIN统计应用；原始值不变。该规则契约已用实际V1布尔mask和独立窗口计算验证：四个周期均无这些观测残留锚点或端点，诊断输出不是数据集。

TRADING_SESSION_CAUSALITY_TEST对旧V1为false：仅靠vendor日19:00假设会通过时钟算术，但4条AU/AG记录违反独立历史证据支持的保守可用性边界，仍在原V1输入/锚点内。最小契约可拒绝已知点，不能覆盖尚未界定的时期；因此完整SESSION_QUARANTINE_CAUSAL仍NOT_VERIFIED，未执行假装完整的隔离/重fit。未知范围登记是复核范围，不能解读为全期皆错或已批准隔离期。

## 重现、Scaler和边界复核

本轮从冻结原始来源独立重建原V1两次，25个数组的.npy SHA256均与旧V1相同，55组TRAIN Scaler参数也逐项精确相同。还实际扰动2021-06-30后的输入检查此前特征前缀，扰动全部VAL特征确认TRAIN参数不变；独立重算全部1/5/10/20标签mask和端点公式，检查TRAIN/VAL、VAL/TEST边界、全部3211个监督origin的padding。上述软件/算术检查通过，不替代真实session因果性。逐项证据在v1_replay_checks.json和causality_tests.json。

三种角色严格分开：原V1只读；此前SN-only内存完整反事实的已封存证据只读复用；完整新候选不存在。本轮V1重现不是新协议发布，SN-only两次复现不是V1.1。scaler_rebuild_audit.csv列55组旧值、SN-only值与完整候选BLOCKED。dataset_version_diff.csv不填造未构建版本的差异和样本数。

原V1有TRAIN2728、VAL483个origin。1/5/10/20有效标签分别为TRAIN57889/57249/56504/55017，VAL12067/11951/11806/11520。此前SN-only TRAIN为57888/57248/56503/55016、VAL不变。本轮最小5规则的纯布尔敏感性统计TRAIN57882/57234/56479/54972、VAL不变；它没有特征或Scaler产物，不能当获准训练集或V1.1数量。新版本的origin和标签数均NOT_VERIFIED。

{test_count}项测试通过，含{new_test_count}项SessionQA测试；80条警告均为原collect.py的NumPy timedelta弃用警告。新增实现修正了本轮审计器最初把SHFE响应内INE记录和SHFE标准化分区比较的问题，交易所范围按原配置固定；专门回归测试覆盖，不修改原名单/协议。首次执行的诊断证据封存在evidence/train_attempt1_exchange_scope.*，实际正式日期比较差异0。原{check['protected_file_count']:,}个文件SHA复核无差异。所有新产物和源码快照见delivery_manifest.json，来源、规则、原环境及历史材料见provenance.json。

## 独立验收

|字段|结果|
|---|---|
{gates}

HORIZON_BOUNDARY_PASS、PREFIX_CAUSALITY_PASS仅指实际重跑的旧V1软件与切分检查；TRAIN_SCALER_REFITTED和DATASET_V1_1_REPRODUCIBLE针对完整新候选，因不存在而BLOCKED。SN_QUARANTINE_APPLIED=false针对发布版本，不否认已验证的规则契约和此前SN-only反事实。HISTORICAL_NIGHT_SESSION_RULES_AUDITED=true只表示已审阅当时材料，OTHER_FUTURES_SESSION_AUDITED=true表示已按证据覆盖逐资产审核，均不等于全历史认证。

## 对16项问题的逐项回答

1. 07-08与夜盘上线在时间和内容上有关联，制度和首夜报道为新旁证；不能唯一归因于新浪提前标记，H3上游档案与H4版本仍未排除。
2. 未证明系统性提前一天。每个AU/AG上线后初始41条仅1条+1 all7，其他40条不支持该映射。
3. 强+1七字段证据局限AU/AG2013-07-08；07-05另有首夜close时间风险。2013–2022及VAL2023–2024实际统计均交付，不能据此认定其他年所有记录错误。
4. 存在无法确定供应商会话日期的记录，特别是初始上线后非匹配片段及各年残留非匹配。DCE无独立比较另列，不是已确认错位。
5. 最终采用不改raw、不批准映射、保留20:00和发布HOLD的决策。仅冻结最小整观测隔离契约供将来完整协议整合。
6. 已知4条AU/AG风险必须隔离；更广时期的安全边界尚无独立证据，不能随意指定结束日期或隔离全部25商品。
7. SN规则已纳入最小规则登记与诊断，但没有完整修订版本可供正式应用。SN_QUARANTINE_APPLIED=false。
8. 原V1全部特征和标签为复现审计真实重建两遍；完整新协议特征/标签没有构建。此前SN-only完整重建保留其独立角色。
9. V1审计重现重新fit55组且与旧参数一致；此前SN-only重fit影响16统计组。完整新候选TRAIN Scaler BLOCKED，没有复用旧值冒充新fit。
10. V1.1未生成，目标目录不存在。
11. V1.1 TRAIN origin数量NOT_VERIFIED，不借用2728冒充。
12. V1.1 VAL origin数量NOT_VERIFIED，不借用483冒充。
13. V1.1四周期标签数量NOT_VERIFIED。上文只列有真实证据的旧V1、SN-only和明确标注的布尔敏感性数量。
14. 原V1仍保留已知重大时间风险和SN异常，不能授予无未来污染结论；没有新版本可声称已修复全期。
15. 不能解除探索性训练禁令；EXPLORATORY_TRAIN_READY=false，原RESEARCH_HOLD原样不变。
16. 即使后续解决时间风险，历史PIT/供应商修订、连续换月、固定股票cohort及VAL参与选池、未复权公司行动、跨市场真实发布时间与重叠标签边界仍限制确认性论文，PIT/ROLL/CONFIRMATORY均false。

## 解除阻断所需的最小外部证据与后续顺序

1. 取得Sina AU0/AG0在2013夜盘上线及其后变更时期的历史日线会话定义、日期标签规则和版本边界说明；至少解释07-05、07-08及非匹配片段，不能只给当前接口函数签名。
2. 对应AU1312/AG1312的2013-07-05/08/09原SHFE日档（含内生业务日期、发布/修订元数据）或同等可核验的会话记录。现有SDK请求日不能替代这项证据。无需为此重复全量采集或反复试受阻接口。
3. 用独立证据确定受影响时期的映射或合理隔离范围，先冻结TRAIN规则，再检查VAL；DCE及FU保留各自证据等级，不强求全部逐合约换月证明。
4. 仅在完整规则成立后，从原始来源重建所有特征/标签、重fit55组TRAIN Scaler、独立验证所有依赖mask和session可用时点，两次完整构建SHA一致后再建立V1.1目录。

本阶段于审计交付处停止，未运行任何模型或MAE。后续需要的是边界明确的外部证据，不是放宽门禁。
''')
    # Freeze SDK function evidence, not any market payload or auth material.
    from akshare.futures.futures_daily_bar import get_shfe_daily
    (out/'evidence/akshare_get_shfe_daily.txt').write_text(inspect.getsource(get_shfe_daily))
    codefiles=['src/triweave/sessionqa_match.py','src/triweave/sessionqa_checks.py','src/triweave/sessionqa_report.py','tests/test_sessionqa.py']
    codeout=out/'code_snapshot';codeout.mkdir(exist_ok=True)
    for name in codefiles:shutil.copyfile(root/name,codeout/Path(name).name)
    docs=['https://www.shfe.com.cn/content/lxjy/main.html','https://www.shfe.com.cn/content/lxjy/mtbd6-7.html',
          'https://www.shfe.com.cn/content/lxjy/mtbd6-6.html','https://www.shfe.com.cn/docview/docview_35218417.htm',
          'https://www.shfe.com.cn/content/lxjy/f2.html','https://www.shfe.com.cn/content/lxjy/f18-3.html']
    old=json.loads((out/'preservation_before.json').read_text())
    provenance=dict(phase='E-SessionQA',created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
      task_file='/home/yangxiaotong/.codex/attachments/e2f7bd2c-31d5-44c5-adda-7320a2f37871/已粘贴的文本.txt',
      task_sha256=sha256_file('/home/yangxiaotong/.codex/attachments/e2f7bd2c-31d5-44c5-adda-7320a2f37871/已粘贴的文本.txt'),
      environment='environment.json',before_inventory_sha256=sha256_file(out/'preservation_before.json'),
      previous_manifests={k:v for k,v in old['files'].items() if 'manifest' in k and k.endswith('.json')},
      source_references=['train_matching_results.json:sources','val_matching_results.json:sources',
                         'v1_replay_checks.json:quote_read_log','../phase_e_sourceqa/candidate_reproducibility.json'],
      documentation_urls=docs,documentation_captures={str(p.relative_to(root)):sha256_file(p) for p in (out/'evidence').glob('web_history_batch*.json')},
      code={k:sha256_file(root/k) for k in codefiles},
      calendar_sha256=sha256_file(root/'data/manifests/baseline15y/reference_trading_calendar.csv'),
      frozen_policy_sha256=sha256_file(out/'frozen_matching_policy.json'),frozen_disposition_sha256=sha256_file(out/'frozen_disposition.json'),
      TEST_numeric_rows_parsed=0,new_market_requests=0,documentation_web_requests=True,model_training=False,MAE=False,
      dataset_created=False,read_only_old_data_and_code=True,
      execution_notes=['First exploratory raw/std check stopped on out-of-exchange INE rows in shared SHFE response. Scope fixed to original exchange configuration and regression added; no source was modified.',
                       'All7 matches are forensic native-number fingerprints; units and historical PIT not inferred from equality.',
                       'Repeated-bar diagnostics may inspect the whole TRAIN split but are not historical features or future-based asset selection.',
                       'VAL matching ran only after frozen_disposition.json; it did not change any rule or release decision.'],
      commands=['.venv/bin/python -m triweave.sessionqa_match --split TRAIN',
                '.venv/bin/python -m triweave.sessionqa_match --split VAL',
                '.venv/bin/python -m triweave.sessionqa_checks',
                '.venv/bin/python -m pytest -q',
                '.venv/bin/python -m triweave.sessionqa_report --verify'],
      replay_notice='Matching/replay builders are append-only and refuse sealed existing results. Verification is read-only. New evidence requires a separately versioned audit directory, not deletion of this delivery.')
    ac.write_json(out/'provenance.json',provenance)
    archive=root/'data/manifests/phase_e_sessionqa';archive.mkdir(exist_ok=True)
    ac.write_json(archive/'provenance.json',provenance)
    files=[p for p in out.rglob('*') if p.is_file() and p.name!='delivery_manifest.json']
    files += [root/name for name in codefiles]+[archive/'provenance.json']
    records={str(p.relative_to(root)):dict(sha256=sha256_file(p),bytes=p.stat().st_size) for p in sorted(files)}
    ac.write_json(out/'delivery_manifest.json',dict(phase='E-SessionQA',self_excluded=True,
      fingerprint_rule='All delivered artifacts and new source/test files are covered; this self-referential index is identified by its external SHA256.',
      files=records,total_indexed_bytes=sum(r['bytes'] for r in records.values()),dataset_V1_1_bytes=0))
    verify(root)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verify',action='store_true')
    args=p.parse_args()
    (verify if args.verify else build)(Path.cwd())
