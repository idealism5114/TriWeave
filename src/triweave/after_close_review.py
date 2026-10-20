"""Independent Phase E integrity review, alignment view and delivery reports."""
from __future__ import annotations
import argparse
import datetime
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from triweave.after_close import (DATA_REL, REPORT_REL, PROTOCOL, KINDS, TARGETS,
                                 SPLITS, HORIZONS, AfterCloseDataset, write_json)
from triweave.universe import sha256_file


def alignment_frame(directory, reference_date):
    """Decode one date into the requested auditable, typed alignment schema."""
    ds=AfterCloseDataset(directory,'TRAIN');a=ds.arrays
    ids=np.flatnonzero(a['reference_dates']==reference_date)
    if len(ids)!=1:raise ValueError('TRAIN/VAL reference date required')
    i=int(ids[0]);sources=json.loads((Path(directory)/'source_manifest.json').read_text())
    rows=[]
    for j,spec in enumerate(ds.meta):
        si=int(a['source_index'][i,j]);day=int(a['observation_day'][i,j]);valid=bool(a['input_valid_mask'][i,j])
        rows.append(dict(reference_date=reference_date,asset_id=spec['asset_id'],market=spec['market'],
                         economic_type=spec['economic_type'],observation_date=str(np.datetime64(day,'D')) if day>=0 else None,
                         trading_session='SOURCE_TRADING_DATE' if spec['market']=='future' else 'SOURCE_OBSERVATION_DATE',
                         source_id=sources[si]['source_id'] if si>=0 else None,
                         source_manifest=sources[si]['manifest'] if si>=0 else None,
                         value=float(a['aligned_values'][i,j]) if valid else None,
                         unit=spec['unit'],availability_mask=valid,available_at_utc_seconds=int(a['available_at'][i,j]),
                         decision_time_utc_seconds=int(a['decision_time'][i]),
                         available_at_status='AVAILABLE_AT_ASSUMED',quote_age_days=int(a['age_days'][i,j]),
                         new_valid_observation=bool(a['new_observation'][i,j])))
    return pd.DataFrame(rows)


def longest_false(x):
    best=n=0
    for b in x:
        n=0 if b else n+1;best=max(best,n)
    return best


def independent_review(root):
    root=Path(root);dest=root/DATA_REL;out=root/REPORT_REL
    ds=AfterCloseDataset(dest,'TRAIN');a=ds.arrays;dates=a['reference_dates'];assets=ds.meta
    manifest=ds.manifest;checks={};evidence={}
    def verify(name,ok):
        checks[name]=bool(ok)
        if not ok:raise AssertionError(name)
    verify('all_stored_file_hashes_verified',all(sha256_file(dest/p)==info['sha256'] for p,info in manifest['files'].items()))
    # Independent cumulative invalid-count algorithm audits true AND false masks.
    q=a['input_valid_mask'][:,-25:]
    c=np.concatenate([np.zeros((1,25),np.int64),np.cumsum(~q,axis=0,dtype=np.int64)],axis=0)
    for split in ('TRAIN','VAL'):
        pre=split.lower();lo,hi=SPLITS[split];expected=np.zeros((len(dates),4,25),bool)
        for k,h in enumerate(HORIZONS):
            for i in range(len(dates)-h):
                if lo<=dates[i]<=dates[i+h]<=hi:
                    expected[i,k]=(c[i+h+1]-c[i])==0
        origins=np.flatnonzero(expected.any((1,2)))
        verify(pre+'_label_eligibility_exact_no_excess_false',np.array_equal(origins,a[pre+'_origin_indices'])
               and np.array_equal(expected[origins],a[pre+'_label_valid_mask']))
        verify(pre+'_last20_h20_always_invalid',not expected[np.flatnonzero((dates>=lo)&(dates<=hi))[-20:],3].any())
    # Recompute group support/mean/std independently from serialized daily panel.
    params=json.loads((dest/'scaler_params.json').read_text());train=dates<=SPLITS['TRAIN'][1]
    raw=a['unscaled_features'];mask=a['unscaled_feature_valid_mask']
    for p in params:
        js=[j for j,s in enumerate(assets) if s['economic_type']==p['economic_type']];k=p['feature_index']
        values=raw[train][:,js,k][mask[train][:,js,k]]
        assert len(values)==p['valid_train_count']
        assert np.isclose(values.mean(),p['mean_train'],rtol=1e-13,atol=1e-13)
        assert np.isclose(values.std(ddof=0),p['std_train'],rtol=1e-13,atol=1e-13)
    verify('saved_unique_daily_train_scaler_stats_independent',True)
    rows=[];anomalies=[]
    for split,(lo,hi) in list(SPLITS.items())[:2]:
        time=(dates>=lo)&(dates<=hi)
        for j,s in enumerate(assets):
            m=a['input_valid_mask'][time,j];valid_dates=dates[time][m]
            rows.append(dict(split=split,asset_id=s['asset_id'],economic_type=s['economic_type'],
                             first_valid_date=str(valid_dates[0]) if len(valid_dates) else None,
                             last_valid_date=str(valid_dates[-1]) if len(valid_dates) else None,
                             valid_quote_dates=int(m.sum()),missing_reference_dates=int((~m).sum()),
                             longest_missing_gap_reference_sessions=longest_false(m),
                             new_valid_observations=int(a['new_observation'][time,j].sum()),
                             historical_quote_reuse_dates=int((m&~a['new_observation'][time,j]).sum())))
            if s['economic_type'] not in (KINDS[0],KINDS[3]):continue
            positions=np.flatnonzero(time)
            for i in positions[1:]:
                if not (a['input_valid_mask'][i,j] and a['input_valid_mask'][i-1,j]):continue
                change=a['aligned_values'][i,j]/a['aligned_values'][i-1,j]-1
                if abs(change)>.2:
                    si=int(a['source_index'][i,j]);sources=json.loads((dest/'source_manifest.json').read_text())
                    anomalies.append(dict(split=split,date=dates[i],asset_id=s['asset_id'],economic_type=s['economic_type'],
                                          previous_reference_date=dates[i-1],previous_price=float(a['aligned_values'][i-1,j]),
                                          current_price=float(a['aligned_values'][i,j]),decimal_price_change=float(change),
                                          source_manifest=sources[si]['manifest'],
                                          interpretation='Unadjusted corporate-action/source risk; not verified' if j<263 else 'Roll/source/market change uncertain; not proof of roll',
                                          action='FLAG_ONLY_PRESERVED_NOT_CLIPPED'))
    pd.DataFrame(rows).to_csv(out/'per_asset_quote_coverage.csv',index=False)
    pd.DataFrame(anomalies).to_csv(out/'price_jump_flags_full_train_val.csv',index=False)
    listing=json.loads((root/'data/manifests/listing_dates_verified.json').read_text())
    listing_results={}
    for j,s in enumerate(assets):
        if s['asset_id'] in listing:
            first=listing[s['asset_id']]['date'];before=dates<first
            listing_results[s['asset_id']]={'listing_date':first,'valid_prelisting_quotes':int(a['input_valid_mask'][before,j].sum())}
    verify('known_listing_boundaries_no_manufactured_quotes',all(v['valid_prelisting_quotes']==0 for v in listing_results.values()))
    evidence['verified_listing_checks']=listing_results
    evidence['price_jump_flags_all_train_val']=len(anomalies)
    evidence['stock_price_jump_flags']=sum(r['economic_type']==KINDS[0] for r in anomalies)
    evidence['futures_price_jump_flags']=sum(r['economic_type']==KINDS[3] for r in anomalies)
    inherited=pd.read_csv(out/'inherited_continuous_quote_quality_issues.csv')
    quality=pd.read_csv(out/'quote_quality_audit.csv')
    verify('selected_future_bad_quotes_match_prior_audit',int(quality.loc[quality.economic_type.eq(KINDS[3]),'bad_quote_rows_including_suspensions'].sum())==len(inherited)==20)
    freeze=json.loads((out/'prebuild_code_freeze.json').read_text())['files']
    unchanged=True
    for p,h in freeze.items():
        if sha256_file(root/p)==h:continue
        if p=='tests/test_after_close.py':
            extension=json.loads((out/'postbuild_test_extension.json').read_text())
            old=(root/extension['original_snapshot']).read_bytes()
            unchanged &= (sha256_file(root/extension['original_snapshot'])==h
                          and (root/p).read_bytes().startswith(old)
                          and sha256_file(root/p)==extension['extended_sha256'])
        else:unchanged=False
    verify('frozen_build_code_protocol_unchanged',unchanged)
    old=json.loads((out/'preservation_before.json').read_text())['files']
    verify('all_prior_13313_files_still_unchanged',all(sha256_file(root/p)==h for p,h in old.items()))
    prior=json.loads((out/'prior_delivery_validation.json').read_text())
    verify('all_three_prior_delivery_manifests_valid',all(not r['mismatches'] for r in prior))
    for date in ('2011-10-10','2023-01-03','2024-12-31'):
        alignment_frame(dest,date).to_csv(out/('alignment_example_'+date+'.csv'),index=False)
    write_json(out/'independent_review.json',{'checks':checks,'evidence':evidence})
    return checks,evidence


def markdown_table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(str(v) for v in r)+' |' for r in rows)+'\n'


def reports(root):
    root=Path(root).resolve();out=root/REPORT_REL;dest=root/DATA_REL
    if (out/'delivery_manifest.json').exists():raise FileExistsError('Audit delivery sealed; no overwrite')
    extra,evidence=independent_review(root)
    real=json.loads((out/'real_data_checks.json').read_text())
    repro=json.loads((out/'reproducibility_audit.json').read_text())
    summary=json.loads((out/'collection_summary.json').read_text())
    manifest=json.loads((dest/'dataset_manifest.json').read_text())
    params=json.loads((out/'scaler_params.json').read_text())
    tests=(out/'tests.txt').read_text()
    test_match=re.search(r'(\d+) passed',tests)
    tests_ok=bool(test_match) and 'failed' not in tests and 'skipped' not in tests
    test_count=int(test_match.group(1)) if test_match else 0
    source_alert=json.loads((out/'source_alert.json').read_text()) if (out/'source_alert.json').exists() else {}
    source_hold=source_alert.get('blocking_for_exploratory_training',False)
    allreal=all(real['checks'].values());allreview=all(extra.values())
    acceptance={
        'TARGET_25_UNIVERSE_PASS':real['checks']['25_target_order'],
        'INPUT_313_ASSETS_PASS':real['checks']['313_asset_order'],
        'THREE_SPLIT_PROTOCOL_PASS':SPLITS=={'TRAIN':('2011-10-01','2022-12-31'),'VAL':('2023-01-01','2024-12-31'),'TEST':('2025-01-01','2026-09-30')},
        'AFTER_CLOSE_PROTOCOL_PASS':allreal,
        'DECISION_ANCHOR_CONSISTENCY_PASS':real['checks']['train_all_actual_label_dates_values_times'] and real['checks']['val_all_actual_label_dates_values_times'],
        'CROSS_MARKET_ALIGNMENT_PASS':real['checks']['asof_available_by_row_decision'] and real['checks']['no_stock_futures_carry'] and real['checks']['bond_staleness_limited'],
        'FEATURE_TYPE_SEMANTICS_PASS':real['checks']['structural_inapplicability_masked'] and tests_ok,
        'MISSINGNESS_MASK_PASS':real['checks']['invalid_feature_placeholders_zero'] and real['checks']['all_masks_boolean'],
        'TRAIN_ONLY_SCALER_PASS':real['checks']['real_VAL_perturbation_scaler_identical'] and extra['saved_unique_daily_train_scaler_stats_independent'],
        'NORMALIZATION_NUMERIC_PASS':real['checks']['all_numeric_outputs_finite'],
        'RETURN_LABEL_PASS':real['checks']['train_all_actual_label_dates_values_times'] and real['checks']['val_all_actual_label_dates_values_times'],
        'HORIZON_BOUNDARY_PASS':extra['train_label_eligibility_exact_no_excess_false'] and extra['val_label_eligibility_exact_no_excess_false'],
        'PREFIX_CAUSALITY_PASS':real['checks']['real_future_prefix_unscaled_exact'] and real['checks']['real_future_prefix_with_frozen_scaler_exact'],
        'TEST_ISOLATION_PASS':real['checks']['no_test_numeric_ingestion'] and tests_ok,
        'DATASET_REPRODUCIBILITY_PASS':all(v['identical'] for v in repro.values()) and extra['all_stored_file_hashes_verified'],
        'TRAIN_DATASET_READY':allreal and allreview and tests_ok and not source_hold,
        'VAL_DATASET_READY':allreal and allreview and tests_ok,
        'EXPLORATORY_TRAIN_READY':allreal and allreview and tests_ok and not source_hold,
        'HISTORICAL_PIT_VERIFIED':False,
        'FUTURES_ROLL_METHOD_VERIFIED':False,
        'CONFIRMATORY_TRAIN_READY':False}
    audit={'protocol_id':PROTOCOL,'acceptance':acceptance,'statistics':summary,
           'scope':'Exploratory fixed-snapshot mathematical vendor returns under assumed publication times; no model trained',
           'tests':tests.strip().splitlines()[-1],
           'real_label_checks':real['evidence'],'independent_review':evidence,
           'TEST_numeric_rows_parsed':0,'TEST_features_or_labels_generated':False,
           'VAL_informed_cohort_design':True,'DECISION_TIME_ASSUMED':True,
           'DATASET_ARTIFACTS_READABLE_AND_REPRODUCIBLE':allreal and allreview and tests_ok,
           'scientific_training_release_hold':source_alert.get('issue_id') if source_hold else None,
           'hold_evidence_file':'source_alert.json' if source_hold else None}
    write_json(out/'audit_results.json',audit)
    footer='\n\n不可省略的限制：263股票是固定研究cohort，非历史逐时点指数成分股；25商品设计参考过VAL覆盖，VAL不独立于研究设计；Sina连续方法、复权与历史修订未核实；available_at为假设，HISTORICAL_PIT_VERIFIED=false；数学连续序列变化不等于可执行期货收益。没有TEST数值开发、训练、MAE、调参、下载、提交或推送。\n'
    protocol='''# TRIWEAVE_25_AFTER_CLOSE_V1_EXPLORATORY

唯一冻结协议见frozen_protocol.json、decision_time_protocol.json与frozen_feature_registry.json，冻结早于真实数值构建。原RAW15及38商品协议/数据不改。

资产顺序：stocks_263.csv原顺序263只；bond_variables_25.csv原顺序18利率+7财富指数；LU SC CF PG HC J JM FU C MA SR AG OI RM RB M CU AL ZN SN NI A AU I Y。25商品均为输入和目标，无其他辅助商品。38减25排除13条，具体为WH及12条原辅助。

仅TRAIN 2011-10-01—2022-12-31，VAL 2023-01-01—2024-12-31，TEST 2025-01-01—2026-09-30三个划分。无WARMUP。20参考日输入；早期不足左侧补计算占位0且mask=false，节点不删除。VAL可回看TRAIN的过去输入，但标签绝不跨split。

BaoStock adjustflag=3不复权，股/元/百分比不混Eastmoney手；18利率保留percent_per_annum，差分乘100为bps；7财富指数原始index_level仅作相对变化；商品Sina symbol0原生连续close，交易日保留夜盘供应商归属，成交量/OI原生lots统计基准未核实。没有合约修订或拼接修正。

固定参考日历为data/manifests/baseline15y/reference_trading_calendar.csv，SHA256在calendar_manifest.json；非逐时点或独立期货日历验证。所有来源raw/standardized/manifest/参数/抓取时刻与SHA见source_manifest.json及provenance.json。引用2026快照回溯历史，不宣称历史当时同版本可得。

北京时间t日20:00决策。假设股票t18:00、商品t19:00可用，锚P(t)在决策前；债券利率/财富指数于严格后一个参考日08:00/08:30可用。周末债券观测不会向前挪，用原始观测日期和假设发布时间as-of；最新已发布记录即使无效也不回退跳过，最多5自然日旧观测。股票/期货只取同一参考日新报价，不延续。所有为AVAILABLE_AT_ASSUMED；20:00被假定在下一夜盘21:00之前，不是实际逐年发布验证。

唯一日期、有限正OHLC且low<=open/close<=high，股票状态1；利率仅有限，不强制正；财富指数有限正。所有重复日期均无效。非参考日股票/期货保留原层，不插入日历。负活动量仅屏蔽对应活动特征。价格变化绝对值>20%仅标记，不裁剪，不据此推断换月。无bfill、插值、全市场dropna(any)或零价格填充。

类型化特征映射26维公共schema。结构不适用feature_applicable_mask独立于feature_valid_mask。quote_available为可观测缺失指示，报价缺失时0本身有效；其余报价依赖通道要求input_valid。input_valid_mask代表有效as-of报价；label_valid_mask独立按origin/h/c判断。原始不可用数值不参与统计，最终计算占位0的对应mask必须false；有效收益恰为0与无效占位由mask区分。

滚动收益要求h+1连续参考日有效报价；MA用w连续点；w日波动用w个单日变化、需w+1连续价格，ddof=0、不年化；所有依赖只来自当前决策截止前。利差两腿须同一原始观测日期并已发布。债券旧观测可有0日变化，但age/new_observation显式区分重新报价与已知旧值。

TRAIN-only按economic_type+feature_name分55组Z-score，唯一date-asset-feature计数，不在重复X窗口fit；mean/std用float64总体方差，epsilon=1e-8、minimum_train_count=2预先冻结。有效样本不足全split屏蔽，常量/小方差记录并用max(std,epsilon)，不裁剪；VAL只transform。拟合后输出float32，保留float64未缩放面板以审计。全TRAIN拟合的参数为离线训练工件，不宣称TRAIN每个历史时点已知；前缀测试固定训练工件，单独测试原始因果特征。

标签R(t,h,c)=P(t+h,c)/P(t,c)-1，h=1/5/10/20冻结参考session、decimal_return（0.02=2%）。标签类型VENDOR_CONTINUOUS_MATHEMATICAL_RETURN；从t到t+h所有h+1报价有效，起止同split。20d无效不删除1d，单商品无效不删全日期；仅所有100个标签均无效的origin不作为监督样本。重叠20日标签非独立统计样本。

date-first读取器先解析ISO日期前缀，丢弃2025起记录，再进行CSV字段解析与数值转换。压缩包/原始档案仅为SHA按不透明字节读取；TEST行缓冲的数字尾部不解析、不生成记录或统计。毒化TEST数字尾部的自动化测试证明它不进入quote parser。TEST构建/标签入口明确拒绝，未生成TEST数据。

只保存每日面板[3217,313,26]、静态applicability[313,26]、日期/来源/时间/年龄、TRAIN/VAL样本索引、Y和mask，不存重复大X。AfterCloseDataset返回[X=20,313,26;Y=4,25]及所有mask、时点、身份。核心构建文件SHA在dataset_manifest.json；后续科研暂停边车README/RESEARCH_HOLD单独记录在audit_supplement_manifest.json，全部最终文件SHA在交付delivery_manifest.json。第二次从源全流程独立重建数值NPY字节相同。目标目录已有数据则拒绝覆盖。本版本不因源审阅而静默重算，最终训练准入见SOURCE_ALERT.md。
'''
    (out/'DATA_PROTOCOL.md').write_text(protocol+footer)
    (out/'DATA_ALIGNMENT_AUDIT.md').write_text('''# 三市场as-of对齐审计

时间协议先冻结再构建；未沿用统一lag2。20:00决策，股票18:00、商品19:00同日假设；债券收益率下一参考日08:00、财富指数08:30。对每个原始观测计算available_at，按决策时点选择最近已发布观测，债券最多5自然日；周末不后填到周五。股票/期货无沿用，停牌保持无效。夜盘保留源交易日，不将夜盘自然日期擅自重映射。

每日有效节点数见train_coverage.csv/val_coverage.csv，分类型有效数及缺失比例见per_market_coverage.csv，逐资产最长缺失见per_asset_quote_coverage.csv。日历不取各市场日期交集。所选25商品继承20条OHLC无效和75条非参考日记录（原38范围26和102），原始文件未改。

完整对齐schema通过按参考日解码数组提供：reference_dates.npy行索引、asset_manifest.json列身份/market/unit、aligned_values.npy（仅有效值）、observation_day.npy（自1970起日期天数）、available_at.npy（UTC秒）、source_index.npy索引source_manifest.json、input_valid_mask.npy、age_days.npy、new_observation.npy和trading_status.npy。缺失时间/索引用-1，不解释为真实时间。可用triweave.after_close_review.alignment_frame(directory,date)得到reference_date/asset_id/market/observation_date/source_id/value/unit/availability_mask等表；三个真实日期的alignment_example文件已落盘。

label anchor为同日商品P(t)，19:00假设可得；20:00预测，未来价格假设发布时间严格晚于决策，已逐标签验证。所有已用输入available_at<=相应决策。unknown发布时间并未被这些算术检查变成已核实PIT。
'''+footer)
    (out/'FEATURE_ENGINEERING_REPORT.md').write_text('''# 类型化特征审计

统一F=26。feature_registry.json逐条包含经济类型、源字段/原单位、变换、lookback、可用性；asset适用性另存静态[313,26]，真实日期有效性[3217,313,26]。逐资产/类型特征覆盖见per_asset_feature_coverage.csv与per_feature_type_coverage.csv。

股票：过去1/5/10/20日简单变化、5/10/20日MA偏离、5/10/20日非年化波动、5日减20日动量、成交量/额单日相对变化及换手率%。股数与元来自BaoStock，不混旧Eastmoney。未复权，公司行动风险另记price_jump_flags_full_train_val.csv，不能称作总收益。

债券18利率：年化百分比水平、过去1/5/10/20参考日差分*100bps；10Y节点的国债10Y−1Y与10Y−3Y利差；10个AAA节点的相同期限AAA−国债信用利差。同一as-of信息集且两腿原始观测日期相同才有效，绝不做利率百分比价格回报。7财富指数：1/5/10/20日变化、MA趋势、波动和5/20动量；不把异基期原始指数水平混缩放。

期货：相同价格变化/MA/波动/动量，成交量变化、OI变化、(high-low)/close区间；只选25品种。每类型还有quote_available和quote_age_days。价格无效传播到整个需要该日的滚动窗口；活动分母0/负或无效独立屏蔽。quote_available缺失时为有效0指示，此通道例外不要求input_valid为真，不能将其误解释成0真实报价。age仅有效报价可用。

短中期动量由return_5d/return_20d和momentum_5_20表示，不人为复制旧21维。ma_deviation用于财富指数时表示相对趋势。有效且零成交可作为0活动，但交易/报价状态独立保存；不从0分母计算变化。没有未来最高/最低/价格/成交或持仓特征。
'''+footer)
    norm=summary['normalization']
    (out/'NORMALIZATION_REPORT.md').write_text(f'''# TRAIN-only标准化审计

55组，TRAIN唯一date-asset-feature总体均值/标准差、epsilon=1e-8，最少2个有效观测。修改全部实际VAL特征为123456789后TRAIN参数完全不变；独立从落盘未缩放日面板重算count/mean/std一致。重复20日窗口不参与拟合。

实际常量组{norm['constant_groups']}、触发floor组{norm['scale_floor_groups']}、不足支持组{norm['insufficient_groups']}：两个常量均为股票/期货quote_age_days，因无旧价沿用恒0，std=0、scale=1e-8、输出0，状态均保存。其余组未触发floor，无NaN/Inf。

固定审计阈值|z|>20：TRAIN {norm['train_extreme_observations']}处，VAL {norm['val_extreme_observations']}处（每日资产特征观测，非独立样本）。所有计数见normalization_audit.csv，每组最大5例及原始观测、来源、mean/scale见extreme_normalized_examples.csv。TRAIN全局最大绝对值281.144257，VAL75.965523，主要为活动量变化；VAL16例集中成交量变化6、成交额5、换手率5。2024-10-14 sh.600837量/额变化分别133.712396/145.453539，标准化72.696625/75.965523；这只是相邻观察值的算术变化，缺少事件证据不能据此断定原因。

股票2016-06-17 sz.300496原不复权变化-0.738897、z=-26.773621，记录公司行动/来源风险而未自动替换。没有VAL重fit、截尾或阈值优化。当前数值可读取且finite，但重尾与公司行动应在未来单独预注册的稳健性研究中评估，本阶段不改变已冻结方案。

前缀测试：改2021-06-30之后真实四类型行情，2366个历史参考日的原始特征/mask完全相同；使用固定TRAIN拟合参数时标准化X也完全相同。重新在改变后的全TRAIN拟合新参数当然可能改变历史z；不把离线拟合工件误称历史PIT。
'''+footer)
    label_rows=[]
    for split in ('TRAIN','VAL'):
        label_rows.append([split,summary[split]['reference_dates'],summary[split]['supervised_origins']]+[summary[split]['valid_labels_by_horizon'][str(h)] for h in HORIZONS])
    label_table=markdown_table(['split','参考日','监督origin','1d','5d','10d','20d'],label_rows)
    (out/'LABEL_CONSTRUCTION_REPORT.md').write_text('''# 探索性数学标签

R=P(t+h)/P(t)-1，供应商连续close、小数收益，0.02即2%。参考session而非自然日。全部h+1报价质量有效才保留。原点及端点同split，mask每个origin/h/c独立。训练末20个参考日h20无效，验证末20日同理；h1可继续有效。标签占位0不当真实0收益，只能配mask消费。

'''+label_table+'''
共274003个true标签已逐项独立复算价格/日期/中间窗口及假设时间；另用累计无效报价计数的独立算法比较全部true/false mask，检查没有错误地多屏蔽或多保留。25商品四周期均有TRAIN/VAL监督。所有标签均无TEST数值，最后VAL日2024-12-31只可作标签端点/特征面板，不是有监督origin。

逐25商品及四周期真实数量见per_target_label_coverage.csv；VAL逐月见val_monthly_label_coverage.csv。不能要求全商品同日有效，原点至少一项true；TRAIN只因此移除5日、VAL1日。20日重叠标签高度相关，样本计数不代表独立样本量。连续换月/事后修订未核实，标签不是实际可交易持仓回报。
'''+footer)
    (out/'CAUSALITY_AUDIT.md').write_text('''# 因果性与隔离审计

tests.txt记录完整实际测试结果，80条仅为旧collect.py日期运算弃用警告。新测试涵盖用户22类要求：日期与跨市场对齐、收盘时间、输入可得时间、锚点、两条split边界和20日跨界、未来前缀、无bfill、TRAIN-only拟合、收益/基点单位、停牌/期货缺失、313/25顺序、mask、finite、SHA复现、旧文件保护和TEST毒化门禁。另有真实全数据检查，非仅合成测试。

真实TRAIN2728+VAL483个样本的shape、mask和时间逐项检查；226659+47344个true标签独立复算；所有输入源时间<=所在决策，所有有效目标未来价格时间>决策且anchor<=决策。独立算法比较完整标签mask，包括false。prefix真实扰动已通过；VAL大幅改变不影响TRAIN scaler。日历/质量/特征方案没有看TEST分布或预测表现。

date-first门禁在quote parser之前丢弃TEST行，测试将TEST尾部改为恶意不可解析文本仍不进入CSV数值记录。压缩数据的块读取或SHA字节读取不等于解释TEST报价；本轮TEST数值解析0行、特征/标签0份。只检查静态TEST边界和manifest。

从原来源独立第二遍计算25个数值NPY文件，字节SHA全部相同，scaler参数完全相同，见reproducibility_audit.json（实际文件数以该JSON为准）。此前13313个数据/报告/源码等文件逐个SHA相同，三轮交付manifest共115项也独立复核通过。旧文件不是在本轮反向修改后自证未变。

日期与假设时钟检查通过不证明真实available_at、数据修订和可交易收益。HISTORICAL_PIT_VERIFIED、FUTURES_ROLL_METHOD_VERIFIED、CONFIRMATORY_TRAIN_READY保持false。股票cohort和VAL参与商品池设计均有已知选择限制，不能消除后宣称完全样本外。SN2015首日存在具体跨来源尺度/身份疑点，见SOURCE_ALERT.md，最终TRAIN/探索训练准入未放行；不能因因果测试通过就忽略该源问题。
'''+footer)
    target=pd.read_csv(out/'per_target_label_coverage.csv')
    target_table=[]
    for c in TARGETS:
        train=target.loc[(target.split=='TRAIN')&(target.asset_id=='FUT_'+c)].iloc[0]
        val=target.loc[(target.split=='VAL')&(target.asset_id=='FUT_'+c)].iloc[0]
        target_table.append([c]+[int(train[f'{h}d_valid_labels']) for h in HORIZONS]+[int(val[f'{h}d_valid_labels']) for h in HORIZONS])
    nt=markdown_table(['commodity','TRAIN1','TRAIN5','TRAIN10','TRAIN20','VAL1','VAL5','VAL10','VAL20'],target_table)
    acc=markdown_table(['验收项','结果'],[[k,str(v).lower()] for k,v in acceptance.items()])
    sha=sha256_file(dest/'dataset_manifest.json');size=sum(p.stat().st_size for p in dest.rglob('*') if p.is_file())
    feature_sha=manifest['files']['features.npy']['sha256']
    report=f'''# Phase E：收盘后25商品三市场预处理交付

完成TRAIN/VAL实际构建及审计；协议{PROTOCOL}。数据文件可读并可复现，但SN2015-03-27重大源数据疑点使TRAIN_DATASET_READY=false、EXPLORATORY_TRAIN_READY=false，科研训练放行暂缓。未进入模型训练。生成时间{datetime.datetime.now(datetime.timezone.utc).isoformat()}。

## 实际规模与数据集

{label_table}
313输入=263股票+18收益率+7财富指数+25商品，F=26；日面板[3217,313,26]。逻辑TRAIN X=[2728,20,313,26]，Y=[2728,4,25]；VAL X=[483,20,313,26]，Y=[483,4,25]。313固定身份、25固定目标、20窗口与4周期已逐样本验证。TRAIN从2011年重新算，无旧2017统计复用。

数据目录data/derived/triweave_25_after_close_v1/，文件合计{size:,}字节（{size/1024**2:.3f}MiB，目录块另计）。共享每日NPY mmap、窗口索引及四种mask，避免重复存大X。dataset_manifest.json SHA256={sha}；features.npy SHA256={feature_sha}；核心构建文件逐项SHA、字节数和两个split shape见dataset_manifest.json；后续科研暂停边车见audit_supplement_manifest.json，未覆盖核心manifest/数组。源、日历、代码、配置、协议和所有最终交付文件指纹见provenance.json及delivery_manifest.json。

```python
from triweave.after_close import AfterCloseDataset
ds = AfterCloseDataset("data/derived/triweave_25_after_close_v1", "TRAIN")
sample = ds[0]  # X, Y, masks, dates, identities and time evidence
assert sample["X"].shape == (20, 313, 26)
```

## 25商品逐目标标签数量

{nt}
25/25均有四周期TRAIN与VAL标签；晚起历史保留缺失，如LU/PG只有上市后TRAIN，SC2018起。不能把2728日期或重叠标签当作独立统计样本。WH不替换、不填补，原38来源全部保持，新增配置单独冻结I/Y为新目标。

## 对最终20项问题的回答

1. 25商品目标已冻结，顺序与用户一致，WH移除、C保留、I/Y加入；配置configs/futures_targets_25_v1.csv，筛选披露docs/TARGET_25_SELECTION_PROTOCOL.md。VAL覆盖参与设计，无MAE选择。
2. 313输入节点顺序完整保留，检查通过；不因早年缺失删节点。
3. 仅TRAIN/VAL/TEST三划分，TRAIN2011-10-01—2022-12-31，VAL2023—2024，TEST2025-01-01—2026-09-30；没有WARMUP集。
4. TRAIN有2733个参考日、2728个监督origin。
5. VAL有484个参考日、483个监督origin。
6. 唯一北京时间20:00收盘后决策，假设商品P(t)19:00可用，早于决策且未来目标价格晚于决策。
7. 股票假设18:00同日；债券下一参考日08:00/08:30，最大旧观测年龄5自然日。所有记录源观测日/交易日/available_at/decision_time，保留AVAILABLE_AT_ASSUMED，历史发布并未验证。
8. 价格类历史1/5/10/20变化、MA/波动/动量；股票量/额变化与换手；利率水平、基点变化和可核对的期限/信用利差；财富指数相对趋势；期货量/OI变化及OHLC区间；可用性/年龄通道。详见feature_registry.json。
9. 按经济类型与特征语义的55组TRAIN-only Z-score，唯一每日资产特征拟合，epsilon1e-8；VAL仅transform。
10. 修改所有实际VAL特征不改变TRAIN scaler；落盘独立复算TRAIN count/mean/std一致，严格通过；离线全TRAIN工件不能声称当时PIT。
11. 2个常量/scale floor组，均无沿用价格的quote_age_days=0；不足组0。|z|>20为TRAIN358、VAL16；最大|z|分别281.144257/75.965523。仅记录来源、不剪裁、不重fit，风险详见NORMALIZATION_REPORT.md。
12. 四周期总标签TRAIN57889/57249/56504/55017；VAL12067/11951/11806/11520，均为decimal_return。
13. 各商品四周期数量见上表与per_target_label_coverage.csv，VAL月份见val_monthly_label_coverage.csv。
14. 没有跨split标签；末20参考日起点h20无效。以全部true/false窗口独立验证，不仅检查已有true。
15. 代码日期/输入依赖与固定scaler前缀测试未发现未来数值泄漏；2366个前缀参考日逐值相同。但历史available_at、供应商回溯修订未核实，不能将此升级为真实PIT。
16. 本轮TEST数值解析、特征开发、标签、统计均0；允许的仅静态边界/manifest与不透明字节SHA。TEST数字尾部在quote parser前丢弃并通过毒化测试。
17. TRAIN/VAL已生成可读文件，独立全流程第二次重建所有数值NPY SHA一致，scaler完全相同。
18. 实际shape/大小/SHA见本报告规模段及dataset_manifest.json，源来源和完整SHA见provenance.json。
19. 文件与程序已具备读入条件，数学/时间/复现检查通过，但EXPLORATORY_TRAIN_READY=false、TRAIN_DATASET_READY=false：SN2015-03-27源报价15905与既有官方SN记录数量级不符，须先解释身份/版本或另冻新版本隔离规则。本版本原数值不偷偷改。见SOURCE_ALERT.md、source_alert.json与数据目录RESEARCH_HOLD.json。
20. 确认性研究仍受固定股票cohort、VAL参与选池、Sina连续/复权/修订未知、历史发布时间未知、未复权公司行动以及重叠标签限制；HISTORICAL_PIT_VERIFIED=false、FUTURES_ROLL_METHOD_VERIFIED=false、CONFIRMATORY_TRAIN_READY=false。

## 真实质量、单位与来源风险

所选25商品继承20条OHLC错误报价屏蔽和75个非参考日记录，原38的26/102证据保留。所选原审计2个>20%变化标记（PG2022-02-24、FU2019-08-06），本轮扩展TRAIN起点后重新按同一阈值统计，商品标记{evidence['futures_price_jump_flags']}条、股票{evidence['stock_price_jump_flags']}条，详见price_jump_flags_full_train_val.csv；不据跳变认定换月。报价有效日期不压缩，2019-04-22连续源缺口未用逐合约补填。利率百分数没有当价格，财富指数基期不同不混水平；期货活动量原生lots统计基准仍未核实。

重大补充：SN0在2015-03-27 close15905、下一参考日118080，单日数学变化{source_alert.get('vendor_one_session_decimal_change',0)*100:.6f}%；已有SHFE同日SN1507 close119270、可读合约low最低115150。raw与standardized一致，不是新处理制造。冻结数值有效规则尚保留以该点为锚的4个标签与相关输入，故表中数学有效数量并不等于科研可用性已放行。不自动换为官方合约，不按涨幅阈值删点，新增显式RESEARCH_HOLD边车并停止训练准入。I/C/FU早期片段亦留证据。

所有市场有效节点数/缺失见per_market_coverage.csv；逐资产的首尾、缺失/最长gap、新报价/沿用见per_asset_quote_coverage.csv。股票不复权的严重单日变化可能是公司行动，未直接纠正。近期债券沿用只是已知旧观测而不是新报价，原日期及年龄可追溯。

## 独立验收

{acc}
true仅代表在冻结探索性时间假设、来源快照和mask语义下通过，不是供应商方法或历史可交易性认证。{test_count}测试通过、80条既有弃用警告；旧13313个文件SHA无差异，三轮115项旧交付逐条验证。没有覆盖既有数据/manifest、改CMGM、训练、计算MAE、下载或Git操作。DATASET_ARTIFACTS_READABLE_AND_REPRODUCIBLE=true与科研训练准入false独立，不能混为全部PASS。

## 后续优先事项（本轮到此停止）

1. 先解决SN2015首日身份/尺度问题及I/C/FU早期跳变来源；取得证据后单独冻结新版本隔离规则，保留本版本和全部异常，不能根据VAL表现临时删点。TEST继续封存。
2. 取得供应商连续构造、换月/复权和历史版本证据，先补文档证据，不把补齐所有逐合约当当前探索任务前置条件。
3. 核实真实历史available_at、夜盘日期和日历版本；当前20:00协议仅是假设性信息集。
4. 单独预注册股票公司行动、活动量重尾与统计基准的稳健性方案，不根据VAL表现悄悄修改本版本。
5. 保留VAL参与商品池设计的披露；正式确认研究须重新冻结研究设计和未经研究选择污染的确认数据，不能回溯声称现VAL独立。

复核命令：.venv/bin/python -m triweave.after_close_review --verify。构建入口triweave.after_close_build对已有数据集拒绝覆盖；本次内部独立calculate已从源完成两遍，不需要重跑覆盖原版本。报告/脚本/原始来源SHA在交付封存中可逐项核对。
'''
    (out/'REPORT.md').write_text(report+footer)
    return acceptance


def verify_delivery(root):
    root=Path(root);out=root/REPORT_REL;dest=root/DATA_REL
    if (out/'delivery_manifest.json').exists():
        delivery=json.loads((out/'delivery_manifest.json').read_text())
        for relative,record in delivery['files'].items():
            assert sha256_file(root/relative)==record['sha256'],relative
    m=json.loads((dest/'dataset_manifest.json').read_text())
    for name,record in m['files'].items():assert sha256_file(dest/name)==record['sha256'],name
    if (dest/'audit_supplement_manifest.json').exists():
        supplement=json.loads((dest/'audit_supplement_manifest.json').read_text())
        assert sha256_file(dest/'dataset_manifest.json')==supplement['original_core_dataset_manifest_sha256']
        for name,record in supplement['files'].items():assert sha256_file(dest/name)==record['sha256'],name
    for name,h in json.loads((out/'preservation_before.json').read_text())['files'].items():assert sha256_file(root/name)==h,name
    print('Verified dataset, delivery (when sealed) and all protected original files; no quote ingestion')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--verify',action='store_true')
    args=p.parse_args()
    if args.verify:verify_delivery(args.root)
    else:reports(args.root)
