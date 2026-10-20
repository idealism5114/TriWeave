"""Report scope is vendor-defined snapshot prediction, distinct from trading returns."""
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from triweave.collect import atomic_json,utc_now


def table(f,columns):
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join('---' for _ in columns)+' |']+
        ['| '+' | '.join(str(r.get(c,'')) for c in columns)+' |' for r in f.to_dict('records')])


def write_report(root,out,futures,targets,daily,roll,cross,anomalies):
    stock=cross.loc[cross.market.eq('stock')];bonds=cross.loc[cross.market.eq('bond')]
    stock_ready=len(stock)==263 and stock.TRAIN_valid_dates.gt(0).all() and stock.VAL_valid_dates.gt(0).all()
    bond_ready=len(bonds)==25 and bonds.TRAIN_valid_dates.gt(0).all() and bonds.VAL_valid_dates.gt(0).all()
    raw_available=len(futures)==38 and futures.record_count.gt(0).all() and futures.valid_reference_dates.gt(0).all()
    train_all=targets[[f'TRAIN_{h}d_candidate_labels' for h in (1,5,10,20)]].gt(0).all(axis=1)
    val_all=targets[[f'VAL_{h}d_candidate_labels' for h in (1,5,10,20)]].gt(0).all(axis=1)
    val2024=targets[[f'VAL_2024_{h}d_candidate_labels' for h in (1,5,10,20)]].gt(0).all(axis=1)
    exploratory=bool(raw_available and stock_ready and bond_ready and (train_all&val_all).any())
    flags={'STOCK_263_READY':bool(stock_ready),'BOND_25_READY':bool(bond_ready),
        'FUTURES_38_RAW_AVAILABLE':bool(raw_available),'FUTURES_CONTINUOUS_METHOD_VERIFIED':False,
        'TARGET_24_LABEL_COVERAGE_AUDITED':len(targets)==24,
        'EXPLORATORY_PREDICTION_READY':exploratory,'CONFIRMATORY_LABEL_READY':False,
        'READY_FOR_DATASET_CONSTRUCTION':exploratory}
    daily_summary=[]
    for split,f in daily.groupby('split',sort=False):
        for h in (1,5,10,20):
            s=f[f'{h}d_candidate_evaluable_targets_24']
            daily_summary.append({'split':split,'horizon_sessions':h,'reference_origins':len(s),
                'origins_with_any_candidate':int(s.gt(0).sum()),'origins_with_all_24_candidates':int(s.eq(24).sum()),
                'origins_with_fewer_than_24_candidates':int(s.lt(24).sum()),'minimum_candidates':int(s.min()),
                'maximum_candidates':int(s.max()),'candidate_asset_origin_pairs':int(s.sum())})
    pd.DataFrame(daily_summary).to_csv(out/'same_date_target_coverage_summary.csv',index=False)
    wh=targets.loc[targets.asset_id.eq('FUT_WH')].to_dict('records')[0]
    train756=futures.loc[futures.TRAIN_at_least_756_valid_quotes,'asset_id'].tolist()
    train252=futures.loc[futures.TRAIN_at_least_252_valid_quotes,'asset_id'].tolist()
    no_train=futures.loc[futures.TRAIN_valid_dates.eq(0),'asset_id'].tolist()
    few_train=futures.loc[futures.TRAIN_valid_dates.lt(252),['asset_id','TRAIN_valid_dates']].to_dict('records')
    by_h={str(h):{'TRAIN':int(targets[f'TRAIN_{h}d_candidate_labels'].gt(0).sum()),
                 'VAL':int(targets[f'VAL_{h}d_candidate_labels'].gt(0).sum()),
                 'VAL_2024':int(targets[f'VAL_2024_{h}d_candidate_labels'].gt(0).sum())} for h in (1,5,10,20)}
    full_val=futures.loc[futures.VAL_coverage_rate.eq(1),'asset_id'].tolist()
    partial_val=futures.loc[futures.VAL_coverage_rate.lt(1),['asset_id','VAL_valid_dates','VAL_coverage_rate']].to_dict('records')
    result={'at':utc_now(),'phase':'D-Continuous','flags':flags,
        'readiness_scope':'EXPLORATORY_FIXED_CAPTURE_VENDOR_CONTINUOUS_PRICE_CHANGE_ONLY; not confirmed real-time PIT, stable commodity economic returns or tradable strategy PnL',
        'official_or_confirmatory_labels_allowed':False,'dataset_constructed_this_round':False,
        'futures_total':38,'targets':24,'auxiliaries':14,'stocks':263,'bond_yields':18,'bond_wealth_indices':7,
        'futures_observation_rows':int(futures.record_count.sum()),
        'targets_with_all_four_train_candidate_horizons':int(train_all.sum()),
        'targets_with_all_four_val_candidate_horizons':int(val_all.sum()),
        'targets_with_all_four_val2024_candidate_horizons':int(val2024.sum()),
        'targets_with_train_and_val2024_candidates':int((train_all&val2024).sum()),
        'targets_with_candidates_by_horizon':by_h,'full_reference_val_coverage_assets':full_val,
        'partial_reference_val_coverage':partial_val,
        'futures_quality_counts':{c:int(futures[c].sum()) for c in ('duplicate_date_count','nonpositive_price_count','nonfinite_close_count','invalid_OHLC_or_duplicate_rows','off_reference_calendar_dates')},
        'training_quote_count_descriptors':{'at_least_252':train252,'at_least_756':train756,'no_training_quotes':no_train,'fewer_than_252':few_train},
        'descriptors_are_not_asset_filters_or_statistical_power_guarantees':True,
        'continuous_methods_verified':0,'roll_method_status':'ROLL_METHOD_NOT_VERIFIED',
        'large_price_change_flags_train_val':len(anomalies),'large_price_change_assets':sorted({r['asset_id'] for r in anomalies}),
        'large_price_change_is_not_roll_confirmation':True,'WH':wh,
        'candidate_label_definition':'All h+1 valid reference session OHLC/close quotes, origin and endpoint in same TRAIN or VAL; masks/counts only',
        'calendar_status':'Existing baseline calendar frozen, not independently certified as futures exchange/PIT calendar',
        'available_at_status':'UNKNOWN; lag2 alignment statistics are assumptions only, not historical publication proof',
        'sealed_candidate_labels_read_or_computed':False,'numeric_forward_labels_saved':False,
        'network_requests':0,'price_imputation':False,'full_asset_dropna':False,
        'training_or_MAE':False,'source_data_modified':False,
        'dataset_construction_restrictions':['Keep original 326 identities and 24 target order',
            'Use per_asset_availability_mask and per_target_label_mask; preserve late listings',
            'Fit transformations on observed TRAIN only; no VAL/SEALED fitting',
            'Unseen training variables need an unfitted flag or predetermined nonfitted transform',
            'Declare fixed-vintage pseudo-out-of-sample scope; do not claim historical real-time availability',
            'Seal 2025-2026 labels and model evaluation; separate 23-target supplementary reporting',
            'Do not clip/remove jumps using VAL outcomes or replace WH']}
    atomic_json(result,out/'audit_results.json')
    proposal='''# TRIWEAVE_CONTINUOUS_DATA_V1（提案）

状态：PROPOSED，探索性固定快照数据集可实施；尚非正式经济收益或确认性标签协议。旧RAW15、D-Rescue、D-Contracts产物及资产配置保持不变。此提案改变研究对象的声明，不追认旧主连为真实可交易收益。

## 预测对象及经济含义

研究跨市场信息融合与“供应商定义的商品连续价格序列变化”。固定当前采集快照，候选数学目标为 y[a,d,h]=P_vendor[a,d+h]/P_vendor[a,d]−1，h为1/5/10/20个冻结参考交易日；值采用小数比例，若以后以百分数报告须明确×100及MAE单位。

本轮仅计算窗口有效性bool与数量，没有保存上述正式数值标签或拟合模型。原始响应没有逐日对应合约、历史换月规则、复权因子或发布时间；每条均ROLL_METHOD_NOT_VERIFIED。即使数学定义明确，也不能声称它等于恒定期限商品收益、同合约收益、总收益或真实策略收益。固定快照可复现，不证明历史当时可获得、供应商未来不会重写历史。

## 来源、版本及资产顺序

股票使用已归档BaoStock不复权报价（adjustflag=3），263只固定Clean V1身份及原顺序。债券18条收益率水平percent_per_annum，7条财富指数index_level；不将利率当价格。商品使用Sina InnerFuturesNewService.getDailyKLine经AKShare返回的38个symbol0快照；接口名/代码0不足以证明其主力选择、复权或换月定义。

24目标和14辅助严格沿用原配置顺序，4个可选国债期货不加入本协议38商品或25债券。provenance.json包含完整顺序、参数、下载时间、每份raw/standardized/manifest的SHA256；data_fingerprints.json记录本轮新产物，preservation_before.json保护此前数据。禁止为改善VAL而选择供应商或替换价格。

## 有效性与mask

每资产每参考日独立availability：日期可解析且唯一、OHLC有限正数且关系一致。股票还要求原交易状态=1；债券利率允许有限零/负值，财富指数须正值。重复日期的全部记录不充作唯一报价，不任意挑一条。仅mask不可用，不覆盖或删除raw。

每目标每h候选mask：从d至d+h的h+1个参考日全部有有效报价；不压缩缺失交易日，不对不规则行号直接shift(-h)。起点和终点都必须位于同一TRAIN或VAL。缺失中间报价、上市前、非正/非有限价格、跨切分边界均无效；不bfill、不用0充作真实价格、不因某资产缺失删除整个日期。

本轮per_asset_availability_mask涵盖预热/TRAIN/VAL的326个原身份；per_target_label_mask只含TRAIN/VAL的24×4列bool。标签列全false的末端日期也保留在审核表中，未来评价时按该h mask处理。候选数量不等于正式标签已通过经济含义验收，重叠20日窗口也不是独立样本数。

## 时间对齐及available_at

日线date是来源交易/观测日，不是已验证available_at。未来构建时必须另记forecast_origin、信息截止时点和各市场发布延迟；不能默认当日股票、债券或期货日线在开盘前可用。

本轮保存同日精确匹配覆盖及滞后2个冻结参考交易日的敏感性计数，不填补价格。lag2是保守假设，不是历史发布时间验证。历史PIT不明时，数据集只能声明为固定采集快照下的伪样本外研究，不能据此声称可实时实现预测优势。跨市场周末银行观测保留在原层并单列，不能悄悄改日期；历史as-of对齐需经过单独版本化设计。

## 特征及标准化建议

先用明确经济类型的因果变换：利率保持原百分比，差分可按百分比点×100转换为基点并记录；财富指数与供应商价格序列变化分开处理。股票不复权价格受公司行为影响，不能当作已验证总收益。商品特征继承供应商拼接不确定性。

以后若拟合缩放器，仅使用TRAIN内真实可用的输入和mask，不用VAL/SEALED均值、标准差、分位数、收益分布或未来完整性；不基于全资产dropna选择样本。TRAIN没有观测或无可拟合方差的变量保留原身份并标记UNFITTED，可采用事先定义且不拟合的单位/因果变化表示，或保持该标准化通道不可用，不能拿VAL补拟合。LC等晚上市辅助变量尤其要记录此状态。当前没有拟合任何缩放器。

## 切分与封存

WARMUP 2011-10-01—2016-12-31；TRAIN 2017—2022；VAL 2023—2024；SEALED 2025-01-01—2026-09-30。本轮不计算或访问SEALED候选标签与模型表现。TRAIN末端h个参考交易日起点无法在TRAIN内完成h步标签，必须mask；VAL末端同理，不借SEALED报价完成标签。以后滚动评估还需另行冻结最长20步purge/embargo与特征回看边界。

## MAE评价建议（本轮未执行）

按h及商品分别计算有效mask上的MAE，并同时报告样本数、日期覆盖、可评价商品数。主配置保留24行，无样本的WH等显示NOT_EVALUABLE/NA，不能将缺失MAE记0。汇总须明确macro使用的实际可评价商品数与micro有效商品-日期数量，不能伪装24/24完整。预先固定23商品（仅排除WH）的辅助报告，与主24配置同时公开，不能根据VAL模型表现选择报告集合。

保留大幅变化并按原20%固定阈值标记复核，不据VAL MAE调整阈值、裁剪极端值或删除日期。未来如果核实了复权/拼接问题，建立新的数据版本与协议ID，不静默修订旧快照。

## 准入结论

无需先补齐DCE/GFEX逐合约，便可建立上述受限的探索性供应商序列数据集。若要把结果升级为已核实经济收益、真实历史可得或确认性结论，仍需连续构造定义、复权/修订说明和发布版本证据；可以先补文档/映射，不等于必须大规模补采行情。本轮不实际构造训练集、不生成正式标签、不训练、不计算MAE。
'''
    (out/'data_protocol_proposal.md').write_text(proposal)
    concise=futures[['asset_id','target_or_auxiliary','TRAIN_valid_dates','VAL_valid_dates','TRAIN_20d_valid_labels','VAL_20d_valid_labels','VAL_2024_status']]
    now=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    text=['# Phase D-Continuous：既有连续行情可用性与预测准备度审核','',f'生成：{now}。仅本地只读审核，网络请求0；不训练、不算MAE、不下载行情。',
        '',table(pd.DataFrame([{'field':k,'value':str(v).lower()} for k,v in flags.items()]),['field','value']),
        '', '**true的范围**：股票/债券可用于现有快照下的日期对齐；探索和数据集准备通过仅限供应商序列、固定快照、独立mask的伪样本外研究。并非全部24目标完整、方法已核实、历史PIT或正式收益标签通过。当前没有实际构造训练数据集。',
        '', '## 38品种实际覆盖', '',table(concise,list(concise.columns)),
        '',f'38条共{int(futures.record_count.sum()):,}行。TRAIN至少252个有效报价的品种{len(train252)}个，至少756个的{len(train756)}个；这只是事先固定的约1/3交易年描述，不是统计功效保证或删资产门槛。',
        '至少756报价：'+', '.join(train756)+'。',
        '少于252报价：'+json.dumps(few_train,ensure_ascii=False)+'；TRAIN完全无观测：'+', '.join(no_train)+'。',
        'SC/LU/PG/SI/LC等上市较晚，上市前缺失不算下载失败；完整窗口覆盖率及上市后缺失分开报告。first/last_valid_date与全期质量计数只用于来源审核，不计算SEALED未来标签。',
        '', '## 目标与日期级样本', '',
        f'四个h都有TRAIN候选的目标{int(train_all.sum())}/24；四个h都至少有一个VAL候选的目标{int(val_all.sum())}/24；2024年四个h均有候选{int(val2024.sum())}/24。按h分别统计的可评价目标数：'+json.dumps(by_h,ensure_ascii=False)+'。候选仅表示结构和日期窗口满足规则，不代表连续收益定义已核实。',
        '',table(pd.DataFrame(daily_summary),['split','horizon_sessions','candidate_asset_origin_pairs','origins_with_any_candidate','origins_with_all_24_candidates']),
        '逐日期可评价数量见cross_market_date_coverage.csv。日期不因一个商品缺失被删除，四个h独立mask；同一日期不是必须24/24同时有效。所有候选mask只有bool，未落盘正式收益标签。',
        '', '## 连续真实性、异常及版本', '',
        '38条均来自Sina symbol0，经AKShare同一日线服务采集。本地采集器/参数和raw字段已逐一核对，原始返回只有日期、OHLC、成交量/持仓及结算等字段，没有历史逐日对应合约、换月公告或复权因子。现有本地证据不足以核实供应商公布的主力选择/复权/连续定义，全部ROLL_METHOD_NOT_VERIFIED。',
        f'TRAIN/VAL相邻参考日价格变化绝对值超过既有20%阈值的标记{len(anomalies)}条，涉及'+', '.join(result['large_price_change_assets'])+'。阈值沿用此前冻结审计规则，未根据VAL分布或预测表现调整；没有裁剪或删除这些报价。',
        '这些标记可能是行情变化、拼接或来源问题，不能单凭涨跌认定换月；没有标记也不能证明无换月污染。已有短间隔RB0快照比较见existing_vendor_vintage_comparison.csv，不能用短间隔一致性证明历史从不修订。',
        f'全研究窗口重复日期、非正close、非有限close均为0；逻辑OHLC异常共{int(futures.invalid_OHLC_or_duplicate_rows.sum())}行，按事先规则在availability中无效，原文件未改。非参考日来源记录{int(futures.off_reference_calendar_dates.sum())}个商品-日期，与数值OHLC异常分开，不能未经核实认定为真实交易日。两类明细只输出到2024年，见continuous_quote_quality_issues.csv与continuous_off_calendar_dates.csv。',
        '2019-04-22等缺失已与已有官方CU/RB/AU/CF逐合约观测交叉核对，支持供应商连续序列存在缺口，不能当作休市；没有用官方价格补洞。证据见missing_date_existing_official_evidence.csv。',
        f'VAL全484参考日有效报价覆盖的品种为{len(full_val)}/38；其余：'+json.dumps(partial_val,ensure_ascii=False)+'。少量缺口会使多个20日窗口失效，因此候选数必须按逐日窗口计算。',
        '描述性研究可以使用有效价格并保留缺失；探索性预测可定义固定快照下供应商价格的数学变化；正式经济收益标签及确认性解释尚未通过。缺逐合约不是探索性研究的一票否决项，供应商方法未核实也不会被悄悄当作已核实。',
        '', '## 股票、债券与发布时间', '',
        '263只股票、18条利率与7条财富指数只读复用。逐资产对齐统计见cross_market_asset_coverage.csv；每日同日和lag2计数见cross_market_date_coverage.csv。利率百分比保持为利率，财富指数保持指数水平，不混合经济类型。',
        'available_at全部保持UNKNOWN，未默认日线开盘前已知；lag2只是提案假设。股票停牌行不作为可交易新报价，未填价。银行非参考交易日观测单列，没有挪到相邻日期。',
        '', '## WH及23商品辅助报告', '',
        'WH原身份保留。VAL2023/2024各h数量：'+json.dumps({k:v for k,v in wh.items() if k.startswith('VAL_202')},ensure_ascii=False)+'。2024无有效候选则NOT_EVALUABLE；2023少量候选单列，不把它说成整个VAL完全无数据。',
        'target_23_auxiliary_report.csv固定排除WH，仅为辅助覆盖报告；主24资产配置及输出顺序未变。没有用其他小麦、填价或零标签补齐24/24。',
        '', '## 对下一阶段的建议', '',
        '可以先构造TRIWEAVE_CONTINUOUS_DATA_V1提案中的探索性数据集：固定来源快照和顺序、使用两类mask、按经济类型生成因果输入、仅TRAIN拟合变换、保持切分与最长20步边界、封存2025—2026。无需先大规模补采逐合约行情。',
        '若要求正式稳定经济收益或确认性结论，优先补齐供应商连续定义、历史复权/修订和available_at证据；是否需要额外行情取决于这些证据，不把DCE缺口等同于当前任务失败。',
        '具体目标单位、标准化、LC无TRAIN观测的处理、逐资产MAE分母及23辅助报告边界见data_protocol_proposal.md。本轮不训练、不算MAE、不改变名单、不提交或推送。',
        '', '## 复现与归档', '',
        '代码入口：`.venv/bin/python -m triweave.continuous_audit`；交付封存后拒绝同目录覆盖。provenance.json列出数据来源、参数、全部相关raw/standardized/manifest SHA256及原始资产顺序；preservation_before.json与final_validation.json用于核查旧数据未变。tests.txt记录实际测试，不能代替供应商方法证明。']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')
    return result
