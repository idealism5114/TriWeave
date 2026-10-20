"""Document a concrete TRAIN source discrepancy without editing frozen prices."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from triweave.after_close import (DATA_REL, REPORT_REL, AfterCloseDataset,
                                 safe_csv, write_json, HORIZONS)
from triweave.universe import sha256_file


def source_alert(root):
    root=Path(root);out=root/REPORT_REL;dest=root/DATA_REL
    if (out/'source_alert.json').exists():raise FileExistsError('Evidence already exists')
    old=json.loads((root/'experiments/phase_d_continuous/provenance.json').read_text())
    logs=[];evidence=[];source_files=[]
    for code,start,end in [('SN','2015-03-27','2015-04-07'),('I','2015-09-25','2015-10-13'),
                           ('C','2016-03-29','2016-04-08'),('FU','2014-08-25','2014-09-02')]:
        entry=next(e for e in old['source_files'] if e['parameters'].get('symbol')==code+'0')
        for key,digest in [('raw_file','raw_sha256'),('standardized_file','standardized_sha256')]:
            assert sha256_file(root/entry[key])==entry[digest]
            f=safe_csv(root/entry[key],start=start,end=end,log=logs)
            name=f'evidence_{code}_{key}.csv';f.to_csv(out/name,index=False)
            evidence.append(name)
        source_files.append(entry)
    # Existing official daily data, restricted physically/date-gated to TRAIN.
    relative='data/standardized/baseline15y/futures_contracts/SHFE/SHFE_20150301_20150331.csv.gz'
    mpath=root/'data/manifests/baseline15y/jobs/futures_contracts/SHFE/SHFE_20150301_20150331.json'
    m=json.loads(mpath.read_text());assert sha256_file(root/relative)==m['standardized_sha256']
    f=safe_csv(root/relative,start='2015-03-27',end='2015-03-30',log=logs)
    f=f.loc[f.variety.eq('SN')];f.to_csv(out/'evidence_SN_existing_SHFE_contracts.csv',index=False)
    first=f.loc[f.date.eq('2015-03-27')];lows=pd.to_numeric(first.low,errors='coerce')
    original=pd.read_csv(out/'evidence_SN_raw_file.csv')
    start=original.loc[original.date.eq('2015-03-27')].iloc[0]
    following=original.loc[original.date.eq('2015-03-30')].iloc[0]
    ds=AfterCloseDataset(dest,'TRAIN');a=ds.arrays
    oi=int(np.flatnonzero(a['reference_dates']=='2015-03-27')[0]);si=int(np.flatnonzero(a['train_origin_indices']==oi)[0]);j=ds.meta[-25:]
    tj=next(i for i,s in enumerate(j) if s['asset_id']=='FUT_SN')
    labels={str(h):float(a['train_targets'][si,k,tj]) for k,h in enumerate(HORIZONS)}
    assert a['train_label_valid_mask'][si,:,tj].all()
    alert={'issue_id':'SN_20150327_SOURCE_IDENTITY_REQUIRES_REVIEW','blocking_for_exploratory_training':True,
           'status':'REQUIRES_REVIEW_NOT_AUTOMATIC_PRICE_CORRECTION','observation_date':'2015-03-27',
           'vendor_close':float(start.close),'next_reference_date':'2015-03-30','next_vendor_close':float(following.close),
           'vendor_one_session_decimal_change':float(following.close/start.close-1),
           'existing_SHFE_observed_contract_count':len(first),
           'existing_SHFE_finite_contract_low_min':float(lows.min()),
           'existing_SHFE_SN1507_close':float(first.loc[first.contract.eq('SN1507'),'close'].iloc[0]),
           'original_raw_and_standardized_agree':True,
           'interpretation':'Concrete cross-source scale/identity discrepancy. Continuous construction is unknown; this does not establish a replacement price or prove every large jump is an error.',
           'why_hold':'The frozen mathematical-validity mask retains four labels anchored to a substantially disputed first SN quote. Do not release this version for scientific training until identity/vintage is resolved or a separately versioned evidence-based quarantine protocol is frozen.',
           'affected_original_origin':'2015-03-27','retained_mathematically_valid_labels':labels,
           'directly_disputed_anchor_label_count':4,
           'input_risk':'Past-price features up to20 sessions may inherit this anchor; overlapping sample windows may reuse those features. No automatic exclusion or clipping.',
           'data_modified':False,'new_downloads':0,'TEST_numeric_rows_parsed':0,
           'source_files':source_files,'extra_official_source':{'manifest':str(mpath.relative_to(root)),
               'manifest_sha256':sha256_file(mpath),'standardized_file':relative,'standardized_sha256':m['standardized_sha256'],
               'raw_sha256':m['raw_sha256'],'api':m.get('api'),'parameters':m.get('api_kwargs')},
           'evidence_files':evidence+['evidence_SN_existing_SHFE_contracts.csv'],'read_log':logs}
    write_json(out/'source_alert.json',alert)
    write_json(dest/'RESEARCH_HOLD.json',{'protocol_id':ds.manifest['protocol_id'],'EXPLORATORY_TRAIN_READY':False,
               'dataset_numeric_arrays_unchanged':True,'issue_id':alert['issue_id'],
               'evidence':'experiments/phase_e_after_close/source_alert.json','evidence_sha256':sha256_file(out/'source_alert.json'),
               'original_dataset_manifest_sha256':sha256_file(dest/'dataset_manifest.json'),
               'scope':'Readable and reproducible research audit artifact; scientific training release held for source review'})
    (out/'SOURCE_ALERT.md').write_text('''# TRAIN源数据重大复核项

SN0在2015-03-27的原始OHLC为16025/16060/15880/15905；2015-03-30为119270/119530/115600/118080，close数学变化+642.408048%。raw与standardized相同，因此不是本轮预处理缩放制造。

同项目已保存的SHFE 2015-03-27逐合约中，SN1507 close=119270，已有可读合约low最小115150，明显不同数量级。未知连续构造定义意味着不能擅自用该合约替换15905，也不能只凭大涨认定换月或确定唯一真价。但这项具体身份/尺度疑点足以暂停科研训练放行。

冻结OHLC/正数规则没有排除该报价，四个以2015-03-27为origin的数学标签仍被mask标为有效；其1d约6.424080。但“数学可算”不能消除重大来源问题。输入历史特征和重叠窗口也可能传播该点。全数值文件保持不变，数据集新增RESEARCH_HOLD.json边车标记，原dataset_manifest与NPY不覆盖。

最终TRAIN_DATASET_READY=false、EXPLORATORY_TRAIN_READY=false表示科研准入未放行；DATASET_ARTIFACTS_READABLE_AND_REPRODUCIBLE=true单独记录结构/复现成功。VAL文件已构建且可读，但不能据此启动模型训练。无需授权/新下载即可保留全部证据；下一轮应先核对历史连续映射/原始来源，若无法解释，以明确证据冻结新版本隔离规则，不能偷偷删除或替价。

I在2015-10-08/09的+50.41%/−28.97%、C在2016-04-05的+20.35%、FU在2014-08-29的+24.22%亦保存原始片段，仍为REQUIRES_REVIEW，并不自动等于错误。没有TEST数值、网络采集或模型评价参与此判断。
''')
    return alert


if __name__=='__main__':source_alert(Path.cwd())
