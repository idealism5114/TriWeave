"""Exhaustive frozen-threshold z audit with existing source-field evidence."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from triweave.after_close import registry,DATA_REL
from triweave.sourceqa_investigate import OUT_REL


def extremes(root,result,old,oldparams):
    out=Path(root)/OUT_REL;dates=result['arrays']['reference_dates'];assets=result['assets']
    reg=registry();sources=json.loads((Path(root)/DATA_REL/'source_manifest.json').read_text())
    rows=[];groups=[]
    sn=int(np.flatnonzero(dates=='2015-03-27')[0])
    for version,arrays,params in [('V1',old,oldparams),('SN_ONLY_IMPACT_SIMULATION',result['arrays'],result['params'])]:
        for p in params:
            k=p['feature_index'];js=[j for j,s in enumerate(assets) if s['economic_type']==p['economic_type']]
            z=arrays['features'][:,js,k];valid=arrays['feature_valid_mask'][:,js,k]
            positions=np.argwhere(valid&(np.abs(z)>20))
            for ti,aj in positions:
                i,j=int(ti),js[int(aj)];asset=assets[j]['asset_id'];f=result['frames'][asset]
                day=str(np.datetime64(int(arrays['observation_day'][i,j]),'D'))
                r=f.loc[f.date.eq(day)].iloc[0]
                prior_date=dates[i-1] if i else None;previous=f.loc[f.date.eq(prior_date)] if prior_date else f.iloc[:0]
                def num(row,field):
                    return float(pd.to_numeric(pd.Series([row.get(field)]),errors='coerce').iloc[0]) if row is not None else np.nan
                prior=previous.iloc[0] if len(previous) else None
                field={'volume_change_1d':'volume','amount_change_1d':'amount','open_interest_change_1d':'open_interest','turnover_pct':'turnover'}.get(p['feature_name'],'close')
                value=num(r,field);pv=num(prior,field);preclose=num(r,'preclose');prevclose=num(prior,'close')
                action_reference_change=bool(np.isfinite(preclose) and np.isfinite(prevclose) and not np.isclose(preclose,prevclose,rtol=1e-6,atol=1e-6))
                if asset=='FUT_SN' and sn<=i<=sn+reg[k]['lookback']-1:
                    cause='DISPUTED_SN_SOURCE_OBSERVATION_DEPENDENCY'
                elif assets[j]['economic_type']=='STOCK_PRICE' and field in ('volume','amount','turnover'):
                    cause='NATIVE_UNADJUSTED_STOCK_ACTIVITY_TAIL_NO_UNIT_SWITCH_FOUND'
                elif action_reference_change:
                    cause='CORPORATE_ACTION_REFERENCE_PRICE_CHANGE_INDICATED_NOT_EVENT_VERIFIED'
                elif assets[j]['economic_type']=='STOCK_PRICE':
                    cause='UNADJUSTED_PRICE_OR_ACTIVITY_RISK_CORPORATE_ACTION_NOT_VERIFIED'
                else:cause='VENDOR_MARKET_ROLL_OR_SOURCE_NOT_VERIFIED'
                si=int(arrays['source_index'][i,j])
                rows.append(dict(version=version,split='TRAIN' if dates[i]<='2022-12-31' else 'VAL',
                                 asset_id=asset,observation_date=day,reference_date=dates[i],economic_type=p['economic_type'],
                                 feature_name=p['feature_name'],unscaled=float(arrays['unscaled_features'][i,j,k]),
                                 z=float(z[int(ti),int(aj)]),mean_train=p['mean_train'],std_train=p['std_train'],
                                 effective_scale=p['effective_scale'],valid_train_count=p['valid_train_count'],
                                 scale_floor_flag=p['scale_floor_flag'],low_variance_floor_amplification=p['scale_floor_flag'],
                                 source_field=field,current_source_value=value,previous_source_value=pv,source_preclose=preclose,
                                 previous_reference_close=prevclose,corporate_action_reference_change=action_reference_change,
                                 original_economic_unit=assets[j]['unit'],unit_switch_evidence=False,
                                 cause_classification=cause,verified_genuine_market_move=False,
                                 source_manifest=sources[si]['manifest'] if si>=0 else None,
                                 action='RETAIN_FLAG_NO_VAL_REFIT_NO_CLIP'))
            for split,selection in [('TRAIN',dates<='2022-12-31'),('VAL',dates>'2022-12-31')]:
                vals=z[selection][valid[selection]]
                groups.append(dict(version=version,split=split,economic_type=p['economic_type'],feature_name=p['feature_name'],
                                   valid_count=len(vals),normalized_min=float(vals.min()) if len(vals) else None,
                                   normalized_max=float(vals.max()) if len(vals) else None,
                                   abs_z_gt20_count=int((np.abs(vals)>20).sum()),constant_flag=p['constant_flag'],
                                   floor_flag=p['scale_floor_flag'],std_train=p['std_train'],effective_scale=p['effective_scale']))
    pd.DataFrame(rows).to_csv(out/'normalization_extreme_audit.csv',index=False)
    pd.DataFrame(groups).to_csv(out/'normalization_group_audit.csv',index=False)
    summary={}
    for version in ['V1','SN_ONLY_IMPACT_SIMULATION']:
        s=pd.DataFrame(rows);s=s.loc[s.version.eq(version)]
        summary[version]={split:{'count_abs_z_gt20':int(s.split.eq(split).sum()),
                                'max_abs_z':float(s.loc[s.split.eq(split),'z'].abs().max())}
                          for split in ['TRAIN','VAL']}
    summary['any_extreme_floor_amplification']=any(r['low_variance_floor_amplification'] for r in rows)
    summary['all_finite']=all(np.isfinite(r['z']) for r in rows)
    summary['no_automatic_extreme_deletion']=True
    return summary
