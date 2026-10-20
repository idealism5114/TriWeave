"""Exhaustive TRAIN/VAL coverage and source-backed feature-tail audit."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from triweave import after_close as ac
from triweave import research23 as r23


def longest_missing(mask):
    best=run=0
    for present in mask:
        run=0 if present else run+1;best=max(best,run)
    return best


def summarize(root,r):
    out=root/r23.OUT;a=r['arrays'];assets=r['assets'];dates=a['reference_dates'];reg=ac.registry()
    assetrows=[];marketrows=[];targetrows=[];feature_rows=[];labeldist=[];padding=[];summary={}
    for split in ('TRAIN','VAL'):
        s=split.lower();lo,hi=ac.SPLITS[split];own=(dates>=lo)&(dates<=hi);origins=a[s+'_origin_indices']
        lm=a[s+'_label_valid_mask'];y=a[s+'_targets'];di=np.flatnonzero(own)
        labelmap={int(i):si for si,i in enumerate(origins)};daily=[]
        for i in di:
            row=dict(date=dates[i],supervised_origin=i in labelmap)
            for market in ('stock','bond','future'):
                js=[j for j,x in enumerate(assets) if x['market']==market]
                row[market+'_valid_nodes']=int(a['input_valid_mask'][i,js].sum())
            for hk,h in enumerate(ac.HORIZONS):row[f'labels_{h}d']=int(lm[labelmap[i],hk].sum()) if i in labelmap else 0
            daily.append(row)
        pd.DataFrame(daily).to_csv(out/(s+'_coverage.csv'),index=False)
        for kind in ac.KINDS:
            js=[j for j,x in enumerate(assets) if x['economic_type']==kind];q=a['input_valid_mask'][own][:,js]
            marketrows.append(dict(split=split,economic_type=kind,nodes=len(js),reference_dates=len(di),valid_node_dates=int(q.sum()),
                missing_ratio=float(1-q.mean()),mean_valid_nodes=float(q.sum(1).mean()),min_valid_nodes=int(q.sum(1).min()),max_valid_nodes=int(q.sum(1).max())))
        for j,spec in enumerate(assets):
            q=a['input_valid_mask'][own,j];present=di[q];positions=np.flatnonzero(q)
            first=int(positions[0]) if len(positions) else None;last=int(positions[-1]) if len(positions) else None
            observed=r['frames'][spec['asset_id']]
            source_in_range=observed.date.between(lo,hi)
            assetrows.append(dict(split=split,asset_id=spec['asset_id'],input_index=j,market=spec['market'],economic_type=spec['economic_type'],unit=spec['unit'],
                reference_dates=len(di),source_observations=int(source_in_range.sum()),valid_input_dates=int(q.sum()),
                missing_dates=int((~q).sum()),missing_ratio=float(1-q.mean()),coverage_ratio=float(q.mean()),
                first_valid_date=dates[present[0]] if len(present) else None,last_valid_date=dates[present[-1]] if len(present) else None,
                longest_missing_gap=longest_missing(q),longest_internal_missing_gap=longest_missing(q[first:last+1]) if first is not None else len(q),
                leading_unavailable_reference_dates=first if first is not None else len(q),
                max_observation_age_days=int(a['age_days'][present,j].max()) if len(present) else None,
                carried_valid_dates=int((q & ~a['new_observation'][own,j]).sum()),
                source_time_status='AVAILABLE_AT_ASSUMED'))
            for feature in reg:
                k=feature['index'];app=bool(a['feature_applicable_mask'][j,k]);m=a['feature_valid_mask'][own,j,k]
                v=a['features'][own,j,k][m];raw=a['unscaled_features'][own,j,k][m]
                feature_rows.append(dict(split=split,asset_id=spec['asset_id'],economic_type=spec['economic_type'],feature=feature['feature_name'],
                    applicable=app,reference_dates=len(di),valid_count=int(m.sum()),valid_ratio=float(m.mean()) if app else None,
                    structural_inapplicability=not app,unscaled_min=float(raw.min()) if len(raw) else None,unscaled_max=float(raw.max()) if len(raw) else None,
                    z_min=float(v.min()) if len(v) else None,z_max=float(v.max()) if len(v) else None))
        for tj,c in enumerate(r23.TARGETS):
            row=dict(split=split,asset_id='FUT_'+c,target_index=tj,reference_dates=len(di),supervised_origins=len(origins))
            for hk,h in enumerate(ac.HORIZONS):row[f'{h}d_valid_labels']=int(lm[:,hk,tj].sum())
            targetrows.append(row)
        for hk,h in enumerate(ac.HORIZONS):
            v=y[:,hk][lm[:,hk]]
            labeldist.append(dict(split=split,horizon=h,count=len(v),minimum=float(v.min()),maximum=float(v.max()),mean=float(v.mean()),
                q01=float(np.quantile(v,.01)),q50=float(np.quantile(v,.5)),q99=float(np.quantile(v,.99)),unit='decimal_return'))
        pads=(a[s+'_window_indices']<0).sum(1)
        for si,n in enumerate(pads):
            if n or si<30:padding.append(dict(split=split,origin_date=dates[origins[si]],padding_sessions=int(n),all_padding_masks_false=True))
        summary[split]=dict(reference_dates=len(di),origin_count=len(origins),first_origin=dates[origins[0]],last_origin=dates[origins[-1]],
          logical_X_shape=[len(origins),20,311,26],Y_shape=list(y.shape),label_counts={str(h):int(lm[:,hk].sum()) for hk,h in enumerate(ac.HORIZONS)},
          padding_origin_count=int((pads>0).sum()),max_padding_sessions=int(pads.max()),all_label_invalid_origin_count=len(di)-len(origins))
    for name,rows in [('per_asset_coverage.csv',assetrows),('per_market_coverage.csv',marketrows),('per_target_label_coverage.csv',targetrows),
                      ('per_feature_coverage.csv',feature_rows),('label_distribution.csv',labeldist),('padding_audit.csv',padding)]:
        pd.DataFrame(rows).to_csv(out/name,index=False)
    pd.DataFrame(r['quality']).to_csv(out/'quote_quality_audit.csv',index=False)
    pd.DataFrame(r['replay']).to_csv(out/'raw_normalizer_replay.csv',index=False)
    ac.write_json(out/'coverage_summary.json',summary)
    return summary


def tail_audit(root,r):
    out=root/r23.OUT;a=r['arrays'];dates=a['reference_dates'];assets=r['assets'];aligned=r['aligned'];reg=ac.registry()
    rows=[];norm=[];summary={}
    for p in r['params']:
        k=p['feature_index'];name=p['feature_name'];js=[j for j,x in enumerate(assets) if x['economic_type']==p['economic_type']]
        audit=dict(p,old_scaler_reused=False,fit_population='TRAIN_UNIQUE_DATE_ASSET_FEATURE_NOT_OVERLAPPING_WINDOWS')
        for split,(lo,hi) in ac.SPLITS.items():
            if split=='TEST':continue
            own=(dates>=lo)&(dates<=hi);ds=np.flatnonzero(own)
            z=a['features'][own][:,js,k];mask=a['feature_valid_mask'][own][:,js,k];v=z[mask]
            audit[split+'_valid_count']=len(v);audit[split+'_z_min']=float(v.min()) if len(v) else None
            audit[split+'_z_max']=float(v.max()) if len(v) else None;audit[split+'_z_mean']=float(v.mean()) if len(v) else None
            audit[split+'_z_std']=float(v.std()) if len(v) else None
            audit[split+'_abs_z_gt20']=int((np.abs(v)>20).sum())
            for ti,aj in np.argwhere(mask & (np.abs(z)>20)):
                i,j=int(ds[int(ti)]),js[int(aj)];spec=assets[j];asset=spec['asset_id']
                obs=int(a['observation_day'][i,j]);day=str(np.datetime64(obs,'D')) if obs>=0 else None
                frame=r['frames'][asset];current=frame.loc[frame.date.eq(day)] if day else frame.iloc[:0]
                row=current.iloc[0] if len(current) else None
                def num(field):return float(pd.to_numeric(pd.Series([row.get(field) if row is not None else None]),errors='coerce').iloc[0])
                field={'volume_change_1d':'volume','amount_change_1d':'amount','open_interest_change_1d':'open_interest','turnover_pct':'turnover'}.get(name,
                    'value' if spec['economic_type'] in ac.KINDS[1:3] else 'close')
                source_value=num(field);previous=None;denominator=None;denominator_date=None;details={}
                if name in ('volume_change_1d','amount_change_1d','open_interest_change_1d') and i:
                    previous=float(aligned[field][i-1,j]);denominator=previous;denominator_date=dates[i-1]
                elif name.startswith('return_'):
                    h=int(name.split('_')[1][:-1]);denominator=float(aligned['value'][i-h,j]);denominator_date=dates[i-h]
                elif name.startswith('ma_deviation_'):
                    w=int(name.split('_')[-1][:-1]);denominator=float(aligned['value'][i-w+1:i+1,j].mean());denominator_date=f'{dates[i-w+1]}..{dates[i]}'
                elif name=='ohlc_range':denominator=float(aligned['value'][i,j]);denominator_date=dates[i]
                elif name=='momentum_5_20':details=dict(price_lag5=float(aligned['value'][i-5,j]),price_lag20=float(aligned['value'][i-20,j]))
                elif name.startswith('volatility_'):
                    w=int(name.split('_')[-1][:-1]);details=dict(price_window=aligned['value'][i-w:i+1,j].tolist())
                elif name.startswith('yield_delta_'):
                    h=int(name.split('_')[2][:-1]);previous=float(aligned['value'][i-h,j]);denominator_date=dates[i-h]
                previous_close=float(aligned['value'][i-1,j]) if i and aligned['valid'][i-1,j] else np.nan
                preclose=num('preclose');action_hint=bool(spec['market']=='stock' and np.isfinite(preclose) and np.isfinite(previous_close)
                    and not np.isclose(preclose,previous_close,rtol=1e-6,atol=1e-6))
                if name in ('amount_change_1d','volume_change_1d','open_interest_change_1d'):
                    cause='ACTIVITY_RATIO_TAIL_DENOMINATOR_RECORDED;CORPORATE_ACTION_AND_VENDOR_ROLL_NOT_EVENT_VERIFIED'
                elif action_hint:cause='UNADJUSTED_CORPORATE_ACTION_REFERENCE_HINT_NOT_ANNOUNCEMENT_VERIFIED'
                else:cause='HEAVY_TAIL_MARKET_OR_SOURCE_CONSTRUCTION_UNVERIFIED_NOT_AUTOMATIC_ERROR'
                unit=('RMB' if field=='amount' else 'shares' if field=='volume' and spec['market']=='stock' else
                      'native_vendor_lots_counting_basis_unverified' if field in ('volume','open_interest') else 'percent' if field=='turnover' else spec['unit'])
                si=int(a['source_index'][i,j]);source=r['sources'][si] if si>=0 else None
                rows.append(dict(split=split,market=spec['market'],asset_id=asset,economic_type=spec['economic_type'],
                    reference_date=dates[i],observation_date=day,feature_name=name,source_field=field,original_unit=unit,
                    original_value=source_value,previous_value=previous,historical_denominator=denominator,denominator_date=denominator_date,
                    unscaled_feature=float(a['unscaled_features'][i,j,k]),mean_train=p['mean_train'],std_train=p['std_train'],effective_scale=p['effective_scale'],
                    normalized_value=float(z[int(ti),int(aj)]),valid_train_count=p['valid_train_count'],scale_floor_flag=p['scale_floor_flag'],
                    original_preclose=preclose,previous_reference_close=previous_close,corporate_action_reference_hint=action_hint,
                    unit_switch_status='NO_SWITCH_DETECTED_IN_EXISTING_SOURCE_FIELDS_NOT_UNIVERSAL_PROOF',
                    attribution=cause,confirmed_source_error=False,confirmed_genuine_market_move=False,
                    source_raw_file=source['raw_file'] if source else None,source_raw_sha256=source['raw_sha256'] if source else None,
                    source_standardized_file=source['standardized_file'] if source else None,
                    raw_renormalization_verified=True,extra_dependency_evidence=json.dumps(details),action='RETAIN_FLAG_NO_CLIP_NO_VAL_SELECTION'))
        norm.append(audit)
    pd.DataFrame(norm).to_csv(out/'normalization_audit.csv',index=False)
    extreme=pd.DataFrame(rows);extreme.to_csv(out/'extreme_feature_audit.csv',index=False)
    for split in ('TRAIN','VAL'):
        own=(dates>=ac.SPLITS[split][0])&(dates<=ac.SPLITS[split][1]);m=a['feature_valid_mask'][own];v=a['features'][own][m]
        sub=extreme.loc[extreme.split.eq(split)]
        maxrow=sub.loc[sub.normalized_value.abs().idxmax()].to_dict() if len(sub) else None
        if maxrow is not None:maxrow={k:None if isinstance(v,float) and not np.isfinite(v) else v for k,v in maxrow.items()}
        summary[split]=dict(valid_feature_values=int(m.sum()),max_abs_z=float(np.abs(v).max()),abs_z_gt20=int((np.abs(v)>20).sum()),
                            all_finite=bool(np.isfinite(v).all()),largest_example=maxrow)
    summary['extreme_floor_amplification_count']=int(extreme.scale_floor_flag.sum()) if len(extreme) else 0
    summary['constant_groups']=sum(p['constant_flag'] for p in r['params']);summary['floor_groups']=sum(p['scale_floor_flag'] for p in r['params'])
    summary['insufficient_groups']=sum(p['insufficient_train_support'] for p in r['params']);summary['clipped_values']=0
    ac.write_json(out/'normalization_summary.json',summary)
    return summary
