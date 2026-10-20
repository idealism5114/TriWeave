"""Rebuild an in-memory SN-only impact counterfactual, never a released V1.1.

The independent timing conflicts make a complete revised protocol inadmissible.
No file in data/derived, previous reports, scalers or manifests is written here.
"""
from __future__ import annotations
import copy
import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

import triweave.after_close as ac
from triweave.sourceqa_investigate import OUT_REL,EVENTS
from triweave.universe import sha256_file


def policy(root):
    root=Path(root);out=root/OUT_REL
    inv=json.loads((out/'investigation_results.json').read_text())
    assert inv['sn_joint_evidence']['joint_identity_contradiction']
    rules=pd.DataFrame([dict(asset_id='FUT_SN',observation_date='2015-03-27',
                            source_id='sina_continuous_exploratory_v1',
                            reason_code='SOURCE_IDENTITY_UNSUPPORTED_PRICE_SETTLEMENT_AND_ACTIVITY',
                            evidence_reference='investigation_results.json:sn_joint_evidence;SN_two_days_all_contracts.csv',
                            review_status='EVIDENCE_SUPPORTED_FOR_SN_ONLY_IMPACT_SIMULATION',
                            rule_version='SOURCE_VERIFICATION_QUARANTINE_SQ1',
                            action='MASK_ENTIRE_ASSET_OBSERVATION_BEFORE_FEATURES_AND_LABELS',
                            application_scope='IN_MEMORY_COUNTERFACTUAL_ONLY_NOT_DATASET_RELEASE',
                            future_price_dependency=False)])
    if (out/'source_quarantine_rules.csv').exists():raise FileExistsError('Rules already frozen')
    rules.to_csv(out/'source_quarantine_rules.csv',index=False)
    ac.write_json(out/'impact_analysis_protocol.json',{
        'frozen_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'analysis':'SN_ONLY_IMPACT_SIMULATION_NOT_V1_1_DATASET',
        'dataset_version_generation_allowed':False,
        'reason':'Additional AU/AG session/date conflicts have not been causally resolved',
        'raw_values_changed':0,'rules_sha256':sha256_file(out/'source_quarantine_rules.csv'),
        'all_original_feature_time_label_scaler_parameters_unchanged_except_refit_statistics':True,
        'normalization_fit':'Recompute all55 groups from source TRAIN unique date-asset-feature; no old scaler reuse',
        'unaffected_unscaled_numeric_comparison':{'absolute_tolerance':1e-9,'relative_tolerance':1e-10,
            'reason':'Existing cumulative-sum rolling variance subtraction may change rounding outside a removed point; counts and maximum error separately audited. No unexplained large changes allowed.'},
        'TEST_numeric_access':False,'model_metrics':False,
        'causality_scope':'SN rule uses same-day observation and archived same-day contract evidence only, not next20 prices. Ex-post reviewed fixed snapshot; not proven historical PIT.'})


def apply_sn_rule(aligned,assets):
    j=next(i for i,a in enumerate(assets) if a['asset_id']=='FUT_SN')
    when=aligned['dates']=='2015-03-27'
    # Exact source date; no price thresholds, future prices or dynamic selections.
    good=aligned['observation_day'][:,j]==np.datetime64('2015-03-27','D').astype(int)
    aligned['valid'][when&good,j]=False
    aligned['new_observation'][when&good,j]=False
    return aligned


def calculate(root,with_quarantine):
    root=Path(root)
    protocol=json.loads((root/'experiments/phase_e_after_close/frozen_protocol.json').read_text())
    assets=ac.asset_definitions(root)
    full=pd.read_csv(root/'data/manifests/baseline15y/reference_trading_calendar.csv',dtype=str).date.tolist()
    frames,sources,logs=ac.load_sources(root,assets)
    aligned,_=ac.align_sources(frames,assets,full,protocol)
    if with_quarantine:apply_sn_rule(aligned,assets)
    x,m,app=ac.feature_panel(aligned,assets)
    params=ac.fit_scaler(x,m,aligned['dates'],assets,protocol)
    z,fm=ac.transform(x,m,assets,params)
    arrays=ac.dataset_arrays(aligned,assets,z,fm,app)
    arrays.update(unscaled_features=x,unscaled_feature_valid_mask=m)
    return dict(arrays=arrays,aligned=aligned,assets=assets,params=params,logs=logs,protocol=protocol,frames=frames)


def dependency_impact(root,result,old,local_change,changed_params):
    out=root/OUT_REL;a=result['arrays'];assets=result['assets'];dates=a['reference_dates']
    ids=[x['asset_id'] for x in assets];reg=ac.registry();rows=[];summaries=[]
    # Existing jump events plus the two newly substantiated timing conflicts.
    roots=[('FUT_'+c,d,'QUARANTINE_CANDIDATE' if c=='SN' else 'RETAINED_REVIEW_ONLY') for c,d in EVENTS]
    roots += [('FUT_AU','2013-07-08','UNRESOLVED_SESSION_MAPPING'),('FUT_AG','2013-07-08','UNRESOLVED_SESSION_MAPPING')]
    for asset,date,status in roots:
        j=ids.index(asset);point=int(np.flatnonzero(dates==date)[0]);daily=[];affected_windows=set();label_counts={str(h):0 for h in ac.HORIZONS}
        def add(origin,feature='',horizon=None,kind='',feature_date=None):
            label_semantics=''
            if kind=='LABEL_WINDOW_DEPENDENCY':
                origin_index=int(np.flatnonzero(dates==origin)[0])
                label_semantics=('ENDPOINT_VALUE_AND_WINDOW_QUALITY' if point in (origin_index,origin_index+horizon)
                                 else 'INTERMEDIATE_QUALITY_ONLY_NOT_ARITHMETIC_ENDPOINT')
            rows.append(dict(asset_id=asset,source_date=date,affected_origin_date=origin,
                             affected_feature=feature,affected_horizon=horizon,impact_type=kind,
                             reason_code='SOURCE_OBSERVATION_QUARANTINE' if status=='QUARANTINE_CANDIDATE' else status,
                             affected_feature_date=feature_date,root_status=status,
                             role='POTENTIAL_DEPENDENCY' if status!='QUARANTINE_CANDIDATE' else 'SN_ONLY_COUNTERFACTUAL',
                             label_dependency_semantics=label_semantics))
        add(date,'RAW_OHLCV_OI_SETTLE',kind='RAW_SOURCE_OBSERVATION')
        for r in reg:
            k=r['index']
            if not old['feature_applicable_mask'][j,k]:continue
            for i in range(point,min(len(dates),point+r['lookback'])):
                if not old['unscaled_feature_valid_mask'][i,j,k]:continue
                daily.append((i,k))
                add(dates[i],r['feature_name'],kind='DERIVED_DAILY_FEATURE_DEPENDENCY',feature_date=dates[i])
                for split in ('train','val'):
                    oi=old[split+'_origin_indices'];wi=old[split+'_window_indices']
                    # One row per affected window/feature, preserve underlying feature date.
                    for origin in oi[(wi==i).any(1)]:
                        affected_windows.add((split,int(origin)))
                        add(dates[origin],r['feature_name'],kind='INPUT_WINDOW_PROPAGATION',feature_date=dates[i])
                if dates[i]<=ac.SPLITS['TRAIN'][1]:
                    add(dates[i],r['feature_name'],kind='TRAIN_SCALER_SAMPLE_DEPENDENCY',feature_date=dates[i])
        for split in ('train','val'):
            origins=old[split+'_origin_indices'];m=old[split+'_label_valid_mask'];tj=j-288
            for si,i in enumerate(origins):
                for hk,h in enumerate(ac.HORIZONS):
                    if i<=point<=i+h and m[si,hk,tj]:
                        label_counts[str(h)]+=1
                        add(dates[i],'target_vendor_close',h,'LABEL_WINDOW_DEPENDENCY')
        summaries.append(dict(asset_id=asset,source_date=date,status=status,
                              dependent_valid_daily_features=len(daily),
                              dependent_input_windows=len(affected_windows),
                              labels_by_horizon=label_counts,
                              actual_local_changed_windows='SN_ONLY' if asset=='FUT_SN' else 'NOT_APPLIED'))
    # Pooled refit propagates normalization to other futures; distinguish that
    # from direct contamination and do not count all those prices as anomalous.
    kinds=[s['economic_type'] for s in assets]
    for p in changed_params:
        if not p['transformation_changed']:continue
        k=p['feature_index'];js=[j for j,t in enumerate(kinds) if t==p['economic_type']]
        daily=a['feature_valid_mask'][:,js,k].any(1)
        for split in ('train','val'):
            for origin,window in zip(a[split+'_origin_indices'],a[split+'_window_indices']):
                if np.any(daily[window[window>=0]]):
                    rows.append(dict(asset_id='FUT_SN',source_date='2015-03-27',affected_origin_date=dates[origin],
                                     affected_feature=p['feature_name'],affected_horizon=None,
                                     impact_type='POOLED_TRAIN_SCALER_REFIT_PROPAGATION',
                                     reason_code='CHANGED_TRAIN_MOMENTS_ALL_FUTURES_IN_GROUP',affected_feature_date=None,
                                     root_status='QUARANTINE_CANDIDATE',role='SN_ONLY_COUNTERFACTUAL'))
    pd.DataFrame(rows).to_csv(out/'anomaly_dependency_impact.csv',index=False)
    ac.write_json(out/'anomaly_impact_summary.json',summaries)
    return summaries


def run(root):
    root=Path(root);out=root/OUT_REL;dest=root/ac.DATA_REL
    if (out/'impact_results.json').exists():raise FileExistsError('Impact audit already exists')
    if (root/'data/derived/triweave_25_after_close_v1_1').exists():raise FileExistsError('V1.1 must not be generated before timing gate passes')
    policy(root)
    old={p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in dest.glob('*.npy')}
    print('Independent V1 reproduction from frozen sources',flush=True)
    baseline=calculate(root,False)
    baseline_sha={k:ac.array_sha(v)==sha256_file(dest/(k+'.npy')) for k,v in baseline['arrays'].items()}
    assert all(baseline_sha.values())
    del baseline
    print('SN-only in-memory masked rebuild and fresh TRAIN scaler fit',flush=True)
    r=calculate(root,True);a=r['arrays'];dates=a['reference_dates'];assets=r['assets']
    oldparams=json.loads((dest/'scaler_params.json').read_text());changed=[]
    for p,previous in zip(r['params'],oldparams):
        if p!=previous:
            changed.append(dict(p,old_valid_train_count=previous['valid_train_count'],
                                old_mean_train=previous['mean_train'],old_std_train=previous['std_train'],
                                old_effective_scale=previous['effective_scale'],
                                transformation_changed=any(p[k]!=previous[k] for k in ['mean_train','effective_scale']),
                                reason='SN_OBSERVATION_MASK_BEFORE_ROLLING_FEATURES_AND_TRAIN_FIT'))
    pd.DataFrame(changed).to_csv(out/'changed_scaler_groups.csv',index=False)
    ac.write_json(out/'candidate_sn_only_scaler_params.json',{'status':'IMPACT_COUNTERFACTUAL_NOT_RELEASED_V1_1','parameters':r['params']})
    x=a['unscaled_features'];m=a['unscaled_feature_valid_mask'];ox=old['unscaled_features'];om=old['unscaled_feature_valid_mask']
    local=(m!=om)|~np.isclose(x,ox,rtol=1e-10,atol=1e-9)
    j=next(i for i,s in enumerate(assets) if s['asset_id']=='FUT_SN');point=int(np.flatnonzero(dates=='2015-03-27')[0])
    cone=np.zeros_like(m)
    for reg in ac.registry():
        if a['feature_applicable_mask'][j,reg['index']]:cone[point:point+reg['lookback'],j,reg['index']]=True
    unexplained=local&~cone
    assert not unexplained.any(),'Unexplained changes outside source dependency cone'
    roundoff=(x!=ox)&~cone&(m&om)
    max_roundoff=float(np.max(np.abs(x[roundoff]-ox[roundoff]))) if roundoff.any() else 0.
    rows=[]
    for k,v in a.items():
        was=old[k];diff=v!=was
        rows.append(dict(array=k,comparison='V1_VS_SN_ONLY_SIMULATION_NOT_V1_1',
                         dataset_v1_1_generated=False,changed_elements=int(diff.sum()),shape=list(v.shape),
                         reason='MASK_AND_LOCAL_DEPENDENCIES' if 'mask' in k or k in ['aligned_values','new_observation'] else
                                'FRESH_TRAIN_SCALER_AND_SOURCE_DEPENDENCIES' if k=='features' else
                                'LOCAL_DEPENDENCIES_AND_CUMULATIVE_SUM_ROUNDOFF' if k=='unscaled_features' else
                                'MASKED_LABEL_PLACEHOLDERS' if 'targets' in k else 'UNCHANGED'))
    pd.DataFrame(rows).to_csv(out/'dataset_v1_v1_1_diff.csv',index=False)
    impact=dependency_impact(root,r,old,local,changed)
    counts={};local_windows={};scaled_windows={}
    for split in ('train','val'):
        oi=a[split+'_origin_indices'];wi=a[split+'_window_indices']
        daylocal=local.any((1,2))|(a['input_valid_mask']!=old['input_valid_mask']).any(1)
        normalized_changed=(a['features']!=old['features']).any((1,2))
        local_windows[split]=int(sum(daylocal[w[w>=0]].any() for w in wi))
        scaled_windows[split]=int(sum(normalized_changed[w[w>=0]].any() for w in wi))
        assert np.array_equal(oi,old[split+'_origin_indices'])
        counts[split]={str(h):int(a[split+'_label_valid_mask'][:,hk].sum()) for hk,h in enumerate(ac.HORIZONS)}
    # Intrinsic prefix causality and TRAIN-only fit; cannot validate source session semantics.
    cutoff='2021-06-30';prefix=dates<=cutoff;perturbed={}
    for name,v in r['aligned'].items():
        if isinstance(v,np.ndarray):perturbed[name]=v.copy()
    for field in ['value','open','high','low','volume','amount','turnover','open_interest']:
        perturbed[field][~prefix]=perturbed[field][~prefix]*1.53+.01
    xx,mm,app=ac.feature_panel(perturbed,assets)
    prefix_pass=np.array_equal(xx[prefix],x[prefix]) and np.array_equal(mm[prefix],m[prefix])
    assert prefix_pass
    zz,zm=ac.transform(xx,mm,assets,r['params'])
    assert np.array_equal(zz[prefix],a['features'][prefix])
    valchange=x.copy();valchange[dates>ac.SPLITS['TRAIN'][1]]=987654321.
    assert ac.fit_scaler(valchange,m,dates,assets,r['protocol'])==r['params']
    boundaries=True
    for split in ('train','val'):
        lo,hi=ac.SPLITS[split.upper()]
        for si,i in enumerate(a[split+'_origin_indices']):
            for hk,h in enumerate(ac.HORIZONS):
                good=a[split+'_label_valid_mask'][si,hk]
                if not good.any():continue
                e=int(a[split+'_endpoint_indices'][si,hk]);assert e==i+h and lo<=dates[i]<=dates[e]<=hi
                assert a['input_valid_mask'][i:e+1,-25:][:,good].all()
                np.testing.assert_allclose(a[split+'_targets'][si,hk,good],
                    a['aligned_values'][e,-25:][good]/a['aligned_values'][i,-25:][good]-1,rtol=1e-7,atol=1e-8)
    # First30 actual origins and masks from the immutable original loader.
    ds=ac.AfterCloseDataset(dest,'TRAIN');padding=[]
    for i in range(30):
        s=ds[i];wi=old['train_window_indices'][i];pad=wi<0
        assert not s['input_valid_mask'][pad].any() and not s['feature_valid_mask'][pad].any()
        assert np.all(s['X'][pad]==0) and np.all(wi[~pad]<=old['train_origin_indices'][i])
        padding.append(dict(sample_index=i,origin_date=s['origin_date'],real_reference_sessions=int((~pad).sum()),
                            left_padding_sessions=int(pad.sum()),padding_input_and_feature_masks_false=True,
                            padding_values_zero=True,future_or_first_valid_quote_copied=False))
    pd.DataFrame(padding).to_csv(out/'early_window_padding_audit.csv',index=False)
    # Exhaustive extreme audits and stock corporate-action references are separate.
    from triweave.sourceqa_normalization import extremes
    norm=extremes(root,r,old,oldparams)
    print('Second independent SN-only calculation, no dataset-version write',flush=True)
    second=calculate(root,True)
    repro={k:{'first_sha256':ac.array_sha(v),'second_sha256':ac.array_sha(second['arrays'][k])} for k,v in a.items()}
    for record in repro.values():record['identical']=record['first_sha256']==record['second_sha256']
    assert all(v['identical'] for v in repro.values()) and second['params']==r['params']
    result=dict(role='SN_ONLY_IMPACT_SIMULATION_NOT_V1_1_RELEASE',V1_1_generated=False,
                changed_raw_values=0,changed_input_masks=int((a['input_valid_mask']!=old['input_valid_mask']).sum()),
                changed_feature_masks=int((a['feature_valid_mask']!=old['feature_valid_mask']).sum()),
                changed_label_masks=int(sum((a[s+'_label_valid_mask']!=old[s+'_label_valid_mask']).sum() for s in ('train','val'))),
                changed_scaler_groups=len(changed),changed_scaler_transformations=sum(x['transformation_changed'] for x in changed),
                local_feature_dependency_windows=local_windows,pooled_refit_changed_X_windows=scaled_windows,
                valid_label_counts_after_SN_only_simulation=counts,source_dependency_summary=impact,
                unexplained_material_unscaled_changes=int(unexplained.sum()),
                cumulative_sum_roundoff_only_values=int(roundoff.sum()),max_unscaled_roundoff=max_roundoff,
                independent_V1_reproduction=baseline_sha,intrinsic_prefix_pass=prefix_pass,
                prefix_cutoff=cutoff,TRAIN_only_scaler_pass=True,TRAIN_VAL_boundary_pass=boundaries,
                early_padding_pass=True,all_arrays_finite=all(np.isfinite(v).all() for v in a.values() if v.dtype.kind=='f'),
                normalization=norm,quote_read_log=r['logs']+second['logs'],TEST_numeric_rows_parsed=0,
                source_timing_validation='FAILED_RELEASE_GATE_AU_AG_UNRESOLVED',
                no_new_dataset_or_old_files_written=True)
    ac.write_json(out/'candidate_reproducibility.json',{'role':result['role'],'arrays':repro,'scaler_identical':second['params']==r['params']})
    ac.write_json(out/'impact_results.json',result)
    print(json.dumps({k:result[k] for k in ['changed_raw_values','changed_input_masks','changed_feature_masks','changed_label_masks','changed_scaler_groups','local_feature_dependency_windows','pooled_refit_changed_X_windows','max_unscaled_roundoff']},ensure_ascii=False),flush=True)


if __name__=='__main__':run(Path.cwd())
