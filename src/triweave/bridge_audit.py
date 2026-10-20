"""Independent mask, clock, label and release checks for the technical branch."""
from __future__ import annotations
import copy
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from triweave import after_close as ac
from triweave import bridge as b
from triweave.universe import sha256_file


def preservation(root):
    old=json.loads((root/b.OUT/'preservation_before.json').read_text())['files']
    changed=[name for name,h in old.items() if not (root/name).is_file() or sha256_file(root/name)!=h]
    if changed:raise RuntimeError('Original artifacts changed: '+repr(changed))
    return dict(protected_files=len(old),changed_files=[],originals_unchanged=True,
                method='OPAQUE_BYTE_SHA256; not TEST numeric ingestion')


def complete(q,w):
    out=np.zeros(q.shape,bool)
    if len(q)>=w:out[w-1:]=np.lib.stride_tricks.sliding_window_view(q,w,axis=0).all(-1)
    return out


def mask_oracle(aligned, assets):
    """Independent boolean oracle using sliding windows, not core rolling sums."""
    q=aligned['valid'];t,n=q.shape;out=np.zeros((t,n,26),bool)
    kinds=np.array([a['economic_type'] for a in assets]);ids=[a['asset_id'] for a in assets]
    for r in ac.registry():
        name=r['feature_name'];k=r['index'];app=np.isin(kinds,r['economic_type'])
        if name.startswith('return_') or name.startswith('yield_delta_'):
            h=int(name.split('_')[1][:-1]) if name.startswith('return_') else int(name.split('_')[2][:-1])
            m=complete(q,h+1)
        elif name.startswith('ma_deviation_'):m=complete(q,int(name.split('_')[-1][:-1]))
        elif name.startswith('volatility_'):m=complete(q,int(name.split('_')[-1][:-1])+1)
        elif name=='momentum_5_20':m=complete(q,21)
        elif name in ('volume_change_1d','amount_change_1d','open_interest_change_1d'):
            field={'volume_change_1d':'volume','amount_change_1d':'amount','open_interest_change_1d':'open_interest'}[name]
            v=aligned[field];m=np.zeros_like(q)
            m[1:]=q[1:]&q[:-1]&np.isfinite(v[1:])&np.isfinite(v[:-1])&(v[1:]>=0)&(v[:-1]>0)
        elif name=='turnover_pct':m=q & np.isfinite(aligned['turnover']) & (aligned['turnover']>=0)
        elif name in ('ohlc_range','quote_age_days','yield_level_pct'):m=q.copy()
        elif name=='quote_available':m=np.ones_like(q)
        elif name.startswith('term_'):
            m=np.zeros_like(q);long=ids.index('CN_GOV_YIELD_10Y');short=ids.index('CN_GOV_YIELD_'+name.split('_')[2])
            m[:,long]=q[:,long]&q[:,short]&(aligned['observation_day'][:,long]==aligned['observation_day'][:,short])
        elif name=='credit_spread_bps':
            m=np.zeros_like(q)
            for j,asset in enumerate(ids):
                if asset.startswith('CN_AAA_'):
                    other=ids.index('CN_GOV_YIELD_'+asset.rsplit('_',1)[1])
                    m[:,j]=q[:,j]&q[:,other]&(aligned['observation_day'][:,j]==aligned['observation_day'][:,other])
        else:raise AssertionError(name)
        out[:,:,k]=m&app[None]
    return out


def label_oracle(result, split):
    """Reconstruct from literal vendor indices and clocks; no label_arrays call."""
    a=result['arrays'];dates=a['reference_dates'];q=a['target_quote_valid_mask'];p=a['target_vendor_values']
    lo,hi=ac.SPLITS[split];deadline=int(ac.clock_seconds(np.array([hi]),20)[0]);m=np.zeros((len(dates),4,23),bool)
    y=np.zeros(m.shape,np.float32)
    for i in range(len(dates)):
        anchor=i+3
        for hk,h in enumerate((1,5,10,20)):
            end=anchor+h
            if end>=len(dates) or not lo<=dates[i]<dates[anchor]<=dates[end]<=hi:continue
            for j in range(23):
                if all(bool(q[k,j]) and 0<=a['target_assumed_available_at'][k,j]<=deadline for k in range(anchor,end+1)):
                    value=p[end,j]/p[anchor,j]-1
                    if np.isfinite(value) and abs(value)<=np.finfo(np.float32).max:
                        m[i,hk,j]=True;y[i,hk,j]=value
    present=np.array([a['input_valid_mask'][max(0,i-19):i+1].any() for i in range(len(dates))])
    origins=np.flatnonzero(m.any((1,2)) & present).astype(np.int32)
    return y,m,origins


def known_risk_audit(root, result):
    f=pd.read_csv(root/'experiments/phase_e_sourceqa/all_exact_vendor_official_fingerprints.csv')
    f=f.loc[f.variety.isin(b.TARGETS)].copy();f['nonflat']=f.high>f.low
    outside=f.loc[f.nonflat & f.reference_session_difference.abs().gt(2)]
    strong=pd.read_csv(root/'experiments/phase_e_sourceqa/timing_match_adjudication.csv')
    strong=strong.loc[strong.variety.isin(b.TARGETS) & strong.adjudication.eq('TIMING_CONFLICT_STRONG_EVIDENCE')]
    # Do not equate the absence of comparison (e.g. DCE) with verified PIT.
    unresolved=outside.to_dict('records')
    unresolved+=strong.loc[strong.reference_session_difference.gt(2)].to_dict('records')
    retained=pd.read_csv(root/'experiments/phase_e_sourceqa/early_futures_anomaly_audit.csv')
    retained=retained.loc[retained.asset_id.isin(['FUT_'+c for c in b.TARGETS])]
    retained.to_csv(root/b.OUT/'retained_source_anomalies.csv',index=False)
    repeated=f.loc[f.reference_session_difference.abs().gt(2)]
    repeated.to_csv(root/b.OUT/'repeated_fingerprint_time_ambiguities.csv',index=False)
    cases=[]
    forward=f.loc[f.reference_session_difference.gt(0)]
    dates=result['arrays']['reference_dates'];idx={d:i for i,d in enumerate(dates)}
    for row in forward.to_dict('records'):
        if row['date_vendor'] not in idx:continue
        source=idx[row['date_vendor']];release=source+2
        official=idx.get(row['date_official'],-1)
        cases.append(dict(asset_id=row['asset_id'],vendor_date=row['date_vendor'],official_match_date=row['date_official'],
                          offset=row['reference_session_difference'],nonflat=row['nonflat'],
                          diagnostic_input_release_date=str(dates[release]) if release<len(dates) else None,
                          known_official_match_precedes_input_release=0<=official<=release,
                          inference='Repeated thin-bar match does not independently prove a date convention'))
    assert not unresolved,'Known substantive session risk outside frozen diagnostic guard'
    assert all(c['known_official_match_precedes_input_release'] for c in cases)
    return dict(known_unbounded_material_session_conflicts_in_remaining23=unresolved,
                known_forward_fingerprint_cases=cases,nonflat_abs_offset_gt2=len(outside),
                repeated_flat_abs_offset_gt2=len(repeated),
                repeated_flat_min_offset=float(repeated.reference_session_difference.min()) if len(repeated) else None,
                repeated_matches_not_timestamps=True,DCE_historical_PIT='NOT_VERIFIED',
                remaining_unverified_time_risk='Vendor construction, revisions and real publication remain unbounded in general; diagnostic validity is only an explicit vendor-index engineering task, not true historical PIT.',
                AU_AG_excluded_not_resolved=True)


def validate(root, r):
    a=r['arrays'];aligned=r['aligned'];base=r['base'];assets=r['assets'];dates=a['reference_dates'];im=a['input_valid_mask']
    assert a['features'].shape==(len(dates),311,26)
    assert [x['asset_id'] for x in assets][288:]==['FUT_'+c for c in b.TARGETS]
    assert not {'FUT_AU','FUT_AG'} & {x['asset_id'] for x in assets}
    assert len(r['params'])==55 and max(dates)<='2024-12-31'
    cutoff=a['base_reference_indices'];expect=np.zeros_like(im);expect[2:]=base['valid'][:-2]
    for j,spec in enumerate(assets):
        if spec['economic_type'] in ac.KINDS[1:3]:expect[:,j]&=aligned['age_days'][:,j]<=5
    assert np.array_equal(im,expect)
    assert not im[:2].any()
    days=dates.astype('datetime64[D]').astype(int)
    cutdays=days[np.maximum(cutoff,0)]
    assert (a['observation_day'][im]<=np.broadcast_to(cutdays[:,None],im.shape)[im]).all()
    assert (a['available_at'][im]<=np.broadcast_to(a['decision_time'][:,None],im.shape)[im]).all()
    assert np.array_equal(a['available_at'],b.shift_publication(np.concatenate([np.full_like(base['available_at'][:2],-1),base['available_at'][:-2]]),r['calendar']))
    oracle=mask_oracle(aligned,assets)
    assert np.array_equal(oracle,a['unscaled_feature_valid_mask'])
    assert np.array_equal(oracle,a['feature_valid_mask']) # all55 groups have sufficient support
    assert not (a['feature_valid_mask'] & ~a['feature_applicable_mask'][None]).any()
    for k in ('features','unscaled_features','aligned_values','target_vendor_values'):
        assert np.isfinite(a[k]).all()
    assert (a['features'][~a['feature_valid_mask']]==0).all()
    assert (a['unscaled_features'][~a['unscaled_feature_valid_mask']]==0).all()
    assert (a['aligned_values'][~im]==0).all()
    # Independently compare every valid input value to the selected source frame.
    source_rows=0
    for j,spec in enumerate(assets):
        f=r['frames'][spec['asset_id']];field='value' if spec['economic_type'] in ac.KINDS[1:3] else 'close'
        lookup=pd.Series(pd.to_numeric(f[field],errors='coerce').to_numpy(),index=f.date.to_numpy())
        lookup=lookup.loc[~lookup.index.duplicated(keep=False)]
        source_dates=a['observation_day'][im[:,j],j].astype('datetime64[D]').astype('U10')
        expected=lookup.reindex(source_dates).to_numpy(float)
        assert np.array_equal(expected,a['aligned_values'][im[:,j],j])
        source_rows+=len(expected)
    coverage=[];label_checks=[];counts={};origin_counts={}
    for split in ('TRAIN','VAL'):
        s=split.lower();y,m,origins=label_oracle(r,split)
        assert np.array_equal(origins,a[s+'_origin_indices'])
        assert np.array_equal(m[origins],a[s+'_label_valid_mask'])
        assert np.array_equal(y[origins],a[s+'_targets'])
        assert np.isfinite(a[s+'_targets']).all() and (a[s+'_targets'][~a[s+'_label_valid_mask']]==0).all()
        counts[s]={str(h):int(m[origins,hk].sum()) for hk,h in enumerate(ac.HORIZONS)};origin_counts[s]=len(origins)
        for si,i in enumerate(origins):
            assert a[s+'_anchor_indices'][si]==i+3
            assert (a[s+'_window_indices'][si]<=i).all()
            for hk,h in enumerate(ac.HORIZONS):
                good=m[i,hk];end=int(a[s+'_endpoint_indices'][si,hk]);anchor=i+3
                if good.any():
                    assert end==anchor+h and dates[i]<dates[anchor]<dates[end]
                    assert a[s+'_window_indices'][si].max()<=i<anchor
                    assert (a['target_assumed_available_at'][anchor:end+1,good]>a['decision_time'][i]).all()
                label_checks.append(dict(split=split,origin_date=dates[i],decision_time_utc=int(a['decision_time'][i]),
                   input_cutoff_vendor_date=dates[i-2] if i>=2 else '',label_anchor_date=dates[anchor],
                   label_end_date=dates[end] if end>=0 else '',horizon=h,valid_targets=int(good.sum()),
                   anchor_offset_sessions=3,publication_purge_sessions=2,same_split=True,
                   task_id=b.TASK,scope='VENDOR_INDEX_ASSUMPTIONS_NOT_VERIFIED_EXCHANGE_SESSION_RETURN'))
        lo,hi=ac.SPLITS[split];in_split=(dates>=lo)&(dates<=hi)
        for j,c in enumerate(b.TARGETS):
            q=a['target_quote_valid_mask'][:,j]
            coverage.append(dict(split=split,asset_id='FUT_'+c,target_index=j,input_index=288+j,
               reference_dates=int(in_split.sum()),valid_vendor_quotes=int((q&in_split).sum()),
               valid_lagged_input_dates=int((im[:,288+j]&in_split).sum()),
               first_valid_vendor_date=dates[np.flatnonzero(q&in_split)[0]] if (q&in_split).any() else None,
               last_valid_vendor_date=dates[np.flatnonzero(q&in_split)[-1]] if (q&in_split).any() else None,
               **{f'valid_labels_{h}d':int(m[origins,hk,j].sum()) for hk,h in enumerate(ac.HORIZONS)}))
    pd.DataFrame(label_checks).to_csv(root/b.OUT/'label_anchor_consistency.csv',index=False)
    for split in ('TRAIN','VAL'):
        pd.DataFrame([x for x in coverage if x['split']==split]).to_csv(root/b.OUT/(split.lower()+'_coverage.csv'),index=False)
    # Actual prefix perturbation happens before the delay, not only on saved X.
    changed=copy.deepcopy(base);prefix=dates<='2021-06-30'
    for k in ('value','open','high','low','volume','amount','turnover','open_interest'):
        changed[k][~prefix]=changed[k][~prefix]*1.53+.01
    delayed=b.lag_inputs(changed,assets,r['calendar']);xx,mm,_=ac.feature_panel(delayed,assets)
    assert np.array_equal(xx[prefix],a['unscaled_features'][prefix])
    assert np.array_equal(mm[prefix],a['unscaled_feature_valid_mask'][prefix])
    zz,_=ac.transform(xx,mm,assets,r['params']);assert np.array_equal(zz[prefix],a['features'][prefix])
    # Frozen TRAIN fit is invariant to any VAL values and feature masks.
    xx=a['unscaled_features'].copy();mm=a['unscaled_feature_valid_mask'].copy();vv=dates>'2022-12-31'
    xx[vv]=987654321.;mm[vv]=~mm[vv]
    assert ac.fit_scaler(xx,mm,dates,assets,r['protocol'])==r['params']
    del xx,mm,zz,changed,delayed,oracle
    assert all(l['TEST_numeric_rows_parsed']==0 and (l['max_parsed_date'] is None or l['max_parsed_date']<=ac.END) for l in r['logs'])
    risks=known_risk_audit(root,r)
    return dict(universe_pass=True,input_lag_pass=True,feature_mask_oracle_pass=True,source_value_matches=source_rows,
                label_oracle_pass=True,horizon_boundaries_pass=True,TRAIN_only_scaler_pass=True,
                prefix_causality_pass=True,TEST_isolation_pass=True,finite_and_placeholders_pass=True,
                origin_counts=origin_counts,label_counts=counts,known_risk_review=risks,
                historical_PIT_verified=False,clock_assumptions_verified_as_declared_only=True)


def sn_audit(root,r,control):
    a=r['arrays'];c=control['arrays'];dates=a['reference_dates'];ids=[a['asset_id'] for a in r['assets']]
    j=ids.index('FUT_SN');rawidx=int(np.flatnonzero(dates=='2015-03-27')[0]);idx=rawidx+2;tj=j-288
    assert r['unmasked_base']['valid'][rawidx,j] and not r['base']['valid'][rawidx,j]
    assert r['base']['value'][rawidx,j]==control['base']['value'][rawidx,j]==15905.
    assert not a['input_valid_mask'][idx,j] and c['input_valid_mask'][idx,j]
    assert int((a['input_valid_mask']!=c['input_valid_mask']).sum())==1
    local=(a['unscaled_feature_valid_mask']!=c['unscaled_feature_valid_mask'])|~np.isclose(a['unscaled_features'],c['unscaled_features'],rtol=1e-10,atol=1e-9)
    cone=np.zeros(local.shape,bool);rows=[]
    for reg in ac.registry():
        k=reg['index']
        if a['feature_applicable_mask'][j,k]:cone[idx:idx+reg['lookback'],j,k]=True
        changed=int((a['unscaled_feature_valid_mask'][:,j,k]!=c['unscaled_feature_valid_mask'][:,j,k]).sum())
        value_changed=int(local[:,j,k].sum())
        if changed or value_changed:
            rows.append(dict(asset_id='FUT_SN',vendor_date='2015-03-27',diagnostic_input_date=dates[idx],
                       dependency_type='INPUT_FEATURE',feature=reg['feature_name'],horizon=None,
                       changed_masks=changed,material_local_changes=value_changed,remaining_bad_observation_usage=0,
                       original_price=15905.,replacement_price_used=False))
    assert not (local&~cone).any(),'Unexplained substantive change outside SN dependency cone'
    roundoff=(a['unscaled_features']!=c['unscaled_features'])&~cone
    stats=[]
    for p,q in zip(r['params'],control['params']):
        stats.append(dict(p,SN_control_valid_train_count=q['valid_train_count'],SN_control_mean=q['mean_train'],
                          SN_control_std=q['std_train'],SN_control_scale=q['effective_scale'],
                          SN_changes_statistics=p!=q,SN_changes_transform=any(p[k]!=q[k] for k in ('mean_train','effective_scale')),
                          fitted_on='TRAIN_ONLY_DIAGNOSTIC_DECISION_DATES',old_scaler_reused=False,
                          VAL_used_for_fit=False,TEST_used_for_fit=False))
    assert all(p['economic_type']=='FUTURES_CONTINUOUS_PRICE' for p in stats if p['SN_changes_statistics'])
    pd.DataFrame(stats).to_csv(root/b.OUT/'normalization_audit.csv',index=False)
    local_windows={};scaled_windows={};label_changes={}
    for split in ('train','val'):
        assert np.array_equal(a[split+'_origin_indices'],c[split+'_origin_indices'])
        lm=a[split+'_label_valid_mask'];cm=c[split+'_label_valid_mask'];origins=a[split+'_origin_indices']
        count={}
        for hk,h in enumerate(ac.HORIZONS):
            bad_uses=[]
            for si,i in enumerate(origins):
                anchor=i+3;end=anchor+h
                if anchor<=rawidx<=end:
                    assert not lm[si,hk,tj]
                    bad_uses.append(si)
            loss=int((lm[:,hk,tj]!=cm[:,hk,tj]).sum());count[str(h)]=loss
            if split=='train':rows.append(dict(asset_id='FUT_SN',vendor_date='2015-03-27',diagnostic_input_date=dates[idx],
                dependency_type='FUTURE_VENDOR_WINDOW_LABEL',feature='',horizon=h,changed_masks=loss,
                material_local_changes=loss,remaining_bad_observation_usage=0,original_price=15905.,replacement_price_used=False))
        label_changes[split]=count
        daily=local.any((1,2));scaled=(a['features']!=c['features']).any((1,2))
        windows=a[split+'_window_indices'];local_windows[split]=int(sum(daily[w[w>=0]].any() for w in windows))
        scaled_windows[split]=int(sum(scaled[w[w>=0]].any() for w in windows))
    pd.DataFrame(rows).to_csv(root/b.OUT/'sn_quarantine_audit.csv',index=False)
    result=dict(SN_quarantine_pass=True,raw_changed_values=0,input_mask_changes=1,
                vendor_date='2015-03-27',delayed_input_date=dates[idx],
                feature_mask_changes=int((a['feature_valid_mask']!=c['feature_valid_mask']).sum()),
                label_mask_changes=label_changes,scaler_statistics_groups_changed=sum(x['SN_changes_statistics'] for x in stats),
                scaler_transform_groups_changed=sum(x['SN_changes_transform'] for x in stats),
                local_changed_origin_windows=local_windows,scaled_changed_origin_windows=scaled_windows,
                unexplained_material_changes=0,outside_cone_roundoff_elements=int(roundoff.sum()),
                outside_cone_roundoff_max=float(np.abs(a['unscaled_features'][roundoff]-c['unscaled_features'][roundoff]).max()) if roundoff.any() else 0.,
                control_role='Unquarantined311/23 lag2 FWD3 diagnostic ablation in memory only; never released')
    ac.write_json(root/b.OUT/'sn_quarantine_details.json',result)
    return result


def publish(root,r,repro,checks,sncheck):
    a=r['arrays'];out=root/b.OUT;dest=root/b.DATA;stage=dest.parent/'.triweave_23_diagnostic_v0_build'
    if dest.exists() or stage.exists():raise FileExistsError('Refuse to overwrite dataset/staging')
    flags=dict(DIAGNOSTIC_UNIVERSE_PASS=checks['universe_pass'],SN_QUARANTINE_PASS=sncheck['SN_quarantine_pass'],
      CONSERVATIVE_INPUT_LAG_PASS=checks['input_lag_pass'],DIAGNOSTIC_LABEL_TIME_CONSISTENCY_PASS=checks['label_oracle_pass'],
      TRAIN_ONLY_SCALER_PASS=checks['TRAIN_only_scaler_pass'],HORIZON_BOUNDARY_PASS=checks['horizon_boundaries_pass'],
      TEST_ISOLATION_PASS=checks['TEST_isolation_pass'],DIAGNOSTIC_DATASET_REPRODUCIBLE=repro['all_identical'],
      DIAGNOSTIC_DATASET_READY=False,EXPLORATORY_25_RESEARCH_READY=False,HISTORICAL_PIT_VERIFIED=False,
      FUTURES_ROLL_METHOD_VERIFIED=False,CONFIRMATORY_TRAIN_READY=False)
    required=list(flags)[:8]
    ready=all(flags[k] is True for k in required) and not checks['known_risk_review']['known_unbounded_material_session_conflicts_in_remaining23']
    flags['DIAGNOSTIC_DATASET_READY']=ready
    ac.write_json(out/'audit_results.json',dict(flags,protocol_id=b.PROTOCOL,task_id=b.TASK,input_nodes=311,targets=23,
       origin_counts=checks['origin_counts'],valid_label_counts=checks['label_counts'],
       use_restriction='CODE_AND_NUMERICAL_DIAGNOSTICS_ONLY; NOT_FORMAL_RESEARCH_OR_PERFORMANCE_CLAIMS',
       time_consistency_scope='EXPLICIT_FROZEN_VENDOR_INDEX_AND_ASSUMED_PUBLICATION_CLOCKS; NOT_REAL_HISTORICAL_PIT'))
    if not ready:return False
    protection=preservation(root);ac.write_json(out/'preservation_check.json',protection)
    files={k+'.npy':dict(sha256=ac.array_sha(v),shape=list(v.shape),dtype=str(v.dtype)) for k,v in a.items()}
    meta=b.metadata(root,r)
    for name,payload in meta.items():files[name]=dict(sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload))
    manifest=dict(protocol_id=b.PROTOCOL,task_id=b.TASK,DIAGNOSTIC_DATASET_READY=True,diagnostic_only=True,
       actual_input_nodes=311,actual_prediction_targets=23,features=26,window=20,horizons=[1,5,10,20],
       stored_daily_shape=list(a['features'].shape),origin_counts=checks['origin_counts'],label_counts=checks['label_counts'],
       assumed_input_lag=2,future_vendor_anchor_offset=3,label_publication_purge_sessions=2,
       TEST_dataset_generated=False,HISTORICAL_PIT_VERIFIED=False,FUTURES_ROLL_METHOD_VERIFIED=False,
       EXPLORATORY_25_RESEARCH_READY=False,CONFIRMATORY_TRAIN_READY=False,
       loader='triweave.bridge.BridgeDataset',files=files,
       protocol_sha256=sha256_file(out/'frozen_protocol.json'))
    stage.mkdir()
    for k,v in a.items():np.save(stage/(k+'.npy'),v,allow_pickle=False)
    for name,payload in meta.items():(stage/name).write_bytes(payload)
    for name,e in files.items():assert sha256_file(stage/name)==e['sha256']
    ac.write_json(stage/'dataset_manifest.json',manifest)
    stage.rename(dest)
    # Exercise actual mmap interface for every retained sample, with padding.
    for split in ('TRAIN','VAL'):
        ds=b.BridgeDataset(dest,split)
        for i in range(len(ds)):
            sample=ds[i];pad=sample['reference_indices']<0
            assert sample['X'].shape==(20,311,26) and sample['Y'].shape==(4,23)
            assert not sample['input_valid_mask'][pad].any() and not sample['feature_valid_mask'][pad].any()
            assert (sample['X'][pad]==0).all()
            assert (sample['assumed_available_at'][pad]==-1).all()
            assert (sample['vendor_observation_day'][pad]==-1).all()
            assert np.isfinite(sample['X']).all() and np.isfinite(sample['Y']).all()
    ac.write_json(out/'dataset_publication.json',dict(dataset_path=b.DATA,files=len(files)+1,
       bytes=sum(p.stat().st_size for p in dest.iterdir() if p.is_file()),all_files_rehashed=True,
       loader_all_origins_padding_finite_shapes_pass=True,diagnostic_only=True))
    return True


def run(root):
    root=Path(root);out=root/b.OUT
    if (out/'build_checks.json').exists() or (root/b.DATA).exists():raise FileExistsError('Append-only Bridge build; use --verify')
    start=datetime.datetime.now(datetime.timezone.utc).isoformat()
    print('Diagnostic source build1:311 inputs /23 targets /lag2 /FWD3',flush=True)
    r=b.calculate(root);checks=validate(root,r)
    print('Independent SN quarantine ablation, no separate dataset',flush=True)
    control=b.calculate(root,False,cached=(r['frames'],r['sources'],r['logs']))
    sncheck=sn_audit(root,r,control);del control
    print('Diagnostic independent source build2',flush=True)
    second=b.calculate(root)
    arrays={k:dict(first_sha256=ac.array_sha(v),second_sha256=ac.array_sha(second['arrays'][k])) for k,v in r['arrays'].items()}
    meta1=b.metadata(root,r);meta2=b.metadata(root,second)
    metadata={k:dict(first_sha256=hashlib.sha256(v).hexdigest(),second_sha256=hashlib.sha256(meta2[k]).hexdigest()) for k,v in meta1.items()}
    for e in list(arrays.values())+list(metadata.values()):e['identical']=e['first_sha256']==e['second_sha256']
    repro=dict(scope='TWO_COMPLETE_INDEPENDENT_SOURCE_REBUILDS_OF_DIAGNOSTIC23_WITH_SN_QUARANTINE',
               arrays=arrays,metadata=metadata,scaler_parameters_identical=r['params']==second['params'],
               all_identical=all(e['identical'] for e in list(arrays.values())+list(metadata.values())) and r['params']==second['params'])
    assert repro['all_identical']
    ac.write_json(out/'reproducibility_audit.json',repro)
    checks['build_started_at']=start;checks['protocol_sha256_at_build']=sha256_file(out/'frozen_protocol.json')
    checks['quote_read_log']=r['logs']+second['logs'];checks['TEST_numeric_rows_parsed']=0
    checks['input_sources']=r['sources'];checks['SN_quarantine']=sncheck
    checks['all_valid_source_values_checked']=True
    del second
    ac.write_json(out/'build_checks.json',checks)
    published=publish(root,r,repro,checks,sncheck)
    print(json.dumps(dict(published=published,origin_counts=checks['origin_counts'],valid_labels=checks['label_counts'],
                    SN_quarantine=sncheck),ensure_ascii=False),flush=True)

