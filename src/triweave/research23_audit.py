"""Scientific release tests, original-anchor oracle, and immutable publication."""
from __future__ import annotations
import copy
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from triweave import after_close as ac
from triweave import research23 as r23
from triweave.bridge_audit import mask_oracle
from triweave.sessionqa_match import match_bars,valid_ohlc
from triweave.sourceqa_investigate import load_official,FIELDS
from triweave.research23_quality import summarize,tail_audit
from triweave.universe import sha256_file


def preservation(root):
    files=json.loads((root/r23.OUT/'preservation_before.json').read_text())['files']
    changed=[name for name,h in files.items() if not (root/name).is_file() or sha256_file(root/name)!=h]
    if changed:raise RuntimeError('Old artifacts changed: '+repr(changed))
    return dict(protected_file_count=len(files),changed_files=[],old_artifacts_unchanged=True,
                comparison='OPAQUE_SHA256_NOT_TEST_QUOTE_INGESTION')


def original_label_oracle(a,split):
    dates=a['reference_dates'];q=a['input_valid_mask'][:,-23:];v=a['aligned_values'][:,-23:]
    lo,hi=ac.SPLITS[split];y=np.zeros((len(dates),4,23),np.float32);m=np.zeros(y.shape,bool)
    for i in range(len(dates)):
        for hk,h in enumerate((1,5,10,20)):
            end=i+h
            if end>=len(dates) or not lo<=dates[i]<=dates[end]<=hi:continue
            for j in range(23):
                if all(q[t,j] for t in range(i,end+1)):
                    value=v[end,j]/v[i,j]-1
                    if np.isfinite(value) and abs(value)<=np.finfo(np.float32).max:
                        m[i,hk,j]=True;y[i,hk,j]=value
    origins=np.flatnonzero(m.any((1,2))).astype(np.int32)
    return y,m,origins


def perturb_future_sources(frames,cutoff):
    changed={}
    for asset,frame in frames.items():
        f=frame.copy();future=f.date>cutoff
        for field in ('value','open','high','low','close','volume','amount','turnover','open_interest'):
            if field in f:
                f[field]=pd.to_numeric(f[field],errors='coerce').astype(float)
                f.loc[future,field]=f.loc[future,field]*1.53+.01
        changed[asset]=f
    return changed


def timing_audit(root,r):
    """Recheck current raw-replayed23 against frozen official ordinary bars."""
    out=root/r23.OUT;official,sources,logs=load_official(root)
    ordinary=official.contract.str.fullmatch(r'[A-Z]+[0-9]{3,4}').fillna(False)
    ordinary &= official.contract.str.extract(r'^([A-Z]+)',expand=False).eq(official.variety)
    official=official.loc[ordinary & official.variety.isin(r23.TARGETS)].copy()
    official['official_source_index']=official['__source']
    vendors=[]
    for c in r23.TARGETS:
        f=r['frames']['FUT_'+c][['date']+FIELDS].copy();f['variety']=c;f['asset_id']='FUT_'+c;vendors.append(f)
    vendor=pd.concat(vendors,ignore_index=True)
    current,matches=match_bars(vendor,official,r['calendar'])
    current.to_csv(out/'current23_session_matching.csv',index=False)
    good=valid_ohlc(official)&np.isfinite(official[FIELDS]).all(1)&official.volume.gt(0)&official.open_interest.gt(0)
    vv=vendor.loc[np.isfinite(vendor[FIELDS]).all(1)&vendor.volume.gt(0)&vendor.open_interest.gt(0)]
    full=vv.merge(official.loc[good,['date','variety','contract']+FIELDS],on=['variety']+FIELDS,suffixes=('_vendor','_official'))
    positions={d:i for i,d in enumerate(r['calendar'])}
    full['offset_sessions']=full.date_official.map(positions)-full.date_vendor.map(positions)
    forward=full.loc[full.offset_sessions.gt(0)].copy()
    rules={(q['asset_id'],q['date']) for q in r['rules']}
    forward['quarantined']=[key in rules for key in zip(forward.asset_id,forward.date_vendor)]
    forward.to_csv(out/'current23_forward_all7_matches.csv',index=False)
    unhandled=forward.loc[~forward.quarantined.astype(bool)].to_dict('records') if len(forward) else []
    rows=[];a=r['arrays'];ids=[x['asset_id'] for x in r['assets']]
    for j,spec in enumerate(r['assets']):
        q=a['input_valid_mask'][:,j];sub=current.loc[current.asset_id.eq(spec['asset_id'])]
        same=int(sub['full7_contract_matches_+0'].gt(0).sum()) if len(sub) else None
        comparable=int(sub['comparable_contracts_+0'].gt(0).sum()) if len(sub) else None
        rows.append(dict(asset_id=spec['asset_id'],market=spec['market'],economic_type=spec['economic_type'],
          declared_publication=r['protocol']['publication_assumptions'][spec['economic_type']]['local_time'],
          reference_session_delay=r['protocol']['publication_assumptions'][spec['economic_type']]['reference_session_delay'],
          available_at_status='AVAILABLE_AT_ASSUMED',true_historical_publication_verified=False,
          valid_inputs=int(q.sum()),clock_violations=int((a['available_at'][q,j]>a['decision_time'][q]).sum()),
          source_dates_after_reference=int((a['observation_day'][q,j]>a['reference_dates'][q].astype('datetime64[D]').astype(int)).sum()),
          ordinary_official_comparable_dates=comparable,same_date_all7_fingerprint_dates=same,
          forward_fingerprint_dates_quarantined=';'.join(q['date'] for q in r['rules'] if q['asset_id']==spec['asset_id'] and spec['asset_id']=='FUT_FU'),
          session_status=('NO_INDEPENDENT_DCE_SESSION_VERIFICATION' if comparable==0 else 'MATCHED_SUBSET_ONLY_NOT_GLOBAL_PIT' if comparable else 'MARKET_PUBLICATION_ASSUMPTION')))
    pd.DataFrame(rows).to_csv(out/'time_alignment_audit.csv',index=False)
    ac.write_json(out/'timing_evidence.json',dict(official_sources=sources,quote_read_log=logs,official_ordinary_rows=len(official),
      vendor_rows=len(vendor),forward_matches=forward.to_dict('records'),unhandled_forward_conflicts=unhandled,
      original_exchange_payload_business_date_verified=False,SDK_raw_date_limitation='Existing SDK exchange dates derive from requested archive date; not original HTTP business-date certification',
      date_remapping_applied=False,TEST_numeric_rows_parsed=0))
    # Every observed forward joint fingerprint is conservatively excluded;
    # lack of any match is not by itself classified as a source error.
    return dict(unhandled_forward_conflicts=unhandled,known_time_risk_handled=not unhandled,
                official_comparison_assets=int(current.groupby('variety')['comparable_contracts_+0'].max().gt(0).sum()),
                observed_forward_match_count=len(forward),official_read_log=logs)


def validate(root,r):
    a=r['arrays'];assets=r['assets'];dates=a['reference_dates'];aligned=r['aligned']
    assert r['calendar']==sorted(set(r['calendar'])) and max(dates)==ac.END
    assert a['features'].shape==(len(dates),311,26) and len(r['params'])==55
    assert [s['asset_id'] for s in assets][-23:]==['FUT_'+c for c in r23.TARGETS]
    expected=mask_oracle(aligned,assets)
    assert np.array_equal(expected,a['unscaled_feature_valid_mask']) and np.array_equal(expected,a['feature_valid_mask'])
    assert not (a['feature_valid_mask'] & ~a['feature_applicable_mask'][None]).any()
    q=a['input_valid_mask'];clock=np.broadcast_to(a['decision_time'][:,None],q.shape)
    assert (a['available_at'][q]<=clock[q]).all()
    for j,spec in enumerate(assets):
        if spec['economic_type'] in (ac.KINDS[0],ac.KINDS[3]):
            assert np.array_equal(a['observation_day'][q[:,j],j],dates[q[:,j]].astype('datetime64[D]').astype(int))
        else:assert (a['age_days'][q[:,j],j]<=5).all()
    for name,v in a.items():
        if v.dtype.kind=='f':assert np.isfinite(v).all(),name
    assert (a['features'][~a['feature_valid_mask']]==0).all()
    assert (a['aligned_values'][~q]==0).all()
    for split in ('TRAIN','VAL'):
        s=split.lower();y,m,origins=original_label_oracle(a,split)
        assert np.array_equal(origins,a[s+'_origin_indices'])
        assert np.array_equal(m[origins],a[s+'_label_valid_mask']) and np.array_equal(y[origins],a[s+'_targets'])
        assert np.array_equal(origins,a[s+'_anchor_indices'])
        for si,i in enumerate(origins):
            assert (a[s+'_window_indices'][si]<=i).all()
            for hk,h in enumerate(ac.HORIZONS):
                good=m[i,hk];end=int(a[s+'_endpoint_indices'][si,hk])
                if good.any():
                    assert end==i+h and ac.SPLITS[split][0]<=dates[i]<dates[end]<=ac.SPLITS[split][1]
                    assert (a['available_at'][i,-23:][good]<=a['decision_time'][i]).all()
                    assert (a['available_at'][end,-23:][good]>a['decision_time'][i]).all()
    # Modify actual future source quotes, re-align and rebuild, then refitTRAIN.
    # A VAL cutoff allows a full fresh scaler fit without changing anyTRAIN row.
    cutoff='2023-06-30';changed=perturb_future_sources(r['frames'],cutoff)
    aa,_=ac.align_sources(changed,assets,r['calendar'],r['protocol']);aa,_=r23.apply_quarantine(aa,assets,r['rules'])
    x,m,_=ac.feature_panel(aa,assets);params=ac.fit_scaler(x,m,dates,assets,r['protocol']);assert params==r['params']
    z,fm=ac.transform(x,m,assets,params);prefix=dates<=cutoff
    assert np.array_equal(x[prefix],a['unscaled_features'][prefix]) and np.array_equal(m[prefix],a['unscaled_feature_valid_mask'][prefix])
    assert np.array_equal(z[prefix],a['features'][prefix]) and np.array_equal(fm[prefix],a['feature_valid_mask'][prefix])
    del changed,aa,x,m,z,fm
    changed_x=a['unscaled_features'].copy();changed_m=a['unscaled_feature_valid_mask'].copy();val=dates>'2022-12-31'
    changed_x[val]=987654321.;changed_m[val]=~changed_m[val]
    assert ac.fit_scaler(changed_x,changed_m,dates,assets,r['protocol'])==r['params']
    del changed_x,changed_m,expected
    for p in r['params']:
        js=[j for j,x in enumerate(assets) if x['economic_type']==p['economic_type']];k=p['feature_index'];train=dates<='2022-12-31'
        values=a['unscaled_features'][train][:,js,k][a['unscaled_feature_valid_mask'][train][:,js,k]]
        assert len(values)==p['valid_train_count'] and float(values.mean())==p['mean_train'] and float(values.std())==p['std_train']
    assert all(l['TEST_numeric_rows_parsed']==0 and (l['max_parsed_date'] is None or l['max_parsed_date']<=ac.END) for l in r['logs'])
    return dict(independent_feature_mask_oracle=True,independent_original_label_oracle=True,all_horizon_boundaries=True,
      all_declared_input_clocks_pass=True,current_anchor_declared_available=True,FWD3_used=False,
      prefix_cutoff=cutoff,prefix_actual_source_perturbation_pass=True,TRAIN_scaler_refit_in_prefix_test=True,
      arbitrary_VAL_values_and_masks_do_not_change_TRAIN_scaler=True,unique_TRAIN_scaler_parameters_independently_recomputed=True,
      finite_values_and_masked_placeholders_pass=True,TEST_numeric_rows_parsed=0)


def quarantine_audit(root,r,control):
    a=r['arrays'];old=control['arrays'];dates=a['reference_dates'];assets=r['assets'];ids=[s['asset_id'] for s in assets]
    cone=np.zeros(a['feature_valid_mask'].shape,bool);records=[];label_usage=[]
    for rule in r['rules']:
        j=ids.index(rule['asset_id']);point=int(np.flatnonzero(dates==rule['date'])[0])
        assert not a['input_valid_mask'][point,j] and old['input_valid_mask'][point,j]
        assert r['aligned']['value'][point,j]==r['original']['value'][point,j]
        for feature in ac.registry():
            k=feature['index']
            if a['feature_applicable_mask'][j,k]:cone[point:point+feature['lookback'],j,k]=True
        for split in ('train','val'):
            for hk,h in enumerate(ac.HORIZONS):
                uses=(a[split+'_origin_indices']<=point)&(point<=a[split+'_origin_indices']+h)
                assert not a[split+'_label_valid_mask'][uses,hk,j-288].any()
                label_usage.append(dict(asset_id=rule['asset_id'],source_date=rule['date'],split=split,horizon=h,
                   possible_origin_windows=int(uses.sum()),retained_invalid_source_labels=0,
                   original_control_valid_labels=int(old[split+'_label_valid_mask'][uses,hk,j-288].sum())))
        records.append(dict(asset_id=rule['asset_id'],date=rule['date'],original_close=float(r['original']['value'][point,j]),
             source_value_changed=False,input_mask_rejected=True,feature_dependencies_masked=True,label_window_dependencies_rejected=True,
             reason=rule['reason'],source_identity_mismatch_supported=rule['asset_id']=='FUT_SN',confirmed_date_shift=False,replacement_price_used=False))
    local=(a['unscaled_feature_valid_mask']!=old['unscaled_feature_valid_mask'])|~np.isclose(a['unscaled_features'],old['unscaled_features'],atol=1e-9,rtol=1e-10)
    assert not (local&~cone).any()
    assert not ((a['feature_valid_mask']!=old['feature_valid_mask'])&~cone).any()
    roundoff=(a['unscaled_features']!=old['unscaled_features'])&~cone
    params=[]
    for p,q in zip(r['params'],control['params']):
        if p!=q:params.append(dict(economic_type=p['economic_type'],feature_name=p['feature_name'],statistics_changed=True,
             transformation_changed=any(p[k]!=q[k] for k in ('mean_train','effective_scale'))))
    assert all(p['economic_type']==ac.KINDS[3] for p in params)
    pd.DataFrame(records).to_csv(root/r23.OUT/'quarantine_dependency_audit.csv',index=False)
    pd.DataFrame(label_usage).to_csv(root/r23.OUT/'quarantine_label_dependency_audit.csv',index=False)
    pd.DataFrame(params).to_csv(root/r23.OUT/'quarantine_scaler_group_changes.csv',index=False)
    result=dict(raw_values_changed=0,observations_masked=len(records),input_mask_changes=int((a['input_valid_mask']!=old['input_valid_mask']).sum()),
       feature_mask_changes=int((a['feature_valid_mask']!=old['feature_valid_mask']).sum()),
       scaler_stat_groups_changed=len(params),scaler_transform_groups_changed=sum(x['transformation_changed'] for x in params),
       no_material_unexplained_changes=True,outside_cone_roundoff_elements=int(roundoff.sum()),
       max_outside_cone_roundoff=float(np.abs(a['unscaled_features'][roundoff]-old['unscaled_features'][roundoff]).max()) if roundoff.any() else 0.,
       label_masks_changed={s:{str(h):int((a[s+'_label_valid_mask'][:,hk]!=old[s+'_label_valid_mask'][:,hk]).sum()) for hk,h in enumerate(ac.HORIZONS)} for s in ('train','val')},
       control_role='Unquarantined current311-node original-return ablation, memory only, not Bridge or formal25 data')
    ac.write_json(root/r23.OUT/'quarantine_impact.json',result)
    return result


def run(root):
    root=Path(root);out=root/r23.OUT;dest=root/r23.DATA
    if (out/'build_results.json').exists() or dest.exists():raise FileExistsError('Append-only research23; use --verify')
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    print('Research23 raw-source build1, original t anchor, no input lag/FWD3',flush=True)
    r=r23.calculate(root);checks=validate(root,r)
    print('Current23 ordinary-contract date audit from frozen exchange files',flush=True)
    timing=timing_audit(root,r)
    print('Unquarantined current23 source ablation for dependency and scaler audit',flush=True)
    control=r23.calculate(root,False,cached=(r['frames'],r['sources'],r['logs'],r['replay']))
    impact=quarantine_audit(root,r,control);del control
    summary=summarize(root,r);normalization=tail_audit(root,r)
    print('Research23 independent raw-source build2',flush=True)
    second=r23.calculate(root)
    arrays={k:dict(first_sha256=ac.array_sha(v),second_sha256=ac.array_sha(second['arrays'][k])) for k,v in r['arrays'].items()}
    md1=r23.metadata(root,r);md2=r23.metadata(root,second)
    metadata={k:dict(first_sha256=hashlib.sha256(v).hexdigest(),second_sha256=hashlib.sha256(md2[k]).hexdigest()) for k,v in md1.items()}
    for e in list(arrays.values())+list(metadata.values()):e['identical']=e['first_sha256']==e['second_sha256']
    identical=all(e['identical'] for e in list(arrays.values())+list(metadata.values())) and r['params']==second['params']
    assert identical
    ac.write_json(out/'reproducibility_audit.json',dict(arrays=arrays,metadata=metadata,scaler_identical=r['params']==second['params'],
       all_identical=identical,scope='TWO_INDEPENDENT_RAW_RENORMALIZATION_BUILDS_NO_OLD_NPY_OR_SCALER_INPUT'))
    logs=r['logs']+second['logs']+timing['official_read_log'];del second
    flags=dict(TARGET_23_UNIVERSE_PASS=True,INPUT_311_ASSETS_PASS=True,THREE_SPLIT_PROTOCOL_PASS=True,
      ORIGINAL_RETURN_TARGET_RESTORED=checks['independent_original_label_oracle'],FWD3_TARGET_NOT_USED=not checks['FWD3_used'],
      AFTER_CLOSE_TIMING_AUDITED=True,CROSS_MARKET_ALIGNMENT_PASS=checks['all_declared_input_clocks_pass'],
      SN_QUARANTINE_PASS=impact['observations_masked']==3,SOURCE_ANOMALY_AUDIT_PASS=timing['known_time_risk_handled'],
      FEATURE_TYPE_SEMANTICS_PASS=checks['independent_feature_mask_oracle'],MISSINGNESS_MASK_PASS=checks['independent_feature_mask_oracle'],
      TRAIN_ONLY_SCALER_PASS=checks['unique_TRAIN_scaler_parameters_independently_recomputed'],NORMALIZATION_NUMERIC_PASS=all(normalization[s]['all_finite'] for s in ('TRAIN','VAL')),
      RETURN_LABEL_PASS=checks['independent_original_label_oracle'],HORIZON_BOUNDARY_PASS=checks['all_horizon_boundaries'],
      PREFIX_CAUSALITY_PASS=checks['prefix_actual_source_perturbation_pass'],TEST_ISOLATION_PASS=True,
      DATASET_REPRODUCIBILITY_PASS=identical,DATASET_TECHNICALLY_VALID=True,EXPLORATORY_RESEARCH_READY=False,
      HISTORICAL_PIT_VERIFIED=False,FUTURES_ROLL_METHOD_VERIFIED=False,CONFIRMATORY_RESEARCH_READY=False)
    flags['EXPLORATORY_RESEARCH_READY']=all(flags[k] is True for k in list(flags)[:19]) and timing['known_time_risk_handled']
    protection=preservation(root)
    ac.write_json(out/'audit_results.json',dict(flags,protocol_id=r23.PROTOCOL,availability_status='AVAILABLE_AT_ASSUMED',
      timing_gate_scope='DECLARED_EXPLORATORY_ASSUMPTIONS_WITH_ALL_OBSERVED_FORWARD_FINGERPRINTS_QUARANTINED_NOT_REAL_HISTORICAL_PIT',
      origin_counts={s:summary[s]['origin_count'] for s in ('TRAIN','VAL')},label_counts={s:summary[s]['label_counts'] for s in ('TRAIN','VAL')},
      known_unhandled_time_risk_records=timing['unhandled_forward_conflicts']))
    result=dict(started_at=started,protocol_sha256=sha256_file(out/'frozen_protocol.json'),checks=checks,timing=timing,
       quarantine=impact,coverage=summary,normalization=normalization,sources=r['sources'],quote_read_log=logs,TEST_numeric_rows_parsed=0,
       raw_source_replay_rows=len(r['replay']),raw_replay_numeric_differences=sum(x['exact_numeric_differences'] for x in r['replay']),
       old_artifact_check=protection,model_runs=0,MAE_computations=0,new_downloads=0)
    ac.write_json(out/'build_results.json',result)
    files={k+'.npy':dict(sha256=ac.array_sha(v),shape=list(v.shape),dtype=str(v.dtype)) for k,v in r['arrays'].items()}
    for name,payload in md1.items():files[name]=dict(sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload))
    manifest=dict(protocol_id=r23.PROTOCOL,DATASET_TECHNICALLY_VALID=True,EXPLORATORY_RESEARCH_READY=flags['EXPLORATORY_RESEARCH_READY'],
      release_status='EXPLORATORY_RESEARCH_RELEASED' if flags['EXPLORATORY_RESEARCH_READY'] else 'QUARANTINED_CANDIDATE',
      input_nodes=311,target_count=23,feature_count=26,window_reference_sessions=20,horizons=[1,5,10,20],
      label_formula='P(t+h)/P(t)-1',label_anchor_offset=0,label_unit='decimal_return',FWD3_used=False,input_additional_lag=0,
      split_summary=summary,files=files,protocol_sha256=sha256_file(out/'frozen_protocol.json'),
      TEST_dataset_generated=False,HISTORICAL_PIT_VERIFIED=False,FUTURES_ROLL_METHOD_VERIFIED=False,CONFIRMATORY_RESEARCH_READY=False,
      asset_availability_mask_alias='input_valid_mask.npy',loader='triweave.research23.Research23Dataset')
    stage=dest.parent/'.triweave_23_after_close_research_v1_build'
    if stage.exists():raise FileExistsError('Refuse existing stage overwrite')
    stage.mkdir()
    for k,v in r['arrays'].items():np.save(stage/(k+'.npy'),v,allow_pickle=False)
    for name,payload in md1.items():(stage/name).write_bytes(payload)
    for name,e in files.items():assert sha256_file(stage/name)==e['sha256']
    ac.write_json(stage/'dataset_manifest.json',manifest);stage.rename(dest)
    ac.write_json(out/'dataset_manifest.json',manifest)
    for split in ('TRAIN','VAL'):
        ds=r23.Research23Dataset(dest,split,allow_quarantined=True)
        for index in range(len(ds)):
            sample=ds[index];pad=sample['reference_indices']<0
            assert sample['X'].shape==(20,311,26) and sample['Y'].shape==(4,23)
            assert not sample['input_valid_mask'][pad].any() and not sample['feature_valid_mask'][pad].any()
            assert (sample['X'][pad]==0).all() and (sample['assumed_available_at'][pad]==-1).all()
            assert sample['label_anchor_date']==sample['origin_date']
    ac.write_json(out/'publication.json',dict(dataset_path=r23.DATA,bytes=sum(p.stat().st_size for p in dest.iterdir()),files=len(files)+1,
       all_files_sha_verified=True,all_sample_mmap_padding_and_original_anchor_pass=True,old_artifacts=protection))
    print(json.dumps(dict(EXPLORATORY_RESEARCH_READY=flags['EXPLORATORY_RESEARCH_READY'],coverage=summary,quarantine=impact),ensure_ascii=False),flush=True)
