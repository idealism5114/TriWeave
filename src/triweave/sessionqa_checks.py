"""Read-only V1 replay and temporal release audit; no new dataset is generated."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
from triweave import after_close as ac
from triweave.sourceqa_impact import calculate
from triweave.sessionqa_match import OUT
from triweave.universe import sha256_file


def mask_minimum_contract(aligned, assets, rules):
    """Diagnostic rule contract only. Does not certify unresolved source dates."""
    a=copy.deepcopy(aligned)
    ids=[s['asset_id'] for s in assets]
    for r in rules:
        j=ids.index(r['asset_id'])
        hit=(a['dates']>=r['start_date']) & (a['dates']<=r['end_date'])
        # Check source observation identity, not a carried quote's alignment day.
        days=a['observation_day'][:,j]
        hit &= (days>=np.datetime64(r['start_date'],'D').astype(int)) & (days<=np.datetime64(r['end_date'],'D').astype(int))
        a['valid'][hit,j]=False
        a['new_observation'][hit,j]=False
    return a


def independent_label_masks(q, dates, split):
    lo,hi=ac.SPLITS[split]
    m=np.zeros((len(dates),4,25),bool)
    for hk,h in enumerate(ac.HORIZONS):
        for i in range(len(dates)-h):
            if lo<=dates[i]<=dates[i+h]<=hi:
                m[i,hk]=np.logical_and.reduce(q[i:i+h+1],axis=0)
    return m


def run(root):
    root=Path(root);out=root/OUT;dest=root/ac.DATA_REL
    if (out/'v1_replay_checks.json').exists():raise FileExistsError('Replay sealed')
    disposition=json.loads((out/'frozen_disposition.json').read_text())
    assert not disposition['new_dataset_generation_allowed']
    assert not (root/'data/derived/triweave_25_after_close_v1_1').exists()
    old={p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in dest.glob('*.npy')}
    print('V1 independent source replay1 (audit only)',flush=True)
    r=calculate(root,False);a=r['arrays'];dates=a['reference_dates'];assets=r['assets']
    hashes={k:dict(old=sha256_file(dest/(k+'.npy')),replay1=ac.array_sha(v)) for k,v in a.items()}
    assert all(v['old']==v['replay1'] for v in hashes.values())
    params=json.loads((dest/'scaler_params.json').read_text())
    assert params==r['params'] and len(params)==55
    assert max(dates)<=ac.END
    prefix=dates<='2021-06-30'
    perturbed=copy.deepcopy(r['aligned'])
    for k in ('value','open','high','low','volume','amount','turnover','open_interest'):
        perturbed[k][~prefix]=perturbed[k][~prefix]*1.53+.01
    xx,mm,_=ac.feature_panel(perturbed,assets)
    assert np.array_equal(xx[prefix],a['unscaled_features'][prefix])
    assert np.array_equal(mm[prefix],a['unscaled_feature_valid_mask'][prefix])
    zz,zm=ac.transform(xx,mm,assets,params)
    assert np.array_equal(zz[prefix],a['features'][prefix])
    # This tests offline TRAIN fit isolation, not historical online scaler PIT.
    xx=a['unscaled_features'].copy();xx[dates>'2022-12-31']=987654321.
    assert ac.fit_scaler(xx,a['unscaled_feature_valid_mask'],dates,assets,r['protocol'])==params
    del xx,mm,zz,zm,perturbed
    rules=disposition['minimum_quarantine_rules']
    masked=mask_minimum_contract(r['aligned'],assets,rules)
    rows=[];counts={};minimal_counts={};known_risk_failures=[]
    for split in ('train','val'):
        origins=a[split+'_origin_indices'];original=independent_label_masks(a['input_valid_mask'][:,-25:],dates,split.upper())
        assert np.array_equal(original[origins],a[split+'_label_valid_mask'])
        minimal=independent_label_masks(masked['valid'][:,-25:],dates,split.upper())
        counts[split]={str(h):int(original[:,hk].sum()) for hk,h in enumerate(ac.HORIZONS)}
        minimal_counts[split]={str(h):int(minimal[:,hk].sum()) for hk,h in enumerate(ac.HORIZONS)}
        for si,i in enumerate(origins):
            assert (a[split+'_window_indices'][si]<=i).all()
            for hk,h in enumerate(ac.HORIZONS):
                good=original[i,hk]
                if not good.any():continue
                e=int(a[split+'_endpoint_indices'][si,hk])
                lo,hi=ac.SPLITS[split.upper()]
                assert e==i+h and lo<=dates[i]<=dates[e]<=hi
                expected=(a['aligned_values'][e,-25:][good]/a['aligned_values'][i,-25:][good]-1).astype(np.float32)
                assert np.array_equal(a[split+'_targets'][si,hk,good],expected)
        for rule in rules:
            j=next(i for i,s in enumerate(assets) if s['asset_id']==rule['asset_id']);tj=j-288
            idx=int(np.flatnonzero(dates==rule['start_date'])[0]);day=dates[idx]
            for hk,h in enumerate(ac.HORIZONS):
                uses=np.array([i<=idx<=i+h for i in range(len(dates))]) & original[:,hk,tj]
                anchors=int(original[idx,hk,tj])
                ends=int(sum(original[i,hk,tj] for i in range(max(0,idx-h),idx+1) if i+h==idx))
                after=int(minimal[uses,hk,tj].sum())
                assert after==0 and not masked['valid'][idx,j]
                if split=='train':
                    rows.append(dict(asset_id=rule['asset_id'],vendor_date=day,split=split,horizon=h,
                                     V1_quote_available=bool(a['input_valid_mask'][idx,j]),
                                     minimum_rule_quote_available=False,V1_anchor_valid=anchors,V1_endpoint_labels=ends,
                                     V1_labels_using_observation=int(uses.sum()),minimum_rule_labels_using_observation=after,
                                     rule_scope='MINIMUM_BOOLEAN_MASK_CONTRACT_ONLY_NO_COMPLETE_CANDIDATE',
                                     complete_candidate_time_consistency='BLOCKED',
                                     V1_input_windows_using_record=int(sum((w==idx).any() for w in a['train_window_indices'])),
                                     minimum_rule_inputs_using_record=0,
                                     effective_available_time_basis=('SOURCE_IDENTITY_REJECTED' if rule['asset_id']=='FUT_SN' else
                                       'DATED_FIRST_NIGHT_CLOSE_SUPPORTS_POST_20_EVENT' if day=='2013-07-05' else
                                       'ALL7_NEXT_ARCHIVE_DAY_CONSERVATIVE_RISK_BOUND_NOT_HISTORICAL_PIT')))
                    if rule['asset_id']!='FUT_SN' and anchors:
                        known_risk_failures.append(dict(asset_id=rule['asset_id'],date=day,horizon=h,
                                                       V1_known_session_risk_retained=True))
    pd.DataFrame(rows).to_csv(out/'input_label_temporal_consistency.csv',index=False)
    # All valid inputs satisfy declared V1 clocks. This is insufficient for real session causality.
    declared_pass=bool((a['available_at'][a['input_valid_mask']]<=np.broadcast_to(a['decision_time'][:,None],a['available_at'].shape)[a['input_valid_mask']]).all())
    assert declared_pass
    # Known conflict regression uses explicit conservative alternative time bounds,
    # never just the potentially wrong vendor date.
    risk_cases=[]
    for rule in rules:
        if rule['asset_id']=='FUT_SN':continue
        idx=int(np.flatnonzero(dates==rule['start_date'])[0]);j=next(i for i,s in enumerate(assets) if s['asset_id']==rule['asset_id'])
        bound=(int(ac.clock_seconds(np.array(['2013-07-06']),2,30)[0]) if rule['start_date']=='2013-07-05' else
               int(ac.clock_seconds(np.array(['2013-07-09']),19)[0]))
        assert bound>int(a['decision_time'][idx])
        assert a['input_valid_mask'][idx,j] and not masked['valid'][idx,j]
        risk_cases.append(dict(asset_id=rule['asset_id'],vendor_date=rule['start_date'],
                               decision_time=int(a['decision_time'][idx]),risk_available_time_bound=bound,
                               V1_input_fails_conservative_bound=True,minimum_rule_input_rejected=True,
                               minimum_rule_anchor_all_horizons_rejected=True,
                               entire_history_covered_by_rule=False))
    padding=[]
    for split in ('TRAIN','VAL'):
        ds=ac.AfterCloseDataset(dest,split)
        for i in range(len(ds)):
            sample=ds[i];pad=sample['reference_indices']<0
            assert not sample['input_valid_mask'][pad].any() and not sample['feature_valid_mask'][pad].any()
            assert (sample['X'][pad]==0).all()
            for k in ('observation_day','available_at','source_index','quote_age_days','trading_status'):
                assert (sample[k][pad]==-1).all()
            if split=='TRAIN' and i<30:
                padding.append(dict(origin=sample['origin_date'],left_padding=int(pad.sum()),future_copy=False))
    pd.DataFrame(padding).to_csv(out/'padding_audit.csv',index=False)
    # Save only hashes and diagnostics. No V1.1 directory, features or scaler.
    result=dict(scope='IMMUTABLE_V1_REPLAY_AND_MINIMUM_RULE_DIAGNOSTICS_NOT_V1_1',
                V1_prefix_causality_pass=True,V1_TRAIN_only_scaler_pass=True,V1_label_masks_independently_recomputed=True,
                V1_horizon_and_split_boundaries_pass=True,V1_padding_all_origins_pass=True,
                V1_declared_available_at_pass=declared_pass,
                TRADING_SESSION_CAUSALITY_TEST=False,
                session_test_reason='V1 retains4 known date-risk observations. Minimum rule excludes them, but unknown affected extent prevents complete candidate.',
                minimum_rule_contract_pass=True,minimum_rule_source_observations_masked=5,
                full_temporal_protocol_verified=False,risk_cases=risk_cases,known_risk_anchor_cases=known_risk_failures,
                V1_origin_counts={s:len(a[s+'_origin_indices']) for s in ('train','val')},
                V1_valid_labels=counts,minimum_rule_boolean_label_counts_NOT_A_DATASET=minimal_counts,
                complete_candidate_origin_counts='NOT_VERIFIED',complete_candidate_label_counts='NOT_VERIFIED',
                TEST_numeric_rows_parsed=0,quote_read_log=r['logs'],raw_values_changed=0,
                scaler_replay_scope='Fresh55-group fit reproduces V1 exactly; no new version or complete candidate refit',
                old_HOLD_sha256=sha256_file(dest/'RESEARCH_HOLD.json'))
    del r,a,masked,old
    print('V1 independent source replay2 (audit only)',flush=True)
    second=calculate(root,False)
    for k,v in second['arrays'].items():hashes[k]['replay2']=ac.array_sha(v)
    assert all(v['old']==v['replay1']==v['replay2'] for v in hashes.values())
    assert params==second['params']
    result['quote_read_log']+=second['logs']
    result['V1_reproducible_twice']=True;result['arrays_sha256']=hashes
    result['DATASET_V1_1_GENERATED']=False
    ac.write_json(out/'v1_replay_checks.json',result)
    print(json.dumps({k:result[k] for k in ['TRADING_SESSION_CAUSALITY_TEST','V1_origin_counts','V1_valid_labels','minimum_rule_boolean_label_counts_NOT_A_DATASET','V1_reproducible_twice']},ensure_ascii=False),flush=True)
    return result


if __name__=='__main__':run(Path.cwd())
