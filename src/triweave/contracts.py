"""Phase D-Contracts read-only audit and TRAIN-only mechanical diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd

from triweave.assets import ALL_CONTRACT_FUTURES,validate_asset_configuration
from triweave.collect import atomic_json,utc_now
from triweave.contract_identity import VERIFIED,build_identity_manifest,shfe_metadata
from triweave.pipeline import environment
from triweave.roll_diagnostics import lagged_activity_selection,same_contract_checks,adjacent_contract_price_relations
from triweave.universe import sha256_file

RUN='contracts_20261015'
START,END='2019-10-09','2019-11-08'


def load_contract_quotes(root):
    frames=[];jobs=[]
    baseline=root/'data/manifests/baseline15y/jobs/futures_contracts'
    paths=list(baseline.rglob('*.json'))
    paths+=list((root/'data/manifests/rescue_20261015/gfex_official_akshare_v1/futures_contracts_daily_partial_v3').glob('*.json'))
    for p in sorted(paths):
        m=json.loads(p.read_text())
        if m.get('status')=='OK':
            group=m['job_group'];exchange=group.split('/')[-1]
            raw=root/'data/raw/baseline15y'/group/(m['name']+'.csv.gz')
            std=root/'data/standardized/baseline15y'/group/(m['name']+'.csv.gz')
            source=exchange.lower()+'_official_akshare_v1';retrieved=m['retrieved_utc']
        elif m.get('category')=='futures_contracts_daily_partial_v3' and m.get('status','').startswith('OBSERVATIONS_SAVED'):
            raw=root/m['raw_file'];std=root/m['standardized_file'];source=m['source_id']
            retrieved=m['params']['upstream_retrieved_at']
        else:continue
        if sha256_file(raw)!=m['raw_sha256'] or sha256_file(std)!=m['standardized_sha256']:
            raise ValueError('Existing archive fingerprint mismatch: '+str(p))
        f=pd.read_csv(std);f['source_id']=source;f['manifest']=str(p.relative_to(root))
        f['raw_file']=str(raw.relative_to(root));f['retrieved_at']=retrieved
        frames.append(f);jobs.append({'manifest':str(p.relative_to(root)),**m})
    if not frames:raise RuntimeError('No verified archive files')
    return pd.concat(frames,ignore_index=True),jobs


def quality_counts(f):
    if f.empty:return dict(record_count=0,active_contract_day_rows=0,zero_volume_rows=0,missing_OI_rows=0,
        negative_activity_rows=0,nonpositive_or_missing_close_rows=0,nonpositive_or_missing_settle_rows=0,
        invalid_ohlc_rows=0,special_mode_rows=0)
    prices=f[['open','high','low','close']]
    invalid=prices.isna().any(axis=1)|prices.le(0).any(axis=1)|prices.isin([float('inf'),float('-inf')]).any(axis=1)
    invalid|=f.high.lt(prices[['open','low','close']].max(axis=1))|f.low.gt(prices[['open','high','close']].min(axis=1))
    special=f.contract.str.contains('_TAS',na=False)
    return {'record_count':len(f),'active_contract_day_rows':int((f.volume.gt(0)&~invalid&~special).sum()),
        'zero_volume_rows':int(f.volume.eq(0).sum()),'missing_OI_rows':int(f.open_interest.isna().sum()),
        'negative_activity_rows':int(f[['volume','open_interest']].lt(0).any(axis=1).sum()),
        'nonpositive_or_missing_close_rows':int((f.close.isna()|f.close.le(0)).sum()),
        'nonpositive_or_missing_settle_rows':int((f.settle.isna()|f.settle.le(0)).sum()),
        'invalid_ohlc_rows':int(invalid.sum()),'special_mode_rows':int(special.sum())}


def vendor_train_diagnostic(root,exchange,variety):
    name=f'{exchange}_{variety}_0'
    path=root/'data/standardized/baseline15y/futures_continuous_exploratory'/(name+'.csv.gz')
    mpath=root/'data/manifests/baseline15y/jobs/futures_continuous_exploratory'/(name+'.json')
    m=json.loads(mpath.read_text())
    if m['status']!='OK' or sha256_file(path)!=m['standardized_sha256']:
        raise RuntimeError('Vendor archive missing or checksum changed')
    f=pd.read_csv(path);f=f.loc[f.date.between(START,END)].copy()
    if f.empty:raise RuntimeError('No real vendor rows in fixed TRAIN window')
    f['variety']=variety;f['vendor_price_level_change_not_return']=f.close.diff()
    f['artifact_classification']='NOT_VERIFIED_VENDOR_ROLL_MAPPING_ABSENT'
    f['manifest']=str(mpath.relative_to(root))
    return f


def real_train_checks(root,quotes,metadata,out):
    tests=[];all_selections=[];activities=[];pairs=[];switches=[];distribution=[];vendor=[];term_gaps=[]
    for variety,exchange in (('CU','SHFE'),('RB','SHFE'),('AU','SHFE'),('CF','CZCE')):
        q=quotes.loc[quotes.variety.eq(variety)&quotes.exchange.eq(exchange)&quotes.date.between(START,END)].copy()
        if q.empty:raise RuntimeError('Required real TRAIN test empty: '+variety)
        if q.duplicated(['date','contract']).any():raise RuntimeError('Duplicate TRAIN source observations')
        days=sorted(q.date.unique());cut=days[len(days)//2]
        for date,f in q.groupby('date'):
            active=f.loc[f.volume.gt(0)&f.close.gt(0)]
            activities.append({'variety':variety,'date':date,'listed_records_observed':len(f),'positive_volume_contracts':len(active),
                'volume_sum_native':f.volume.sum(),'OI_sum_native':f.open_interest.sum(),
                'calendar_status':'OBSERVED_EXCHANGE_DAYS_NOT_A_VERIFIED_PIT_CALENDAR'})
            for r in f.itertuples():distribution.append({'variety':variety,'date':date,'contract':r.contract,'volume':r.volume,
                'open_interest':r.open_interest,'close':r.close,'settle':r.settle,'raw_file':r.raw_file,'manifest':r.manifest})
        check=same_contract_checks(q,days)
        for col in ('variety','contract','date','previous_date','previous_close','close','diagnostic_same_contract_price_change','diagnostic_status','raw_file','manifest','official_label'):
            if col not in check:check[col]=variety if col=='variety' else None
        pairs.append(check[['variety','contract','date','previous_date','previous_close','close','diagnostic_same_contract_price_change','diagnostic_status','raw_file','manifest','official_label']])
        vendor.append(vendor_train_diagnostic(root,exchange,variety))
        registry=metadata.loc[metadata.variety.eq(variety)&metadata.verification_status.eq(VERIFIED)].copy()
        if registry.empty:
            tests.append({'variety':variety,'exchange':exchange,'first_date':q.date.min(),'last_date':q.date.max(),
                'rows':len(q),'rule_A_status':'BLOCKED_PIT_CALENDAR_AND_IDENTITY_NOT_VERIFIED',
                'rule_B_status':'BLOCKED_IDENTITY_NOT_VERIFIED','prefix_test':'BLOCKED_IDENTITY_NOT_VERIFIED',
                'same_code_pairs':int(check.diagnostic_same_contract_price_change.notna().sum()),'official_labels':False})
            continue
        before=lagged_activity_selection(q,registry,days)
        term=adjacent_contract_price_relations(q,registry);term['variety']=variety;term_gaps.append(term)
        mutated=q.copy();future=mutated.date.gt(cut)
        mutated.loc[future,['volume','open_interest','close']]=[999999999,999999999,999999999]
        after=lagged_activity_selection(mutated,registry,days)
        pd.testing.assert_frame_equal(before.loc[before.decision_date<=cut],after.loc[after.decision_date<=cut])
        prefix_days=[d for d in days if d<=cut]
        prefix=lagged_activity_selection(q.loc[q.date<=cut],registry,prefix_days)
        pd.testing.assert_frame_equal(before.iloc[:len(prefix_days)].reset_index(drop=True),prefix)
        before['variety']=variety;all_selections.append(before)
        old=None
        for r in before.itertuples():
            new=r.contract
            if pd.isna(new):continue
            if old and new!=old:
                f=q.loc[q.date.eq(r.decision_date)].set_index('contract')
                a=f.loc[old,'close'] if old in f.index else None;b=f.loc[new,'close'] if new in f.index else None
                switches.append({'variety':variety,'date':r.decision_date,'old_contract':old,'new_contract':new,
                    'old_same_day_close':a,'new_same_day_close':b,
                    'same_day_term_price_difference':b-a if a is not None and b is not None else None,
                    'meaning':'Term basis only; not cross-contract return or executable PnL','official_label':False})
            old=new
        tests.append({'variety':variety,'exchange':exchange,'first_date':q.date.min(),'last_date':q.date.max(),
            'rows':len(q),'directory_contracts_verified':len(registry),
            'quote_contracts_outside_snapshot':q.loc[~q.contract.isin(registry.contract_code),'contract'].nunique(),
            'rule_A_status':'BLOCKED_EX_ANTE_CALENDAR_NOT_VERIFIED',
            'rule_B_status':'MECHANICAL_DIAGNOSTIC_ONLY_EXECUTION_PIT_NOT_VERIFIED',
            'prefix_test':'PASS','prefix_cutoff':cut,'past_decisions_compared':len(prefix_days),
            'days_with_selected_contract':int(before.contract.notna().sum()),
            'same_code_pairs':int(check.diagnostic_same_contract_price_change.notna().sum()),'official_labels':False})
    pd.DataFrame(tests).to_csv(out/'train_contract_checks.csv',index=False)
    pd.DataFrame(activities).to_csv(out/'train_active_contracts.csv',index=False)
    pd.DataFrame(distribution).to_csv(out/'train_activity_distribution.csv',index=False)
    pd.concat(pairs,ignore_index=True).to_csv(out/'train_same_contract_arithmetic.csv',index=False)
    pd.concat(all_selections,ignore_index=True).to_csv(out/'train_lagged_selection_diagnostics.csv',index=False)
    pd.DataFrame(switches,columns=['variety','date','old_contract','new_contract','old_same_day_close','new_same_day_close','same_day_term_price_difference','meaning','official_label']).to_csv(out/'train_switch_price_checks.csv',index=False)
    if vendor:pd.concat(vendor,ignore_index=True).to_csv(out/'train_vendor_continuous_diagnostics.csv',index=False)
    if term_gaps:pd.concat(term_gaps,ignore_index=True).to_csv(out/'train_adjacent_contract_price_relations.csv',index=False)
    return tests


def report(root):
    root=Path(root).resolve();out=root/'experiments/phase_d_contracts';out.mkdir(parents=True,exist_ok=True)
    if (out/'delivery_manifest.json').exists():raise FileExistsError('Existing delivery sealed; use a new phase version')
    validate_asset_configuration(root)
    quotes,jobs=load_contract_quotes(root)
    metadata=shfe_metadata(root,root/'data/raw'/RUN/'source_access/shfe_contract_metadata_20191008.response')
    metadata.to_csv(out/'verified_metadata_20191008.csv',index=False)
    metadata.loc[metadata.verification_status.ne(VERIFIED)].to_csv(out/'metadata_validation_issues.csv',index=False)
    identities=build_identity_manifest(quotes,metadata)
    identities.to_csv(out/'contract_identity_manifest.csv',index=False)
    coverage=[];quality=[]
    for spec in ALL_CONTRACT_FUTURES:
        f=quotes.loc[quotes.exchange.eq(spec.exchange)&quotes.variety.eq(spec.variety)]
        ids=identities.loc[identities.exchange.eq(spec.exchange)&identities.variety.eq(spec.variety)]
        ordinary=f.loc[~f.contract.str.contains('_TAS',na=False)]
        active=ordinary.loc[ordinary.volume.gt(0)&ordinary.close.gt(0)]
        coverage.append({'asset_id':'FUT_'+spec.variety,'variety':spec.variety,'exchange':spec.exchange,'role':spec.role,
            'record_count':len(f),'ordinary_contract_records':len(ordinary),'positive_volume_rows':len(active),
            'first_observed_date':f.date.min() if len(f) else None,'last_observed_date':f.date.max() if len(f) else None,
            'unique_trading_dates':f.date.nunique(),'contract_code_count':f.contract.nunique(),
            'identity_episode_count':len(ids),'verified_identity_episodes':int(ids.verification_status.eq(VERIFIED).sum()),
            'has_continuous_series':spec.role!='bond_future','has_contract_observations':bool(len(f)),
            'full_history_coverage_verified':False,'history_and_identity_sufficient_for_official_labels':False,
            'causal_selection_status':'TRAIN_SUBSET_MECHANICS_VERIFIED_ONLY' if spec.variety in ('CU','RB','AU') else 'NOT_VERIFIED',
            'official_labels_available':False,'collection_status':'PARTIAL' if len(f) else 'BLOCKED',
            'source_ids':';'.join(sorted(f.source_id.unique()))})
        quality.append({'asset_id':'FUT_'+spec.variety,'exchange':spec.exchange,'role':spec.role,**quality_counts(f),
                        'action':'PRESERVE_NO_AUTOMATIC_PRICE_REPAIR'})
    pd.DataFrame(coverage).to_csv(out/'contract_coverage.csv',index=False)
    pd.DataFrame(quality).to_csv(out/'contract_quality_summary.csv',index=False)
    gfex=pd.DataFrame(coverage).loc[lambda f:f.exchange.eq('GFEX')].copy()
    gfex['accepted_source_category']='futures_contracts_daily_partial_v3'
    gfex['new_quote_rows_this_phase']=0;gfex['recovery_status']='PARTIAL_NO_ACCESSIBLE_NEW_AUTHORIZED_SOURCE'
    gfex.to_csv(out/'gfex_recovery_status.csv',index=False)
    tests=real_train_checks(root,quotes,metadata,out)
    wh=quotes.loc[quotes.variety.eq('WH')&quotes.date.ge('2023-01-01')].copy()
    wh['year']=wh.date.str[:4]
    yearly=wh.groupby('year').agg(records=('date','size'),positive_volume_records=('volume',lambda s:int(s.gt(0).sum())),
        positive_close_records=('close',lambda s:int(s.gt(0).sum())),positive_volume_dates=('date',lambda s:int(wh.loc[s.index].loc[lambda f:f.volume.gt(0),'date'].nunique())))
    yearly.to_csv(out/'wh_2023_2026_activity.csv')
    wh.loc[wh.volume.gt(0),['date','contract','close','settle','volume','open_interest','raw_file','manifest']].to_csv(out/'wh_positive_volume_records.csv',index=False)
    preservation=json.loads((out/'preservation_before.json').read_text())
    mismatches=[p for p,m in preservation['files'].items() if not (root/p).is_file() or sha256_file(root/p)!=m['sha256']]
    if mismatches:raise RuntimeError('Previously sealed data/report changed: '+str(mismatches[:5]))
    flags={'DCE_SOURCE_AVAILABLE':'BLOCKED','DCE_IMPORTER_READY':True,'DCE_CONTRACT_COVERAGE_PASS':False,
        'GFEX_CONTRACT_COVERAGE_PASS':False,'CONTRACT_IDENTITY_PASS':False,
        'ROLL_CAUSALITY_TEST_PASS':all(t['prefix_test']=='PASS' for t in tests if t['variety']!='CF'),
        'WH_MAPPING_PASS':False,'STOCK_BOND_DATA_PRESERVED':not mismatches,'READY_FOR_24_TARGET_LABELS':False}
    results={'at':utc_now(),'phase':'D-Contracts','flags':flags,'prior_files_checked':len(preservation['files']),
        'prior_fingerprint_mismatches':mismatches,'existing_contract_rows':len(quotes),'new_contract_quote_rows':0,
        'targets_with_contract_observations':18,'auxiliaries_with_contract_observations':8,
        'targets_passing_full_history_identity_and_label_requirements':0,'auxiliaries_passing_full_history_identity_and_label_requirements':0,
        'verified_identity_episodes':int(identities.verification_status.eq(VERIFIED).sum()),'identity_episodes':len(identities),
        'shfe_metadata_scoped_records':len(metadata),'shfe_metadata_verified_records':int(metadata.verification_status.eq(VERIFIED).sum()),
        'train_window':[START,END],'train_checks':tests,
        'dce_blocker':'DCE_DATA_SOURCE_ACCESS_REQUIRED','tushare_status':'DATA_PROVIDER_AUTH_REQUIRED_NO_TOKEN_NO_API_CALLS',
        'importer_scope':'Executable generic CSV/Excel importer, synthetic-tested; official native export format NOT_VERIFIED',
        'roll_pass_scope':'Synthetic rules A/B and actual TRAIN B prefix checks; not data completeness, historical publication proof, execution or label approval',
        'no_formal_labels':True,'no_models_or_mae':True,'no_sealed_prediction_evaluation':True,'no_source_fusion':True}
    atomic_json(results,out/'audit_results.json')
    atomic_json({'at':utc_now(),'previous_inventory_sha256':sha256_file(out/'preservation_before.json'),
        'prior_inventory':'preservation_before.json','source_access':'source_access_probe.json','contract_jobs':jobs,
        'source_sha256':{str(p.relative_to(root)):sha256_file(p) for p in (root/'src/triweave').rglob('*.py')},
        'configuration_sha256':{str(p.relative_to(root)):sha256_file(p) for p in (root/'configs').glob('*') if p.is_file()},
        'environment':environment(root),'research_windows_unchanged':True,
        'gfex_admission':'Only v3; v1/v2 quarantined, neither read nor counted',
        'future_information_note':'All-period data used only for fixed-rule source audit; all selection diagnostics restricted to TRAIN'},out/'provenance.json')
    print(json.dumps(results,ensure_ascii=False,indent=2))
    return results


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['report']);p.add_argument('--root',default='.')
    a=p.parse_args(argv);report(a.root)


if __name__=='__main__':main()
