"""Run once into a new Phase E directory. Offline only; no TEST quote parser."""
from __future__ import annotations
import argparse
import copy
import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from triweave.after_close import (
    DATA_REL, REPORT_REL, END, PROTOCOL, TARGETS, SPLITS, HORIZONS, KINDS,
    asset_definitions, load_sources, align_sources, feature_panel, fit_scaler,
    transform, dataset_arrays, registry, write_json, array_sha, AfterCloseDataset,
    clock_seconds,
)
from triweave.universe import sha256_file


def calculate(root, protocol):
    assets = asset_definitions(root)
    full = pd.read_csv(root/"data/manifests/baseline15y/reference_trading_calendar.csv", dtype=str).date.tolist()
    if full != sorted(set(full)):
        raise ValueError("Reference calendar not unique and sorted")
    frames, sources, readlog = load_sources(root, assets)
    aligned, quality = align_sources(frames, assets, full, protocol)
    x, mask, applicable = feature_panel(aligned, assets)
    params = fit_scaler(x, mask, aligned["dates"], assets, protocol)
    z, final_mask = transform(x, mask, assets, params)
    arrays = dataset_arrays(aligned, assets, z, final_mask, applicable)
    arrays["unscaled_features"] = x
    arrays["unscaled_feature_valid_mask"] = mask
    return dict(assets=assets, full_calendar=full, frames=frames, sources=sources,
                readlog=readlog, aligned=aligned, quality=quality, params=params, arrays=arrays)


def table(path, rows, columns=None):
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def summaries(root, result, protocol):
    out = root/REPORT_REL
    a = result["arrays"]
    assets = result["assets"]
    dates = a["reference_dates"]
    ids = [s["asset_id"] for s in assets]
    frows, markets, target_rows, months = [], [], [], []
    summary = {}
    for split in ("TRAIN", "VAL"):
        pre = split.lower()
        lo, hi = SPLITS[split]
        time = (dates >= lo) & (dates <= hi)
        origins = a[pre+"_origin_indices"]
        m = a[pre+"_label_valid_mask"]
        counts = m.sum((0, 2))
        rows = []
        source_origin_map = {int(o): i for i, o in enumerate(origins)}
        for i in np.flatnonzero(time):
            row = dict(reference_date=dates[i], supervised_origin=i in source_origin_map)
            for market in ("stock", "bond", "future"):
                js = [j for j, s in enumerate(assets) if s["market"] == market]
                row[market+"_valid_nodes"] = int(a["input_valid_mask"][i, js].sum())
            for k, h in enumerate(HORIZONS):
                row[f"labels_{h}d"] = int(m[source_origin_map[i], k].sum()) if i in source_origin_map else 0
            rows.append(row)
        table(out/(pre+"_coverage.csv"), rows)
        for kind in KINDS:
            js = [j for j, s in enumerate(assets) if s["economic_type"] == kind]
            valid = a["input_valid_mask"][time][:, js]
            markets.append(dict(split=split, economic_type=kind, nodes=len(js), reference_dates=int(time.sum()),
                                valid_node_dates=int(valid.sum()), quote_missing_ratio=float(1-valid.mean()),
                                min_valid_nodes=int(valid.sum(1).min()), mean_valid_nodes=float(valid.sum(1).mean()),
                                max_valid_nodes=int(valid.sum(1).max())))
        for j, asset in enumerate(assets):
            for r in registry():
                k = r["index"]
                if not a["feature_applicable_mask"][j, k]:
                    continue
                raw = a["unscaled_features"][time, j, k]
                mask = a["feature_valid_mask"][time, j, k]
                vals = a["features"][time, j, k][mask]
                rawvals = raw[a["unscaled_feature_valid_mask"][time, j, k]]
                frows.append(dict(split=split, asset_id=asset["asset_id"], economic_type=asset["economic_type"],
                                  feature_name=r["feature_name"], reference_dates=int(time.sum()),
                                  valid_count=int(mask.sum()), missing_ratio=float(1-mask.mean()),
                                  unscaled_min=float(rawvals.min()) if len(rawvals) else None,
                                  unscaled_max=float(rawvals.max()) if len(rawvals) else None,
                                  normalized_min=float(vals.min()) if len(vals) else None,
                                  normalized_max=float(vals.max()) if len(vals) else None))
        for j, commodity in enumerate(TARGETS):
            row = dict(split=split, asset_id="FUT_"+commodity, target_order=j,
                       reference_dates=int(time.sum()), supervised_origins=len(origins),
                       valid_quotes=int(a["input_valid_mask"][time, -25+j].sum()))
            for k, h in enumerate(HORIZONS):
                row[f"{h}d_valid_labels"] = int(m[:, k, j].sum())
                row[f"{h}d_label_coverage_reference_dates"] = float(m[:, k, j].sum()/time.sum())
            target_rows.append(row)
        if split == "VAL":
            for month in sorted(set(dates[origins].astype("U7"))):
                for k, h in enumerate(HORIZONS):
                    selection = np.char.startswith(dates[origins], month)
                    months.append(dict(month=month, horizon=h, valid_labels=int(m[selection, k].sum()),
                                       origins=int(selection.sum())))
        summary[split] = dict(reference_dates=int(time.sum()), supervised_origins=len(origins),
                              logical_X_shape=[len(origins),20,313,len(registry())],
                              Y_shape=list(a[pre+"_targets"].shape),
                              valid_labels_by_horizon={str(h):int(counts[k]) for k,h in enumerate(HORIZONS)},
                              origins_removed_all_labels_invalid=int(time.sum()-len(origins)),
                              horizon_origins_with_any_label={str(h):int(m[:,k].any(1).sum()) for k,h in enumerate(HORIZONS)})
    table(out/"per_market_coverage.csv", markets)
    table(out/"per_asset_feature_coverage.csv", frows)
    table(out/"per_target_label_coverage.csv", target_rows)
    table(out/"val_monthly_label_coverage.csv", months)
    table(out/"quote_quality_audit.csv", result["quality"])
    table(out/"source_read_log.csv", result["readlog"])
    # Normalize audit from daily unique panel, never the overlapping sample windows.
    normrows, extreme = [], []
    train = dates <= SPLITS["TRAIN"][1]
    for p in result["params"]:
        js = [j for j,s in enumerate(assets) if s["economic_type"] == p["economic_type"]]
        k = p["feature_index"]
        row = dict(p)
        for split, time in [("TRAIN", train), ("VAL", ~train)]:
            vals = a["features"][time][:,js,k]
            mask = a["feature_valid_mask"][time][:,js,k]
            good = vals[mask]
            row[split+"_normalized_min"] = float(good.min()) if len(good) else None
            row[split+"_normalized_max"] = float(good.max()) if len(good) else None
            flagged = mask & (np.abs(vals)>protocol["normalization"]["extreme_z_audit_threshold_flag_only"])
            row[split+"_extreme_count_abs_z_gt20"] = int(flagged.sum())
            # All counts, up to5 largest examples per group/split; no clipping.
            where = np.argwhere(flagged)
            if len(where):
                magnitude = np.abs(vals[flagged])
                order = np.argsort(-magnitude, kind="stable")[:5]
                tids = np.flatnonzero(time)
                for entry in order:
                    ti, aj = map(int, where[entry]);i,j = int(tids[ti]),js[aj]
                    si = int(a["source_index"][i,j])
                    extreme.append(dict(split=split, date=dates[i], asset_id=ids[j],economic_type=p["economic_type"],
                                        feature_name=p["feature_name"],unscaled=float(a["unscaled_features"][i,j,k]),
                                        normalized=float(vals[ti,aj]),mean_train=p["mean_train"],effective_scale=p["effective_scale"],
                                        observation_date=str(np.datetime64(int(a["observation_day"][i,j]),"D")),
                                        source_manifest=result["sources"][si]["manifest"] if si>=0 else "MASK_DERIVED",
                                        action="RETAINED_NO_VAL_REFIT_NO_CLIP"))
        normrows.append(row)
    table(out/"normalization_audit.csv", normrows)
    table(out/"extreme_normalized_examples.csv", extreme)
    feature_groups = pd.DataFrame(frows).groupby(["split","economic_type","feature_name"],sort=False).agg(
        valid_count=("valid_count","sum"),possible_dates=("reference_dates","sum")).reset_index()
    feature_groups["missing_ratio"] = 1-feature_groups.valid_count/feature_groups.possible_dates
    feature_groups.to_csv(out/"per_feature_type_coverage.csv",index=False)
    summary["normalization"] = dict(groups=len(normrows),constant_groups=sum(p["constant_flag"] for p in normrows),
                                     scale_floor_groups=sum(p["scale_floor_flag"] for p in normrows),
                                     insufficient_groups=sum(p["insufficient_train_support"] for p in normrows),
                                     train_extreme_observations=sum(p["TRAIN_extreme_count_abs_z_gt20"] for p in normrows),
                                     val_extreme_observations=sum(p["VAL_extreme_count_abs_z_gt20"] for p in normrows))
    # Prior issues are already TRAIN/VAL-only; subset, preserve old audit files.
    for name in ("continuous_quote_quality_issues.csv","continuous_off_calendar_dates.csv","continuous_price_anomalies.csv"):
        f = pd.read_csv(root/"experiments/phase_d_continuous"/name)
        f = f.loc[f.asset_id.isin(["FUT_"+c for c in TARGETS])].copy()
        if "date" in f and (f.date>END).any():
            raise ValueError("Prior audit outside development scope")
        f.to_csv(out/("inherited_"+name),index=False)
        summary[name] = len(f)
    write_json(out/"collection_summary.json",summary)
    return summary


def real_checks(root,result,protocol):
    """Actual full-panel checks plus real future/VAL perturbation, not just toys."""
    assets=result["assets"];a=result["arrays"];aligned=result["aligned"];dates=aligned["dates"]
    ids=[s["asset_id"] for s in assets];checks={};evidence={}
    def check(name,condition):
        checks[name]=bool(condition)
        if not condition:raise AssertionError(name)
    check("sorted_unique_dates",list(dates)==sorted(set(dates)))
    check("313_asset_order",ids==json.loads((root/"experiments/phase_d_continuous/provenance.json").read_text())["stock_order"]
          +list(__import__('triweave.assets',fromlist=['YIELD_SPECS']).YIELD_SPECS)
          +list(__import__('triweave.assets',fromlist=['INDEX_SPECS']).INDEX_SPECS)+['FUT_'+c for c in TARGETS])
    check("25_target_order",ids[-25:]==['FUT_'+c for c in TARGETS] and 'FUT_WH' not in ids)
    check("all_masks_boolean",all(v.dtype==bool for k,v in a.items() if 'mask' in k))
    check("all_numeric_outputs_finite",all(np.isfinite(v).all() for v in a.values() if v.dtype.kind=='f'))
    check("invalid_feature_placeholders_zero",np.all(a['features'][~a['feature_valid_mask']]==0))
    check("structural_inapplicability_masked",not np.any(a['feature_valid_mask']&~a['feature_applicable_mask'][None]))
    check("no_test_numeric_ingestion",all((r['max_parsed_date'] is None or r['max_parsed_date']<=END) and r['TEST_numeric_rows_parsed']==0 for r in result['readlog']))
    check("asof_available_by_row_decision",np.all(a['available_at'][a['input_valid_mask']]<=np.broadcast_to(a['decision_time'][:,None],a['input_valid_mask'].shape)[a['input_valid_mask']]))
    check("observation_never_after_reference",np.all(a['observation_day'][a['input_valid_mask']]<=np.broadcast_to(dates.astype('datetime64[D]').astype(int)[:,None],a['input_valid_mask'].shape)[a['input_valid_mask']]))
    check("bond_staleness_limited",np.all(a['age_days'][:,263:288][a['input_valid_mask'][:,263:288]]<=5))
    check("no_stock_futures_carry",np.all(a['age_days'][:,:263][a['input_valid_mask'][:,:263]]==0) and np.all(a['age_days'][:,-25:][a['input_valid_mask'][:,-25:]]==0))
    check("suspensions_not_new_quotes",not np.any(a['input_valid_mask'][:,:263] & (a['trading_status'][:,:263]!=1)))
    for split in ('TRAIN','VAL'):
        pre=split.lower();origins=a[pre+'_origin_indices'];mask=a[pre+'_label_valid_mask'];y=a[pre+'_targets'];ends=a[pre+'_endpoint_indices']
        check(pre+'_origin_has_label',mask.any((1,2)).all())
        check(pre+'_target_shapes',y.shape==(len(origins),4,25) and mask.shape==y.shape)
        check(pre+'_window_shape',a[pre+'_window_indices'].shape==(len(origins),20))
        check(pre+'_invalid_labels_zero',np.all(y[~mask]==0))
        lo,hi=SPLITS[split];count=0
        for si,i in enumerate(origins):
            for k,h in enumerate(HORIZONS):
                m=mask[si,k]
                if not m.any():continue
                e=int(ends[si,k]);assert e==i+h and lo<=dates[i]<=dates[e]<=hi
                assert a['input_valid_mask'][i:e+1,-25:][:,m].all()
                expected=a['aligned_values'][e,-25:][m]/a['aligned_values'][i,-25:][m]-1
                np.testing.assert_allclose(y[si,k,m],expected,rtol=1e-7,atol=1e-8)
                decision=int(a['decision_time'][i]);anchor=a['available_at'][i,-25:][m];future=a['available_at'][e,-25:][m]
                assert np.all(anchor<=decision) and np.all(future>decision)
                count+=int(m.sum())
        check(pre+'_all_actual_label_dates_values_times',True)
        evidence[pre+'_individually_verified_labels']=count
        # Actual mmap loader, every origin (no giant repeated X tensor).
        ds=AfterCloseDataset(root/DATA_REL,split)
        for si in range(len(ds)):
            sample=ds[si]
            assert sample['X'].shape==(20,313,len(registry()))
            assert sample['Y'].shape==(4,25)
            assert sample['input_max_available_time']<=sample['decision_time']
            assert np.isfinite(sample['X']).all() and np.all(sample['X'][~sample['feature_valid_mask']]==0)
        check(pre+'_every_mmap_sample_verified',True)
    # Perturb actual future quotes beyond a fixed pre-VAL date, all four types.
    cutoff='2021-06-30';mutated={}
    for asset,f in result['frames'].items():
        f=f.copy();future=f.date.gt(cutoff)
        for field in ('close','open','high','low','value','volume','amount','turnover','open_interest'):
            if field in f:
                z=pd.to_numeric(f.loc[future,field],errors='coerce')
                f.loc[future,field]=(z*1.37+0.001).astype(str)
        mutated[asset]=f
    changed,_=align_sources(mutated,assets,result['full_calendar'],protocol)
    rx,rm,rapp=feature_panel(changed,assets)
    prefix=dates<=cutoff
    check('real_future_prefix_unscaled_exact',np.array_equal(rx[prefix],a['unscaled_features'][prefix]) and np.array_equal(rm[prefix],a['unscaled_feature_valid_mask'][prefix]))
    rz,rzm=transform(rx,rm,assets,result['params'])
    check('real_future_prefix_with_frozen_scaler_exact',np.array_equal(rz[prefix],a['features'][prefix]) and np.array_equal(rzm[prefix],a['feature_valid_mask'][prefix]))
    evidence['prefix_cutoff']=cutoff
    evidence['prefix_reference_days']=int(prefix.sum())
    # Change every actual VAL feature value; TRAIN fit stays byte-identical.
    xx=a['unscaled_features'].copy();xx[dates>SPLITS['TRAIN'][1]]=123456789.
    pp=fit_scaler(xx,a['unscaled_feature_valid_mask'],dates,assets,protocol)
    check('real_VAL_perturbation_scaler_identical',pp==result['params'])
    # All previously existing data, reports and source/config/docs bytes preserved.
    old=json.loads((root/REPORT_REL/'preservation_before.json').read_text())['files']
    differences=[p for p,h in old.items() if not (root/p).is_file() or sha256_file(root/p)!=h]
    evidence['protected_old_files']=len(old);evidence['old_file_differences']=differences
    check('all_old_files_preserved',not differences)
    return checks,evidence


def build(root):
    root=Path(root).resolve();out=root/REPORT_REL;dest=root/DATA_REL
    if (dest/'dataset_manifest.json').exists() or list(dest.glob('*.npy')):
        raise FileExistsError('Existing dataset cannot be overwritten; use a new version')
    protocol=json.loads((out/'frozen_protocol.json').read_text())
    assert protocol['protocol_id']==PROTOCOL and protocol['splits']=={k:list(v) for k,v in SPLITS.items()}
    assert (dest/'decision_time_protocol.json').read_bytes()==(out/'frozen_protocol.json').read_bytes()
    frozen_sha=sha256_file(out/'frozen_protocol.json')
    print('Loading date-gated sources; TEST quote fields excluded',flush=True)
    r=calculate(root,protocol)
    print('Building deterministic daily panels, scaler and TRAIN/VAL targets',flush=True)
    for name,a in r['arrays'].items():np.save(dest/(name+'.npy'),a,allow_pickle=False)
    write_json(dest/'asset_manifest.json',r['assets'])
    write_json(dest/'target_manifest.json',{'target_ids':['FUT_'+c for c in TARGETS],'horizons':list(HORIZONS),'label_type':protocol['label_type'],'unit':'decimal_return'})
    write_json(dest/'feature_schema.json',registry())
    write_json(dest/'scaler_params.json',r['params']);write_json(out/'scaler_params.json',r['params'])
    write_json(out/'feature_registry.json',registry())
    calendar=root/'data/manifests/baseline15y/reference_trading_calendar.csv'
    write_json(dest/'calendar_manifest.json',{'source':str(calendar.relative_to(root)),'sha256':sha256_file(calendar),'frozen_reference_not_independent_futures_PIT_verification':True,'calendar_dates_in_dataset':'TRAIN and VAL only'})
    write_json(dest/'source_manifest.json',r['sources'])
    stats=summaries(root,r,protocol)
    files={p.name:{'sha256':sha256_file(p),'bytes':p.stat().st_size} for p in sorted(dest.iterdir()) if p.is_file()}
    manifest={'protocol_id':PROTOCOL,'split_statistics':stats,'stored_daily_panel_shape':list(r['arrays']['features'].shape),
              'storage':'Nonredundant daily panel NPY mmap + TRAIN/VAL window indices, targets, masks. Loader yields logical X windows with left-padding masks.',
              'loader':'from triweave.after_close import AfterCloseDataset; ds=AfterCloseDataset(directory, "TRAIN"); sample=ds[0]',
              'files':files,'files_bytes_excluding_manifest':sum(f['bytes'] for f in files.values()),
              'TEST_dataset_generated':False,'protocol_sha256':frozen_sha,
              'DECISION_TIME_ASSUMED':True,'HISTORICAL_PIT_VERIFIED':False,'FUTURES_ROLL_METHOD_VERIFIED':False}
    write_json(dest/'dataset_manifest.json',manifest);write_json(out/'dataset_manifest.json',manifest)
    print('Auditing every real sample/label; perturbing actual future inputs and VAL',flush=True)
    checks,evidence=real_checks(root,r,protocol)
    write_json(out/'real_data_checks.json',{'checks':checks,'evidence':evidence})
    print('Independent second source read and full calculation for SHA256 reproducibility',flush=True)
    rr=calculate(root,protocol)
    reproduction={name:{'first_sha256':sha256_file(dest/(name+'.npy')),'second_sha256':array_sha(arr)} for name,arr in rr['arrays'].items()}
    for item in reproduction.values():item['identical']=item['first_sha256']==item['second_sha256']
    reproduction['scaler_params_identical']={'identical':r['params']==rr['params']}
    write_json(out/'reproducibility_audit.json',reproduction)
    if not all(e['identical'] for e in reproduction.values()):raise AssertionError('Reproduction failed')
    assert sha256_file(out/'frozen_protocol.json')==frozen_sha
    provenance={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol_sha256':frozen_sha,
                'source_files':r['sources'],'source_file_count':len(r['sources']),
                'source_reader':'date-first gate before csv quote parser; excluded TEST tails discarded. Raw archives hashed as opaque bytes only.',
                'source_read_log':'source_read_log.csv','TEST_numeric_rows_parsed':0,'TEST_labels_generated':False,
                'network_calls':0,'training_or_metrics_executed':False,'asset_order':[s['asset_id'] for s in r['assets']],
                'target_order':TARGETS,'calendar_sha256':sha256_file(calendar),
                'source_code_sha256':{str(p.relative_to(root)):sha256_file(p) for p in sorted((root/'src').rglob('*.py'))},
                'old_files_protection_inventory':'preservation_before.json','old_files_unchanged':checks['all_old_files_preserved'],
                'prior_reports_reviewed':['README.md','configs/project.json','docs/data_protocol.md','experiments/data_rescue/REPORT.md','experiments/phase_d_contracts/REPORT.md','experiments/phase_d_continuous/REPORT.md'],
                'historical_PIT_verified':False,'VAL_informed_cohort_design':True}
    write_json(out/'provenance.json',provenance)
    print(json.dumps({'statistics':stats,'real_checks':checks,'dataset_manifest_sha256':sha256_file(dest/'dataset_manifest.json')},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    args=parser.parse_args();build(args.root)
