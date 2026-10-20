"""Source-quality and coverage audit. No imputation, features, labels or model metrics."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd

from triweave.assets import TARGET_FUTURES, AUX_FUTURES, ALL_CONTRACT_FUTURES, YIELD_SPECS, INDEX_SPECS
from triweave.collect import atomic_json, utc_now, make_jobs
from triweave.pipeline import environment, read_events
from triweave.universe import sha256_file, load_stock_universe


# Fixed before full-history inspection. Review flags never alter observations.
RULES = {"constant_run_observations": 20, "price_level_jump_fraction": 0.20,
         "yield_level_jump_percentage_points": 1.0,
         "action": "Flag only; never replace, fill, winsorize or drop observations"}
FIELDS = ['asset_id', 'asset_name', 'market', 'role', 'series_type', 'exchange', 'source', 'unit',
          'first_observation_date', 'last_observation_date', 'record_count', 'unique_date_count',
          'duplicate_date_count', 'missing_value_count', 'invalid_value_count', 'longest_missing_gap',
          'coverage_ratio', 'coverage_years', 'collection_status', 'calendar_basis',
          'valid_observation_count', 'required_field_missing_count', 'listing_date',
          'listing_date_verified', 'prelisting_observation_count', 'longest_constant_run',
          'source_request_status', 'status_reason', 'manifest_references']


def csv_content_fingerprint(path):
    """Exclude gzip timestamp/filename metadata when comparing data vintages."""
    digest = hashlib.sha256()
    with gzip.open(path,'rb') as source:
        for chunk in iter(lambda:source.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def valid_quote_mask(frame, kind):
    field='value' if kind in ('yield_curve','treasury_wealth_index') else 'close'
    values=pd.to_numeric(frame[field],errors='coerce')
    valid=values.notna() & ~values.isin([float('inf'),float('-inf')])
    if kind!='yield_curve':
        valid &= values.gt(0)
    if 'contract' in frame:
        # Preserve TAS rows as source evidence, never count another trading mode
        # as an outright futures quote or use it to fill outright coverage.
        ordinary=frame['contract'].astype('string').str.fullmatch(r'[A-Z]+[0-9]{3,4}').fillna(False)
        valid &= ordinary
        if 'variety' in frame:
            valid &= frame['contract'].str.extract(r'^([A-Z]+)',expand=False).eq(frame['variety'])
    return valid


def valid_observation_dates(frame, kind):
    if frame.empty:
        return pd.DatetimeIndex([])
    valid=valid_quote_mask(frame,kind)
    return pd.DatetimeIndex(pd.to_datetime(frame.loc[valid,'date'],errors='coerce').dropna().unique()).sort_values()


def _maximum_run(mask):
    longest = current = 0
    for value in mask:
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return longest


def inspect_series(frame, record, calendar, start, end, listing=None):
    """Keep contract identities separate, and measure missingness without inventing dates."""
    row = {**record}
    row.update({key: None for key in FIELDS if key not in row})
    row['record_count'] = len(frame)
    row['listing_date'] = listing.get('date') if listing else None
    row['listing_date_verified'] = bool(listing)
    lower = max(pd.Timestamp(start), pd.Timestamp(row['listing_date'])) if listing else pd.Timestamp(start)
    calendar = pd.DatetimeIndex(calendar)
    expected = calendar[(calendar >= lower) & (calendar <= pd.Timestamp(end))]
    dates = pd.to_datetime(frame['date'], errors='coerce') if not frame.empty else pd.Series(dtype='datetime64[ns]')
    observed = pd.DatetimeIndex(dates.dropna().unique()).sort_values()
    valid_dates = valid_observation_dates(frame,record['series_type'])
    row['unique_date_count'] = len(observed)
    row['first_returned_date'] = str(observed.min().date()) if len(observed) else None
    row['last_returned_date'] = str(observed.max().date()) if len(observed) else None
    row['valid_unique_date_count'] = len(valid_dates)
    row['first_observation_date'] = str(valid_dates.min().date()) if len(valid_dates) else None
    row['last_observation_date'] = str(valid_dates.max().date()) if len(valid_dates) else None
    row['coverage_years'] = round((valid_dates.max()-valid_dates.min()).days/365.2425, 4) if len(valid_dates) else 0.0
    row['coverage_ratio'] = round(len(expected.intersection(valid_dates))/len(expected), 8) if len(expected) else None
    row['longest_missing_gap'] = _maximum_run(~expected.isin(valid_dates)) if len(expected) else None
    row['prelisting_observation_count'] = int((dates < pd.Timestamp(listing['date'])).sum()) if listing else None
    row['missing_value_count'] = int(frame.isna().sum().sum()) if not frame.empty else 0
    row['duplicate_date_count'] = 0
    row['invalid_value_count'] = 0
    row['required_field_missing_count'] = 0
    row['valid_observation_count'] = 0
    row['longest_constant_run'] = 0
    row['longest_zero_volume_run'] = 0
    row['special_trading_mode_row_count'] = 0
    row['unrecognized_contract_code_count'] = 0
    anomalies = []
    if frame.empty:
        row['collection_status'] = 'NOT_VERIFIED'
        row['status_reason'] = 'No validated observations; retrieval failure is not evidence of nonexistence'
        return row, anomalies
    frame = frame.copy()
    frame['date'] = dates
    keys = ['date'] + (['contract'] if 'contract' in frame else [])
    row['duplicate_date_count'] = int(frame.duplicated(keys).sum())
    is_yield = record['series_type'] == 'yield_curve'
    is_bond = record['market'] == 'bond'
    required = ['value'] if is_bond else ['open', 'high', 'low', 'close', 'volume']
    if record['market'] == 'stock':
        required += ['amount', 'turnover']
    elif not is_bond:
        required += ['open_interest']
    if 'contract' in frame:
        required += ['settle']
    for col in required:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col],errors='coerce')
    row['required_field_missing_count'] = sum(int(frame[col].isna().sum()) if col in frame else len(frame) for col in required)
    invalid = dates.isna().reindex(frame.index, fill_value=True)
    value_col = 'value' if is_bond else 'close'
    values = pd.to_numeric(frame[value_col], errors='coerce')
    bad_value = ~valid_quote_mask(frame,record['series_type'])
    invalid |= bad_value
    row['valid_observation_count'] = int((~bad_value & dates.notna()).sum())
    if 'contract' in frame:
        codes=frame['contract'].astype('string')
        special=codes.str.fullmatch(r'[A-Z]+_TAS[0-9]{3,4}').fillna(False)
        ordinary=codes.str.fullmatch(r'[A-Z]+[0-9]{3,4}').fillna(False)
        row['special_trading_mode_row_count']=int(special.sum())
        row['unrecognized_contract_code_count']=int((~ordinary & ~special).sum())
        row['delivery_year_verification_status']='NOT_VERIFIED'
        for count,name in ((row['special_trading_mode_row_count'],'SPECIAL_TRADING_MODE_NOT_OUTRIGHT_QUOTE'),
                           (row['unrecognized_contract_code_count'],'UNRECOGNIZED_CONTRACT_CODE_REVIEW')):
            if count:
                anomalies.append({'asset_id':row['asset_id'],'series_type':row['series_type'],
                    'anomaly':name,'count':count,'action':'PRESERVED_SOURCE_NOT_ELIGIBLE_AS_OUTRIGHT_QUOTE'})
    if not is_bond:
        for col in ('volume', 'open_interest', 'amount', 'turnover', 'turnover_rate'):
            if col in frame:
                numeric = pd.to_numeric(frame[col], errors='coerce')
                invalid |= numeric.lt(0) | numeric.isin([float('inf'),float('-inf')])
        if 'contract' in frame and 'settle' in frame:
            invalid |= frame['settle'].le(0) | frame['settle'].isin([float('inf'),float('-inf')])
        if all(col in frame for col in ('open', 'high', 'low', 'close')):
            invalid |= (frame['high'] < frame[['open', 'close', 'low']].max(axis=1)) | (frame['low'] > frame[['open', 'close', 'high']].min(axis=1))
            for col in ('open', 'high', 'low'):
                invalid |= frame[col].le(0) | frame[col].isin([float('inf'), float('-inf')])
    if listing:
        invalid |= frame['date'] < pd.Timestamp(listing['date'])
    row['invalid_value_count'] = int(invalid.sum())
    for index in frame.index[invalid]:
        anomalies.append({'asset_id': row['asset_id'], 'series_type': row['series_type'],
                          'date': str(frame.loc[index, 'date']), 'contract': frame.loc[index].get('contract'),
                          'anomaly': 'INVALID_VALUE_OR_OHLC_REVIEW', 'action': 'PRESERVED_RAW_NO_REPLACEMENT'})
    groups = frame.groupby('contract', dropna=False) if 'contract' in frame else [(None, frame)]
    for contract, group in groups:
        group = group.sort_values('date')
        values = pd.to_numeric(group[value_col], errors='coerce')
        constant = _maximum_run(values.eq(values.shift()) & values.notna())
        constant = constant + 1 if constant else (1 if len(values) else 0)
        row['longest_constant_run'] = max(row['longest_constant_run'], constant)
        if 'volume' in group:
            zero_run = _maximum_run(pd.to_numeric(group['volume'], errors='coerce').eq(0))
            row['longest_zero_volume_run'] = max(row['longest_zero_volume_run'], zero_run)
            if zero_run >= RULES['constant_run_observations']:
                anomalies.append({'asset_id': row['asset_id'], 'series_type': row['series_type'], 'contract': contract,
                                  'anomaly': 'LONG_ZERO_VOLUME_RUN_REVIEW', 'count': zero_run, 'action': 'NO_AUTOMATIC_REPAIR'})
        if constant >= RULES['constant_run_observations']:
            anomalies.append({'asset_id': row['asset_id'], 'series_type': row['series_type'], 'contract': contract,
                              'anomaly': 'LONG_CONSTANT_RUN_REVIEW', 'count': constant, 'action': 'NO_AUTOMATIC_REPAIR'})
        previous = values.shift()
        jumps = (values-previous).abs() > (RULES['yield_level_jump_percentage_points'] if is_yield else previous.abs()*RULES['price_level_jump_fraction'])
        for index in group.index[jumps.fillna(False)]:
            anomalies.append({'asset_id': row['asset_id'], 'series_type': row['series_type'], 'contract': contract,
                              'date': str(group.loc[index, 'date']),
                              'anomaly': 'POSSIBLE_ROLL_OR_SOURCE_JUMP' if row['series_type'] == 'vendor_continuous_unverified_roll' else 'LEVEL_JUMP_REVIEW',
                              'action': 'AUDIT_ONLY_NO_RETURN_LABEL'})
    complete = (row['coverage_ratio'] == 1 and row['invalid_value_count'] == 0 and
                row['duplicate_date_count'] == 0 and row['required_field_missing_count'] == 0)
    row['collection_status'] = 'SUCCESS' if complete else 'PARTIAL'
    row['status_reason'] = ('All reference-calendar observations present' if complete else
                            'Inspect reference-calendar gaps, required fields and anomalies; missing history is never filled')
    if row['valid_observation_count']==0:
        row['collection_status']='FAILED'
        row['status_reason']='Returned rows contain no finite valid quote; source existence remains unverified'
    if row.get('mapping_review_required'):
        row['mapping_status'] = 'TARGET_MAPPING_NOT_VERIFIED'
    return row, anomalies


def _table(frame, columns):
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join('---' for _ in columns) + ' |']
    for item in frame.to_dict('records'):
        lines.append('| ' + ' | '.join(str(item.get(k, '')).replace('|', '/') for k in columns) + ' |')
    return '\n'.join(lines)


def exchange_summary(jobs, manifests, decisions, coverage):
    """Request completeness and asset coverage are separate measurements."""
    rows = []
    for exchange in ('SHFE', 'DCE', 'CZCE', 'INE', 'GFEX', 'CFFEX'):
        group = 'futures_contracts/' + exchange
        planned = {j.name for j in jobs if j.group == group}
        saved = {m['name'] for m in manifests if m['job_group'] == group}
        latest = {d['name']: d['status'] for d in decisions if d['group'] == group}
        table = coverage.loc[coverage['series_type'].eq('exchange_listed_contract') & coverage['exchange'].eq(exchange)]
        dates_first = table['first_observation_date'].dropna()
        dates_last = table['last_observation_date'].dropna()
        rows.append({'exchange': exchange, 'planned_months': len(planned),
                     'saved_response_months': len(planned & saved),
                     'failed_months': sum(latest.get(n) == 'FAILED' for n in planned - saved),
                     'blocked_months': sum(latest.get(n, '').startswith('BLOCKED') for n in planned - saved),
                     'not_attempted_months': sum(n not in latest for n in planned - saved),
                     'configured_varieties': len(table),
                     'varieties_with_observations': int(table['valid_observation_count'].gt(0).sum()),
                     'standardized_rows': int(table['record_count'].sum()),
                     'first_observation_date': dates_first.min() if len(dates_first) else None,
                     'last_observation_date': dates_last.max() if len(dates_last) else None})
    return pd.DataFrame(rows)


def comprehensive_audit(root, run_id='baseline15y'):
    root = Path(root)
    cfg = json.loads((root/'configs/project.json').read_text())
    start, end = cfg['research_start'], cfg['research_end']
    out = root/'experiments/data_collection'/run_id
    out.mkdir(parents=True, exist_ok=True)
    mirror = root/'data/audit'/run_id
    mirror.mkdir(parents=True, exist_ok=True)
    stocks = load_stock_universe(root/'data/manifests/stocks_263.csv')
    manifests = []
    fingerprints = {}
    integrity = []
    data = defaultdict(list)
    references = defaultdict(list)
    for path in sorted((root/'data/manifests'/run_id/'jobs').rglob('*.json')):
        item = json.loads(path.read_text())
        group = item['job_group']
        job_integrity = True
        for folder, field in (('raw', 'raw_sha256'), ('standardized', 'standardized_sha256')):
            file = root/'data'/folder/run_id/group/(item['name']+'.csv.gz')
            if not item.get(field):
                continue
            valid = file.exists() and sha256_file(file) == item[field]
            fingerprints[str(file.relative_to(root))] = {'sha256': sha256_file(file) if file.exists() else None,
                                                        'bytes': file.stat().st_size if file.exists() else 0, 'matches_manifest': valid}
            if valid:
                table_hash = csv_content_fingerprint(file)
                fingerprints[str(file.relative_to(root))]['csv_content_sha256'] = table_hash
                item[folder+'_csv_content_sha256'] = table_hash
            if not valid:
                integrity.append(str(file.relative_to(root)))
                job_integrity = False
        manifests.append({**item, 'manifest_path': str(path.relative_to(root))})
        if item['status'] != 'OK' or not job_integrity:
            continue
        file = root/'data/standardized'/run_id/group/(item['name']+'.csv.gz')
        frame = pd.read_csv(file, dtype={'asset_id':str})
        for asset, sub in frame.groupby('asset_id', sort=False):
            if 'series_type' in sub:
                kind = str(sub['series_type'].iloc[0])
            elif 'instrument_type' in sub:
                kind = str(sub['instrument_type'].iloc[0])
            else:
                kind = 'stock'
            data[(asset,kind)].append(sub)
            references[(asset,kind)].append(str(path.relative_to(root)))
    frames = {key:pd.concat(chunks,ignore_index=True) for key,chunks in data.items()}
    # Bonds can publish on bank working weekends. Their calendar is never forced to stock dates.
    import akshare.futures.futures_daily_bar as daily
    trading_calendar = pd.to_datetime(pd.Series(daily.calendar), format='%Y%m%d')
    calendar_file = root/'data/manifests'/run_id/'reference_trading_calendar.csv'
    if not calendar_file.exists():
        pd.DataFrame({'date':trading_calendar.dt.strftime('%Y-%m-%d')}).to_csv(calendar_file,index=False)
    bond_dates = defaultdict(set)
    for (asset,kind),frame in frames.items():
        if kind in ('yield_curve','treasury_wealth_index'):
            bond_dates[kind].update(pd.to_datetime(frame['date']).tolist())
    listing_file = root/'data/manifests/listing_dates_verified.json'
    listings = json.loads(listing_file.read_text()) if listing_file.exists() else {}
    records = []
    names_path = root/'data/manifests/stock_descriptive_metadata.csv'
    names = (pd.read_csv(names_path,dtype=str).set_index('asset_id')['security_name'].to_dict()
             if names_path.exists() else {})
    bond_identity_path=root/'data/manifests/bond_index_identity.json'
    bond_identities=json.loads(bond_identity_path.read_text()) if bond_identity_path.exists() else {}
    source_identity_errors = []
    for item in manifests:
        if item['job_group'] != 'stocks' or item['status'] != 'OK':
            continue
        raw = root/'data/raw'/run_id/'stocks'/(item['name']+'.csv.gz')
        if not raw.exists():
            continue
        original = pd.read_csv(raw,dtype=str)
        code_col = next((c for c in ('股票代码','stock_code','symbol') if c in original),None)
        if code_col and not original[code_col].str.zfill(6).eq(item['api_kwargs']['symbol']).all():
            source_identity_errors.append(item['name'])
    for asset in stocks['asset_id']:
        records.append({'asset_id':asset,'asset_name':names.get(asset,asset),'market':'stock','role':'input','series_type':'stock',
                        'security_name_time_status':'DESCRIPTIVE_LEGACY_NAME_NOT_VERIFIED_PIT',
                        'exchange':asset.split('.')[0].upper(),'source':'Eastmoney via AKShare stock_zh_a_hist',
                        'unit':'RMB_per_share_unadjusted'})
    for asset,(curve,tenor) in YIELD_SPECS.items():
        records.append({'asset_id':asset,'asset_name':curve+' '+tenor,'market':'bond','role':'input','series_type':'yield_curve',
                        'source':'ChinaBond via AKShare bond_china_yield','unit':'percent_per_annum'})
    for asset,period in INDEX_SPECS.items():
        identity=bond_identities.get(asset,{})
        node=identity.get('official_node') or {}
        records.append({'asset_id':asset,'asset_name':node.get('name','ChinaBond treasury '+period)+' (财富指标)',
                        'official_index_id':identity.get('indexid'),'api_period':period,
                        'economic_identity_status':'VERIFIED_OFFICIAL_ID_AND_NAME_NOT_PIT' if node else 'NOT_VERIFIED',
                        'market':'bond','role':'input','series_type':'treasury_wealth_index',
                        'source':'ChinaBond via AKShare bond_treasury_index_cbond','unit':'index_level'})
    for kind,specs in [('vendor_continuous_unverified_roll',TARGET_FUTURES+AUX_FUTURES),('exchange_listed_contract',ALL_CONTRACT_FUTURES)]:
        for spec in specs:
            records.append({'asset_id':'FUT_'+spec.variety,'asset_name':spec.name_en,'market':'bond_future' if spec.role=='bond_future' else 'commodity_future',
                            'role':spec.role,'series_type':kind,'exchange':spec.exchange,'mapping_review_required':spec.review_required,
                            'source':'Sina via AKShare futures_zh_daily_sina' if kind.startswith('vendor') else spec.exchange+' official via AKShare get_futures_daily',
                            'unit':'exchange_quoted_price_unit_NOT_INDEPENDENTLY_VERIFIED'})
    execution_file = root/'data/manifests'/run_id/'execution_summary.json'
    execution = json.loads(execution_file.read_text()) if execution_file.exists() else {'decisions':[],'counts':{},'execution_incomplete':True}
    all_events = read_events(root,run_id)
    smoke_file = root/'experiments/data_collection/smoke_test/results.json'
    smoke = json.loads(smoke_file.read_text()) if smoke_file.exists() else {'results':[]}
    rows, anomalies, missing_intervals = [], [], []
    for record in records:
        key = (record['asset_id'],record['series_type'])
        frame = frames.get(key,pd.DataFrame())
        if record['market']=='bond':
            calendar = sorted(set(trading_calendar.tolist()) | bond_dates[record['series_type']])
            record['calendar_basis']='AKSHARE_TRADING_CALENDAR_PLUS_ACTUAL_BANK_WORKING_DATES_NOT_INDEPENDENTLY_VERIFIED'
        else:
            calendar = trading_calendar
            record['calendar_basis']='AKSHARE_BUNDLED_TRADING_CALENDAR_NOT_INDEPENDENTLY_EXCHANGE_CERTIFIED'
        record['manifest_references']=';'.join(references[key])
        row, issues = inspect_series(frame,record,calendar,start,end,listings.get(record['asset_id']))
        observed = set(valid_observation_dates(frame,record['series_type']).tolist())
        lower = max(pd.Timestamp(start),pd.Timestamp(row['listing_date'])) if row['listing_date_verified'] else pd.Timestamp(start)
        expected = [d for d in pd.DatetimeIndex(calendar).sort_values() if lower <= d <= pd.Timestamp(end)]
        interval = []
        for date in [*expected,None]:
            if date is not None and date not in observed:
                interval.append(date)
            elif interval:
                missing_intervals.append({'asset_id':record['asset_id'],'series_type':record['series_type'],
                                          'first_missing_reference_date':str(interval[0].date()),
                                          'last_missing_reference_date':str(interval[-1].date()),
                                          'reference_observations_missing':len(interval),
                                          'calendar_basis':record['calendar_basis'],
                                          'interpretation':'RETRIEVAL_OR_AVAILABILITY_NOT_INDEPENDENTLY_VERIFIED'})
                interval=[]
        row['contract_count'] = int(frame['contract'].nunique()) if 'contract' in frame else 0
        if frame.empty:
            if record['series_type']=='stock':
                matching = [j for j in execution['decisions'] if j['group']=='stocks' and j['name']==record['asset_id'].replace('.','_')]
            elif record['series_type']=='exchange_listed_contract':
                matching = [j for j in execution['decisions'] if j['group']=='futures_contracts/'+record['exchange']]
            elif record['series_type'].startswith('vendor'):
                matching = [j for j in execution['decisions'] if j['name']==record['exchange']+'_'+record['asset_id'].removeprefix('FUT_')+'_0']
            else:
                matching = [j for j in execution['decisions'] if j['group']==('bond_yield_curves' if record['series_type']=='yield_curve' else 'bond_wealth_indices') and (record['series_type']=='yield_curve' or j['name']==record['asset_id'])]
            states=Counter(j['status'] for j in matching)
            row['source_request_status']=json.dumps(dict(states))
            if states.get('FAILED'):
                row['collection_status']='FAILED'
            elif row['listing_date_verified'] and pd.Timestamp(row['listing_date'])>pd.Timestamp(end):
                row['collection_status']='UNAVAILABLE'
                row['status_reason']='Verified listing occurs after requested interval'
        else:
            row['source_request_status']='VALIDATED_FILES_PRESENT_CHECK_COVERAGE_SEPARATELY'
        rows.append(row)
        anomalies.extend(issues)
    coverage=pd.DataFrame(rows)
    selections={
        'stock_coverage.csv':coverage['series_type'].eq('stock'),
        'bond_coverage.csv':coverage['market'].eq('bond'),
        'futures_target_coverage.csv':coverage['series_type'].eq('vendor_continuous_unverified_roll')&coverage['role'].eq('target'),
        'futures_auxiliary_coverage.csv':coverage['series_type'].eq('vendor_continuous_unverified_roll')&coverage['role'].eq('auxiliary'),
        'exchange_contract_coverage.csv':coverage['series_type'].eq('exchange_listed_contract')}
    for name,mask in selections.items():
        coverage.loc[mask].to_csv(out/name,index=False)
    planned_jobs = make_jobs(['all'], start, end, stock_universe=stocks)
    decisions = execution['decisions'] or read_events(root,run_id,'execution_decisions.jsonl')
    exchanges = exchange_summary(planned_jobs, manifests, decisions, coverage)
    exchanges.to_csv(out/'exchange_collection_summary.csv',index=False)
    coverage.to_csv(mirror/'asset_coverage_by_source_type.csv',index=False)
    coverage.loc[:,[x for x in FIELDS if x in coverage]].to_csv(out/'missingness_report.csv',index=False)
    pd.DataFrame(missing_intervals,columns=['asset_id','series_type','first_missing_reference_date','last_missing_reference_date',
                                          'reference_observations_missing','calendar_basis','interpretation']).to_csv(out/'missing_intervals.csv',index=False)
    pd.DataFrame(anomalies,columns=['asset_id','series_type','date','contract','anomaly','count','action']).to_csv(out/'anomaly_report.csv',index=False)
    errors=[{**e,'evidence_run':run_id} for e in all_events if e.get('event') in ('RETRY_FAILED','JOB_FAILED')]
    smoke_run=smoke.get('run_id')
    if smoke_run:
        errors += [{**e,'evidence_run':smoke_run} for e in read_events(root,smoke_run) if e.get('event') in ('RETRY_FAILED','JOB_FAILED')]
    http=[]
    capture_runs=list(dict.fromkeys(filter(None,(smoke_run,run_id,'smoke_cache_202310','bulk_probe_202310',
                                                   'czce_annual_probe','bond_identity_probe','bond_identity_tree'))))
    for capture_run in capture_runs:
        for event in read_events(root,capture_run):
            evidence = event.get('response_evidence')
            if isinstance(evidence,dict) and evidence.get('path'):
                path = root/evidence['path']
                matched = path.exists() and sha256_file(path)==evidence['sha256']
                fingerprints[evidence['path']] = {'sha256': evidence['sha256'],
                    'bytes': path.stat().st_size if path.exists() else 0,'matches_manifest': matched,
                    'purpose': 'FAILED_SCHEMA_RESPONSE_PRESERVED'}
                if not matched: integrity.append(evidence['path'])
        for event in read_events(root,capture_run,'http_events.jsonl'):
            http.append({**event,'run_id':capture_run})
            if event.get('event')=='HTTP_ERROR' or event.get('status_code',200)>=400:
                errors.append({'evidence_run':capture_run,'event':event['event'],'group':'HTTP_SOURCE_EVIDENCE',
                               'name':event['url'],'api':event.get('method'),'api_kwargs':event.get('request_url',event['url']),
                               'error':event.get('error','HTTP '+str(event.get('status_code'))),
                               'at':event['at'],'response_evidence':event.get('archive')})
            if event.get('archive'):
                path=root/event['archive']
                matched=path.exists() and sha256_file(path)==event['archive_sha256']
                fingerprints[event['archive']]={'sha256':event['archive_sha256'],'bytes':path.stat().st_size if path.exists() else 0,'matches_manifest':matched}
                if not matched: integrity.append(event['archive'])
    pd.DataFrame(errors,columns=['evidence_run','event','group','name','api','api_kwargs','requested_start','requested_end','attempt','error','at','response_evidence','traceback']).to_csv(out/'failed_requests.csv',index=False)
    for path in sorted((root/'data/manifests').glob('*')):
        if path.is_file():
            fingerprints[str(path.relative_to(root))] = {'sha256': sha256_file(path), 'bytes': path.stat().st_size,
                'matches_manifest': None, 'purpose': 'FROZEN_SOURCE_METADATA_INVENTORY'}
    recovery_path=root/'data/manifests'/run_id/'interrupted_write_recovery.json'
    recovery=json.loads(recovery_path.read_text()) if recovery_path.exists() else None
    if recovery:
        path=root/recovery['archive_path']
        matched=path.exists() and sha256_file(path)==recovery['sha256']
        fingerprints[recovery['archive_path']]={'sha256':recovery['sha256'],
            'bytes':path.stat().st_size if path.exists() else 0,'matches_manifest':matched,
            'purpose':'INTERRUPTED_WRITE_PRESERVED_NOT_A_VERIFIED_JOB'}
        if not matched: integrity.append(recovery['archive_path'])
    atomic_json({'run_id':run_id,'captured_utc':utc_now(),'files':fingerprints,'integrity_failures':integrity},out/'data_fingerprints.json')
    revisions=[]
    def identity(job):
        return (job['job_group'],job['name'],job.get('requested_start'),job.get('requested_end'),
                job['api'],json.dumps(job['api_kwargs'],sort_keys=True,ensure_ascii=True))
    current_jobs={identity(job):job for job in manifests}
    for previous_file in (root/'data/manifests').glob('*/jobs/**/*.json'):
        previous_run=previous_file.relative_to(root/'data/manifests').parts[0]
        if previous_run==run_id:
            continue
        previous=json.loads(previous_file.read_text())
        current=current_jobs.get(identity(previous))
        if not current or previous.get('status')!='OK':
            continue
        entry={'group':previous['job_group'],'name':previous['name'],'previous_run':previous_run,'current_run':run_id,
               'normalizer_changed':previous.get('normalizer_source_sha256')!=current.get('normalizer_source_sha256')}
        for folder in ('raw','standardized'):
            file=root/'data'/folder/previous_run/previous['job_group']/(previous['name']+'.csv.gz')
            entry[folder+'_csv_changed']=(csv_content_fingerprint(file)!=current.get(folder+'_csv_content_sha256')) if file.exists() else None
        revisions.append(entry)
    pd.DataFrame(revisions,columns=['group','name','previous_run','current_run','normalizer_changed','raw_csv_changed','standardized_csv_changed']).to_csv(out/'data_revisions.csv',index=False)
    by_url=defaultdict(set)
    for event in http:
        if event.get('body_sha256'): by_url[event['url']].add(event['body_sha256'])
    atomic_json({'run_id':run_id,'software_environment':'environment.json','jobs':manifests,'http_requests':http,
                 'stock_universe':json.loads((root/'data/manifests/stocks_263.meta.json').read_text()),
                 'verified_listing_dates':listings,
                 'stock_name_metadata':str(names_path.relative_to(root)) if names_path.exists() else None,
                 'official_bond_index_identities':bond_identities,
                 'source_crosscheck':'experiments/data_collection/d1/source_research/shfe_annual_crosscheck.json',
                 'zip_cache_validation':'experiments/data_collection/smoke_test/cache_validation.json',
                 'execution_manifest':str(execution_file.relative_to(root)),
                 'interrupted_write_recovery':recovery,
                 'changed_response_fingerprints':{url:sorted(values) for url,values in by_url.items() if len(values)>1},
                 'change_detection_note':'Different bytes at one URL are only response-change candidates: errors, dynamic HTML and distinct POST parameters can differ. Only same-scope API/parameter CSV comparisons test market-table vintage changes.',
                 'table_version_comparisons':revisions,
                 'table_fingerprint_note':'CSV content hashes exclude gzip headers; physical file hashes separately check immutable storage integrity.',
                 'licenses':'Provider-specific terms and redistribution permission NOT_VERIFIED; HTTP accessibility is not a data license.',
                 'documentation':['https://akshare.akfamily.xyz/data/bond/bond.html','https://akshare.akfamily.xyz/data/stock/stock.html','https://akshare.akfamily.xyz/data/futures/futures.html']},out/'source_manifest.json')
    atomic_json(environment(root),out/'environment.json')
    summary={}
    for name,mask in selections.items():
        table=coverage.loc[mask]
        summary[name.removesuffix('_coverage.csv')]={'configured':len(table),'with_observations':int(table['valid_observation_count'].gt(0).sum()),
                    'with_returned_rows':int(table['record_count'].gt(0).sum()),
                    'without_valid_observations':int(table['valid_observation_count'].eq(0).sum()),
                    'with_observations_ratio':float(table['valid_observation_count'].gt(0).mean()),'status_counts':table['collection_status'].value_counts().to_dict(),
                    'standardized_rows':int(table['record_count'].sum())}
    base=root/'data'
    sizes={folder:sum(p.stat().st_size for p in (base/folder/run_id).rglob('*') if p.is_file()) for folder in ('raw','standardized','manifests','audit')}
    checks_path=root/'experiments/data_collection/d1/environment_checks.final.json'
    if not checks_path.exists():
        checks_path=root/'experiments/data_collection/d1/environment_checks.json'
    checks=json.loads(checks_path.read_text()) if checks_path.exists() else []
    flags={'PROJECT_ENV_PASS':bool(checks) and all(x['returncode']==0 for x in checks),
           'STOCK_UNIVERSE_PASS':len(stocks)==263 and stocks['asset_id'].nunique()==263,
           'STOCK_COLLECTION_PASS':summary['stock']['status_counts'].get('SUCCESS',0)==263,
           'BOND_25_COLLECTION_PASS':summary['bond']['status_counts'].get('SUCCESS',0)==25,
           'FUTURES_24_TARGET_PASS':False,  # WH mapping remains explicitly unverified.
           'FUTURES_14_AUX_PASS':summary['futures_auxiliary']['status_counts'].get('SUCCESS',0)==14,
           'EXCHANGE_CONTRACT_PASS':summary['exchange_contract']['status_counts'].get('SUCCESS',0)==42,
           'DATA_SCHEMA_PASS':bool(frames) and not integrity and not source_identity_errors and int(coverage['required_field_missing_count'].sum())==0 and int(coverage['invalid_value_count'].sum())==0 and int(coverage['duplicate_date_count'].sum())==0,
           'DATA_CAUSALITY_PASS':'NOT_VERIFIED','DATA_AUDIT_PASS':False,
           'READY_FOR_CAUSAL_PREPROCESSING':'BLOCKED'}
    flags={key:(value if isinstance(value,str) else bool(value)) for key,value in flags.items()}
    results={'run_id':run_id,'requested_start':start,'requested_end':end,'flags':flags,
             'summary':summary,'execution':{k:v for k,v in execution.items() if k!='decisions'},'storage_bytes':sizes,
             'integrity_failures':integrity,'anomaly_flags':len(anomalies),'audit_executed':True,
             'raw_stock_identity_errors':source_identity_errors,
             'unverified':['Historic publication/PIT vintages','Stock listing dates, names, suspensions and corporate actions',
                           'Independent exchange calendar completeness','WH/legacy wheat economic mapping','Continuous roll rules',
                           'Price/turnover units by exchange and contract specification','Provider usage/redistribution permissions'],
             'no_training':True,'no_official_return_labels':True,'no_filled_prices':True}
    atomic_json(results,out/'audit_results.json')
    atomic_json({k:v for k,v in results.items() if k not in ('flags','unverified')},out/'collection_summary.json')
    atomic_json(RULES,mirror/'fixed_audit_rules.json')
    lines=['# TriWeave 15 年历史数据采集与科研审计报告','',f'采集窗口：{start} 至 {end}。生成时间：{utc_now()}。',
           '首段需求中的 2011-01-01 至 2026-01-01 与 D3/D9 冲突；本轮使用项目冻结配置及 D3/D9 的窗口。',
           '', '## 验收结果','',_table(pd.DataFrame([{'field':k,'value':str(v).lower() if isinstance(v,bool) else v} for k,v in flags.items()]),['field','value']),
           '', '## 实际采集量与完成率','',_table(pd.DataFrame([{'category':k,**v} for k,v in summary.items()]),['category','configured','with_observations','with_observations_ratio','status_counts','standardized_rows']),
           '',f'审计报告写入前的数据层体积快照（不含虚拟环境）：{json.dumps(sizes,ensure_ascii=False)} 字节；最终落盘体积另见 final_storage.json。',
           '完成率分开报告“已有观测资产比例”和 SUCCESS 数量；产生文件不自动认定完整成功。',
           '覆盖率、首末真实观测和缺失区间基于有限有效利率/正指数或正收盘价；零价和无效值只保留在返回行数与异常记录中，不计为真实覆盖。',
           '跨度年数仅表示首末有效日期间隔，并不表示中间无缺失；请同时读取 coverage_ratio、missing_intervals.csv 和 collection_status。',
           '代码核查、修复与测试证据见 ../d1/CODE_REVIEW_AND_FIXES.md 和 ../d1/final_validation.json。',
           '', '## 股票（263 个固定身份）','',f"实际取得观测：{summary['stock']['with_observations']}/263。名单直接读取最终 Clean V1，顺序与最终 node_order 和协议清单一致。",
           '该名单依据旧 TRAIN 历史资格筛选，不是 2011 年逐时点成分股名单；存在历史选择/幸存者偏差限制。',
           '不复权价格单独保留；未下载复权因子、公司行为及完整停牌/更名历史，不能声称总收益或历史可交易性已验证。',
           '股票证券名称若接口未提供，则以代码标识并标记待核验；不能从缺失行情推断停牌或退市。',
           '已从原 CMGM 名册补充描述性证券名称；旧名册 first_appear_date 是指数成分股首次出现日期，绝不是 IPO/上市日期。名称历史有效时段仍未核验。',
           '', '## 债券：逐变量真实覆盖','',_table(coverage.loc[selections['bond_coverage.csv']],['asset_id','unit','first_observation_date','last_observation_date','coverage_years','record_count','coverage_ratio','collection_status']),
           '18 条收益率始终保存百分比年化利率水平，未当作价格；7 条财富指数保持指数点位，未生成收益标签。',
           '七个财富指数已按官方目录 ID 核对名称；5Y、7Y 是对应年期国债指数，不改写为 5–7 年期限段。具体名称和 ID 见 bond_coverage.csv 与 source_manifest.json。',
           '债券参考日历采用股票交易日基准与同一源类别实际观测日的并集，保留银行工作周末。覆盖率不是独立来源对全历史完整性的证明。',
           '', '## 24 个预测目标（原顺序不变）','',_table(coverage.loc[selections['futures_target_coverage.csv']],['asset_id','first_observation_date','last_observation_date','coverage_years','record_count','collection_status']),
           '', '## 14 个辅助商品','',_table(coverage.loc[selections['futures_auxiliary_coverage.csv']],['asset_id','first_observation_date','last_observation_date','coverage_years','record_count','collection_status']),
           '所有主连均为 EXPLORATORY_CONTINUOUS_SERIES，不能作为正式监督标签或回测收益。',
           'WH 与旧代码存在代码名线索，但尚未独立核对历史合约规格、代码迁移和连续构造，保持 TARGET_MAPPING_NOT_VERIFIED。',
           'SC、LU、PG、SI、LC 的上市前缺失必须保留；未被可靠来源核实的上市日期不推断、不填补。',
           '',_table(pd.DataFrame([{'asset_id':asset,**evidence} for asset,evidence in listings.items()]),['asset_id','date','source']),
           '', '## 交易所逐合约数据与不稳定来源','',_table(pd.DataFrame(smoke['results']),['group','name','pass','record_count']),
           '',_table(exchanges,['exchange','planned_months','saved_response_months','failed_months','blocked_months','not_attempted_months','standardized_rows']),
           'DCE/GFEX 的真实 smoke 失败证据见 failed_requests.csv；未绕过访问限制，也未静默更换供应商。',
           '额外验证：上期所 2023-10 官方年度归档与逐日接口 2,618 行、11 个字段完全一致；仅合约标识合并单元格做向前展开，没有填补价格或日期。该归档未替换生产逐日来源。',
           '上期所 2023 年报表注明成交金额为万元、成交/持仓量为手且单边计算；不能将该当前口径未经核验外推到早期历史，特别是统计口径变更期。',
           '中金所月度 ZIP 缓存已真实复测：204 行完全一致，一次下载与 16 次同快照复用；新快照仍重新请求检查版本。',
           '郑商所年度归档补充探测返回 412，已保留证据；生产仍使用已通过的原逐日接口。',
           '交易所数据与主连分开统计；仅有连续序列而缺逐合约记录的品种可通过 exchange_contract_coverage.csv 的零记录行定位。',
           '逐合约覆盖不足、WH 映射与历史换月协议未完成，当前不能进入正式收益标签构建。',
           '能源中心返回含 SC_TAS 特殊交易方式记录，已保留并单列标记，不计入普通合约有效报价覆盖；不能与 SC 普通合约报价混算。交割月份可读取，交割年份历史解释仍为 NOT_VERIFIED。',
           'TAS 经济定义依据[能源中心官方通知](https://www.ine.com.cn/eng/circularnews/circular/202303/t20230329_822378.html)，不是本轮调整预测目标。',
           '', '## 异常、缺失与真实性','',f'审计标记 {len(anomalies)} 条；明细见 anomaly_report.csv。完整资产缺失统计见 missingness_report.csv。',
           '',_table(pd.DataFrame(anomalies,columns=['series_type','anomaly']).value_counts().rename('flags').reset_index(),['series_type','anomaly','flags']),
           '异常只标记，不替换、不插值、不反向填充。逐合约跳变仅在同一合约内检查；主连跳变仅供人工复核。',
           '没有将下载失败解释为资产不存在；UNAVAILABLE 仅用于可靠上市证据证明整个请求区间尚不可用的情形。',
           '上市日期尚未核验的资产，上市前真实性检查保持 NOT_VERIFIED；不能把“本程序不造价”写成“所有上游历史已证实无伪造”。',
           'SUCCESS 只表示相对于所列参考日历的数据检查通过，不能证明经济身份、PIT 发布时点或许可证已确认。',
           '中断留下的上期所 2025-03 raw 孤立文件已保留在 _interrupted/，附原 SHA256 和恢复记录；没有把缺失的任务 manifest 追认为成功，而是按原来源重新请求。',
           '', '## 时间隔离与科研归档','',
           '2011-10-01—2016-12-31 预热；2017—2022 TRAIN；2023—2024 VAL；2025-01-01—2026-09-30 SEALED CONFIRMATION。',
           '只采集与审核原始观测；没有训练、拟合标准化参数、调模型阈值、选 checkpoint 或生成 MAE/TEST 评价。',
           'standardized 层只统一字段、日期、代码和已声明单位，没有使用全历史拟合任何数值标准化参数。',
           '最长 20 步标签边界及交易日历 embargo 留待单独冻结协议；当前 derived 不含正式标签。',
           '', '## 未解决问题与优先级','',
           '1. 解决 DCE/GFEX 官方访问与逐合约覆盖；先重新做同源 smoke，再继续缺失交易所任务。',
           '2. 按覆盖表补采缺失区间；核对数据源原始单位、品种上市日期、股票停复牌与公司行为。',
           '3. 核对 WH/旧“小麦”经济身份，冻结逐合约换月与可执行收益定义。',
           '4. 核验 ChinaBond、Eastmoney、Sina 和交易所数据使用及再分发授权；若公开接口不足，再由用户选择获授权的数据源，不能由失败推断必须付费。',
           '5. 补齐 PIT 发布时点和修订版本规则，再评估 TriWeave V2 因果预处理准入。',
           '', '## 可继续执行的命令','', '```bash','source .venv/bin/activate',
           'python -m triweave.pipeline smoke --run-id smoke_202310_retry1',
           'python -m triweave.pipeline collect --run-id baseline15y',
           'python -m triweave.research_audit --run-id baseline15y','```',
           '成功历史任务会先校验指纹再复用；更新接口或历史快照时必须使用新的 run-id。',
           '', '来源文档：[债券](https://akshare.akfamily.xyz/data/bond/bond.html)、[股票](https://akshare.akfamily.xyz/data/stock/stock.html)、[期货](https://akshare.akfamily.xyz/data/futures/futures.html)。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    for path in out.iterdir():
        if path.is_file():
            (mirror/path.name).write_bytes(path.read_bytes())
    sealed=root/'data/sealed'/run_id
    sealed.mkdir(parents=True,exist_ok=True)
    policy=sealed/'policy.json'
    if not policy.exists():
        atomic_json({'confirmation_start':'2025-01-01','confirmation_end':end,'raw_download_allowed':True,
                     'model_selection_allowed':False,'fitted_transform_parameters':None,'labels_created':False,
                     'maximum_future_label_horizon_steps':20,'boundary_protocol':'NOT_YET_FROZEN'},policy)
    return results


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='.')
    parser.add_argument('--run-id',default='baseline15y')
    args=parser.parse_args()
    print(json.dumps(comprehensive_audit(Path(args.root),args.run_id),indent=2,ensure_ascii=False))
