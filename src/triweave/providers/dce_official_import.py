"""Fail-closed DCE file importer. Generic table schemas are only synthetic-tested.

Run with --help. An input file alone cannot establish contract existence, delivery
year, permissions or units: a verified identity registry and source sidecar are
required. No network calls, price filling, continuous series or return labels.
"""
import argparse
import csv
from datetime import datetime
import io
import json
from pathlib import Path
import re
import shutil
from urllib.parse import urlsplit

import pandas as pd

from triweave.assets import ALL_CONTRACT_FUTURES
from triweave.collect import atomic_csv, atomic_json, utc_now
from triweave.normalize import SchemaError, finite, verify_unique
from triweave.universe import sha256_file

IDENTITY_VERIFIED = 'VERIFIED_FROM_EXCHANGE_OR_AUTHORIZED_DIRECTORY'
FIELDS = {
    'date': ('date', 'trade_date', 'trading_date', '交易日期', '日期'),
    'contract': ('contract', 'contract_code', 'symbol', '合约', '合约代码'),
    'exchange': ('exchange', '交易所'),
    'open': ('open', '开盘价'), 'high': ('high', '最高价'),
    'low': ('low', '最低价'), 'close': ('close', '收盘价'),
    'settle': ('settle', 'settlement', '结算价'),
    'pre_settle': ('pre_settle', '前结算价', '昨结算'),
    'volume': ('volume', 'vol', '成交量'),
    'open_interest': ('open_interest', 'oi', '持仓量'),
    'amount': ('amount', 'turnover', '成交额'),
    'trading_status': ('trading_status', 'status', '交易状态'),
}
REQUIRED = ('date', 'contract', 'open', 'high', 'low', 'close', 'settle', 'volume', 'open_interest')
DCE_VARIETIES = {s.variety for s in ALL_CONTRACT_FUTURES if s.exchange == 'DCE'}


def strict_dates(values):
    s = values.astype('string').str.strip()
    compact = s.str.fullmatch(r'\d{8}').fillna(False)
    iso = s.str.fullmatch(r'\d{4}-\d{2}-\d{2}(?: 00:00:00)?').fillna(False)
    if not (compact | iso).all(): raise SchemaError('Missing or ambiguous date; no filename/date guessing permitted')
    result = pd.to_datetime(s.where(iso).str[:10], format='%Y-%m-%d', errors='coerce')
    result.loc[compact] = pd.to_datetime(s.loc[compact], format='%Y%m%d', errors='coerce')
    if result.isna().any(): raise SchemaError('Invalid calendar date')
    return result.dt.strftime('%Y-%m-%d')


def read_table(path):
    path = Path(path)
    if path.suffix.lower() in ('.xlsx', '.xls'):
        book = pd.ExcelFile(path)
        if len(book.sheet_names) != 1: raise SchemaError('Multiple Excel sheets require an explicit separately reviewed adapter')
        cells = pd.read_excel(book, dtype=str, keep_default_na=False, header=None)
        if cells.empty: raise SchemaError('Empty Excel sheet')
        headers = cells.iloc[0].astype(str).tolist()
        if len(set(headers)) != len(headers): raise SchemaError('Duplicate raw headers')
        out = cells.iloc[1:].reset_index(drop=True); out.columns = headers
        return out
    if path.suffix.lower() not in ('.csv', '.txt', '.tsv'): raise SchemaError('Unsupported format; no genuine sample-specific adapter')
    data = path.read_bytes(); text = None
    for encoding in ('utf-8-sig', 'gb18030'):
        try: text = data.decode(encoding); break
        except UnicodeDecodeError: pass
    if text is None: raise SchemaError('Unknown file encoding')
    try: sep = csv.Sniffer().sniff(text[:16384], delimiters=',\t;').delimiter
    except csv.Error: raise SchemaError('Ambiguous delimiter or non-table file') from None
    headers = next(csv.reader(io.StringIO(text), delimiter=sep), [])
    if len(set(headers)) != len(headers): raise SchemaError('Duplicate raw headers')
    return pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=False)


def validate_sidecar(sidecar, synthetic=False):
    if any(re.search(r'token|secret|password|credential|cookie|api_key|authorization', str(k), re.I) for k in sidecar):
        raise SchemaError('Source sidecar must not contain credentials')
    required = ('source_id', 'exchange', 'source_url', 'acquired_at', 'permission_status',
                'data_origin', 'price_unit', 'volume_unit', 'open_interest_unit', 'amount_unit', 'statistics_basis')
    if any(not sidecar.get(k) for k in required): raise SchemaError('Incomplete source/permission/unit sidecar')
    if sidecar['exchange'] != 'DCE': raise SchemaError('Only DCE files are accepted')
    if not re.fullmatch(r'dce_official_file_[A-Za-z0-9_-]+', sidecar['source_id']):
        raise SchemaError('Use a distinct versioned DCE official-file source_id')
    if synthetic:
        if sidecar['data_origin'] != 'SYNTHETIC_TEST': raise SchemaError('Synthetic dry-run must declare SYNTHETIC_TEST')
    elif sidecar['data_origin'] != 'DCE_OFFICIAL_EXPORT' or sidecar['permission_status'] != 'AUTHORIZED_RESEARCH_USE':
        raise SchemaError('An authorized official export declaration is required; importer cannot grant a license')
    u = urlsplit(sidecar['source_url'])
    if u.scheme not in ('http', 'https') or not (u.hostname == 'dce.com.cn' or (u.hostname or '').endswith('.dce.com.cn')):
        raise SchemaError('Official DCE provenance URL required')
    if u.username or u.password or u.query or u.fragment: raise SchemaError('Credential/session-bearing source URLs are forbidden')
    try: stamp = datetime.fromisoformat(sidecar['acquired_at'])
    except ValueError: raise SchemaError('acquired_at must be an ISO timestamp') from None
    if stamp.tzinfo is None: raise SchemaError('acquired_at timezone required')


def load_identities(path):
    path = Path(path); f = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = ('exchange', 'variety', 'contract_code', 'delivery_year', 'delivery_month',
                'listing_date', 'last_trade_date', 'verification_status',
                'identity_evidence_file', 'identity_evidence_sha256')
    if set(required) - set(f): raise SchemaError('Verified contract directory fields missing')
    if f.empty or not f.exchange.eq('DCE').all(): raise SchemaError('Empty or wrong-exchange identity registry')
    f['contract_code'] = f.contract_code.str.strip().str.upper()
    if not f.verification_status.eq(IDENTITY_VERIFIED).all(): raise SchemaError('Delivery year and existence NOT_VERIFIED; import refused')
    if not f.contract_code.str.fullmatch(r'[A-Z]+\d{4}').all(): raise SchemaError('Ordinary DCE YYMM contract identity required')
    verify_unique(f, ['exchange', 'contract_code'])
    evidence = {}
    for row in f.itertuples():
        match = re.fullmatch(r'([A-Z]+)(\d{2})(\d{2})', row.contract_code)
        if row.variety != match[1] or row.variety not in DCE_VARIETIES: raise SchemaError('Identity variety is outside the frozen DCE universe')
        try: year, month = int(row.delivery_year), int(row.delivery_month)
        except ValueError: raise SchemaError('Unexplained delivery year/month') from None
        if not 1900 <= year <= 2200 or not 1 <= month <= 12 or year % 100 != int(match[2]) or month != int(match[3]):
            raise SchemaError('Directory delivery year/month contradicts the ordinary contract code')
        p = (path.parent / row.identity_evidence_file).resolve()
        if not p.is_file() or sha256_file(p) != row.identity_evidence_sha256:
            raise SchemaError('Contract-directory evidence missing or checksum mismatch')
        evidence[str(p)] = row.identity_evidence_sha256
    f['listing_date'] = strict_dates(f.listing_date); f['last_trade_date'] = strict_dates(f.last_trade_date)
    if (f.listing_date > f.last_trade_date).any(): raise SchemaError('Listing date after last trading date')
    delivery_year = pd.to_numeric(f.delivery_year)
    expiry_year = pd.to_datetime(f.last_trade_date).dt.year
    if not (delivery_year - expiry_year).between(0, 1).all():
        raise SchemaError('Delivery century/year incompatible with verified trading lifecycle')
    return f, evidence


def normalize_table(raw, identities, sidecar):
    if raw.empty: raise SchemaError('Empty file is not collection success')
    if raw.columns.duplicated().any(): raise SchemaError('Duplicate raw headers')
    mapping = {}
    for field, aliases in FIELDS.items():
        found = [c for c in raw if str(c).strip() in aliases]
        if len(found) > 1: raise SchemaError('Conflicting aliases for ' + field)
        if found: mapping[found[0]] = field
    out = raw.rename(columns=mapping).copy()
    if set(REQUIRED) - set(out): raise SchemaError('Missing required daily fields: ' + ','.join(sorted(set(REQUIRED)-set(out))))
    if 'exchange' in out and not out.exchange.astype(str).str.strip().eq('DCE').all():
        raise SchemaError('Input exchange contradicts DCE sidecar/directory')
    out['date'] = strict_dates(out.date); out['exchange'] = 'DCE'
    out['contract'] = out.contract.astype(str).str.upper().str.strip()
    if not out.contract.str.fullmatch(r'[A-Z]+\d{4}').all(): raise SchemaError('Continuous/special/invalid contracts forbidden')
    if not out.contract.isin(identities.contract_code).all(): raise SchemaError('Contract absent from verified directory; existence/year not established')
    verify_unique(out, ['exchange', 'contract', 'date'])
    registry = identities.set_index('contract_code')
    for field in ('variety', 'delivery_year', 'delivery_month', 'listing_date', 'last_trade_date'):
        out[field] = out.contract.map(registry[field])
    if (out.date < out.listing_date).any() or (out.date > out.last_trade_date).any():
        raise SchemaError('Observation outside verified contract listing/trading dates')
    for field in ('open', 'high', 'low', 'close', 'settle', 'pre_settle', 'volume', 'open_interest', 'amount'):
        if field not in out: out[field] = float('nan'); continue
        text = out[field].astype(str).str.strip()
        grouped = text.str.fullmatch(r'[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?').fillna(False)
        text = text.where(~grouped, text.str.replace(',', '', regex=False))
        out[field] = pd.to_numeric(text, errors='coerce')
        if out[field].isin([float('inf'), float('-inf')]).any() or out[field].lt(0).any():
            raise SchemaError('Negative/nonfinite numerical value in ' + field)
        # Unknown text is never silently accepted as an inactive missing value.
        missing_token = text.isin(['', '-', '--', 'nan', 'None'])
        if (out[field].isna() & ~missing_token).any(): raise SchemaError('Unrecognized numerical text in ' + field)
    activity = out[['volume', 'open_interest']]
    if activity.isna().any().any() or (activity % 1).ne(0).any().any(): raise SchemaError('Volume/OI must be finite nonnegative lot counts')
    status = out.get('trading_status', pd.Series('UNSPECIFIED', index=out.index)).astype(str).str.strip().str.upper()
    # Native numeric flags require a genuine sample and documented semantics.
    # The generic adapter only accepts explicitly named canonical states.
    if not status.isin(['UNSPECIFIED', 'ACTIVE', 'INACTIVE', 'NO_TRADES', 'HALTED']).all(): raise SchemaError('Unknown explicit trading status')
    inactive = status.isin(['INACTIVE', 'NO_TRADES', 'HALTED'])
    if (inactive & out.volume.ne(0)).any(): raise SchemaError('Inactive status contradicts positive volume')
    prices = out[['open', 'high', 'low', 'close', 'settle']]
    bad = prices.isna().any(axis=1) | prices.le(0).any(axis=1)
    if (bad & ~inactive).any(): raise SchemaError('Invalid/missing quote without an explicit inactive status')
    complete = prices[['open', 'high', 'low', 'close']].gt(0).all(axis=1)
    relations = out.high.lt(out[['open', 'low', 'close']].max(axis=1)) | out.low.gt(out[['open', 'high', 'close']].min(axis=1))
    if (complete & relations).any(): raise SchemaError('Inconsistent OHLC relation')
    out['trading_status'] = status; out['valid_trade_quote'] = ~inactive & ~bad
    out['quote_status'] = 'STRUCTURAL_CHECKS_ONLY_NOT_EXECUTABILITY_CERTIFIED'
    out.loc[inactive, 'quote_status'] = 'EXPLICIT_INACTIVE_PLACEHOLDERS_PRESERVED'
    out['asset_id'] = 'FUT_' + out.variety; out['series_type'] = 'exchange_listed_contract'
    out['source_id'] = sidecar['source_id']; out['eligible_for_official_labels'] = False
    for field in ('price_unit', 'volume_unit', 'open_interest_unit', 'amount_unit', 'statistics_basis'):
        out[field] = sidecar[field]
    return out.sort_values(['contract', 'date']).reset_index(drop=True)


def import_file(input_path, identity_path, source_path, root, run_id, dry_run=False, synthetic=False):
    input_path, identity_path, source_path, root = map(Path, (input_path, identity_path, source_path, root))
    if synthetic and not dry_run: raise SchemaError('Synthetic demonstrations are dry-run only; no research archive writes')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id): raise SchemaError('Unsafe run ID')
    sidecar = json.loads(source_path.read_text()); validate_sidecar(sidecar, synthetic)
    source = sidecar['source_id']; fingerprint = sha256_file(input_path)
    scope = {'input_sha256': fingerprint, 'identity_sha256': sha256_file(identity_path),
             'sidecar_sha256': sha256_file(source_path)}
    raw_dir = root / 'data/raw' / run_id / source / fingerprint
    standard = root / 'data/standardized' / run_id / source / (fingerprint + '.csv.gz')
    manifest = root / 'data/manifests' / run_id / source / (fingerprint + '.json')
    if not dry_run and manifest.exists():
        old = json.loads(manifest.read_text())
        if old['scope'] != scope: raise FileExistsError('Existing input has different provenance/identity; use a new explicit version')
        for p, digest in old['archive_fingerprints'].items():
            if sha256_file(root / p) != digest: raise SchemaError('Archive checksum failure')
        if old['status'] != 'VALIDATED_FILE_NOT_AUTHENTICITY_OR_HISTORY_CERTIFIED':
            raise FileExistsError('Failed version preserved; use a new version after correction')
        if sha256_file(standard) != old['standardized_sha256']: raise SchemaError('Standardized archive checksum failure')
        return {**old, 'resume_status': 'VERIFIED_CACHE_REUSED'}
    archives = {}
    def preserve(p, name):
        dst = raw_dir / name; dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if sha256_file(dst) != sha256_file(p): raise FileExistsError('Immutable archive name collision')
        else:
            shutil.copyfile(p, dst); dst.chmod(0o444)
        archives[str(dst.relative_to(root))] = sha256_file(dst)
    doc = {'source_id': source, 'scope': scope, 'retrieved_at': sidecar['acquired_at'],
           'imported_at': utc_now(), 'adapter_sha256': sha256_file(Path(__file__)),
           'format_validation_status': 'GENERIC_CSV_EXCEL_SYNTHETIC_TESTED_NO_REAL_OFFICIAL_SAMPLE',
           'license_note': 'Checks a recorded permission declaration; cannot authenticate offline documents or grant rights',
           'archive_fingerprints': archives}
    try:
        if not dry_run:
            preserve(input_path, 'original' + input_path.suffix.lower())
            preserve(identity_path, 'identity_registry.csv'); preserve(source_path, 'source.json')
        identities, evidence = load_identities(identity_path)
        if not dry_run:
            for path, digest in evidence.items(): preserve(Path(path), 'identity_evidence_' + digest)
        raw = read_table(input_path); out = normalize_table(raw, identities, sidecar)
        if out.date.gt(datetime.fromisoformat(sidecar['acquired_at']).date().isoformat()).any():
            raise SchemaError('Observation date is later than the declared acquisition date')
        doc.update(raw_rows=len(raw), raw_columns=list(raw), standardized_rows=len(out),
                   first_date=out.date.min(), last_date=out.date.max(), contracts=out.contract.nunique(),
                   invalid_placeholder_rows=int((~out.valid_trade_quote).sum()),
                   status='SYNTHETIC_DRY_RUN_ONLY' if synthetic else 'VALIDATED_FILE_NOT_AUTHENTICITY_OR_HISTORY_CERTIFIED')
        if dry_run: return {**doc, 'dry_run': True, 'research_archive_writes': 0}
        keys = ['exchange', 'contract', 'date']
        for prior in manifest.parent.glob('*.json'):
            m = json.loads(prior.read_text())
            if m.get('status') == 'VALIDATED_FILE_NOT_AUTHENTICITY_OR_HISTORY_CERTIFIED':
                p = root / m['standardized_file']
                if sha256_file(p) != m['standardized_sha256']: raise SchemaError('Existing source checksum failure')
                if not out[keys].merge(pd.read_csv(p, usecols=keys), on=keys).empty:
                    raise FileExistsError('Overlapping source/date/contract batch; preserve and version explicitly')
        if standard.exists(): raise FileExistsError('Untracked standardized file preserved; new version required')
        out['raw_file'] = str((raw_dir / ('original' + input_path.suffix.lower())).relative_to(root))
        out['manifest'] = str(manifest.relative_to(root)); out['retrieved_at'] = sidecar['acquired_at']
        atomic_csv(out, standard)
        doc.update(standardized_file=str(standard.relative_to(root)), standardized_sha256=sha256_file(standard))
        atomic_json(doc, manifest)
        return doc
    except Exception as exc:
        if not dry_run:
            doc.update(status='REJECTED_PRESERVED', error_type=type(exc).__name__, error=str(exc))
            atomic_json(doc, manifest)
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True); p.add_argument('--identities', required=True)
    p.add_argument('--source', required=True); p.add_argument('--root', default='.')
    p.add_argument('--run-id', default='contracts_20261015'); p.add_argument('--dry-run', action='store_true')
    p.add_argument('--synthetic-test', action='store_true', help='Explicit synthetic demonstration; requires --dry-run')
    args = p.parse_args(argv)
    try:
        doc = import_file(args.input, args.identities, args.source, args.root, args.run_id, args.dry_run, args.synthetic_test)
    except (ValueError, OSError, KeyError) as exc:
        p.exit(2, type(exc).__name__ + ': ' + str(exc) + '\n')
    print(json.dumps(doc, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
