"""Source-backed identities; code hypotheses never become verified delivery years."""
import json
from pathlib import Path
import re

import pandas as pd

from triweave.assets import ALL_CONTRACT_FUTURES
from triweave.normalize import SchemaError
from triweave.universe import sha256_file

VERIFIED = 'VERIFIED_FULL_DATES_FROM_OFFICIAL_METADATA'


def parse_code(code, exchange):
    code = str(code).strip().upper()
    if '_TAS' in code:
        return {'code_status': 'SPECIAL_MODE_EXCLUDED', 'delivery_month': None, 'code_format': 'TAS'}
    match = re.fullmatch(r'([A-Z]+)(\d{3,4})', code)
    if not match: return {'code_status': 'INVALID_OR_CONTINUOUS_CODE', 'delivery_month': None, 'code_format': None}
    digits = match[2]; month = int(digits[-2:])
    if not 1 <= month <= 12 or (exchange != 'CZCE' and len(digits) != 4):
        return {'code_status': 'INVALID_MONTH_OR_EXCHANGE_FORMAT', 'delivery_month': None, 'code_format': digits}
    return {'code_status': 'ORDINARY_CODE_PATTERN_ONLY', 'delivery_month': month,
            'code_format': 'YMM_DECADE_AMBIGUOUS' if len(digits) == 3 else 'YYMM_CENTURY_UNVERIFIED',
            'variety_from_code': match[1]}


def shfe_metadata(root, raw_path):
    """Full delivery dates, not YYMM substring inference, establish the year."""
    root, raw_path = Path(root), Path(raw_path)
    data = json.loads(raw_path.read_text())
    if str(data.get('report_date')) != '20191008' or not data.get('ContractBaseInfo'):
        raise SchemaError('Unexpected official SHFE metadata snapshot')
    f = pd.DataFrame(data['ContractBaseInfo']); results = []
    required = {'INSTRUMENTID', 'OPENDATE', 'EXPIREDATE', 'STARTDELIVDATE', 'ENDDELIVDATE', 'TRADINGDAY', 'UPDATE_DATE'}
    if required - set(f): raise SchemaError('Official contract metadata schema changed')
    if f.INSTRUMENTID.duplicated().any(): raise SchemaError('Duplicate metadata contract')
    configured_shfe = {s.variety for s in ALL_CONTRACT_FUTURES if s.exchange == 'SHFE'}
    for row in f.to_dict('records'):
        code = row['INSTRUMENTID'].strip().upper(); parsed = parse_code(code, 'SHFE')
        if parsed['code_status'] != 'ORDINARY_CODE_PATTERN_ONLY': continue
        # This endpoint also contains SC. Hostname alone does not establish its
        # exchange; only the fixed SHFE scope is certified by this adapter.
        if parsed['variety_from_code'] not in configured_shfe: continue
        dates = {c: pd.to_datetime(row[c], format='%Y%m%d', errors='raise')
                 for c in ('OPENDATE', 'EXPIREDATE', 'STARTDELIVDATE', 'ENDDELIVDATE', 'TRADINGDAY')}
        delivery = dates['STARTDELIVDATE']; end = dates['ENDDELIVDATE']
        digits = re.search(r'\d+$', code)[0]
        matched = delivery.year == end.year and delivery.month == end.month and delivery.month == parsed['delivery_month'] and delivery.year % 100 == int(digits[:2])
        if dates['OPENDATE'] > dates['EXPIREDATE'] or str(row['TRADINGDAY']) != '20191008':
            raise SchemaError('Invalid listing/trading lifecycle')
        results.append({'exchange': 'SHFE', 'variety': parsed['variety_from_code'], 'contract_code': code,
            'delivery_year': delivery.year if matched else None, 'delivery_month': parsed['delivery_month'],
            'listing_date_if_verified': dates['OPENDATE'].strftime('%Y-%m-%d'),
            'last_trade_date_if_verified': dates['EXPIREDATE'].strftime('%Y-%m-%d'),
            'metadata_available_date_for_diagnostic': '2019-10-09',
            'source_reported_update_at': str(row['UPDATE_DATE']) + '+08:00',
            'metadata_pit_status': 'SOURCE_REPORTED_UPDATE_TIME_NO_INDEPENDENT_VINTAGE_PROOF',
            'identity_source_id': 'shfe_official_contract_metadata_20191008_v1',
            'identity_evidence_file': str(raw_path.relative_to(root)), 'identity_evidence_sha256': sha256_file(raw_path),
            'source_delivery_start': str(row['STARTDELIVDATE']), 'source_delivery_end': str(row['ENDDELIVDATE']),
            'verification_status': VERIFIED if matched else 'NOT_VERIFIED',
            'metadata_check': 'FULL_DATES_AND_CODE_MATCH' if matched else 'DELIVERY_DATE_CODE_RELATION_REQUIRES_VARIETY_SPECIFIC_RULE'})
    return pd.DataFrame(results)


def build_identity_manifest(quotes, metadata):
    specs = {(s.exchange, s.variety): s for s in ALL_CONTRACT_FUTURES}
    meta = {(r.exchange, r.contract_code): r._asdict() for r in metadata.itertuples(index=False)
            if r.verification_status == VERIFIED}
    rows = []
    for (exchange, variety, code, source), frame in quotes.groupby(['exchange', 'variety', 'contract', 'source_id'], sort=True):
        frame = frame.sort_values('date').copy()
        # CZCE YMM codes recur in different decades. Never pool those observations
        # into one economic contract. Episode separation is diagnostic, not proof.
        boundaries = pd.to_datetime(frame.date).diff().dt.days.gt(730).cumsum() if exchange == 'CZCE' else pd.Series(0, index=frame.index)
        for episode, f in frame.groupby(boundaries, sort=True):
            parsed = parse_code(code, exchange); known = meta.get((exchange, code), {})
            if known and (f.date.min() < known['listing_date_if_verified'] or f.date.max() > known['last_trade_date_if_verified']):
                known = {}  # Do not certify contradictory or another-lifecycle rows.
            identity = {'exchange': exchange, 'variety': variety, 'contract_code': code,
                'contract_identity_key': f'{exchange}:{code}:observation_episode_{int(episode)}',
                'observation_episode': int(episode), 'episode_rule': 'CZCE gaps >730 calendar days split for audit only; no inferred year',
                'delivery_year': known.get('delivery_year'), 'delivery_month': parsed.get('delivery_month'),
                'first_observed_date': f.date.min(), 'last_observed_date': f.date.max(), 'record_count': len(f),
                'listing_date_if_verified': known.get('listing_date_if_verified'),
                'last_trade_date_if_verified': known.get('last_trade_date_if_verified'),
                'price_unit': 'NATIVE_EXCHANGE_QUOTE_UNIT_HISTORICAL_SPEC_NOT_VERIFIED',
                'volume_unit': 'source_reported_lots_STATISTICAL_BASIS_NOT_VERIFIED',
                'open_interest_unit': 'source_reported_lots_STATISTICAL_BASIS_NOT_VERIFIED',
                'contract_multiplier_if_verified': None, 'source_id': source,
                'verification_status': known.get('verification_status', 'NOT_VERIFIED'),
                'code_status': parsed['code_status'], 'code_format': parsed.get('code_format'),
                'exchange_variety_match': (exchange, variety) in specs and parsed.get('variety_from_code') == variety,
                'trading_mode': 'TAS_SPECIAL' if parsed['code_status'] == 'SPECIAL_MODE_EXCLUDED' else 'OUTRIGHT',
                'identity_evidence_file': known.get('identity_evidence_file'),
                'identity_evidence_sha256': known.get('identity_evidence_sha256'),
                'quote_manifests': ';'.join(sorted(f.manifest.unique())),
                'official_label_eligible': False}
            rows.append(identity)
    return pd.DataFrame(rows)
