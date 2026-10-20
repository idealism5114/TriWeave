"""Candidate roll mechanics for diagnostics only; no official labels or backtest."""
import pandas as pd

from triweave.contract_identity import VERIFIED
from triweave.normalize import SchemaError, verify_unique


def validate_registry(registry):
    required = {'contract_code', 'listing_date_if_verified', 'last_trade_date_if_verified',
                'metadata_available_date_for_diagnostic', 'verification_status'}
    if required - set(registry): raise SchemaError('Verified lifecycle registry required')
    if registry.empty or not registry.verification_status.eq(VERIFIED).all(): raise SchemaError('Identity/year NOT_VERIFIED; no roll selection')
    if registry.contract_code.duplicated().any(): raise SchemaError('Ambiguous repeated contract identity')


def known_registry(registry, date):
    return registry.loc[(registry.metadata_available_date_for_diagnostic <= date) &
                        (registry.listing_date_if_verified <= date) &
                        (registry.last_trade_date_if_verified >= date)]


def fixed_expiry_selection(registry, calendar, n=5, calendar_verified=False):
    validate_registry(registry)
    if not calendar_verified: raise SchemaError('Ex-ante exchange trading calendar NOT_VERIFIED')
    if n < 1: raise ValueError('Positive fixed roll lead required')
    days = sorted(set(calendar)); position = {d: i for i, d in enumerate(days)}
    if not registry.last_trade_date_if_verified.isin(days).all(): raise SchemaError('Calendar must include verified last-trading dates')
    result = []
    for i, date in enumerate(days):
        available = known_registry(registry, date).copy()
        available = available.loc[available.last_trade_date_if_verified.map(position).sub(i).gt(n)]
        available = available.sort_values(['last_trade_date_if_verified', 'contract_code'])
        result.append({'decision_date': date, 'contract': available.contract_code.iloc[0] if len(available) else None,
                       'rule': f'A_FIXED_EXPIRY_N{n}_DESIGN', 'signal_cutoff': None, 'official_label': False})
    return pd.DataFrame(result)


def lagged_activity_selection(quotes, registry, calendar, lag=2, confirmation=2):
    validate_registry(registry)
    if lag < 1 or confirmation < 1: raise ValueError('Lag and confirmation must be positive')
    required = {'date', 'contract', 'close', 'volume', 'open_interest'}
    if required - set(quotes): raise SchemaError('Daily activity schema incomplete')
    verify_unique(quotes, ['date', 'contract'])
    days = sorted(set(calendar)); current = None; results = []
    for i, date in enumerate(days):
        known = known_registry(registry, date); known_codes = set(known.contract_code)
        if current not in known_codes: current = None
        winners = []; cutoff = days[i-lag] if i >= lag else None
        # Exactly the already observed lagged signal days, never today's close/OI.
        for j in range(max(0, i-lag-confirmation+1), i-lag+1):
            signal = days[j]
            q = quotes.loc[quotes.date.eq(signal) & quotes.contract.isin(known_codes)].copy()
            # A future identity-publication date cannot make a historical quote
            # eligible before the registry itself became available.
            signal_known = known_registry(registry, signal)
            q = q.loc[q.contract.isin(signal_known.contract_code) & q.close.gt(0) &
                      q.volume.gt(0) & q.open_interest.ge(0)]
            q = q.sort_values(['volume', 'open_interest', 'contract'], ascending=[False, False, True])
            winners.append(q.contract.iloc[0] if len(q) else None)
        if len(winners) == confirmation and winners[-1] and len(set(winners)) == 1:
            current = winners[-1]
        results.append({'decision_date': date, 'contract': current, 'rule': f'B_ACTIVITY_LAG{lag}_CONFIRM{confirmation}_DESIGN',
                        'signal_cutoff': cutoff, 'confirmation': confirmation, 'lag_sessions': lag,
                        'official_label': False, 'execution_status': 'DIAGNOSTIC_ONLY_PUBLICATION_AND_FILL_UNVERIFIED'})
    return pd.DataFrame(results)


def same_contract_checks(quotes, calendar):
    """Arithmetic diagnostics; skip gaps and never use another contract as lag."""
    f = quotes.sort_values(['contract', 'date']).copy(); group = f.groupby('contract')
    f['previous_date'] = group.date.shift(); f['previous_close'] = group.close.shift()
    positions = {d: i for i, d in enumerate(sorted(set(calendar)))}
    adjacent = f.date.map(positions).sub(f.previous_date.map(positions)).eq(1)
    valid = adjacent & f.close.gt(0) & f.previous_close.gt(0)
    f['diagnostic_same_contract_price_change'] = (f.close / f.previous_close - 1).where(valid)
    f['diagnostic_status'] = valid.map({True: 'SAME_CODE_ADJACENT_SESSION_ARITHMETIC_ONLY', False: 'NO_VALID_ADJACENT_SAME_CONTRACT_PAIR'})
    f['official_label'] = False
    return f


def adjacent_contract_price_relations(quotes, registry):
    """Same-day term gaps between verified adjacent maturities, never returns."""
    validate_registry(registry)
    ordered=registry.sort_values(['last_trade_date_if_verified','contract_code']).to_dict('records')
    chunks=[]
    for old,new in zip(ordered,ordered[1:]):
        left=quotes.loc[quotes.contract.eq(old['contract_code']),['date','close']].rename(columns={'close':'old_close'})
        right=quotes.loc[quotes.contract.eq(new['contract_code']),['date','close']].rename(columns={'close':'new_close'})
        f=left.merge(right,on='date',validate='one_to_one')
        available=max(old['metadata_available_date_for_diagnostic'],new['metadata_available_date_for_diagnostic'])
        expiry=min(old['last_trade_date_if_verified'],new['last_trade_date_if_verified'])
        f=f.loc[f.date.ge(available)&f.date.le(expiry)&f.old_close.gt(0)&f.new_close.gt(0)].copy()
        f['old_contract']=old['contract_code'];f['new_contract']=new['contract_code']
        f['same_day_term_gap']=f.new_close-f.old_close;f['official_label']=False
        chunks.append(f)
    columns=['date','old_close','new_close','old_contract','new_contract','same_day_term_gap','official_label']
    return pd.concat(chunks,ignore_index=True) if chunks else pd.DataFrame(columns=columns)
