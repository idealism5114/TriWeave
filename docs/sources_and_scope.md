# Raw source specification (auditable assumptions)

Research requested: 2011-10-01 through 2026-09-30. Source existence does not imply full-window availability.

## A shares: 263

Source adapter: `ak.stock_zh_a_hist(symbol='600038', period='daily', start_date='20111001', end_date='20260930', adjust='')` to Eastmoney data, one output table per security. Fields include OHLC, volume, value and turnover. The **identities of the 263 stocks are not in CMGM's tracked GitHub tree**. Obtain the old final Clean V1 manifest. No dynamic scraping of contemporary CSI300 constituents to replace the chosen universe. Shares listed after 2011 have genuine prelisting missing values.

## Bond variables: exactly 25

**18 yield series**, AKShare `bond_china_yield` from ChinaBond historical curves, requested in <=300-day chunks. These are all yield levels, unit percent p.a. (not prices):

- `CN_GOV_YIELD_*`: 3M, 6M, 1Y, 3Y, 5Y, 7Y, 10Y, 30Y = 8
- `CN_AAA_NOTE_YIELD_*`: 3M, 6M, 1Y, 3Y, 5Y = 5
- `CN_AAA_BANK_YIELD_*`: 3M, 6M, 1Y, 3Y, 5Y = 5

The official source's exact *curve names* and *tenor labels* are matched in `assets.py`. Some tenors and curve categories can be absent/NaN in early history; do not fabricate.

**7 wealth indices**, AKShare `bond_treasury_index_cbond(indicator='wealth', period=...)` where indicator is passed in Chinese as the official AKShare enum; tenors: `0-1Y`, `1-3Y`, `3-5Y`, `5Y`, `7Y`, `7-10Y`, `10Y`. Levels are index units. ChinaBond's data terms/licensing must be respected.

Total 18+7 = **25**. No 30-year treasury futures, late-listed ETFs, or third-market price series are silently counted toward these 25. Four listed CFFEX treasury futures (TF,T,TS,TL) can also be collected with the contracts group as optional exchange price data, but have genuine later start dates.

## Commodity futures: exactly 24 targets + 14 auxiliaries

*24 targets* (variety @ exchange):

- INE: `LU`, `SC` = 2
- CZCE: `WH` (legacy wheat identity to review), `CF`, `MA`, `SR`, `OI`, `RM` = 6
- DCE: `PG`, `J`, `JM`, `C`, `M`, `A` = 6
- SHFE: `HC`, `FU`, `AG`, `RB`, `CU`, `AL`, `ZN`, `SN`, `NI`, `AU` = 10

*14 auxiliaries*:

- DCE: `I`, `EG`, `PP`, `L`, `Y`, `P`
- CZCE: `FG`, `SA`, `TA`
- SHFE: `RU`, `SP`, `BU`
- GFEX: `SI`, `LC`

Vendor continuous retrieval: AKShare `futures_zh_daily_sina(symbol='RB0')` etc, exploratory only, individual results subject to upstream availability and existence. 2011 history is not guaranteed.

Exchange contract retrieval: `ak.get_futures_daily(start_date='20231001', end_date='20231031', market='DCE')`, allowed exchanges DCE, SHFE, CZCE, INE, GFEX, CFFEX; select variety codes only after preserving complete raw response. Output includes per-contract prices, volume, open interest, settlement. Rolling is NOT implemented.

## Source documentation

- AKShare Bond: https://github.com/akfamily/akshare/blob/main/docs/data/bond/bond.md
- AKShare Futures: https://github.com/akfamily/akshare/blob/main/docs/data/futures/futures.md
- AKShare Stocks: https://github.com/akfamily/akshare/blob/main/docs/data/stock/stock.md
- CMGM historical code: https://github.com/idealism5114/cmgm/tree/981de35aa9a0472e6974d5a6e4f98e1c5242e465

The availability of a Python endpoint is not a warranty of source uptime, a license to bulk collect, or a complete 15-year time series.
