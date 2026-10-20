# Local verification report

Environment verified: Python 3.13 / pandas available / pytest available; AKShare NOT installed in the sandbox. Outbound package-index connectivity was unavailable, so **no real remote data download was run** here. This does not establish provider endpoint uptime.

Offline tests executed:

```
cd TriWeave
PYTHONPATH=src python -m pytest -q
22 passed
```

Verified without network:

- Exactly 18 historical curve tenors + 7 bond wealth indices = 25 bond series.
- Exactly 24 commodity targets + 14 auxiliary varieties = 38 commodity futures varieties; 4 separately marked government-bond futures varieties.
- Stable fifteen-year date request boundaries (2011-10-01 to 2026-09-30), 19 curve chunks, 7 index queries, 38 continuous queries, and 846 exchange-month jobs over six markets as of this source configuration.
- Missing real 263 ticker manifest fails explicitly. Optional 248 legacy + 15 Phase A-Fix reported tickers reconstruction must be verified against final Clean V1 manifest.
- Unit-tested Eastmoney stock, ChinaBond yield/index, Sina continuous and exchange listed-contract **fake DataFrame** schemas.
- Unavailable values remain missing; no forward-to-past `bfill` and no fake pre-listing prices.
- Atomic raw/standardized gzip CSV, metadata SHA256, reuse of complete snapshots, corruption detection, JSONL error logs.
- Coverage audit produces no model labels, prediction metrics or futures continuous trading P&L.

Not verified here: installation from online PyPI, live AKShare response schemas, rate limits, licensed historical source permissions, asset coverage over 15 years, actual contract-roll policies, historical point-in-time announcement availability, 263 real source tickers. Run the documented one-job live smoke tests before starting bulk collection.
