# TRIWEAVE_RAW15_V1: Data safety and provenance rules

1. All timestamps represent **source observation dates**, not confirmed data publication times. Preserve upstream source URL, API parameters, API version, capture time and SHA256 with each file.
2. Research bounds: `2011-10-01` to `2026-09-30`. Real first observations decide actual availability. No synthetic missing years or prices.
3. Input stock identities are imported, checked for exactly 263 members and order, and hash-frozen. Do not select assets based on downstream validation or confirmation data completeness.
4. Bonds include yield **levels** in percent and wealth-index **levels** in index points; feature-building will be type-aware (yield changes in basis points vs price/wealth-index returns). No price-based 21-feature replication for yields.
5. Exchange-listed futures contracts and vendor continuous quotes stay separate. Never interpret a vendor continuous series close-to-close pct-change around a contract roll as executable future return without a verified roll/proceeds/cost protocol.
6. Never `bfill` or treat an invalid price/late-listed asset as a genuine historical observation. Do not `nan_to_num` missing labels into 0.
7. Never silently reassign `WH` strong-wheat futures to the old wheat target: explicitly reconcile commodity code and trading history.
8. Use point-in-time *known* past prices, valid source-time masks and explicit missingness; do not discard an entire date merely because one asset's data are absent.
9. This raw collection itself creates no training labels, no normalization, no confirmation-period prediction/metric, and no automated model runs.
10. Future model protocol should separately freeze splits, vintage availability, feature causality, corporate actions and the longest-horizon label embargo. The vintage agreement must be verified before claiming historic tradeability.
11. Missing APIs, source changes, missing contract dates and limits are reported `FAILED`, `NOT_COLLECTED_OR_NO_DATA`, or `NO_SELECTED_VARIETIES`. Never interpret nonretrieval as no exchange trading.
12. `run_id` is append-only by convention. Existing raw files will be skipped only after hash verification. To refresh, use a new run_id.
