# TRIWEAVE_23_AFTER_CLOSE_RESEARCH_V1

23个目标按固定顺序：LU SC CF PG HC J JM FU C MA SR OI RM RB M CU AL ZN SN NI A I Y。它们同时为商品输入；263股票及18债券收益率+7财富指数身份和顺序不变，共311节点，无额外辅助期货。

新配置configs/futures_targets_23_research_v1.csv；原25配置保留。AU/AG因已记录的重大历史会话日期风险及未知安全边界移出本研究版本，不替换其他品种，不依据VAL MAE选择。原商品池曾使用VAL覆盖进行设计；此次修改发生在原协议数据质量审计之后，不能称为事前完全独立设计，也不能将23结果当作原25完全同任务比较。

预测定义恢复北京时间20:00的P(t+h)/P(t)-1，h=1/5/10/20参考session，decimal_return，锚点就是t。没有FWD3、没有未来锚点、没有统一lag2。股票当日18:00、商品当日19:00为明确的AVAILABLE_AT_ASSUMED假设；债券严格下一参考日08:00/08:30，最多沿用5自然日的已知观测，无fallback跨过无效最新记录。历史PIT与连续换月均未验证。

本版本整观测隔离SN2015-03-27；额外预防性隔离FU2014-12-31及2016-04-11两条已存在次日全字段指纹匹配的薄量记录。FU为具体时间不确定性防护，并非声称所有薄量或相同价格都错位。无替价、无涨幅阈值删点；隔离须先于全部输入特征、滚动、标签和TRAIN统计。

26维经济类型schema及55组TRAIN-only Z-score沿用审计实现；新参数从raw重新标准化后独立fit，未复用25或Bridge Scaler。标签h+1报价质量窗逐目标判断，origin与endpoint必须在同split。TRAIN2011-10-01–2022-12-31，VAL2023-01-01–2024-12-31，TEST2025-01-01–2026-09-30不读取数值、不生成标签。

三协议独立：原TRIWEAVE_25_AFTER_CLOSE_V1_EXPLORATORY保持原RESEARCH_HOLD；TRIWEAVE_23_DIAGNOSTIC_V0及其FWD3标签只读保留；本协议为新的23商品研究任务。数据技术有效、探索性科研准入、确认性研究准入分别审核，不凭软件通过认证真实PIT。本轮不训练、不计算MAE、不新下载、不提交或推送。

执行冻结参数见experiments/phase_e_23_research/frozen_protocol.json；完整质量报告在同目录，数据目标路径data/derived/triweave_23_after_close_research_v1/。
