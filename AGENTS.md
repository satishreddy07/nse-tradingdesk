# NSE Trading Desk — Mandatory Engineering Rules

## 1. Core Principle

Data integrity, financial calculation accuracy, and preservation of existing functionality take priority over implementation speed.

Never invent, assume, approximate, or silently substitute a business rule, trading parameter, market price, financial value, multiplier, model score, or performance metric.

When the correct value cannot be established from the repository, an authoritative data source, or an explicitly approved specification, stop and report the uncertainty.

### Mandatory Compliance with Permanent Anti-Fabrication Policy

All future code modifications, additions, audits, and agent workflows touching:
1. **Financial Market Data** (live quotes, historical OHLCV, ticks, order-book depth, option chains, Greeks),
2. **Quantitative Analytics** (decay curves, volume profiles, GEX, IV ranks, technical indicators),
3. **Machine Learning & AI** (feature pipelines, empirical training sets, forecast calibration, model registries),
4. **Risk Management** (position sizing, risk limits, stop losses, drawdowns, capital budgets), and
5. **Trading & Order Execution** (execution daemons, broker orders, leg adjustments, state transitions)

MUST strictly comply with the comprehensive rules established in [docs/ANTI_FABRICATION_POLICY.md](docs/ANTI_FABRICATION_POLICY.md).

No pull request, patch, or agent turn is complete without proving adherence to the core policy domains defined in [docs/ANTI_FABRICATION_POLICY.md](docs/ANTI_FABRICATION_POLICY.md).

## 2. Prohibited Behaviors

NEVER:

- Introduce arbitrary numeric constants into trading, financial, prediction, risk-management, or market-data logic.
- Invent ATR multipliers, stop-loss percentages, target percentages, probability thresholds, confidence scores, risk limits, lot sizes, strike-selection rules, or position-sizing parameters.
- Replace unavailable live market data with sample prices, fabricated OHLCV candles, stale values presented as current, or randomly generated values.
- Return mock, placeholder, simulated, or hardcoded financial results from production API endpoints.
- Fabricate model predictions, backtest results, trading signals, execution confirmations, or ML performance statistics.
- Change an existing strategy parameter without explicit authorization and evidence supporting the change.
- Introduce fallback values that make failed data fetching, failed calculations, or failed model inference appear successful.
- Hide exceptions and return plausible-looking default values.
- Create duplicate configuration values that can drift away from the authoritative setting.
- Change a failing test merely to make an incorrect implementation pass.
- Remove existing functionality to avoid fixing a difficult defect.

## 3. Establish the Source of Truth

Before modifying code:

1. Inspect the existing implementation and configuration.
2. Trace the complete data flow from its original source to its final consumer.
3. Locate existing strategy specifications, configuration variables, environment variables, database records, broker responses, and relevant tests.
4. Identify the authoritative source for each important value.
5. Reuse existing implementations and approved parameters whenever possible.
6. If multiple conflicting sources exist, report the conflict instead of silently choosing one.

Do not assume a value is correct merely because it already exists in the codebase. Verify it against the approved strategy or authoritative specification.

## 4. Numeric Constants and Multipliers

Every numeric literal used in business logic must be classified as one of:

- Mathematical or technical constant.
- Protocol, schema, or API requirement.
- Approved strategy parameter.
- Operational setting.
- Test fixture or explicitly isolated simulation.
- Unverified or unauthorized business value.

Legitimate mathematical constants and protocol requirements are permitted.

Approved strategy parameters must have a clear source of truth, suitable validation, and traceable usage.

For example, if an approved configuration specifies an ATR stop multiplier of 2.0, do not introduce 1.5 or 2.5 elsewhere.

Do not blindly replace every numeric literal with a configuration variable. The objective is correct and traceable behavior, not unnecessary abstraction.

If a required business parameter has no authoritative value, report it and request clarification.

## 5. Market Data Integrity

- Obtain live prices and market data from the configured, authorized data provider.
- Respect broker connection state and the application's explicitly configured delayed-data policy.
- Distinguish live, delayed, stale, historical, simulated, and unavailable data.
- Preserve the actual timestamp, source, and relevant instrument identity.
- Never describe delayed or stale data as live.
- Never fabricate prices, OHLCV records, option-chain values, volume, open interest, or implied volatility.
- If fetching fails, return an explicit error or unavailable status.
- Validate missing values, timestamps, data types, ranges, and instrument mappings before using data in trading decisions.
- Do not silently substitute another data source unless the fallback is explicitly authorized and its data status is preserved.

## 6. Trading and Financial Calculations

- Preserve the existing approved trading strategy.
- Use the authoritative configuration and existing calculation implementations.
- Validate units, percentages, decimal fractions, quantities, lot sizes, tick sizes, and rounding rules.
- Keep option-leg calculations, re-entry rules, stop-loss rules, and risk limits consistent with their approved specifications.
- Never report an order as executed unless the execution state is supported by the broker or authoritative execution record.
- Never silently convert a calculation failure into a zero profit, zero risk, successful trade, or valid signal.
- Verify calculations against independently calculated test cases.

## 7. Machine Learning and AI

- Distinguish actual model outputs from placeholders and unavailable results.
- Never fabricate training records, labels, prediction outcomes, evaluation metrics, or model-performance statistics.
- Track model identity, training-data provenance, evaluation period, sample size, and relevant model version where applicable.
- Do not claim a model has trained, improved, or passed evaluation without evidence.
- Prevent unavailable model inference from silently becoming a fabricated trading recommendation.
- Treat AI-generated recommendations as untrusted inputs that must pass deterministic application-level validation.

## 8. API and UI Integrity

- Do not use fake financial data to populate production dashboards.
- Show unavailable, stale, or incomplete data explicitly.
- Preserve existing API contracts unless a change is necessary and authorized.
- Do not display a successful status when the underlying operation failed.
- Clearly separate demo data from production data.
- Ensure every displayed financial metric has a traceable source or reproducible calculation.

## 9. Required Workflow Before Every Change

Before editing:

1. State the observed problem.
2. Identify the relevant files and execution paths.
3. Establish the expected behavior from existing specifications, tests, and [docs/ANTI_FABRICATION_POLICY.md](docs/ANTI_FABRICATION_POLICY.md).
4. Identify any uncertainty or conflicting requirements.
5. Propose the smallest safe change.

During implementation:

1. Make minimal, targeted changes.
2. Reuse existing services and configuration.
3. Preserve unrelated behavior.
4. Do not invent missing business requirements.
5. Do not weaken security, validation, or tests to make the change pass.

After implementation:

1. Run relevant unit tests.
2. Run integration tests where available.
3. Run regression tests for affected trading and data flows.
4. Verify boundary cases and failure handling.
5. Inspect the Git diff for unintended changes and verify strict compliance with [docs/ANTI_FABRICATION_POLICY.md](docs/ANTI_FABRICATION_POLICY.md).
6. Report exactly which tests passed, failed, or could not be run.

Never claim a test passed unless it was actually executed successfully.

## 10. Audit and Evidence Requirements

When auditing the repository, classify each finding as:

- Confirmed defect: direct evidence proves incorrect behavior.
- Suspected defect: evidence suggests a problem but is insufficient for confirmation.
- Approved constant: legitimate and supported by a specification.
- Test-only fixture: isolated from production execution.
- Unused code: not currently reachable from the identified execution paths.
- Unverified business value: no authoritative source has been established.

For each confirmed defect, provide the file path, line number, affected execution path, evidence, impact, and recommended correction.

Do not claim that a repository is free of fabricated values based solely on a text search.

## 11. Change Management

- Never overwrite user data, the production SQLite database, or backups during an audit.
- Never modify secrets or expose credentials.
- Never make unrelated changes during a targeted fix.
- Preserve existing broker integrations, schedulers, database behavior, and application functionality.
- Work on the designated Git branch.
- Do not deploy changes automatically.
- Require explicit authorization before changing approved trading rules or production configuration.

## 12. Definition of Done

A change is complete only when:

- Its behavior is supported by an authoritative requirement.
- Important numeric parameters have traceable sources.
- Invalid or unavailable inputs fail safely.
- Relevant tests pass, or remaining failures are explicitly documented.
- Existing functionality is preserved.
- The Git diff contains only justified changes.
- The final report accurately describes what was verified and what remains uncertain.
- It strictly adheres to all domains of [docs/ANTI_FABRICATION_POLICY.md](docs/ANTI_FABRICATION_POLICY.md).

When evidence is insufficient, do not guess. Report the missing evidence and stop before making an unsupported change.
