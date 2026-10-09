# Gold (XAU/USD) Price Movement Prediction Using Machine Learning

## Dissertation Report Draft

**Author:** [Student name]  
**Programme / Institution:** [Programme and institution]  
**Date:** 29 September 2026  
**Status:** System and methodology draft; empirical evaluation pending

> **Evidence note.** This draft describes the implementation currently present in the project repository. At the time of writing, no historical feature dataset, fitted model, test predictions, confusion matrix, classification report, or metrics JSON is present. Accordingly, this report makes no quantitative claim about predictive accuracy or superiority. Replace the marked evaluation fields only after running the training pipeline and validating its test design.

## Abstract

This project develops a full-stack research prototype for classifying the four-hour direction of the XAU/USD spot price as **Up**, **Down**, or **Neutral**. The motivation is that gold prices respond to interacting market and macroeconomic conditions—including real interest rates, the US dollar, trading-session liquidity, and price and volume dynamics—while short-horizon returns are noisy and potentially non-stationary. The proposed system combines intraday gold OHLCV data with lagged US 10-year Treasury Inflation-Protected Securities (TIPS) yield and US Dollar Index (DXY) observations, then derives a set of technical and institutional-mechanics-inspired features. These include London–New York overlap volume-weighted average price (VWAP) bands, rolling liquidity-sweep proxies, a real-yield/spot divergence measure, average true range, candle momentum, fair-value-gap proxies, and optional basis, options open-interest, Commitments of Traders (COT), and macro-release features.

The classification model is an XGBoost multiclass gradient-boosted tree pipeline. Labels are based on the simple return between a bar’s close and a close near four hours later, with a configurable neutral band. A chronological holdout is specified to represent a forward-in-time evaluation. The repository also contains a FastAPI service backed by PostgreSQL, a React dashboard using Lightweight Charts, and a query-time web-grounded macroeconomic chat assistant. Because model artifacts and held-out evaluation outputs are not currently available, the present report documents the model specification and evaluation protocol but does not assert measured predictive performance. The system is positioned as an educational quantitative decision-support prototype, not an automated trading system or a source of personalized financial advice. Its principal research limitations include proxy measurement, incomplete direct market microstructure data, possible temporal leakage around overlapping labels, execution frictions, and regime change.

**Keywords:** XAU/USD, gold, machine learning, XGBoost, real yields, VWAP, liquidity, decision support, financial time series

## 1. Introduction

### 1.1 Background and problem statement

Gold is traded globally as a store of value, a portfolio diversifier, and a hedge in some macroeconomic and financial regimes. Its US-dollar-denominated spot price is influenced by a changing combination of monetary policy expectations, inflation compensation, real yields, currency movements, risk sentiment, official-sector demand, and market-specific liquidity. The relationship between these variables is neither fixed nor necessarily contemporaneous. For example, lower real yields are often expected to reduce the opportunity cost of holding a non-interest-bearing asset, but empirical findings indicate that the observed relationship can depend on the country and economic regime; one regime-switching study of G7 gold prices reports a positive relationship in its sample and finds gold’s hedging properties vary across regimes ([Beckmann, Berger, & Czudaj, 2019](https://doi.org/10.1016/j.intfin.2018.12.014)). Thus, a single fixed-sign rule is not an adequate general model of the gold–yield relationship.

The research problem is to determine whether a supervised machine-learning classifier using multi-source market and macroeconomic features can provide useful, calibrated information about the **direction** of XAU/USD over a four-hour horizon. The task is a three-class prediction problem: Up, Down, or Neutral. It is not a price-targeting exercise, and directional classification alone cannot establish that a strategy is profitable. Transaction costs, spread, slippage, market impact, forecast calibration, and the cost of incorrect signals would have to be studied separately before any economic interpretation could be made.

### 1.2 Market efficiency and prediction framing

The efficient-market hypothesis provides a useful null framing: if publicly available information is rapidly reflected in prices, persistent out-of-sample directional predictability should be difficult to demonstrate after accounting for estimation error and implementation costs. Fama’s review describes the theoretical and empirical foundations of market-efficiency tests ([Fama, 1970](https://doi.org/10.1111/j.1540-6261.1970.tb00518.x)). This project does not assume that the market is inefficient, nor that a complex feature set necessarily improves forecasts. Instead, it treats predictability as an empirical question to be assessed on data strictly later than the training observations and against transparent baselines.

The chosen output is therefore a **quantitative decision-support signal**. It summarizes model-estimated class probabilities from a defined information set. It does not direct a user to open, close, size, or time a position. A classifier may identify statistical associations while failing to deliver a robust economic edge, particularly when the data-generating process changes or the model is exposed to different liquidity, volatility, or policy regimes.

### 1.3 Objectives and scope

The project objectives are to:

1. Construct a reproducible data and feature pipeline for intraday XAU/USD and selected macroeconomic series.
2. Operationalize selected market-structure concepts as measurable, auditable proxies rather than treat discretionary labels as directly observed facts.
3. Train a multiclass XGBoost classifier for a four-hour directional label and evaluate it using chronological out-of-sample data.
4. Expose live prices, chart history, predictions, prediction outcomes, and macroeconomic context through a web application.
5. Document data, validation, model-risk, and user-safety limitations.

The current implementation does not ingest a complete consolidated order book, broker-level OTC flow, a full COMEX options surface, or a direct FX-swap/central-bank basis feed. Those concepts are approximated only where the available data and code permit; this distinction is material to interpreting the results.

## 2. Literature Review and Conceptual Framework

### 2.1 Price-only forecasting and nonlinear tree ensembles

Traditional price-only forecasting commonly uses lagged returns, moving averages, volatility measures, and autoregressive structures. Such models are useful baselines because they are parsimonious and comparatively easy to interpret. Their limitation is that price history alone does not represent contemporaneous macroeconomic state, cross-asset relationships, or intraday liquidity conditions. Adding variables can improve the information set, but it also creates risks: data alignment errors, look-ahead bias, feature redundancy, and overfitting from repeated experimentation.

Gradient-boosted decision trees can represent nonlinear interactions and threshold effects without requiring the analyst to specify a linear response. XGBoost is a scalable tree-boosting system that uses regularization and sparsity-aware learning mechanisms ([Chen & Guestrin, 2016](https://doi.org/10.1145/2939672.2939785)). These properties motivate its use for heterogeneous tabular inputs. They do not establish that XGBoost is optimal for this instrument, horizon, or class definition; that must be tested empirically against baselines and alternative models.

### 2.2 Macroeconomic state: real yields and the US dollar

The opportunity-cost account of gold suggests a possible link between gold and real yields, while the dollar can affect the dollar price of globally traded gold and the purchasing power of non-US investors. These relationships are often discussed as directional macro drivers, but they are not deterministic. The cited regime-switching evidence cautions against treating the sign and strength of the relationship as invariant ([Beckmann et al., 2019](https://doi.org/10.1016/j.intfin.2018.12.014)). The project therefore makes the yield relationship a rolling feature rather than a fixed trading rule and includes DXY relative returns where available.

The data pipeline uses the US 10-year TIPS yield series and DXY observations at daily frequency, joined backward to intraday bars with an availability lag. This lag is intended to reduce the chance of using a daily observation before it would have been known. Nevertheless, production use requires checking each source’s publication timestamp, revision policy, holiday calendar, and timezone; a date label alone is not proof that a value was available at the joined timestamp.

### 2.3 Intraday liquidity and practitioner market-structure constructs

The project also incorporates concepts often described in practitioner literature as Smart Money Concepts (SMC), including equal highs and lows, liquidity sweeps, Change of Character (ChoCH), order blocks, and fair-value gaps. These terms are not directly observed institutional actions in the available dataset. The implementation converts selected concepts into deterministic OHLCV rules: a rolling prior high or low is treated as a reference liquidity level; a bar that pierces the level and closes back through it is treated as a sweep; a high relative-volume threshold conditions the signal. A bullish reclaim after a downside sweep is labeled a ChoCH/reclaim proxy. A three-candle price-gap condition is used as an FVG proxy.

This operationalization improves reproducibility over visual or discretionary labeling, but it does not prove that institutional orders caused the observed pattern. Equal-high/low clusters based on a tolerance and rolling extrema are simplified approximations; the feature set does not identify resting stop orders, dealer inventory, or participant identity. The terms “order block” and “smart money” should consequently be interpreted as conceptual motivation rather than measured ground truth.

### 2.4 Cross-asset divergence and basis features

The yield-divergence feature estimates a rolling association between gold returns and changes in the TIPS yield. It computes a rolling covariance-to-variance slope, forms a yield-implied return, and compares this expected response with the realized gold return. A positive standardized divergence during a falling-yield observation is represented as a “gold lag” feature. This is a statistical residual proxy; it is neither a risk-free arbitrage nor evidence that a price correction must follow.

An EFP-style feature is computed from futures minus spot less a simplified financing carry term when futures and a risk-free rate are supplied. This is a coarse carry-adjusted basis proxy. It is not a complete FX-swap or central-bank basis-arbitrage measure: it omits instrument-specific financing, lease rates, collateral, delivery, maturity matching, transaction costs, and potentially time-consistent futures/spot observations. Accordingly, claims about FX swaps or central-bank basis must remain outside the empirical conclusion until appropriate data are integrated.

### 2.5 Session-overlap VWAP and mean reversion

Volume-weighted average price is commonly used as an execution and intraday reference measure. This project calculates cumulative, session-reset VWAP and volume-weighted standard-deviation bands for the 12:00–16:00 UTC London–New York overlap. It constructs deviation-in-sigma features and a flag when price is at or above the +2σ band. The flag is only a feature; it does not itself establish a mean-reversion effect or automatically issue a sell recommendation. The current code does not verify the existence of a contemporaneous macro catalyst before setting that feature. “Without a catalyst” should therefore not be claimed as implemented unless a properly timestamped catalyst calendar and exclusion rule are added.

An important data limitation is that OTC spot gold volume is not a single consolidated exchange volume series. If the supplied `volume` field is absent, broker-specific, or synthetic, the VWAP and relative-volume features may not represent market-wide activity. The findings must identify the volume source and its construction.

### 2.6 Multi-layered feature design

The proposed framework differs from a price-only design by combining four layers:

| Layer | Implemented inputs or proxies | Interpretation and boundary |
|---|---|---|
| Price and volatility | ATR(14), candle body scaled by ATR, bullish/bearish momentum, three-candle FVG flags and sizes | OHLC-derived state; does not identify participant intent |
| Intraday liquidity | Prior rolling high/low, relative volume, sweep/reclaim flags, equal-level cluster flags, overlap VWAP and sigma bands | Rule-based proxies dependent on data quality and chosen windows |
| Macro/cross-asset | Lagged TIPS yield changes, rolling gold–yield beta/divergence, lagged DXY returns, relative return | Association and lag features, not causal estimates or arbitrage proofs |
| Optional institutional context | Futures carry-adjusted basis, options strike/open-interest proximity, COT net-long changes, macro-release-window buying proxy | Only available when external files are supplied; current proxies omit full market structure |

The architecture is multi-layered in its feature design, but feature availability varies by dataset. A report of the trained model must state exactly which columns survived into the fitted pipeline and which optional inputs were present. Merely including code paths for optional data is not evidence that these features contributed to a trained model.

## 3. Data and Methodology

### 3.1 Data inputs and preprocessing

The feature-engineering interface accepts timestamped intraday gold OHLCV data and optional timestamped files for TIPS yield, DXY, futures, COT positioning, options strike/open interest, and macro-release times. Timestamps are parsed as UTC, rows are sorted, and duplicate timestamps are reduced to the last record. Daily TIPS and DXY observations are shifted by one day and joined backward to the intraday timeline. The intended primary input is an intraday series because the session-overlap VWAP cannot be meaningfully reconstructed from daily bars.

Before empirical use, the data appendix should report provider, symbol definition, bid/ask or mid-price convention, sampling interval, timezone, missing-data policy, market closures, sample dates, and row counts. It should also distinguish spot from futures prices and describe the origin and meaning of any volume field. No source dataset is currently available in the repository, so these details remain to be supplied from the actual data run.

### 3.2 Feature construction

#### 3.2.1 Session-overlap VWAP

For each UTC date, the implementation restricts observations to 12:00 inclusive through 16:00 exclusive UTC. It uses typical price (P_t=(H_t+L_t+C_t)/3), weights each observation by non-negative volume (V_t), and calculates cumulative session VWAP:

\[
VWAP_t=\frac{\sum_{i\le t}P_iV_i}{\sum_{i\le t}V_i}.
\]

The weighted variance is calculated from cumulative weighted first and second moments. Bands are defined as (VWAP_t\pm k\sigma_t) for (k\in\{1,2,3\}). The model receives overlap membership, VWAP, sigma, deviation in sigma units, and a +2σ condition flag. These quantities are reset by date and are missing outside the target session.

#### 3.2.2 Liquidity-sweep proxies

The prior rolling high and low are computed over a configurable lookback and shifted by one bar so the current candle does not define its own reference. Relative volume is current volume divided by the prior rolling mean volume. A high sweep is flagged where the current high exceeds the prior rolling high but the close finishes below it; a low sweep is the mirror condition. The sweep also requires relative volume of at least 1.5. The low-sweep flag is reused as a bullish reclaim/ChoCH proxy, and the high-sweep flag as a bearish rejection proxy. Equal-level clusters use a price tolerance of 0.05% around the prior rolling extremum.

These are reproducible rule definitions rather than labels from an order book. The rolling-window size and relative-volume threshold are model assumptions and should be subjected to sensitivity analysis.

#### 3.2.3 Real-yield divergence and DXY

The pipeline computes the gold simple return (r_t), change in the TIPS yield \(\Delta y_t\), and a rolling slope:

\[
\hat\beta_t=\frac{\operatorname{Cov}(r,\Delta y)}{\operatorname{Var}(\Delta y)}.
\]

The rolling yield-implied return is \(\hat\beta_t\Delta y_t\). Divergence is defined as that implied return minus the observed return, then standardized using a rolling z-score. A flag is emitted when the yield falls and the divergence z-score is at least 1.0. DXY returns and gold-minus-DXY relative returns are also computed where the joined DXY series exists. The rolling slope is an adaptive descriptive association; it should not be interpreted as a causal structural coefficient.

#### 3.2.4 Additional and optional features

ATR(14) is calculated from true range. Candle body divided by ATR produces normalized momentum; thresholded bullish and bearish flags use ±0.5 ATR. A bullish FVG proxy is a current low above the high two bars earlier; the bearish case is defined symmetrically. Optional futures data produce an EFP/carry proxy and rolling basis z-score. Optional options data produce strike distance and open-interest-weighted proximity; a separate round-number squeeze proxy uses distance from a configurable $100 increment and unusually large price acceleration. Optional COT data produce net-long change and z-score. Optional macro-release timestamps flag a ±30-minute window and combine it with an up candle and relative volume ≥1.5 to make an institutional-buying proxy.

The current feature code does not ingest actual order-block labels or dealer gamma/delta exposures. It does not perform a catalyst exclusion for the +2σ mean-reversion flag. These items should be classified as future data/instrumentation work, not completed measurements.

### 3.3 Target definition

The target uses the future close near (t+4\) hours and its simple return from the current close:

\[
R_{t,4h}=\frac{C_{t+4h}}{C_t}-1.
\]

With the default neutral threshold \(\tau=0.001\), observations are labeled Down if (R_{t,4h}<-\tau), Up if (R_{t,4h}>\tau), and Neutral otherwise. This threshold is 0.10% in absolute return. The implementation aligns to the nearest available bar within a configurable five-minute tolerance. Because nearest-neighbor alignment can select a bar slightly before the nominal target timestamp as well as after it, this should be changed or carefully audited for the final experiment; a strictly forward-only alignment or an explicitly defined bar index is preferable.

### 3.4 Model and pipeline

The training script defines a scikit-learn pipeline consisting of median imputation with missingness indicators, `StandardScaler`, and `XGBClassifier`. The default XGBoost configuration uses `multi:softprob`, multiclass log loss, 500 estimators, maximum depth 5, learning rate 0.03, row subsampling 0.85, column subsampling 0.85, L2 regularization 1.0, random seed 42, all available CPU threads, and histogram tree construction. The pipeline stores the training feature order and class mapping for inference.

These values are code defaults, not the result of a reported hyperparameter search. Scaling is included for a consistent preprocessing pipeline, although tree split decisions generally do not require standardized numeric inputs. Hyperparameter selection should be done using only training-period data and a time-ordered validation design. The present script performs a chronological train/test split, but it does not implement a separate validation window, nested temporal tuning, purging, or an embargo for overlapping four-hour labels.

The pipeline removes features with no observed values in the complete labeled sample before the split. This uses information about missingness in the test period, albeit not test target values. For a strict evaluation, feature selection and preprocessing decisions should be learned only within the training folds. Furthermore, the 4-hour labels can overlap heavily when bars are more frequent than four hours; samples around the train/test boundary can therefore share portions of their forward outcome interval. A gap/purge at the split boundary and a walk-forward evaluation are recommended before treating the holdout as a definitive estimate.

### 3.5 Evaluation protocol and metrics

The configured protocol holds out the most recent 20% of labeled samples and trains on the earlier 80%. Confusion-matrix rows correspond to actual classes and columns to predicted classes, ordered Down, Neutral, Up. The training script exports the confusion matrix, per-class precision, recall, F1-score, support, aggregate accuracy, test predictions, and a JSON summary. Accuracy is not sufficient where class frequencies are imbalanced; the final report should include per-class precision/recall/F1, macro-F1, balanced accuracy, class support, and a simple baseline such as majority-class prediction. Given probabilistic outputs, calibration and log loss or Brier score are also useful. A strategy-level backtest would need to incorporate spread, fees, slippage, and a clear position/exit rule, and is outside the current classifier evaluation.

## 4. Model Development and Evaluation

### 4.1 Available empirical results

No training artifacts are present in the project workspace as of the report date. Therefore, actual scores cannot be responsibly reported yet.

| Evaluation item | Current status | Required evidence before final submission |
|---|---|---|
| Sample period and usable row count | Not available | Data manifest and labeled row counts |
| Train/test sizes and class balance | Not available | `metrics.json` plus class support in report |
| Confusion matrix (Down / Neutral / Up) | Not available | `confusion_matrix.csv` from a completed run |
| Per-class precision, recall, and F1 | Not available | `classification_report.csv` |
| Overall accuracy and macro-F1 | Not available | Holdout metrics, with exact split dates |
| Majority-class baseline | Not implemented in training script | Same split evaluated with a training majority classifier |
| Calibration / probabilistic score | Not available | Reliability assessment and log loss or Brier score |
| Economic performance after costs | Not evaluated | Separate, pre-specified execution-aware backtest |

The appropriate dissertation statement at this stage is that the model has been specified and the evaluation pipeline has been implemented, while empirical predictive performance remains unverified. The output labels “UP”, “DOWN”, and “NEUTRAL” must not be presented as a validated forecasting capability until the data run, baseline comparison, and leakage review are complete.

### 4.2 Confusion matrix interpretation

When available, the matrix should be reported numerically with actual classes as rows and predicted classes as columns. Diagonal cells are correct classifications. Off-diagonal cells expose direction errors—for example, a true Down observation predicted Up—and Neutral-class behavior. Precision for class (k) is (TP_k/(TP_k+FP_k)), while recall is (TP_k/(TP_k+FN_k)). Precision measures the fraction of predictions for a class that were correct; recall measures the fraction of actual class observations recovered. In a three-class setting, both should be reported per class rather than reduced to a single undifferentiated score.

### 4.3 Baseline comparison

The current training script does not compute a baseline. A minimum benchmark is a majority-class classifier fitted on the training partition and evaluated on the identical test dates. A stronger benchmark is a price-only feature model using a small, predeclared set of lagged returns and volatility inputs. Optional additional baselines include regularized multinomial logistic regression and a simple persistence/neutral rule. The proposed multi-layer feature model should be compared on identical observations, labels, split boundaries, and metrics. Only then can any incremental value of macro or liquidity features be evaluated. Feature ablation—price-only, price plus macro, and full available feature set—would help determine whether the added complexity contributes out of sample.

### 4.4 Recommended final reporting table

Populate this table from actual artifacts; do not fill cells by visual estimate.

| Model | Accuracy | Balanced accuracy | Macro-F1 | Down P/R/F1 | Neutral P/R/F1 | Up P/R/F1 |
|---|---:|---:|---:|---:|---:|---:|
| Majority-class baseline | Pending | Pending | Pending | Pending | Pending | Pending |
| Price-only baseline | Pending | Pending | Pending | Pending | Pending | Pending |
| Full available feature model | Pending | Pending | Pending | Pending | Pending | Pending |

## 5. System Architecture

### 5.1 Data and prediction pipeline

The research workflow begins with external market and macroeconomic data. Historical gold OHLCV, TIPS yields, and DXY values are transformed by the feature-engineering module and passed to the model-training script. The fitted preprocessing/model pipeline is serialized as a joblib artifact. Inference uses FastAPI services to load features and the model, return a directional class and confidence, and store prediction records in PostgreSQL. SQLAlchemy models represent price history and prediction logs. APScheduler is configured to refresh live-price data every minute, generate predictions every four hours, and periodically score predictions against later prices.

The current implementation should be understood as an MVP integration, not yet a fully reconciled research-to-production feed. In particular, the historical fetcher and intraday feature pipeline have different data granularity requirements, and the live-price polling path can store quote snapshots rather than exchange-quality OHLCV candles. Chart-history integrity and model-feature parity must be verified before deployment claims are made.

### 5.2 Backend API

The FastAPI application exposes `/price/live` for a current quote, spread, active session and source/staleness metadata; `/history` for stored candle records; `/predict` for a model inference and persisted prediction; `/accuracy-log` for scored predictions and a running accuracy summary; and `/chat` for current macro-news retrieval and grounded response generation. CORS configuration admits configured local and deployed frontend origins. PostgreSQL persistence is accessed through SQLAlchemy sessions. A production deployment should use managed migrations rather than relying only on table creation at application startup, and should protect operational endpoints, validate data freshness, and monitor provider failures.

### 5.3 Frontend dashboard

The Vite/React dashboard organizes a dark quantitative interface around a Lightweight Charts candlestick view, a live prediction panel, an accuracy tracker, session clocks, and a floating assistant. The session clocks use Luxon to present Sydney, Tokyo, London, and New York time zones and mark the UTC London–New York overlap. The UI communicates model output and data history; it does not execute trades.

### 5.4 Retrieval-augmented macro assistant

The chat service performs query-time retrieval using Tavily or Serper news search and supplies recent snippets to an OpenAI or Anthropic language-model API. The system prompt instructs the model to ground current-event claims in retrieved evidence, cite source indices, identify uncertainty, and refuse direct personalized trade instructions. Search snippets are treated as untrusted evidence. This implementation is web-grounded retrieval at request time; it does not include a persistent document corpus or vector database. Retrieved snippets can be incomplete, stale, duplicated, or misleading, and citations should be checked against the linked original source.

## 6. Risk Mitigation, Limitations, and Ethical Considerations

### 6.1 Model and data risk

Financial time series are non-stationary: volatility, monetary regimes, market participants, liquidity, and relationships among gold, yields, and the dollar can change. Rolling features may adapt to recent observations but cannot guarantee future stability. Model performance should be monitored over time, tested across distinct regimes, and recalibrated only using a documented process. Confidence scores from `predict_proba()` are model class probabilities; absent calibration analysis they must not be interpreted as empirically calibrated probabilities of success.

Potential leakage arises from target alignment, overlapping four-hour label horizons, test-aware removal of wholly missing features, revised macro series, and timestamps that do not reflect actual publication availability. A robust final experiment should use point-in-time data, forward-only labels, a boundary purge or gap at least as long as the forecast horizon, rolling-origin or walk-forward evaluation, and a final untouched test period. All transformations and feature selection should be fitted using training data only. Hyperparameter selection must not use the final test set.

Data-source limitations include heterogeneous OTC spot quotes, uncertain volume meaning, missing optional inputs, as-of joins with publication lags, rate limits, stale fallback quotes, and imperfect alignment between spot and COMEX futures. Options open interest and COT reports have their own publication delays and represent different participant universes. The system should store source, observation time, retrieval time, and staleness metadata and fail visibly when essential inputs are missing.

### 6.2 Economic and operational risk

Classification performance is not investment performance. The current pipeline does not include an execution policy, position sizing, leverage, transaction costs, bid–ask spread in historical labels, slippage, market impact, financing, or drawdown controls. A directional signal can be correct yet economically unusable, or wrong in a way that creates outsized loss. No result should be described as an arbitrage opportunity unless the relevant instruments, funding, execution, and costs are measured and the trade is shown to be executable.

### 6.3 Concept validity and interpretability

Features named after SMC or institutional flows are rule-based proxies, not direct measurements of hidden liquidity, dealer hedging, or institutional intent. Their labels may embed researcher discretion through rolling windows, thresholds, and round-number definitions. This creates specification risk and opportunities for data mining. Sensitivity tests and ablation studies should be pre-specified and documented. Feature importance should not be interpreted causally; correlated predictors and tree split behavior can distort simple importance rankings.

### 6.4 User safety and financial disclaimer

The dashboard and assistant are designed for educational research and decision support. They must not be represented as a broker, investment adviser, guaranteed forecasting service, or automated execution system. The interface should prominently state: **“Decision Support System — Not Financial Advice.”** The chat assistant must not give personalized buy/sell/hold, entry/exit, position-sizing, or timing instructions. Outputs may be wrong, delayed, incomplete, or unavailable; users remain responsible for independent verification and any decisions they make. These safeguards reduce communication risk but do not remove model, data, or financial risk.

## 7. Conclusion and Future Work

This project specifies an end-to-end prototype for four-hour XAU/USD direction classification, combining price-action features with lagged macroeconomic information and rule-based liquidity/session proxies. It pairs the modeling workflow with a FastAPI/PostgreSQL backend, React visualization dashboard, and a web-grounded assistant intended to explain macroeconomic context without making personalized trading recommendations. The architecture is suitable as a research MVP, while the current evidence does not yet establish forecasting performance.

The next research priorities are to obtain and document synchronized intraday data; verify point-in-time feature availability and model/data parity; correct target alignment to be strictly forward-looking; add a temporal gap or purge; run baselines and ablations; report class-wise metrics and calibration; evaluate robustness across time regimes; and conduct a separate cost-aware backtest only after predefining a decision policy. Direct options, futures, and financing inputs would be required before making claims about dealer gamma, FX swaps, or central-bank basis. The dissertation’s central conclusion should ultimately be determined by those experiments, including the possibility that the multi-layer model does not outperform simple baselines.

## References

Beckmann, J., Berger, T., & Czudaj, R. (2019). Do gold prices respond to real interest rates? Evidence from the Bayesian Markov Switching VECM model. *Journal of International Financial Markets, Institutions & Money, 60*, 134–148. [https://doi.org/10.1016/j.intfin.2018.12.014](https://doi.org/10.1016/j.intfin.2018.12.014)

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). [https://doi.org/10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)

Fama, E. F. (1970). Efficient capital markets: A review of theory and empirical work. *The Journal of Finance, 25*(2), 383–417. [https://doi.org/10.1111/j.1540-6261.1970.tb00518.x](https://doi.org/10.1111/j.1540-6261.1970.tb00518.x)

## Appendix A. Reproducibility checklist for the final empirical chapter

- Record exact data provider, instrument identifiers, sampling interval, timezone, coverage dates, and retrieval date.
- Archive the raw data or immutable checksums and document revisions, missing values, duplicate handling, and source latency.
- Record the exact feature columns used by the fitted artifact, not just all columns supported by the feature script.
- Fix the forward label alignment and report its tolerance; ensure it cannot select a price from before the target horizon.
- Use chronological validation with an explicit gap/purge for overlapping target intervals; preserve a final untouched test period.
- Fit imputation, feature selection, scaling, and tuning only within training data.
- Report confusion matrix, per-class precision/recall/F1, macro-F1, balanced accuracy, baseline metrics, class support, and test dates.
- Assess probability calibration and, if claiming economic utility, separately specify and test a cost-aware execution policy.
- Run ablation/sensitivity analyses for optional macro and liquidity features and disclose unavailable data sources.
- Include the user-facing disclaimer and describe model/data freshness, failure modes, and non-stationarity limitations.
