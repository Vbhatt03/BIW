# Multi-Station, Part-Wise Quality Analysis

## 1. Understand each measurement's distribution

For every measurement column, calculate:

- count, mean, median, standard deviation, minimum, maximum, IQR, skewness, and kurtosis;
- normality-test results using Shapiro–Wilk (where sample size permits) and D’Agostino–Pearson as a check; and
- a histogram with KDE overlay and a Q–Q plot.

The plots matter as much as the tests. With a large sample, normality tests can flag tiny departures that are not practically important; with a small sample, they may miss meaningful ones.

Classify each column as normal, roughly normal, or non-normal, and retain the test statistic and p-value. For non-normal columns, describe the pattern where possible: skew, heavy tails, a hard cutoff at a specification limit, or two peaks. A bimodal distribution can indicate mixed populations such as different tool heads, shifts, or product variants, and should be investigated before treating it as ordinary noise.

If a measurement is clearly non-normal, decide whether a transform such as log or Box–Cox is appropriate before applying methods that assume normality, including Grubbs’ test and three-sigma control limits.

**Output:** a distribution summary for all measurement columns and reproducible plots for each one.

---

## 2. Find unusual measurements

Use two complementary views: one for individual measurements and one for unusual combinations of measurements.

### 2.1 Per-column outliers

Assess every part column independently.

- For approximately normal data, use Z-scores or the more robust modified Z-score (median and MAD). Typical starting thresholds are `|z| > 3` and `|modified z| > 3.5`.
- For non-normal data, use IQR fences (`Q1 - 1.5 × IQR`, `Q3 + 1.5 × IQR`) or percentile cutoffs rather than sigma rules.
- As an optional robust comparison, run Isolation Forest or Local Outlier Factor on each single column.

Store an anomaly flag and score for every unit/column pair. That makes it easy to see which measurements were unusual for a unit without rerunning the analysis.

### 2.2 Unit-level, cross-part anomalies

A unit can be within range on every individual measurement but still have an unusual combination of values. Model this separately.

- Fit Isolation Forest or LOF using all measurement columns after scaling them with `StandardScaler`.
- Use PCA reconstruction error as a second, more interpretable signal. Large error suggests that the unit does not fit the usual relationship among measurements.

**Output:** a unit-by-column anomaly table, a unit-level composite anomaly score, and a short summary of anomaly rates by column and station.

---

## 3. Forecast whether anomalies are becoming more likely

For each part column, forecast either the raw reading or the likelihood of an anomaly. Which approach is useful depends on the timestamps, the length of the history, and how common anomalies are.

1. **Trend forecasting on the raw measurement.** Treat each `(timestamp, value)` sequence as a time series. If tool wear or calibration drift precedes failures, rolling means and control-chart rules may be enough and are easy to interpret. If the history is long and regularly sampled, ARIMA or Prophet may also be worth testing.
2. **Prediction of the anomaly flag.** Frame the question as: “Will this part be anomalous in the next N units or hours?” Use lagged features such as recent mean, variation, slope, time since the last anomaly, and recent readings from other stations. Gradient-boosted trees (XGBoost or LightGBM) are a sensible baseline for this type of tabular data and can also indicate which features are driving predictions.

First verify that the data supports forecasting. Sparse or short time series may not be usable, and rare anomalies create a class-imbalance problem. In those cases, report whether anomaly rates are rising rather than claiming a reliable forward prediction.

**Output:** for each part, either a validated trend/prediction model with a clear forecast window and appropriate metrics (MAE/RMSE for readings; precision/recall for anomaly prediction), or a brief explanation of why forecasting is not supported.

---

## 4. Root-cause analysis: framework only until labels exist

Without real root-cause or failure-mode labels, this work should be presented as a demonstration framework, not a confirmed causal analysis.

1. Draft a small, clearly hypothetical causal graph for anomaly-prone measurements. For example: `Tool_Wear → Measurement_Drift → Anomaly_Flag`, `Upstream_Station_Value → Downstream_Station_Value → Anomaly_Flag`, or `Shift/Time_of_Day → Anomaly_Flag`.
2. Test the parts that the data can support, such as whether STN08 readings are associated with later anomalies at STN18 or BIW.
3. Use DoWhy for causal estimation and refutation, or CausalNex for structure learning and Bayesian-network inference. Any counterfactual result should make its assumptions explicit.
4. Keep data-supported associations separate from the placeholder causal story in every result and chart.

**Output:** a small causal graph and a DoWhy/CausalNex notebook that demonstrates the workflow with placeholder causes, plainly labelled as a framework rather than a validated root-cause finding.
