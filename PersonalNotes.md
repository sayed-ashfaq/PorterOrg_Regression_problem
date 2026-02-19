# Concepts I have faced and Understanding them in depth

## Skewness vs Kurtosis


### Skewness
Skewness measures the asymmetry in the distribution. It tell you wheter data leans more to the left or right.

#### **Types of skewness**
-> Positive skewness(Right skew): More data is towards the left side. Long tail on the right.
	- Mean > median
-> Negative Skewness(Left skew): Long tail on left side, where as vast amount data leaning towards the right.
	- Mean < median.

#### How does this effect the models
- Many models assumes normality. Coefficient become unstable.
- prediction biased toward tail
- Loss dominated by extreme values.

NNs don't assume normality - But skewness still matters because
- Larger values dominates the gradient by targeting loss functions.
- focus more will be on rare extremes
- Tree models - least sensitive - split based on order, not distribution.

### Kurtosis
Kurtosis measure how heavy the tails and peakness of a distribution are. Length and height. 
- it tells you how extreme the outliers might be.

#### Types of Kurtosis
-> High kurtosis(Leptokurtic)
- Sharp peak | Heavy tails | More extreme outliers. 

-> Low Kurtosis (Platykurtic)
- Flatter shape | Fewer extreme values

#### How does this effect the models
* High Kurtosis = many outliers => results in MSE loss exploding. Outliers dominate training
* increases Overfitting risk
---

### QQ plots

The Quantile-Quantile method(QQ) is plot technique to determine whether dataset is following a particular distribution(Usually normal) or not. Whether two samples of data came from the same population or not.

#### How to read a QQ plot

1. Points follow a straight line => data is approximately normal.

2. S-Shape Curve => Upward curve : Positive skew, downward curve: negative skew

3. Ends bend away from line => high Kurtosis/heavy tails.

4. Flat middle, curved ends => Non normal distribution.

#### Effects on model
* Neural networks and tree models do not require normality.
* It answers `Should I transform my target before modeling?`

---

### ANOVA Testing
 Analysis of Variance is a statistical method used to determine whether there are significant differences between the means of three or more independent groups by analyzing the variability within each group and between the groups.


### More EDA

#### Great Question — Let Me Give You the Real World Picture

You're right to sense there's more. What we did covers the **regression/prediction workflow.** But statistical testing in real data science is much broader. Let me map it out properly.

---

#### The Tests a Senior Data Scientist Actually Uses

##### Category 1 — Distribution Tests (What we did)
Shapiro-Wilk, Anderson-Darling — formally testing normality. QQ plots for visual confirmation. Skewness/kurtosis for shape analysis.

##### Category 2 — Group Comparison Tests (What we did)
ANOVA, Kruskal-Wallis for multiple groups. T-test, Mann-Whitney for two groups. These answer "are these groups actually different?"

##### Category 3 — Relationship Tests
Pearson correlation (linear, normal data). Spearman correlation (non-linear, skewed data). Chi-square test — are two **categorical** variables related? For example: is food category related to market? We didn't do this but should have for our categoricals.

##### Category 4 — A/B Testing (Huge in industry)
Did our new delivery algorithm actually improve times? Companies run these constantly. Two sample tests with power analysis, confidence intervals, multiple testing corrections (Bonferroni).

##### Category 5 — Time Series Tests
Augmented Dickey-Fuller — is the data stationary? Ljung-Box — is there autocorrelation? Critical for anything with dates.

##### Category 6 — Model Validation Tests
Residual analysis — are our model errors random or patterned? If patterned, model is missing something. Levene's test for variance equality across predictions.

---

#### For Your SCB Example — Anomaly Detection in Banking

This is a completely different problem type and the statistical toolkit shifts significantly. Let me walk through how a senior would approach it.

**The Core Challenge:** You don't know what fraud looks like in advance. You're not predicting a known outcome — you're finding needles in a haystack where you're not sure what the needle looks like.

##### Step 1 — Establish Normal Behavior First
Before finding anomalies you must define "normal." This is statistical profiling:

```
For each customer:
- Average transaction amount (mean, median)
- Typical transaction frequency per day/week
- Common merchant categories
- Geographic patterns (city, country)
- Time patterns (usually transacts 9am-6pm?)
```

You use **percentiles and rolling statistics** to build a behavioral baseline per customer, not just globally.

##### Step 2 — Statistical Tests You'd Run

**Z-score on transaction amounts** — but *per customer*, not globally. A ₹500,000 transaction is normal for a business account, anomalous for a student account. Global Z-score misses this entirely.

**Benford's Law test** — legitimate transactions follow a natural law where leading digits follow a predictable distribution (1 appears ~30% of the time, 9 appears ~5%). Fraudulent/manipulated transactions violate this. Chi-square test against Benford's expected distribution.

**Isolation Forest / Local Outlier Factor** — these are ML-based anomaly detection, not pure statistics, but grounded in statistical distance concepts.

**Time series anomaly detection** — sudden spike in transaction frequency? Augmented Dickey-Fuller + CUSUM (cumulative sum control charts) to detect when behavior shifts from baseline.

**Graph-based analysis** — who is transacting with whom? Sudden new connections between accounts that never interacted = suspicious. This uses network statistics.

##### Step 3 — Cyber Security Specific Tests

**Entropy analysis** — legitimate network traffic has predictable patterns. High entropy in packet sizes or timing = possible encrypted malware communication.

**Sequential probability ratio test (SPRT)** — real-time test that raises alarm as soon as enough evidence accumulates, without waiting for a fixed sample size. Critical for real-time threat detection.

**Markov chain modeling** — model normal user behavior as state transitions (login → browse → transact → logout). Deviations from the expected transition probabilities flag suspicious sessions.

---

#### The Key Difference From What We Did

In our Porter project we had a **labeled target** — delivery time was known. We were doing supervised learning with statistical validation.

In fraud/anomaly detection you're mostly doing **unsupervised statistics** — no labels, no "correct answer." You're asking "does this observation look like it belongs to the same distribution as everything else?" That's a fundamentally different statistical question.

The tests shift from "are these groups different" to "does this point belong to this distribution" — Mahalanobis distance, isolation scores, reconstruction error from autoencoders.

---

#### Bottom Line

In a real data science job the statistical tests you use depend entirely on the problem type:

| Problem Type | Core Tests |
|---|---|
| Regression/Classification | Distribution tests, correlation, residual analysis |
| A/B Testing | T-test, Mann-Whitney, power analysis, Bonferroni |
| Anomaly Detection | Z-score per entity, Benford's Law, Isolation Forest |
| Time Series | ADF, Ljung-Box, CUSUM |
| NLP | Chi-square for feature selection, KL divergence |

















