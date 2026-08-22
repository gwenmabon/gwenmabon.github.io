---
title: "The Counterfactual"
weight: 60
math: true
draft: true
---

Over the previous five chapters we built a bidding system piece by piece. Chapter 1 derived the optimal bid :

$$
b^* = v\,\mu(x) - \frac{F(b^*)}{f(b^*)}
$$

where \(\mu(x) = P(Y = 1 \mid X = x)\) is the probability of conversion, and \(F\) and \(f\) are the CDF and density of the highest competing bid. Chapter 2 showed how to estimate and calibrate \(\mu(x)\). Chapter 3 addressed the feedback loop in its training data. Chapter 4 estimated \(F\) and \(f\) from censored auction data. Chapter 5 added the budget constraint, discounting the value by \(1/(1+\lambda)\).

One assumption has been present since Chapter 1 and never questioned : that the value of showing an ad is \(v \cdot \mu(x)\). This is wrong. A user who would have converted without the ad produces the same revenue but was not caused by the ad. The DSP pays \(b^*\) for an impression that created no value. The correct value is not \(v \cdot \mu(x)\) but \(v \cdot \tau(x)\), where \(\tau(x)\) is the causal effect of the ad. This chapter derives what \(\tau(x)\) is and how to estimate it.

## The Fundamental Problem

### Potential outcomes

Let \(T \in \{0, 1\}\) denote treatment : \(T = 1\) if the ad is shown, \(T = 0\) if not. Each user has two **potential outcomes** :

- \(Y^{(1)}\) : conversion if the ad is shown.
- \(Y^{(0)}\) : conversion if the ad is not shown.

The **individual treatment effect** is :

$$
\tau_i = Y_i^{(1)} - Y_i^{(0)}
$$

For each user, we observe only one of \(Y^{(1)}\) and \(Y^{(0)}\). If we showed the ad (\(T = 1\)), we observe \(Y^{(1)}\) but never \(Y^{(0)}\). If we did not show it (\(T = 0\)), we observe \(Y^{(0)}\) but never \(Y^{(1)}\). The individual treatment effect is never directly observable. This is the **fundamental problem of causal inference**.

### From individual to average effects

Since \(\tau_i\) is unobservable, we target population averages. The **Average Treatment Effect** :

$$
\text{ATE} = \mathbb{E}\bigl[Y^{(1)} - Y^{(0)}\bigr]
$$

Under randomization (the ad is assigned randomly, independent of \(X\)), the potential outcomes are independent of the treatment. Then :

$$
\text{ATE} = \mathbb{E}[Y \mid T = 1] - \mathbb{E}[Y \mid T = 0]
$$

This is the difference in conversion rates between the treated and control groups. Simple to compute, but it gives a single number for the entire population. The ATE for a retargeting campaign might be 0.5%, but some user segments might have \(\tau = 0\) (would have converted anyway) and others \(\tau = 3\%\) (truly incremental). Bidding the same amount on both wastes budget on the first group.

### Conditional ATE

The **Conditional Average Treatment Effect** captures this heterogeneity :

$$
\tau(x) = \mathbb{E}\bigl[Y^{(1)} - Y^{(0)} \mid X = x\bigr]
$$

This is the quantity the bid formula needs. For a user segment with \(\tau(x) = 0\), the ad creates no value and the bid should be zero. For a segment with \(\tau(x) = 0.03\), the incremental value is \(v \cdot 0.03\) and the bid should be positive. The bid formula becomes :

$$
b^* = \frac{v\,\tau(x)}{1 + \lambda} - \frac{F(b^*)}{f(b^*)}
$$

This is the same structure as Chapter 1, with \(\tau(x)\) replacing \(\mu(x)\).

## Why the Naive Bid Is Wrong

The quantity \(\mu(x) = P(Y = 1 \mid X = x, T = 1)\) from Chapter 2 is the probability of conversion given that the ad was shown. It includes both incremental conversions (caused by the ad) and organic conversions (would have happened without the ad). The naive bid uses \(v \cdot \mu(x)\) as the value. The correct value is \(v \cdot \tau(x)\). The difference is :

$$
v\,\mu(x) - v\,\tau(x) = v \cdot P(Y^{(0)} = 1 \mid X = x)
$$

This is \(v\) times the organic conversion rate : the probability the user converts without the ad. For retargeting campaigns, where the DSP shows ads to users who already visited the advertiser's website, the organic rate can be high. A retargeted user with \(\mu(x) = 0.05\) and organic rate 0.04 has \(\tau(x) = 0.01\). The naive bid is 5 times too high.

**A numerical example.** A retargeting campaign with payout \(v = 50\) euros. The CTR model estimates \(\mu(x) = 0.05\). The naive value is \(50 \times 0.05 = 2.50\) euros. But if the organic conversion rate is 0.04, the true incremental value is \(50 \times 0.01 = 0.50\) euros. The DSP overbids by 2.00 euros per impression. At 1 million daily impressions on retargeted users, that is 2 million euros per day of overspend.

This is not a hypothetical number. In retargeting, organic rates of 80% or more of the observed conversion rate are common. The DSP's attribution dashboard shows a healthy ROAS, but most of the attributed conversions would have happened anyway.

## Generating the Data : Randomized Experiments

To estimate \(\tau(x)\), we need observations under both \(T = 0\) and \(T = 1\) for users with similar features \(x\). The only clean way to get this is a **randomized experiment** : randomly withhold ads from a fraction of users and compare their conversion rates to the treated group.

### Ghost bidding

The standard design in AdTech is **ghost bidding**. The DSP runs the full bidding pipeline on all impressions but, for a random subset (the control group), submits a bid of zero. The auction runs normally for both groups. The DSP observes :

- **Treatment group** : the ad was shown (if the bid won). Outcome \(Y\) is observed.
- **Control group** : the ad was not shown (bid was zero). Outcome \(Y^{(0)}\) is observed.

The randomization happens at the bid level, so it is within the DSP's control. No coordination with the publisher or exchange is needed.

The cost of the experiment is the revenue lost on the control group. If 10% of traffic is held out and the campaign generates 100,000 euros per day, the experiment costs roughly 10,000 euros per day. The ML engineer must justify this cost against the expected improvement in bid accuracy.

### Intent-to-treat vs. as-treated

Ghost bidding measures the **intent-to-treat** (ITT) effect : the effect of *intending* to show the ad, not the effect of the ad being seen. In the treatment group, not all bids win. Some users in the treatment group were never actually shown the ad because the DSP lost the auction. The ITT effect is diluted by the win rate.

Write \(w\) for the win rate. The observed ITT is :

$$
\text{ITT} = \mathbb{E}[Y \mid \text{treatment}] - \mathbb{E}[Y \mid \text{control}]
$$

The effect of actually seeing the ad is the **Local Average Treatment Effect** (LATE), recovered by dividing by the compliance rate :

$$
\text{LATE} = \frac{\text{ITT}}{w}
$$

If the win rate is 30%, the ITT is 0.3% (1.3% treated vs. 1.0% control), and the LATE is \(0.003 / 0.30 = 0.01\), or 1%. The ad actually lifts conversion by 1% for users who see it, but the raw experiment shows only 0.3% because 70% of the treatment group never saw the ad.

## Estimating CATE

The ATE tells us the average effect. For bidding, we need \(\tau(x)\) for each impression. Three **meta-learners** estimate CATE using any supervised model as a building block.

### S-Learner

Train a single model on \((X, T) \to Y\), treating \(T\) as a feature. The CATE estimate is :

$$
\hat{\tau}_S(x) = \hat{\mu}(x, T = 1) - \hat{\mu}(x, T = 0)
$$

The model sees all the data, which is good for statistical power. But if the treatment effect is small relative to the baseline conversion rate, the feature \(T\) contributes little to the loss. Regularisation (L2, max depth, dropout) can suppress \(T\) entirely. In AdTech, treatment effects are often 0.1-1% on a baseline of 1-5%. The S-Learner tends to estimate \(\hat{\tau}_S(x) \approx 0\) everywhere.

### T-Learner

Train two separate models :

$$
\hat{\mu}_1(x) \text{ on } \{(x_i, y_i) : T_i = 1\}, \qquad \hat{\mu}_0(x) \text{ on } \{(x_i, y_i) : T_i = 0\}
$$

$$
\hat{\tau}_T(x) = \hat{\mu}_1(x) - \hat{\mu}_0(x)
$$

The treatment effect is a first-class quantity : it is the difference between two models, each trained to estimate the conditional mean under one treatment arm. But each model sees only half the data. The CATE is a difference of two noisy estimates, and the noise is amplified : \(\text{Var}(\hat{\tau}_T) = \text{Var}(\hat{\mu}_1) + \text{Var}(\hat{\mu}_0)\).

### X-Learner

The X-Learner (Künzel et al., 2019) fixes the T-Learner's variance problem by using cross-imputation. The idea : use the model trained on the larger group to impute treatment effects for the smaller group.

1. Train \(\hat{\mu}_0\) and \(\hat{\mu}_1\) as in the T-Learner.
2. For treated units (\(T_i = 1\)), impute the individual effect :
$$
\tilde{D}_i^1 = Y_i - \hat{\mu}_0(X_i)
$$
3. For control units (\(T_i = 0\)), impute :
$$
\tilde{D}_i^0 = \hat{\mu}_1(X_i) - Y_i
$$
4. Train two CATE models : \(\hat{\tau}_1(x)\) on \(\{(X_i, \tilde{D}_i^1) : T_i = 1\}\) and \(\hat{\tau}_0(x)\) on \(\{(X_i, \tilde{D}_i^0) : T_i = 0\}\).
5. Combine :
$$
\hat{\tau}_X(x) = e(x)\,\hat{\tau}_0(x) + (1 - e(x))\,\hat{\tau}_1(x)
$$

where \(e(x) = P(T = 1 \mid X = x)\) is the **propensity score**. This is the same propensity score from Chapter 3, estimated at serving time. The X-Learner weights the two CATE estimates by the propensity : in regions where most users are treated, it relies more on \(\hat{\tau}_0\) (which was trained on the control group, using the treated model's predictions to impute effects). This is efficient when the groups are imbalanced, which is the norm in AdTech : most users are treated (shown an ad), and the control group is a small holdout.

### Summary

| | S-Learner | T-Learner | X-Learner |
|---|:---:|:---:|:---:|
| Models trained | 1 | 2 | 4 |
| Data efficiency | High | Low | High |
| Small effects | Suppressed by regularisation | Noisy | Handles well |
| Imbalanced groups | Poor | Poor | Good |

## Observational Estimation

Running experiments is expensive. The DSP holds out traffic, loses revenue, and must run the experiment long enough for the treatment effects to be detectable. In practice, some DSPs estimate CATE from observational data (logged bids and outcomes, without randomization).

The problem is confounding. The DSP does not bid uniformly : it bids more on users it thinks will convert. Users who see the ad are systematically different from users who do not. The observed difference \(\mathbb{E}[Y \mid T = 1, X = x] - \mathbb{E}[Y \mid T = 0, X = x]\) is not \(\tau(x)\) because treatment is not random.

Under the assumption of **no unobserved confounders** (all variables that affect both treatment and outcome are in \(X\)), we can correct for the bias using the propensity score \(e(x) = P(T = 1 \mid X = x)\). The IPW estimator from Chapter 3 becomes :

$$
\hat{\tau}_{\text{IPW}} = \frac{1}{n}\sum_{i=1}^{n}\left[\frac{T_i\,Y_i}{e(X_i)} - \frac{(1 - T_i)\,Y_i}{1 - e(X_i)}\right]
$$

The assumption of no unobserved confounders is strong and usually violated in AdTech. User intent is a confounder (users who are about to convert are more likely to see retargeting ads), and intent is rarely fully captured by features. Observational estimates are biased upward. The ML engineer should treat them as an upper bound on the true effect, and use experiments whenever possible.

## From CATE to the Bid Formula

We can now write the full bid formula with incrementality :

$$
b^* = \frac{v\,\hat{\tau}(x)}{1 + \lambda} - \frac{F(b^*)}{f(b^*)}
$$

Compare this to the naive formula \(b^* = v\,\hat{\mu}(x)/(1+\lambda) - F/f\). The only change is replacing \(\hat{\mu}(x)\) with \(\hat{\tau}(x)\). But the consequences are large.

For **prospecting campaigns** (targeting new users), the organic conversion rate is near zero. Most conversions are incremental : \(\tau(x) \approx \mu(x)\). The incremental bid is close to the naive bid.

For **retargeting campaigns**, the organic rate is high. The incremental bid can be 5-10x lower than the naive bid. A DSP that switches from \(\mu\) to \(\tau\) in its retargeting bids will bid less, win fewer auctions, but pay only for impressions that create value. The ROAS reported to the advertiser drops (fewer attributed conversions), but the true incremental ROAS improves.

**A numerical comparison.** Two user segments, same campaign with \(v = 50\) euros, \(\lambda = 0\) :

| Segment | \(\mu(x)\) | Organic rate | \(\tau(x)\) | Naive value | Incremental value |
|---|:---:|:---:|:---:|:---:|:---:|
| Prospecting | 0.005 | 0.001 | 0.004 | 0.25 € | 0.20 € |
| Retargeting | 0.050 | 0.045 | 0.005 | 2.50 € | 0.25 € |

The naive bid on retargeting is 10x the naive bid on prospecting. The incremental bid is nearly the same. The entire allocation flips : budget moves from retargeting (where the DSP is paying for organic conversions) to prospecting (where every conversion is incremental).

## What the ML Engineer Monitors

Incrementality estimation adds a new layer to the bidding system. The ML engineer must track :

- **Experiment coverage.** What fraction of traffic is in the control group ? Too small and the CATE estimates are noisy. Too large and the revenue loss is high.
- **Lift by segment.** \(\hat{\tau}(x)\) by publisher, geo, audience type. Segments with \(\hat{\tau}(x) \leq 0\) should be excluded from bidding.
- **Incremental ROAS vs. attributed ROAS.** The attributed ROAS (from the attribution dashboard) is always higher. The gap is the cost of organic conversions the DSP was paying for.
- **Experiment duration.** Treatment effects in AdTech are small. Detecting a 0.5% lift requires large sample sizes. The ML engineer must compute the required sample size before starting the experiment, or the experiment will be inconclusive and the budget wasted.

## Key Takeaways

1. The bid formula should use \(\tau(x) = \mathbb{E}[Y^{(1)} - Y^{(0)} \mid X = x]\), not \(\mu(x) = P(Y = 1 \mid X = x, T = 1)\). The difference is the organic conversion rate, which the DSP pays for but does not cause.
2. The individual treatment effect \(\tau_i\) is never observable. We estimate the conditional average \(\tau(x)\) from randomized experiments.
3. Ghost bidding is the standard experimental design : the DSP randomly bids zero on a holdout group. The cost is the forgone revenue on the control traffic.
4. The X-Learner handles the imbalanced treatment/control groups that are typical in AdTech, using cross-imputation and the propensity score from Chapter 3.
5. Replacing \(\mu(x)\) with \(\tau(x)\) reallocates budget from retargeting (low incrementality) to prospecting (high incrementality). The attributed ROAS drops, but the true incremental ROAS improves.
