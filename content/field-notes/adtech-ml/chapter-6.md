---
title: "The Counterfactual"
weight: 60
math: true
draft: true
---

Not every conversion attributed to an ad was caused by that ad.
This chapter introduces the causal inference framework for measuring
*incrementality* : the true causal effect of ad exposure. We
formalize the problem using the potential outcomes framework (Rubin),
define Average Treatment Effect (ATE) and Conditional ATE (CATE), and
present the S-Learner, T-Learner, and X-Learner meta-algorithms for
heterogeneous treatment effect estimation.

## The Attribution Illusion

### Correlation Is Not Causation

A user sees an ad and then installs the app. Was the ad responsible ?

- Maybe the user was already going to install (organic intent).
- Maybe a friend recommended the app (social influence).
- Maybe another ad on a different channel drove the conversion.

Standard attribution models (last-click, multi-touch) assign
*credit* but do not measure *causation*.

> [!WARNING]
> If 80% of users attributed to your ad would have converted
> anyway, your true incremental CPA is 5x what your
> attribution dashboard shows. This is not a theoretical concern ;
> it is the norm in retargeting.

### The Business Impact

Over-attribution leads to :

1. Inflated ROAS (Return on Ad Spend) reported to advertisers.
2. Bidding too aggressively on non-incremental users.
3. Wasting budget on users who would convert organically.

## The Potential Outcomes Framework

### Notation

Let \(T \in \{0, 1\}\) denote treatment (ad shown or not), and
\(Y^{(t)}\) the potential outcome under treatment \(t\).

| User | \(Y^{(0)}\) | \(Y^{(1)}\) | ITE |
|------|:-----------:|:-----------:|:---:|
| Alice | ? | 1 (conv.) | ? |
| Bob | 0 (no conv.) | ? | ? |
| Carol | ? | 1 (conv.) | ? |
| Dave | 1 (conv.) | ? | ? |

For any individual user, we can observe at most one potential outcome.
The Individual Treatment Effect (ITE) \(\tau_i = Y_i^{(1)} - Y_i^{(0)}\)
is **never directly observable**. This is the fundamental problem of causal inference.

### Average Treatment Effect (ATE)

Since individual effects are unobservable, we target population averages :

$$
\text{ATE} = \mathbb{E}\bigl[Y^{(1)} - Y^{(0)}\bigr]
$$

Under randomization (the ad is shown randomly), ATE simplifies to :

$$
\text{ATE} = \mathbb{E}[Y \mid T=1] - \mathbb{E}[Y \mid T=0]
$$

This is the difference in conversion rates between the treated and
control groups in a randomized experiment (A/B test).

### Conditional ATE (CATE)

CATE captures *heterogeneous* treatment effects : the effect may
vary across user segments :

$$
\tau(x) = \mathbb{E}\bigl[Y^{(1)} - Y^{(0)} \mid X = x\bigr]
$$

> [!NOTE]
> CATE is what we really want in AdTech. If \(\tau(x) \leq 0\) for a
> user segment, showing them ads is wasteful (or actively harmful).
> If \(\tau(x)\) is high, that segment is incrementally valuable :
> bid more aggressively there.

## \(P(Y \mid \text{do}(X))\) vs. \(P(Y \mid X)\)

Pearl's do-calculus formalizes the difference between :

$$
P(Y \mid X) \quad \text{(observational : what we see)}
$$

$$
P(Y \mid \text{do}(X)) \quad \text{(interventional : what happens when we act)}
$$

| | **Observational** \(P(Y \mid X)\) | **Interventional** \(P(Y \mid \text{do}(X))\) |
|---|---|---|
| Question | "Among users who saw the ad, what fraction converted ?" | "If we *force* showing the ad, what fraction converts ?" |
| Confounders | Included (user intent, targeting bias) | Removed by intervention |
| Use case | Attribution dashboards | Incrementality measurement |

## Meta-Learners for CATE Estimation

When we have experimental data (randomized treated/control), we can
estimate \(\tau(x)\) using "meta-learners" : frameworks that use
any supervised learner as a building block.

### S-Learner (Single Model)

Train one model on \((X, T) \to Y\), then compute CATE as :

$$
\hat{\tau}_S(x) = \hat{\mu}(x, T\!=\!1) - \hat{\mu}(x, T\!=\!0)
$$

- **Pro** : simple, uses all data.
- **Con** : if \(T\) has a small effect, the model may ignore it
  entirely (regularization can suppress \(T\)).

### T-Learner (Two Models)

Train separate models for treated and control :

$$
\hat{\tau}_T(x) = \hat{\mu}_1(x) - \hat{\mu}_0(x)
$$

where \(\hat{\mu}_1\) is trained on \(\{(x_i, y_i) : T_i = 1\}\) and
\(\hat{\mu}_0\) on \(\{(x_i, y_i) : T_i = 0\}\).

- **Pro** : treatment effect is a first-class citizen.
- **Con** : each model sees half the data ; noisy difference
  of two noisy estimates.

### X-Learner (Cross-Learner)

The X-Learner (Kunzel et al., 2019) improves upon the T-Learner by
using cross-imputed treatment effects :

1. Train \(\hat{\mu}_0\) and \(\hat{\mu}_1\) as in the T-Learner.
2. Impute individual treatment effects :
$$
\tilde{D}_i^1 = Y_i^{(1)} - \hat{\mu}_0(X_i) \quad \text{for treated units}
$$
$$
\tilde{D}_i^0 = \hat{\mu}_1(X_i) - Y_i^{(0)} \quad \text{for control units}
$$
3. Train two CATE models : \(\hat{\tau}_1(x)\) on \(\tilde{D}^1\)
   and \(\hat{\tau}_0(x)\) on \(\tilde{D}^0\).
4. Combine : \(\hat{\tau}_X(x) = g(x)\,\hat{\tau}_0(x) + (1-g(x))\,\hat{\tau}_1(x)\),
   where \(g(x) = \hat{e}(x)\) is the propensity score.

> [!NOTE]
> The X-Learner excels when treatment and control groups are
> **imbalanced** (common in AdTech, where most users are
> treated). It uses the larger group's predictions to
> impute effects in the smaller group.

| **Property** | **S-Learner** | **T-Learner** | **X-Learner** |
|---|:---:|:---:|:---:|
| Models trained | 1 | 2 | 4 |
| Data efficiency | High | Low | High |
| Handles imbalance | Poorly | Poorly | Well |
| Treatment effect | Implicit | Explicit | Explicit |
| Complexity | Low | Medium | High |

## From CATE to Bidding

Once we have \(\hat{\tau}(x)\), the incremental value replaces the naive
predicted conversion probability in the bid formula :

$$
\text{bid}_{\text{incremental}} = \hat{\tau}(x) \times \text{payout}
$$

Compare with the naive bid :

$$
\text{bid}_{\text{naive}} = \hat{p}(\text{conv} \mid x, T\!=\!1) \times \text{payout}
$$

The difference can be dramatic. For retargeting campaigns where
organic conversion rates are high, \(\hat{\tau}(x) \ll \hat{p}(x)\),
meaning the incremental bid is much lower, correctly reflecting
that most of these users would have converted anyway.

## Key Takeaways

1. Attribution \(\neq\) causation. Standard attribution models
   (last-click, multi-touch) overcount by ignoring organic
   conversions.
2. The potential outcomes framework formalizes "what would have
   happened without the ad", but the individual counterfactual
   is never observable.
3. ATE measures the average causal effect ; CATE measures
   heterogeneous effects across user segments.
4. Meta-learners (S, T, X) estimate CATE from experimental data
   using standard ML models as building blocks.
5. In bidding, replacing \(\hat{p}(\text{conv})\) with \(\hat{\tau}(x)\)
   avoids overpaying for non-incremental users.
