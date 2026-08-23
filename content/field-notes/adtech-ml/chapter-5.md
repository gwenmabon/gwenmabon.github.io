---
title: "Budget Pacing"
weight: 50
math: true
draft: true
---

In Chapter 4, we derived the shaded bid
\(b^* = v - F(b^*)/f(b^*)\), where \(v = \hat{\mu}(x) \cdot \text{payout}\)
is the impression value from Chapter 2, \(F\) and \(f\) are the CDF and
density of the highest competing bid, and \(b^*\) is the optimal bid. The derivation maximised the expected surplus per impression
without constraint. But every campaign has a daily budget \(B\). If we bid
the full shaded value on all available traffic, the expected daily spend is :

$$
\text{Spend} = \sum_{i=1}^{N} b_i^*\,F_i(b_i^*)
$$

For a typical campaign seeing millions of bid requests per day, this sum
exceeds \(B\) by orders of magnitude. The DSP must reduce its bids so that total expected spend fits within \(B\). This is the **pacing problem**.

## The Constrained Optimisation

**The objective.** We want to maximise total surplus over a day of
\(N\) impressions, subject to a budget constraint :

$$
\max_{b_1, \ldots, b_N} \quad \sum_{i=1}^{N} (v_i - b_i)\,F_i(b_i)
\qquad \text{s.t.} \quad \sum_{i=1}^{N} b_i\,F_i(b_i) \leq B
$$

The term \(b_i\,F_i(b_i)\) is the expected cost of impression \(i\) :
the bid times the win probability. The constraint says that total
expected spend must not exceed \(B\).

**The Lagrangian.** We introduce a multiplier \(\lambda \geq 0\) :

$$
\mathcal{L} = \sum_{i=1}^{N}(v_i - b_i)\,F_i(b_i) - \lambda\left(\sum_{i=1}^{N} b_i\,F_i(b_i) - B\right)
$$

Collecting terms for each impression :

$$
\mathcal{L} = \sum_{i=1}^{N}\bigl[v_i - (1+\lambda)\,b_i\bigr]\,F_i(b_i) + \lambda B
$$

**The KKT conditions.** The solution \((b_1^*, \ldots, b_N^*, \lambda)\) must satisfy four conditions :

1. **Stationarity** : the gradient of \(\mathcal{L}\) with respect to each \(b_i\) is zero.
2. **Primal feasibility** : the budget constraint holds, \(\sum_i b_i\,F_i(b_i) \leq B\).
3. **Dual feasibility** : the multiplier is non-negative, \(\lambda \geq 0\).
4. **Budget either binds or it does not** : \(\lambda\,(\sum_i b_i\,F_i(b_i) - B) = 0\).

We use each of these in what follows.

**Stationarity.** Differentiating \(\mathcal{L}\) with respect to
\(b_i\) and setting to zero :

$$
\frac{\partial \mathcal{L}}{\partial b_i}
= -(1+\lambda)\,F_i(b_i) + \bigl[v_i - (1+\lambda)\,b_i\bigr]\,f_i(b_i) = 0
$$

Solving for \(b_i\) :

$$
b_i^* = \frac{v_i}{1 + \lambda} - \frac{F_i(b_i^*)}{f_i(b_i^*)}
$$

This is the shading formula from Chapter 4, applied to a discounted
value \(\tilde{v}_i = v_i/(1+\lambda)\). The shading term \(F/f\)
is unchanged : it depends on the market, not on our budget.

**Budget either binds or it does not.** The fourth condition requires :

$$
\lambda \left(\sum_{i=1}^{N} b_i^*\,F_i(b_i^*) - B\right) = 0
$$

Either \(\lambda = 0\) and we bid the unconstrained shaded value, or \(\lambda > 0\) and the budget is spent exactly.

**Dual feasibility** (\(\lambda \geq 0\)) will appear later in the gradient descent update, where we project \(\lambda\) back to zero whenever the update would make it negative.

## The Shadow Price of Budget

The multiplier \(\lambda\) is the marginal value of one additional
euro of budget :

$$
\lambda = \frac{\partial}{\partial B}\left[\max \sum_{i}(v_i - b_i^*)\,F_i(b_i^*)\right]
$$

A campaign with \(\lambda = 0\) has more budget than it can spend.
Adding euros does not help. A campaign with \(\lambda = 2\) divides
every impression value by 3 before shading. Each additional euro of
budget would generate 2 euros of surplus.

| \(\lambda\) | Budget situation | Effect on bid |
|---|---|---|
| 0 | Excess budget | Unconstrained shading |
| 0.5 | Moderate pressure | Value discounted by 33% |
| 2 | Tight budget | Value discounted by 67% |
| \(\to \infty\) | Exhausted | Bids go to zero |

**A numerical example.** Campaign payout is 50 euros per conversion.
The value model gives \(v_i = 50 \times 0.003 = 0.15\) euros for a
specific impression. In the unconstrained case (\(\lambda = 0\)),
suppose the shading term \(F/f\) equals 0.05. The bid is
\(0.15 - 0.05 = 0.10\) euros. With \(\lambda = 1\), the discounted
value is \(0.15/2 = 0.075\). The bid becomes
\(0.075 - 0.05 = 0.025\) euros. The win probability drops, and the
campaign spends four times slower on this impression.

## From Batch to Online

The Lagrangian assumes all \(N\) impressions are known in advance.
In production, impressions arrive one by one. The DSP must bid
immediately. We do not know the volume or quality of traffic that
will arrive in three hours.

**The offline optimum.** If we had perfect foresight (all \(N\)
impressions known), we would solve the constrained problem exactly.
Call the resulting total surplus \(S^*\). This is the best any
algorithm can achieve.

**The online problem.** The pacer sees impressions sequentially
and must commit to a bid before seeing the next impression. It
maintains a running estimate of \(\lambda\) and updates it as the
campaign spends. The total surplus \(S_{\text{online}}\) is
necessarily less than \(S^*\).

**Online regret.** The gap \(S^* - S_{\text{online}}\) measures
how much surplus we lose by not knowing the future, similar
to the bandit regret from Chapter 3. If the per-impression loss stays constant over the day, the total loss grows linearly with \(T\). A pacer with sublinear regret improves its estimate of \(\lambda\) as the day progresses, so the per-impression loss shrinks and the average cost goes to zero.

The two ways a pacer fails illustrate this. If \(\lambda\) reacts too slowly to a traffic spike, the budget binds early : a campaign that exhausts its budget at 2pm leaves roughly half its potential surplus on the table. Symmetrically, if \(\lambda\) stays too high all day, 40% of the budget remains at 6pm and the pacer drops \(\lambda\) close to zero, bidding aggressively on whatever traffic is left at 3-5x the daily average cost per conversion. Both failures are linear regret : the pacer does not learn \(\lambda\) fast enough. We need a controller that converges.

## Adapting \(\lambda\) in Real Time

### Dual Gradient Descent

The dual function is \(g(\lambda) = \max_{\{b_i\}} \mathcal{L}(\{b_i\}, \lambda)\).
It is concave in \(\lambda\). We minimise \(g(\lambda)\) by gradient
descent on the dual :

$$
\lambda_{t+1} = \max\!\bigl(0,\; \lambda_t + \eta\,(\text{spend}_t - B_t)\bigr)
$$

where \(B_t = B \cdot t/T\) is the target cumulative spend at time
\(t\) and \(\eta\) is a step size. The gradient of the dual with
respect to \(\lambda\) is exactly \(\text{spend} - B\), so the
update moves \(\lambda\) in the direction that tightens or relaxes
the constraint.

### PID Control

Dual gradient descent uses only the current error. A PID controller
adds memory and anticipation. The error signal is :

$$
e_t = \frac{\text{spend}_t}{t} - \frac{B}{T}
$$

The PID update combines three terms :

$$
\Delta\lambda_t = K_p\,e_t + K_i\sum_{s=1}^{t} e_s + K_d\,(e_t - e_{t-1})
$$

| Term | Coefficient | Role |
|---|---|---|
| Proportional | \(K_p\) | Reacts to the current spend rate error |
| Integral | \(K_i\) | Corrects accumulated drift over the day |
| Derivative | \(K_d\) | Dampens oscillations when \(e_t\) changes fast |

The gains \(K_p, K_i, K_d\) are hyperparameters. The integral term matters most in practice. Without it, the pacer systematically under- or over-shoots. The accumulated error in \(K_i \sum e_s\) corrects this drift.

## Non-Stationary Traffic

The uniform target \(B_t = B \cdot t/T\) assumes traffic is constant
over the day, but impression volume varies by hour. Let \(q(t)\) be the density of impression
volume over time, normalised so that \(\int_0^T q(t)\,dt = 1\). The
fraction of impressions arriving before time \(t\) is :

$$
Q(t) = \int_0^t q(s)\,ds
$$

A pacing target that tracks traffic volume sets :

$$
B_t = B \cdot Q(t)
$$

If 60% of impressions arrive before noon, the target at noon is
\(0.6 \cdot B\), not \(0.5 \cdot B\). The pacer spends faster in
high-volume hours and slower in low-volume hours, instead of
fighting the traffic pattern.

**Estimating \(q(t)\).** The volume density is estimated from
historical data : same day of week, same campaign type, same geo.
Even a rough day-of-week average is better than the uniform assumption and reduces the regret of the online pacer.

**Value also varies.** Impression volume is not the only thing that changes over the day. Evening traffic on e-commerce sites converts at higher rates. Spending budget uniformly means buying cheap morning impressions and missing expensive evening ones. The optimal \(B_t\) should account for both volume and value, which requires forecasting their joint distribution over time.

## Multi-Campaign Allocation

A DSP runs hundreds of campaigns with overlapping targeting. Each
campaign has its own \(\lambda_j\). When two campaigns target the
same impression, the DSP must choose which one bids. This is a
second-level allocation problem : given impression \(x\), assign it
to the campaign \(j\) that maximises :

$$
j^* = \arg\max_j \left[\frac{v_j(x)}{1+\lambda_j} - \frac{F(b^*)}{f(b^*)}\right]
$$

The campaign with the highest budget-adjusted value wins the right
to bid. This couples all pacers together : a change in one
campaign's \(\lambda\) shifts impressions to other campaigns.

The single-assignment model is a simplification. A DSP can bid multiple campaigns on the same impression. The coupling only matters when campaigns share targeting.

This connects to the exploration-exploitation trade-off from
Chapter 3 : each campaign's \(\lambda_j\) affects which impressions
it wins, which determines its training data. A campaign with high
\(\lambda_j\) wins less traffic, accumulates fewer labels, and its
model uncertainty grows. The pacing and exploration problems are
coupled through the win rate.

## Key Takeaways

1. The budget constraint discounts the value in the bid formula :
   \(b_i^* = v_i/(1+\lambda) - F_i/f_i\), where \(\lambda\) is
   the shadow price of budget.
2. Budget either binds or it does not : a campaign either has excess budget
   (\(\lambda = 0\)) or spends exactly \(B\).
3. In production, \(\lambda\) is adapted online via dual gradient
   descent or PID control. The gap between online and offline
   surplus is the pacing regret.
4. Traffic is non-stationary. The pacing target should track the
   volume density \(q(t)\), not a uniform line.
5. Multi-campaign allocation couples all pacers : impression
   assignment depends on the relative \(\lambda_j\) values across
   campaigns, and the resulting win rates feed back into each
   campaign's training data.
