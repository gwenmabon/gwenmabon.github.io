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
exceeds \(B\) by orders of magnitude. The DSP must decide which
impressions to bid on and how much to reduce each bid. This is the
pacing problem.

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
\mathcal{L} = \sum_{i=1}^{N}(v_i - b_i)\,F_i(b_i)
- \lambda\left(\sum_{i=1}^{N} b_i\,F_i(b_i) - B\right)
$$

Collecting terms for each impression :

$$
\mathcal{L} = \sum_{i=1}^{N}\bigl[v_i - (1+\lambda)\,b_i\bigr]\,F_i(b_i)
+ \lambda B
$$

**The first-order condition.** Differentiating with respect to
\(b_i\) :

$$
\frac{\partial \mathcal{L}}{\partial b_i}
= -(1+\lambda)\,F_i(b_i) + \bigl[v_i - (1+\lambda)\,b_i\bigr]\,f_i(b_i) = 0
$$

Setting to zero and solving for \(b_i\) :

$$
b_i^* = \frac{v_i}{1 + \lambda} - \frac{F_i(b_i^*)}{f_i(b_i^*)}
$$

This is the shading formula from Chapter 4, applied to a discounted
value \(\tilde{v}_i = v_i/(1+\lambda)\). The shading term \(F/f\)
is unchanged : it depends on the market, not on our budget.

**Complementary slackness.** The KKT conditions require :

$$
\lambda \left(\sum_{i=1}^{N} b_i^*\,F_i(b_i^*) - B\right) = 0
$$

Either \(\lambda = 0\) (the budget is not binding and we bid the
unconstrained shaded value) or the budget binds with equality.
There is no middle ground : a campaign either has excess budget
or it spends exactly \(B\).

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
how much surplus we lose by not knowing the future. This is analogous
to the bandit regret from Chapter 3. A good pacer has sublinear
regret : the per-impression loss shrinks as the day progresses and
the estimate of \(\lambda\) stabilises.

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

The gains \(K_p, K_i, K_d\) are hyperparameters. In practice, the
integral term matters most : without it, the pacer can systematically
under- or over-shoot all day long. With it, the accumulated error
forces \(\lambda\) back on target.

## Non-Stationary Traffic

The uniform target \(B_t = B \cdot t/T\) assumes traffic is constant
over the day. It is not. Let \(q(t)\) be the density of impression
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
This is a forecasting problem. The forecast does not need to be
precise ; it needs to be better than uniform. Even a rough
day-of-week average reduces the regret of the online pacer.

**Value also varies.** Volume is not the only non-stationary
quantity. The average impression value \(\bar{v}(t)\) changes over
the day. Evening traffic on e-commerce sites converts at higher
rates. Spending budget uniformly over time means buying cheap
morning impressions and missing expensive evening ones. The optimal
\(B_t\) should account for both volume and value, which requires
forecasting \(q(t) \cdot \bar{v}(t)\) jointly.

## Failure Modes

**Early exhaustion.** If \(\lambda\) reacts too slowly to a traffic
spike, the budget binds early. A campaign that exhausts its budget
at 2pm leaves \(\sum_{i \in \text{afternoon}} v_i\,F_i(b_i^*)\)
euros of potential surplus on the table. For a 10,000-euro daily
budget, losing 8 hours of a 16-hour day means roughly 5,000 euros
of missed surplus, assuming uniform value.

**Forced spend.** The symmetric failure : \(\lambda\) is too high
all day, and 40% of the budget remains at 6pm. The pacer drops
\(\lambda\) close to zero, bidding aggressively on whatever traffic
is left. The cost per conversion in the last two hours can be 3-5x
the daily average because the pacer is buying whatever remains, not
what is valuable.

**Multi-campaign allocation.** A DSP runs hundreds of campaigns
with overlapping targeting. Each campaign has its own \(\lambda_i\).
When two campaigns target the same impression, the DSP must choose
which one bids. This is a second-level allocation problem : given
impression \(x\), assign it to the campaign \(j\) that maximises :

$$
j^* = \arg\max_j \left[\frac{v_j(x)}{1+\lambda_j} - \frac{F(b^*)}{f(b^*)}\right]
$$

The campaign with the highest budget-adjusted value wins the right
to bid. This couples all pacers together : a change in one
campaign's \(\lambda\) shifts impressions to other campaigns.

## Key Takeaways

1. The budget constraint discounts the value in the bid formula :
   \(b_i^* = v_i/(1+\lambda) - F_i/f_i\), where \(\lambda\) is
   the shadow price of budget.
2. Complementary slackness : a campaign either has excess budget
   (\(\lambda = 0\)) or spends exactly \(B\). No middle ground.
3. In production, \(\lambda\) is adapted online via dual gradient
   descent or PID control. The gap between online and offline
   surplus is the pacing regret.
4. Traffic is non-stationary. The pacing target should track the
   volume density \(Q(t)\), not a uniform line.
5. Multi-campaign allocation couples all pacers : impression
   assignment depends on the relative \(\lambda\) values across
   campaigns.
