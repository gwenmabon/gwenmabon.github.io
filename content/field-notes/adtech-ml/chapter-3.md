---
title: "Explore or Exploit"
weight: 30
math: true
draft: true
---

In Chapters 1 and 2, we derived the optimal bid as a function of \(\mu(x) = \mathbb{E}[Y \mid X = x]\) and showed that the loss function for estimating \(\mu(x)\) is the log-loss, the natural likelihood of a Bernoulli problem. We assumed one thing without stating it : that the training data is drawn from the same distribution as the production traffic.

This assumption fails. The model is trained on outcomes of impressions it chose to bid on. It never observes what happens on traffic it skipped. Formally, the training distribution \(P_{\text{train}}\) is not \(P_X\), the marginal over all bid requests. It is \(P_{X \mid \text{bid}}\), the distribution conditional on the model having bid. Everything that follows in this chapter comes from this mismatch.

## Selection Bias in Training Data

### The retraining loop

A DSP retrains its CTR model daily :

1. Day \(t\) : model \(M_t\) scores bid requests, bids on the subset where the expected value exceeds a threshold.
2. Outcomes are observed only for won impressions.
3. Day \(t{+}1\) : model \(M_{t+1}\) is trained on those outcomes.

The training set of \(M_{t+1}\) is a function of \(M_t\). Bid requests that \(M_t\) skipped have no outcome label. They do not enter the loss function. The model cannot learn from data it did not generate.

Write \(\pi_t(x) = P(\text{bid} \mid x, M_t)\) for the probability that model \(M_t\) bids on impression \(x\). The training distribution at day \(t+1\) is :

$$
P_{\text{train}}^{(t+1)}(x) \propto \pi_t(x) \cdot P_X(x)
$$

Segments where \(\pi_t(x)\) is small are under-represented. Segments where \(\pi_t(x) = 0\) are absent entirely. The model minimises log-loss under \(P_{\text{train}}\), not under \(P_X\). It converges to the right \(\mu(x)\) on traffic it sees often, and to an arbitrary value on traffic it never sees.

### Why standard metrics do not detect it

The test set has the same bias. It is sampled from the same \(P_{\text{train}}\) distribution. Log-loss on this test set measures how well the model predicts on the traffic it already bids on. It says nothing about the segments it ignores.

AUC, calibration error, Brier Score : all computed on \(P_{\text{train}}\). The model can have perfect calibration on observed traffic and be completely wrong everywhere else. The metrics are not lying ; they are answering the wrong question.

### Survivor bias in practice

**Creative rotation.** An advertiser has 5 creatives. Creative A gets 3 early clicks out of 100 impressions ; creative B gets 0 out of 20. The model estimates \(\hat{\mu}_A = 0.03\) with a tight confidence interval, and \(\hat{\mu}_B = 0\) with an interval that includes 0.15. The point estimate for B is zero, so B stops getting traffic. The system has no way to discover that B might have a true CTR of 0.2%.

**Audience segments.** Retargeted users have thousands of labeled examples. Prospecting users have dozens. The model's uncertainty on prospecting is high, but nothing in the pipeline acts on that uncertainty. The bid formula takes \(\hat{\mu}(x)\) as a point estimate. A low estimate produces a low bid. A low bid wins nothing. No new data is generated.

**Publisher inventory.** The model bid on a publisher 5 times and observed 0 clicks. The estimated CTR is 0. But 5 observations cannot distinguish a true rate of 0% from a true rate of 1%. The model treats the point estimate as truth and never returns.

### The cost is invisible

The feedback loop does not degrade a visible metric. It creates an opportunity cost. The DSP never discovers the segments it should have been bidding on. A segment with true CTR of 0.3% and a payout of 50 euros is worth \(0.003 \times 50 = 0.15\) euros per impression. At 100,000 daily impressions on that segment, that is 15,000 euros per day of value the system cannot see, let alone capture. The loss does not appear in any dashboard because the data to compute it was never collected.

## The Multi-Armed Bandit

### Setup

We have \(K\) arms. Each arm \(a \in \{1, \dots, K\}\) has an unknown mean reward \(\mu_a\). At each round \(t = 1, \dots, T\), we choose arm \(a_t\) and observe a reward \(r_t\) with \(\mathbb{E}[r_t \mid a_t = a] = \mu_a\).

In our context, an arm is a repeated decision : which creative to show, which audience segment to target. The reward is the event value (click, conversion). The mean reward is the true expected value of that decision, which we do not know.

The bidding problem maps to this framework directly. At each impression, the DSP picks an action. It observes the outcome. It updates its estimates. The question is how to allocate impressions between actions it believes are good (exploit) and actions it knows little about (explore).

### Regret

**Regret** measures the total cost of not knowing which arm is best :

$$
R_T = T\,\mu^* - \sum_{t=1}^{T} \mu_{a_t}
$$

where \(\mu^* = \max_a \mu_a\). A pure exploitation strategy (always play the arm with the highest \(\hat{\mu}\)) has zero exploration cost but may lock onto a suboptimal arm forever. A pure exploration strategy (play uniformly at random) gathers information but wastes it on arms that are clearly bad.

A good algorithm has sublinear regret : \(R_T = o(T)\). The per-round cost \(R_T / T \to 0\). The theoretical lower bound is \(\Omega(\sqrt{KT})\). The algorithms below achieve \(O(\sqrt{KT \log T})\).

## Three Algorithms

### \(\varepsilon\)-greedy

With probability \(1 - \varepsilon\), play the arm with the highest \(\hat{\mu}_a\). With probability \(\varepsilon\), play a uniform random arm.

$$
a_t = \begin{cases}
\arg\max_a \hat{\mu}_a & \text{with probability } 1 - \varepsilon \\
\text{Uniform}(\{1, \dots, K\}) & \text{with probability } \varepsilon
\end{cases}
$$

The regret is \(O(\varepsilon T)\). It is linear because the exploration does not adapt. At round \(T = 10^6\), we still spend \(\varepsilon \cdot 10^6\) pulls on random arms, even if we identified the best arm at round 1,000. The exploration is blind : it allocates the same probability to an arm we pulled 50,000 times and an arm we pulled 5 times.

Most DSPs start here. Set \(\varepsilon = 0.05\), reserve 5% of traffic for random bids. It breaks the loop. But the linear regret means we keep paying for exploration long after it has stopped being useful.

### Upper Confidence Bound (UCB)

The waste in \(\varepsilon\)-greedy is that it explores uniformly. UCB allocates exploration to where the uncertainty is highest.

By Hoeffding's inequality, the true mean \(\mu_a\) satisfies :

$$
P\!\left(\mu_a > \hat{\mu}_a + \sqrt{\frac{2\ln t}{N_a(t)}}\right) \leq t^{-4}
$$

where \(N_a(t)\) is the number of times arm \(a\) has been played. The quantity \(\hat{\mu}_a + \sqrt{2\ln t / N_a(t)}\) is an upper confidence bound on \(\mu_a\). UCB plays the arm with the highest upper bound :

$$
a_t = \arg\max_a \left[\hat{\mu}_a + \sqrt{\frac{2\ln t}{N_a(t)}}\right]
$$

When \(N_a\) is small, the bonus is large : the arm could plausibly be the best, so we try it. When \(N_a\) is large, the bonus shrinks and the decision is driven by \(\hat{\mu}_a\) alone. Exploration concentrates on arms where data is scarce. Arms that are clearly suboptimal, even accounting for uncertainty, are not explored.

The regret is \(O(\sqrt{KT \log T})\). The improvement over \(\varepsilon\)-greedy is not a constant factor. For a DSP choosing among \(K = 20\) creatives over \(T = 10^6\) impressions, \(\varepsilon\)-greedy with \(\varepsilon = 0.05\) has regret proportional to 50,000. UCB has regret proportional to \(\sqrt{20 \times 10^6 \times 14} \approx 16,700\). The gap grows with \(T\).

### Thompson Sampling

UCB constructs a confidence bound and acts on it. Thompson Sampling maintains a full posterior distribution over each arm's reward and acts on samples.

For binary rewards (click / no click), the conjugate prior is Beta. After observing \(s_a\) successes and \(f_a\) failures on arm \(a\), the posterior is \(\text{Beta}(\alpha_a, \beta_a)\) with \(\alpha_a = 1 + s_a\), \(\beta_a = 1 + f_a\).

The algorithm :

1. For each arm \(a\), sample \(\theta_a \sim \text{Beta}(\alpha_a, \beta_a)\).
2. Play \(a_t = \arg\max_a \theta_a\).
3. Observe reward. Update : \(\alpha_{a_t} \mathrel{+}= r_t\), \(\beta_{a_t} \mathrel{+}= (1 - r_t)\).

An arm with few observations has a wide posterior. Its sample is sometimes high, sometimes low. It gets explored occasionally. An arm with many observations has a concentrated posterior. Its sample is close to \(\hat{\mu}_a\) every time. The probability that arm \(a\) is played at round \(t\) is :

$$
P(a_t = a) = \mathbb{E}\!\left[\prod_{a' \neq a} F_{a'}(\theta_a)\right]
$$

where \(F_{a'}\) is the CDF of the posterior of arm \(a'\). This probability is high when \(a\) is likely to be optimal given the current data. Thompson Sampling explores in proportion to the posterior probability of optimality. This is a stronger property than UCB's uniform confidence bound.

The regret bound is the same : \(O(\sqrt{KT \log T})\). In practice, Thompson Sampling often outperforms UCB because it adapts faster when one arm is clearly dominant. It also has no hyperparameter beyond the prior, which for binary rewards is typically \(\text{Beta}(1, 1)\) (uniform).

### Summary

| | \(\varepsilon\)-greedy | UCB | Thompson Sampling |
|---|:---:|:---:|:---:|
| Regret | \(O(\varepsilon T)\) | \(O(\sqrt{KT \log T})\) | \(O(\sqrt{KT \log T})\) |
| Exploration | Uniform random | Confidence-driven | Posterior sampling |
| Adapts to data | No | Yes | Yes |
| Parameters | \(\varepsilon\) | None | Prior |

## From Bandits to the Bidding System

### Contextual bandits

The basic bandit assumes the reward of each arm is fixed. In a DSP, the reward depends on context : the same creative has different CTR on a sports site and a cooking blog. The problem becomes a **contextual bandit**. At each round, we observe features \(x\), choose action \(a\), and observe reward \(r\) with \(\mathbb{E}[r \mid x, a] = \mu(x, a)\).

This connects directly to the CTR model from Chapter 2. Replace the fixed \(\hat{\mu}_a\) with the model's prediction \(\hat{\mu}(x, a)\). The UCB rule becomes :

$$
a_t = \arg\max_a \left[\hat{\mu}(x_t, a) + \beta\;\hat{\sigma}(x_t, a)\right]
$$

where \(\hat{\sigma}(x, a)\) is the model's predictive uncertainty. For Thompson Sampling, we need the model to produce a posterior distribution over \(\mu(x, a)\), not just a point estimate.

This has a direct consequence on model architecture. A gradient-boosted tree outputs a scalar. It does not give uncertainty. A Bayesian model (variational inference, MC dropout, or an ensemble with calibrated variance) gives both a prediction and a distribution. Some DSPs run a separate model for uncertainty alongside the main CTR model. Others redesign the CTR model to output distributional predictions.

### Creative optimisation with Thompson Sampling

The most common bandit application in a DSP is creative selection. An advertiser uploads \(K\) creatives. For each impression, the system picks one.

With Thompson Sampling, each creative \(a\) has a posterior \(\text{Beta}(\alpha_a, \beta_a)\) updated in real time as clicks arrive. After 200 impressions, creative A has 4 clicks : posterior \(\text{Beta}(5, 197)\), mean 0.025. Creative B has 0 clicks in 50 impressions : posterior \(\text{Beta}(1, 51)\), mean 0.019. The posteriors overlap. B still gets sampled above A roughly 30% of the time. After 2,000 impressions, if A has 50 clicks and B has 2, the posteriors separate. B gets sampled above A less than 1% of the time. The system has converged to A, having spent a controlled amount of budget on B to confirm it is worse.

### Audience exploration

The same framework applies to audience segments. Instead of creatives, the arms are user segments defined by recency, demographics, or intent signals. Thompson Sampling on segment-level bid multipliers allocates exploration budget toward under-observed segments. The posterior width determines how much exploration each segment receives : segments with thousands of observations get almost none ; segments with dozens get significant exploration.

The feedback loop from the beginning of this chapter is strongest on audience segments. Retargeted users accumulate data quickly because the model bids on them aggressively. Prospecting users stay data-poor because the model bids on them rarely. Thompson Sampling corrects this asymmetry through the posterior : the wide posterior on prospecting users generates occasional high samples that trigger exploration bids.

### Correcting the training data

The bandit algorithms above change how the DSP selects traffic. A complementary approach changes how the model learns from biased data.

**Propensity logging.** For every bid request, we log the probability that the model bids on it : \(\pi(x) = P(\text{bid} \mid x)\). This is the **propensity score**. It must be logged at serving time, before the outcome is known.

**Inverse propensity weighting (IPW).** When retraining the model, we weight each example by \(1/\pi(x_i)\). The weighted loss becomes :

$$
\mathcal{L}_{\text{IPW}}(\theta) = -\frac{1}{n}\sum_{i=1}^{n} \frac{1}{\pi(x_i)} \bigl[y_i \log \hat{\mu}_\theta(x_i) + (1 - y_i)\log(1 - \hat{\mu}_\theta(x_i))\bigr]
$$

This is the standard importance sampling correction. Under the biased sampling, \(\mathbb{E}_{P_{\text{train}}}[1/\pi(x) \cdot \ell(x)] = \mathbb{E}_{P_X}[\ell(x)]\). The weighted loss is an unbiased estimate of the loss under the full population \(P_X\).

The correction has high variance when \(\pi(x)\) is small (rare bids get large weights). In practice, we clip the weights : \(w_i = \min(1/\pi(x_i), \; C)\) for some cap \(C\). This introduces bias but reduces variance. The bias-variance trade-off is controlled by \(C\).

**Counterfactual policy evaluation.** Propensity scores also allow offline evaluation of new bidding strategies. Given a new policy \(\pi'\) and historical data logged under policy \(\pi\), the importance-weighted estimator of the new policy's value is :

$$
\hat{V}(\pi') = \frac{1}{n}\sum_{i=1}^{n} \frac{\pi'(a_i \mid x_i)}{\pi(a_i \mid x_i)} \cdot r_i
$$

This allows testing whether a new exploration strategy or a new model would improve revenue, without deploying it. The estimator has high variance when \(\pi'\) and \(\pi\) are very different, because the ratio \(\pi'/\pi\) can be large. Doubly robust estimators combine IPW with a reward model to reduce this variance, but the core idea is the same.

## The Exploration Budget and the Bid

In Chapters 1 and 2, the bid was :

$$
b^* = v \cdot \hat{\mu}(x) - \frac{F_M(b^*)}{f_M(b^*)}
$$

When we add exploration, the system sometimes bids on impressions where \(\hat{\mu}(x)\) is low or uncertain. If the bid wins, we pay for an impression we do not expect to convert. The short-term expected utility of an exploration bid is negative.

The total exploration cost over a period is :

$$
C_{\text{explore}} = \sum_{i \in \mathcal{E}} b_i \cdot \mathbf{1}_{\{b_i > M_i\}}
$$

where \(\mathcal{E}\) is the set of exploration bids. This cost must be offset by the long-term information gain : better estimates of \(\mu(x)\) on previously unexplored segments, leading to better bids in the future.

In practice, DSPs set the exploration budget between 5% and 10% of campaign spend. With Thompson Sampling, the budget is implicit : it depends on the posterior uncertainty, so campaigns with more uncertainty explore more. With \(\varepsilon\)-greedy, the budget is explicit : \(\varepsilon\) fraction of impressions go to random arms.

## Key Takeaways

1. The training distribution \(P_{\text{train}}\) is not \(P_X\). It is \(P_{X \mid \text{bid}}\), biased by the model's own decisions. Standard metrics computed on \(P_{\text{train}}\) do not detect this.
2. The multi-armed bandit formalises the exploration-exploitation trade-off. Regret \(R_T\) measures the total cost of not knowing which arm is best.
3. \(\varepsilon\)-greedy has linear regret. UCB and Thompson Sampling have \(O(\sqrt{KT \log T})\) regret. The difference is structural, not marginal.
4. In the contextual setting, the CTR model from Chapter 2 must provide uncertainty estimates for UCB or posterior distributions for Thompson Sampling. This constrains model architecture.
5. IPW corrects the selection bias in training data. The weighted loss \(\mathcal{L}_{\text{IPW}}\) is an unbiased estimate of the loss under \(P_X\). Counterfactual evaluation uses propensity scores to test new strategies offline.
6. Exploration has a measurable short-term cost. The ML engineer must budget it, track it, and show that the system converges to better long-term performance.
