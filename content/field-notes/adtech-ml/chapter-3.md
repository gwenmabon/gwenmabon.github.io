---
title: "Explore or Exploit"
weight: 30
math: true

---

In Chapters 1 and 2, we derived the optimal bid as a function of \(\mu(x) = \mathbb{E}[Y \mid X = x]\) and showed that the loss function for estimating \(\mu(x)\) is the log-loss, the natural likelihood of a Bernoulli problem. We assumed one thing without stating it : that the training data is drawn from the same distribution as the production traffic.

This assumption fails in practice. The model is trained on outcomes of impressions it won and never observes what happens on traffic it lost. The training distribution \(P_{\text{train}}\) is \(P_{X \mid \text{win}}\), the distribution conditional on the model having won the auction, not \(P_X\), the marginal over all bid requests. The rest of this chapter deals with this mismatch.

## Selection Bias in Training Data

### The retraining loop

A DSP bids on every request it receives, but it only wins a fraction of them. Assume it retrains its CTR model daily :

1. Day \(t\) : model \(M_t\) scores bid requests and submits a bid for each one.
2. Outcomes (click or no click) are observed only for won impressions.
3. Day \(t{+}1\) : model \(M_{t+1}\) is trained on those outcomes.

The training set of \(M_{t+1}\) is a function of \(M_t\). Impressions that \(M_t\) lost have no outcome label and never enter the loss function, so the model cannot learn from traffic it did not win.

Write \(\pi_t(x) = P(\text{win} \mid x, M_t)\) for the probability that model \(M_t\) wins impression \(x\). This depends on the bid amount : a low \(\hat{\mu}(x)\) produces a low bid, which wins less often. The training distribution at day \(t+1\) is :

$$
P_{\text{train}}^{(t+1)}(x) \propto \pi_t(x) \cdot P_X(x)
$$

Segments where \(\pi_t(x)\) is small are under-represented. Segments where the bid is consistently below the market price have \(\pi_t(x) \approx 0\) and are nearly absent. The model minimises log-loss under \(P_{\text{train}}\), not under \(P_X\). It converges to the right \(\mu(x)\) on traffic it wins often, and to an arbitrary value on traffic it rarely wins.

Consider a creative B with \(\hat{\mu}_B = 0\) after 20 impressions : the posterior \(\text{Beta}(1, 21)\) has a mean of 0.045 but the bid formula uses the point estimate, so B gets a near-zero bid and stops winning. A publisher with 5 won impressions and 0 clicks has an estimated CTR of 0, but 5 observations cannot distinguish a true rate of 0% from 1%. Prospecting users accumulate dozens of labeled examples while retargeted users accumulate thousands, because the higher bids on retargeted users win more often. In each case, \(\pi_t(x) \approx 0\) and the segment disappears from the training data.

### The metrics cannot detect it

The test set has the same bias. It is sampled from the same \(P_{\text{train}}\) distribution. All the metrics we introduced in Chapter 2 (AUC, calibration error, Brier Score) are computed on the same biased dataset and only measure performance on traffic the model already wins. The model can have perfect calibration on observed traffic and be completely wrong everywhere else. This creates a feedback loop : the model's predictions determine the bids, the bids determine the win rate, the win rate determines the training data, and the training data determines the next predictions. At every step the metrics look correct because they only measure inside the loop.

### The cost of the feedback loop

The cost is pure opportunity cost, invisible in every dashboard. A segment with true CTR of 0.3% and a payout of 50 euros is worth \(0.003 \times 50 = 0.15\) euros per impression. At 100,000 daily impressions on that segment, that is 15,000 euros per day the DSP is not capturing, and the data to compute this loss was never collected.

To break the loop, the model must sometimes bid higher than \(\hat{\mu}(x)\) alone justifies, in order to win impressions on under-observed segments and gather new labels. The question is how much to overbid, and on which segments. This is the **exploration-exploitation trade-off**.

## Regret : the Cost of Not Knowing

At each impression, the DSP chooses an action \(a\) from a set of \(K\) options (which creative to show, which audience segment to target). Each action \(a \in \{1, \dots, K\}\) has an unknown mean reward \(\mu_a\). At round \(t\), we choose \(a_t\) and observe reward \(r_t\) with \(\mathbb{E}[r_t \mid a_t = a] = \mu_a\).

**Regret** measures the total cost of not knowing which action is best :

$$
R_T = T\,\mu^* - \sum_{t=1}^{T} \mu_{a_t}
$$

where \(\mu^* = \max_a \mu_a\). A pure exploitation strategy (always play the action with the highest \(\hat{\mu}\)) has zero exploration cost but may lock onto a suboptimal action forever. A pure exploration strategy (play uniformly at random) gathers information but wastes it on actions that are clearly bad.

What we want is sublinear regret : \(R_T = o(T)\), meaning the average cost per round \(R_T / T\) goes to zero as \(T\) grows. The algorithm eventually stops paying for what it learned. The theoretical lower bound is \(\Omega(\sqrt{KT})\). The algorithms below achieve \(O(\sqrt{KT \log T})\).

## Three Strategies

### \(\varepsilon\)-greedy

The simplest approach : with probability \(1 - \varepsilon\), play the action with the highest \(\hat{\mu}_a\) ; with probability \(\varepsilon\), play a uniform random action.

$$
a_t = \begin{cases}
\arg\max_a \hat{\mu}_a & \text{with probability } 1 - \varepsilon \\
\text{Uniform}(\{1, \dots, K\}) & \text{with probability } \varepsilon
\end{cases}
$$

The regret is \(O(\varepsilon T)\), linear, because the exploration does not adapt. At round \(T = 10^6\), we still spend \(\varepsilon \cdot 10^6\) pulls on random actions, even if we identified the best one at round 1,000. The exploration is blind : it allocates the same probability to an action we pulled 50,000 times and one we pulled 5 times.

Most DSPs start here : set \(\varepsilon = 0.05\), reserve 5% of traffic for random bids. This breaks the feedback loop, but the linear regret means the exploration cost stays constant even after the best action has been identified.

### Upper Confidence Bound (UCB)

The waste in \(\varepsilon\)-greedy is that it explores uniformly. UCB allocates exploration to where the uncertainty is highest.

By Hoeffding's inequality, the true mean \(\mu_a\) satisfies :

$$
P\!\left(\mu_a > \hat{\mu}_a + \sqrt{\frac{2\ln t}{N_a(t)}}\right) \leq t^{-4}
$$

where \(N_a(t)\) is the number of times action \(a\) has been played. The quantity \(\hat{\mu}_a + \sqrt{2\ln t / N_a(t)}\) is an upper confidence bound on \(\mu_a\). UCB plays the action with the highest upper bound :

$$
a_t = \arg\max_a \left[\hat{\mu}_a + \sqrt{\frac{2\ln t}{N_a(t)}}\right]
$$

When \(N_a\) is small, the bonus is large : the action could plausibly be the best, so we try it. When \(N_a\) is large, the bonus shrinks and the decision is driven by \(\hat{\mu}_a\) alone. Exploration concentrates on actions where data is scarce. Actions that are clearly suboptimal, even accounting for uncertainty, are not explored.

In a DSP, the reward depends on context : the same creative has different CTR on a sports site and a cooking blog. The fixed \(\hat{\mu}_a\) becomes the model's prediction \(\hat{\mu}(x, a)\) from Chapter 2, and the UCB rule becomes :

$$
a_t = \arg\max_a \left[\hat{\mu}(x_t, a) + \beta\;\hat{\sigma}(x_t, a)\right]
$$

where \(\hat{\sigma}(x, a)\) is the model's predictive uncertainty. This is a **contextual bandit** : at each round, we observe features \(x\), choose action \(a\), and observe reward \(r\) with \(\mathbb{E}[r \mid x, a] = \mu(x, a)\).

The regret is \(O(\sqrt{KT \log T})\). For a DSP choosing among \(K = 20\) creatives over \(T = 10^6\) impressions, \(\varepsilon\)-greedy with \(\varepsilon = 0.05\) has regret proportional to 50,000. UCB has regret proportional to \(\sqrt{20 \times 10^6 \times 14} \approx 16,700\). The gap grows with \(T\).

### Thompson Sampling

UCB relies on a worst-case bound and treats every action with low data the same way, regardless of how likely that action is to actually be the best. Thompson Sampling maintains a full posterior distribution over each action's reward and acts on samples.

For binary rewards (click / no click), the conjugate prior is Beta. After observing \(s_a\) successes and \(f_a\) failures on action \(a\), the posterior is \(\text{Beta}(\alpha_a, \beta_a)\) with \(\alpha_a = 1 + s_a\), \(\beta_a = 1 + f_a\).

The algorithm :

1. For each action \(a\), sample \(\theta_a \sim \text{Beta}(\alpha_a, \beta_a)\).
2. Play \(a_t = \arg\max_a \theta_a\).
3. Observe reward. Update : \(\alpha_{a_t} \mathrel{+}= r_t\), \(\beta_{a_t} \mathrel{+}= (1 - r_t)\).

An action with few observations has a wide posterior ; its sample is sometimes high, sometimes low, so it gets explored occasionally. An action with many observations has a concentrated posterior ; its sample is close to \(\hat{\mu}_a\) every time. The probability that action \(a\) is played at round \(t\) is :

$$
P(a_t = a) = \mathbb{E}\!\left[\prod_{a^{\prime} \ne a} F_{a^{\prime}}(\theta_a)\right]
$$

where \(F_{a^{\prime}}\) is the CDF of the posterior of action \(a^{\prime}\). Thompson Sampling explores in proportion to the posterior probability of optimality.

Applied to creative selection : an advertiser uploads \(K\) creatives. After 200 impressions, creative A has 4 clicks, giving posterior \(\text{Beta}(5, 197)\) with mean 0.025. Creative B has 0 clicks in 50 impressions, giving posterior \(\text{Beta}(1, 51)\) with mean 0.019. The posteriors overlap ; B still gets sampled above A roughly 30% of the time. After 2,000 impressions, if A has 50 clicks and B has 2, the posteriors separate and B gets sampled above A less than 1% of the time. The system has converged to A, having spent a controlled amount of budget on B to confirm it is worse.

The regret bound is the same as UCB : \(O(\sqrt{KT \log T})\). In practice, Thompson Sampling often outperforms UCB because it adapts faster when one action is clearly dominant. It also has no hyperparameter beyond the prior, which for binary rewards is typically \(\text{Beta}(1, 1)\) (uniform).

### Summary

| | \(\varepsilon\)-greedy | UCB | Thompson Sampling |
|---|:---:|:---:|:---:|
| Regret | \(O(\varepsilon T)\) | \(O(\sqrt{KT \log T})\) | \(O(\sqrt{KT \log T})\) |
| Exploration | Uniform random | Confidence-driven | Posterior sampling |
| Adapts to data | No | Yes | Yes |
| Parameters | \(\varepsilon\) | None | Prior |

## Model Requirements

Both UCB and Thompson Sampling require the model to produce uncertainty estimates, not just a point prediction. UCB needs \(\hat{\sigma}(x, a)\) ; Thompson Sampling needs a posterior distribution over \(\mu(x, a)\). The gradient-boosted trees from Chapter 2 output a scalar \(\hat{\mu}(x, a)\) with no measure of confidence. Bayesian approaches (variational inference, MC dropout, or ensembles with calibrated variance) give both a prediction and a distribution. In practice, some DSPs run a separate model for uncertainty alongside the main CTR model ; others redesign the CTR model to output distributional predictions.

The feedback loop from the beginning of this chapter is strongest on audience segments. Retargeted users accumulate data quickly because the model bids high and wins often. Prospecting users stay data-poor because the model bids low and wins rarely. Thompson Sampling corrects this asymmetry through the posterior : the wide posterior on prospecting users generates occasional high samples that trigger exploration bids.

## Correcting the Training Data

The strategies above change how the DSP allocates bids. A complementary approach changes how the model learns from biased data.

**Propensity logging.** For every bid request, we log the probability that the model wins it : \(\pi(x) = P(\text{win} \mid x)\). This is the **propensity score**. In practice, \(\pi(x)\) is a function of the bid amount and the market price distribution. It must be logged at serving time, before the outcome is known.

**Inverse propensity weighting (IPW).** When retraining the model, we weight each example by \(1/\pi(x_i)\). The weighted loss becomes :

$$
\mathcal{L}_{\text{IPW}}(\theta) = -\frac{1}{n}\sum_{i=1}^{n} \frac{1}{\pi(x_i)} \bigl[y_i \log \hat{\mu}_\theta(x_i) + (1 - y_i)\log(1 - \hat{\mu}_\theta(x_i))\bigr]
$$

This is the standard importance sampling correction. Under the biased sampling, \(\mathbb{E}_{P_{\text{train}}}[1/\pi(x) \cdot \ell(x)] = \mathbb{E}_{P_X}[\ell(x)]\). The weighted loss is an unbiased estimate of the loss under the full population \(P_X\).

The correction has high variance when \(\pi(x)\) is small (rarely won segments get large weights). In practice, we clip the weights : \(w_i = \min(1/\pi(x_i), \; C)\) for some cap \(C\). This introduces bias but reduces variance. The bias-variance trade-off is controlled by \(C\).

**Counterfactual policy evaluation.** Propensity scores also allow evaluating a new bidding strategy before deploying it. Given a new policy \(\pi'\) and historical data logged under policy \(\pi\), the importance-weighted estimator of the new policy's value is :

$$
\hat{V}(\pi') = \frac{1}{n}\sum_{i=1}^{n} \frac{\pi'(a_i \mid x_i)}{\pi(a_i \mid x_i)} \cdot r_i
$$

This allows testing whether a new exploration strategy or a new model would improve revenue, without deploying it. The estimator has high variance when \(\pi'\) and \(\pi\) are very different, because the ratio \(\pi'/\pi\) can be large. Doubly robust estimators combine IPW with a reward model to reduce this variance.

## The Exploration Budget and the Bid

In Chapters 1 and 2, the bid was :

$$
b^* = v \cdot \hat{\mu}(x) - \frac{F(b^*)}{f(b^*)}
$$

where \(F\) and \(f\) are the CDF and density of the highest competing bid.

When we add exploration, the system sometimes bids on impressions where \(\hat{\mu}(x)\) is low or uncertain. If the bid wins, we pay for an impression we do not expect to convert. The short-term expected utility of an exploration bid is negative.

The total exploration cost over a period is :

$$
C_{\text{explore}} = \sum_{i \in \mathcal{E}} b_i \cdot \mathbf{1}_{\{b_i > M_i\}}
$$

where \(\mathcal{E}\) is the set of exploration bids. This cost must be offset by the long-term information gain : better estimates of \(\mu(x)\) on previously unexplored segments, leading to better bids in the future.

In practice, DSPs set the exploration budget between 5% and 10% of campaign spend. With Thompson Sampling, the budget is implicit : it depends on the posterior uncertainty, so campaigns with more uncertainty explore more. With \(\varepsilon\)-greedy, the budget is explicit : \(\varepsilon\) fraction of impressions go to random actions.

## Key Takeaways

1. The training distribution \(P_{\text{train}}\) is \(P_{X \mid \text{win}}\), biased by the model's own bid amounts, and standard metrics computed on it do not detect this.
2. The feedback loop (predictions \(\to\) bids \(\to\) win rate \(\to\) training data \(\to\) predictions) makes the bias self-reinforcing.
3. Regret \(R_T\) measures the total cost of not knowing which action is best. \(\varepsilon\)-greedy has linear regret \(O(\varepsilon T)\) while UCB and Thompson Sampling achieve \(O(\sqrt{KT \log T})\).
4. UCB and Thompson Sampling require the CTR model to produce uncertainty estimates, which rules out standard gradient-boosted trees.
5. IPW corrects the selection bias in training data ; the weighted loss \(\mathcal{L}_{\text{IPW}}\) is an unbiased estimate of the loss under \(P_X\). Counterfactual evaluation uses propensity scores to test new strategies offline.
6. Exploration has a measurable short-term cost that the ML engineer must budget and track.
