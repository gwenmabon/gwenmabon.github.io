---
title: "Bid Shading"
weight: 40
math: true
draft: true
---

In Chapter 1 we derived the optimal bid in a first-price auction :

$$
b^* = v\,\mu(x) - \frac{F_M(b^*)}{f_M(b^*)}
$$

Chapters 2 and 3 addressed \(\mu(x)\) : how to estimate it, how to calibrate it, and how to protect its training data from feedback loops. The formula still contains one unknown : \(F_M\), the CDF of the highest competing bid. Without an estimate of \(F_M\), the formula is unusable. This chapter is about estimating it.

## What We Observe

After each auction, the DSP learns whether it won or lost.

- **Win** : the highest competing bid \(m\) was below our bid \(b\). Some exchanges report the clearing price, giving the exact value of \(m\). Others only confirm the win.
- **Loss** : the competing bid was above ours. We know \(m > b\) but not the value of \(m\).

We want to estimate \(F_M(b \mid x) = P(m \leq b \mid x)\), the probability of winning at bid level \(b\) given auction context \(x\). But for most auctions, we do not observe \(m\) directly. We observe a binary outcome at a single bid level.

A regression model trained to predict \(m\) would need the target variable to be observed. It is not. What we have is a threshold and an indicator : did the competing bid fall below the threshold ? This is **censored data**. The observation is incomplete, and the incompleteness is systematic : we censor more when we bid low, less when we bid high.

## The Censored Data Likelihood

The standard tool for incomplete observations is the likelihood, written carefully to account for what we know and what we do not know.

For each auction \(i\), we observe a triple \((t_i, x_i, \delta_i)\). When the exchange reports the clearing price and we win, \(t_i = m_i\) is the exact competing bid and \(\delta_i = 1\). When we lose, we know only \(m_i > b_i\), so \(t_i = b_i\) and \(\delta_i = 0\) : the observation is **right-censored** at our bid. The likelihood contribution of auction \(i\) is :

$$
L_i = f_M(t_i \mid x_i)^{\delta_i} \cdot S(t_i \mid x_i)^{1 - \delta_i}
$$

where \(S(t \mid x) = 1 - F_M(t \mid x)\) is the survival function : the probability that the competing bid exceeds \(t\). An uncensored observation contributes the density at \(t_i\). A censored observation contributes the probability that the event has not yet occurred by \(t_i\).

The full log-likelihood over \(n\) auctions is :

$$
\ell = \sum_{i=1}^{n} \bigl[\delta_i \log f_M(t_i \mid x_i) + (1 - \delta_i) \log S(t_i \mid x_i)\bigr]
$$

Compare this to the Bernoulli log-likelihood from Chapter 2 :

$$
\ell_{\text{Bernoulli}} = \sum_{i=1}^{n} \bigl[y_i \log \mu(x_i) + (1 - y_i) \log(1 - \mu(x_i))\bigr]
$$

The structure is identical. The Bernoulli likelihood estimates a probability from binary labels. The censored likelihood estimates a distribution from incomplete observations of a continuous variable. The indicator \(\delta_i\) plays the role of \(y_i\), the density plays the role of \(\mu\), and the survival function plays the role of \(1 - \mu\).

## The Survival Framework

To maximize \(\ell\), we need to model \(f_M\) and \(S\). These are not independent : they are two views of the same distribution, connected through the **hazard function** \(h\).

The hazard at bid level \(b\) is the instantaneous rate of "winning" given that the competing bid is at least \(b\) :

$$
h(b \mid x) = \frac{f_M(b \mid x)}{S(b \mid x)}
$$

It answers a precise question : among auctions where the competition is at least \(b\), how dense is the competition right at level \(b\) ? The hazard is the quantity that makes the framework tractable, because \(S\) and \(f_M\) can both be recovered from it :

$$
S(b \mid x) = \exp\!\left(-\int_0^b h(u \mid x)\,du\right), \qquad f_M(b \mid x) = h(b \mid x) \cdot S(b \mid x)
$$

Substituting into the log-likelihood, using \(\log f = \log h + \log S\) :

$$
\ell = \sum_{i=1}^{n} \left[\delta_i \log h(t_i \mid x_i) - \int_0^{t_i} h(u \mid x_i)\,du\right]
$$

The problem reduces to estimating \(h(b \mid x)\). This is where the Cox model enters.

## The Cox Model

### The proportional hazards assumption

The **Cox PH model** assumes that covariates act multiplicatively on the hazard :

$$
h(b \mid x) = h_0(b)\,\exp(\beta^\top x)
$$

The baseline hazard \(h_0(b)\) captures the overall shape of the competing-bid distribution. The exponential term shifts it up or down depending on the auction context \(x\). A premium publisher has more competition : \(\exp(\beta^\top x)\) is large, the hazard is higher at every bid level, and the win probability at any given bid is lower.

The log-likelihood becomes :

$$
\ell(\beta, h_0) = \sum_{i:\delta_i=1} \bigl[\log h_0(t_i) + \beta^\top x_i\bigr] - \sum_{i=1}^{n} \exp(\beta^\top x_i) \int_0^{t_i} h_0(u)\,du
$$

This depends on \(\beta\) (a finite-dimensional parameter) and \(h_0(\cdot)\) (a function, infinite-dimensional). Maximising jointly is hard. Cox's insight was to avoid it.

### The partial likelihood

Order the uncensored events by bid level : \(t_{(1)} < t_{(2)} < \cdots < t_{(D)}\). At each event time \(t_{(j)}\), define the **risk set** \(\mathcal{R}_j = \{i : t_i \geq t_{(j)}\}\), the set of auctions that could have produced a win at bid level \(t_{(j)}\). The probability that the observed winner is auction \((j)\), given that exactly one event occurred at \(t_{(j)}\), is :

$$
\frac{h(t_{(j)} \mid x_{(j)})}{\sum_{i \in \mathcal{R}_j} h(t_{(j)} \mid x_i)}
= \frac{h_0(t_{(j)})\,\exp(\beta^\top x_{(j)})}{\sum_{i \in \mathcal{R}_j} h_0(t_{(j)})\,\exp(\beta^\top x_i)}
= \frac{\exp(\beta^\top x_{(j)})}{\sum_{i \in \mathcal{R}_j} \exp(\beta^\top x_i)}
$$

The baseline hazard \(h_0(t_{(j)})\) cancels. The **partial likelihood** is the product of these terms over all events :

$$
L_P(\beta) = \prod_{j=1}^{D} \frac{\exp(\beta^\top x_{(j)})}{\sum_{i \in \mathcal{R}_j} \exp(\beta^\top x_i)}
$$

This depends on \(\beta\) alone. We estimate \(\beta\) by maximising \(\log L_P\), using standard gradient-based optimisation. The partial likelihood has the same structure as a softmax : at each event, the model assigns probabilities to all auctions in the risk set, and we maximise the probability of the observed outcome. Compare this to the cross-entropy loss from Chapter 2 : the mathematical structure of learning is the same. What changes is the data.

### Recovering the baseline

Once \(\hat{\beta}\) is estimated, the baseline hazard is recovered by the **Breslow estimator** :

$$
\hat{H}_0(b) = \sum_{j: t_{(j)} \leq b} \frac{1}{\sum_{i \in \mathcal{R}_j} \exp(\hat{\beta}^\top x_i)}
$$

where \(\hat{H}_0(b) = \int_0^b h_0(u)\,du\) is the cumulative baseline hazard. The baseline survival is \(S_0(b) = \exp(-\hat{H}_0(b))\). Putting it together :

$$
S(b \mid x) = S_0(b)^{\exp(\hat{\beta}^\top x)}
$$

$$
F_M(b \mid x) = 1 - S_0(b)^{\exp(\hat{\beta}^\top x)}
$$

$$
f_M(b \mid x) = h_0(b)\,\exp(\hat{\beta}^\top x)\,S_0(b)^{\exp(\hat{\beta}^\top x)}
$$

The shading term in the bid formula is now computable :

$$
\frac{F_M(b \mid x)}{f_M(b \mid x)} = \frac{1 - S_0(b)^{\exp(\hat{\beta}^\top x)}}{h_0(b)\,\exp(\hat{\beta}^\top x)\,S_0(b)^{\exp(\hat{\beta}^\top x)}}
$$

The model has two components : \(\hat{\beta}\) (which features shift the competition) and \(\hat{H}_0\) (the shape of the baseline distribution). The first is estimated by the partial likelihood, the second by the Breslow estimator. The separation is what makes the Cox model semi-parametric : we never assume a functional form for \(F_M\).

### What the features capture

The features \(x\) enter through the multiplicative term \(\exp(\beta^\top x)\). Each coefficient \(\beta_k\) measures how feature \(k\) shifts the hazard. A positive \(\beta_k\) means higher competition (harder to win at any bid level). A negative \(\beta_k\) means less competition.

| Feature | Sign of \(\beta\) | Effect on shading |
|---------|:-:|---|
| Premium publisher | \(+\) | More competition, shade less |
| Video format | \(+\) | Higher CPMs, shade less |
| Long-tail app | \(-\) | Less competition, shade more |
| Night hours | \(-\) | Fewer bidders, shade more |

A video ad on a premium publisher at 8pm has \(\exp(\beta^\top x)\) much higher than a banner on a long-tail app at 3am. The first impression has dense competition and little room to shade. The second has sparse competition and large potential savings.

## Sensitivity to Density Errors

The bid formula divides by \(f_M\). Errors in the density are amplified. We can quantify this.

Write \(g(b) = F_M(b)/f_M(b)\), the shading term. A first-order approximation of the bid error from an error \(\Delta f\) in the density estimate gives :

$$
\Delta b^* \approx \frac{F_M(b^*)}{f_M(b^*)^2}\,\Delta f
$$

The error is proportional to \(1/f_M^2\). Where competition is sparse (\(f_M\) small), the bid error is large.

### A concrete example

Take the simplest case : competing bids uniform on \([0, c]\). Then \(F_M(b) = b/c\), \(f_M(b) = 1/c\), and the shading term is \(F/f = b\). The optimal bid satisfies \(b^* = v\mu(x) - b^*\), giving :

$$
b^* = \frac{v\mu(x)}{2}
$$

An impression with \(v\mu(x) = 0.15\) euros and uniform competition on \([0, 0.20]\) euros has optimal bid \(b^* = 0.075\) euros. The win probability is \(F_M(0.075) = 37.5\%\). The expected surplus per auction :

$$
(0.15 - 0.075) \times 0.375 = 0.028\text{ euros}
$$

If the DSP bids 0.10 euros (too high) : surplus = \((0.15 - 0.10) \times 0.50 = 0.025\) euros. If the DSP bids 0.05 euros (too low) : surplus = \((0.15 - 0.05) \times 0.25 = 0.025\) euros. Both errors cost 0.003 euros per auction. At 10 million daily auctions, that is 30,000 euros per day.

The optimum is sharp, and the loss is symmetric around it. Overshading (bid too low) loses auctions. Undershading (bid too high) wins auctions at negative margin.

## Beyond Cox : The Direct Win-Rate Model

The proportional hazards assumption says that the ratio of hazards between two auction contexts is constant across all bid levels. This can be violated. If competition on publisher A is concentrated around 0.05 euros and on publisher B is spread between 0.01 and 0.20 euros, a multiplicative shift cannot capture the difference in shape.

An alternative is to skip the survival framework entirely and train a classifier directly on the binary outcome.

### The win-rate classifier

Train a model \(\hat{F}(b, x)\) on tuples \((b_i, x_i, \delta_i)\) where the bid \(b\) is a feature alongside the auction context \(x\), and the label \(\delta_i\) indicates a win. The model output is interpreted as \(F_M(b \mid x) = P(\text{win} \mid b, x)\).

This is appealing : it uses the same binary classification pipeline as the CTR model from Chapter 2, trained with the same log-loss. No survival analysis, no proportional hazards assumption. But the bid formula needs both \(F_M\) and \(f_M\). Recovering the density requires differentiating the model output with respect to \(b\) :

$$
\hat{f}_M(b \mid x) = \frac{\partial \hat{F}(b, x)}{\partial b}
$$

For a neural network, this is a gradient computation : tractable but noisy. For a gradient-boosted tree, the output is piecewise constant in \(b\), so the derivative is zero almost everywhere and undefined at the splits. Smoothing (kernel density estimation on the tree output) introduces a bandwidth hyperparameter and can bias the density.

### Comparison

| | Cox PH | Win-rate classifier |
|---|---|---|
| Assumption | Proportional hazards | None (nonparametric in \(b\)) |
| \(F_M(b \mid x)\) | Closed-form from \(S_0, \beta\) | Direct model output |
| \(f_M(b \mid x)\) | Closed-form from \(h_0, S_0, \beta\) | Requires \(\partial \hat{F}/\partial b\) |
| Interpretability | \(\beta\) coefficients | Black-box |
| Proportional hazards violated | Misspecified | Handles it |

In practice, many DSPs start with a Cox model (interpretable, stable density estimates) and move to a win-rate classifier as the system matures and the engineering team can handle the density estimation problem.

## Non-Stationarity and the Feedback Loop

### The competition shifts

Other DSPs update their models. New campaigns start. Seasonal patterns shift demand. The distribution \(F_M^{(t)}\) at time \(t\) is not the same as \(F_M^{(t+1)}\) at time \(t+1\). A shading model trained on Monday's data may already be miscalibrated by Wednesday.

The staleness problem is analogous to the feedback loop from Chapter 3, but for a different model. In Chapter 3, the CTR model's training data was biased by the model's own bid decisions (\(P_{\text{train}} \neq P_X\)). Here, the shading model's training data is biased by time : the distribution it learned on no longer matches the current market.

The fix is the same in spirit : monitor and retrain. Most DSPs retrain the shading model daily. The key diagnostic is the **win-rate residual** : for each context \(x\) and bid level \(b\), compare the predicted win probability \(\hat{F}_M(b \mid x)\) to the observed win rate. A systematic gap signals staleness.

### Selection bias in shading data

Chapter 3 showed that the CTR model only sees outcomes for impressions it chose to bid on. The same selection bias affects the shading model. The DSP only observes win/loss for auctions where it bid. If it systematically avoids a publisher (because \(\hat{\mu}(x)\) is low there), it collects no data on that publisher's competing-bid distribution. The shading model for that publisher is based on old data or nothing at all.

The exploration policy from Chapter 3 helps both models. Exploration bids generate data on unfamiliar traffic, which improves both the CTR estimate \(\hat{\mu}(x)\) and the competition estimate \(\hat{F}_M(b \mid x)\).

## Computing the Bid

The equation \(b^* = v\mu(x) - F_M(b^*)/f_M(b^*)\) is implicit : \(b^*\) appears on both sides. In production, the DSP solves it numerically for each bid request.

**Bisection** on the interval \([0, v\mu(x)]\) converges in \(\lceil\log_2(v\mu(x)/\epsilon)\rceil\) steps. For \(v\mu(x) = 0.15\) euros and \(\epsilon = 0.001\) euros, that is about 8 iterations. Each iteration evaluates \(F_M\) and \(f_M\) once, so the computational cost of shading is roughly 8 lookups in the survival model.

**Guardrails.** The computed bid is clipped to \([\text{floor}, v\mu(x)]\). The floor is the exchange minimum. The ceiling prevents bidding above the impression value, which would guarantee negative surplus. In practice, an additional cap at a fraction of \(v\mu(x)\) (e.g., 90%) protects against density estimation errors in the tail.

Advertisers have daily budgets. The budget constraint modifies the formula by discounting the value :

$$
b^* = \frac{v\mu(x)}{1 + \lambda} - \frac{F_M(b^*)}{f_M(b^*)}
$$

where \(\lambda \geq 0\) is set by the pacer. This is the subject of Chapter 5.

## Key Takeaways

1. Estimating \(F_M\) is a censored data problem. The censored likelihood has the same structure as the Bernoulli likelihood from Chapter 2, with the density and survival function replacing \(\mu\) and \(1-\mu\).
2. The Cox PH partial likelihood \(L_P(\beta)\) eliminates the baseline hazard \(h_0\), making the model semi-parametric. This is critical because the shape of the competing-bid distribution varies widely across contexts.
3. The bid formula divides by \(f_M\). Errors in the density estimate are amplified by \(1/f_M^2\) ; at 10M daily auctions, a suboptimal shade costs tens of thousands of euros per day.
4. The direct win-rate classifier avoids the proportional hazards assumption, but recovering \(f_M = \partial F/\partial b\) from a classifier is noisy.
5. The shading model suffers from the same feedback loop as the CTR model (Chapter 3) : the DSP only observes competition on traffic it bid on. Exploration helps both models.
