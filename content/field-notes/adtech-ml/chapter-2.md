---
title: "The Price of a Probability"
weight: 20
math: true
draft: true
---

In Chapter 1, we derived the optimal bid as a function of \(\mu(x) = \mathbb{E}[Y \mid X = x]\), the expected value of a target event given the impression context. We left \(\mu(x)\) abstract. This chapter makes it concrete : what exactly do we estimate, what does the data look like, and what properties must the estimate satisfy for the bid to be correct ?

## What We Estimate

### CTR and CVR

The target event \(Y\) depends on the advertiser's campaign objective. Two quantities dominate :

- **Click-through rate (CTR)** : the probability that a user clicks on the ad denoted \(p(\text{click} \mid x)\).
- **Conversion rate (CVR)** : the probability that a user converts (purchases, installs, signs up) denoted \(p(\text{conversion} \mid x)\).

Here there are two ways of modeling the problem. The first one is to consider that conversion happens after an impression. Indeed, a user may see an ad, convert later through another channel. This is called *view-through conversions* or post-impression conversions. The second one is considering that a conversion is posterior to a click. We model a particular funnel: serve an impression, click on the ad then convert. The dominant signal in most campaigns remains the post-click modeling. You can consider that a click better demonstrates the impact of the ad leading to a conversion. In practice, most DSPs decompose the conversion probability along this path :

$$
p(\text{conversion} \mid x) = p(\text{click} \mid x) \cdot p(\text{conversion} \mid \text{click}, x)
$$

The two factors are estimated by separate models, trained on different datasets and at different scales. The CTR model sees every impression. The post-click conversion model sees only clicks. This factorisation is not strictly necessary, but it is practical : training a single end-to-end model directly on \(p(\text{conversion} \mid x)\) is possible in principle, but conversion labels are so sparse that the signal-to-noise ratio makes learning difficult. Decomposing the problem gives each model a richer supervision signal at its own stage of the funnel.

### The Class Imbalance Reality

Clicks and conversions are extremely rare events. Typical orders of magnitude :

| Event | Rate | Positive examples per million impressions |
|-------|:----:|:-----------------------------------------:|
| Click | 0.1 -- 0.3% | 1,000 -- 3,000 |
| Post-click conversion | 1 -- 5% of clicks | 10 -- 150 |
| View-through conversion | 0.001 -- 0.01% | 10 -- 100 |

The CTR model predicts probabilities around 0.1 -- 0.3%. The end-to-end conversion probability \(p(\text{conversion} \mid x)\) is often around 0.001 -- 0.01%. This extreme imbalance has direct consequences :

- **Evaluation** : accuracy is meaningless here. A model that always predicts 0 achieves 99.9% accuracy. We need metrics that are sensitive to the positive class.
- **Calibration** : a small absolute error on a small probability is a large relative error. If the true conversion rate is 0.01% and the model predicts 0.02%, the prediction is off by 100%.
- **Training** : gradient updates are dominated by the negative class. Techniques like negative downsampling are common, but they shift the predicted probabilities and require post-hoc correction.

### The Quantity That Enters the Bid

In the bid formula from Chapter 1, \(\mu(x)\) is the probability of the advertiser's target event. Then if we are running a campaign for a client who wants us to generate conversions, we will compute:

$$
\hat{\mu}(x) = \hat{p}(\text{click} \mid x) \cdot \hat{p}(\text{conversion} \mid \text{click}, x)
$$

Ultimately, $\hat{\mu}(x)$ is estimated via the product of two different models. Then if either model is miscalibrated, the bid we offer will be off. The rest of this chapter focuses on what calibration means, how to measure it, and how to enforce it.

## Why Calibration Matters More Than AUC

In most ML applications, we care about *ranking* : which user is more likely to convert than which other. The metric for ranking quality is AUC. In adtech, ranking is not enough. The predicted probability enters the bid as a multiplicative factor, so we need its *value* to be correct, not just its *order*.

In a first-price auction :

$$
b^{*} = v\,\hat{\mu}(x) - \frac{F(b^{*})}{f(b^{*})}
$$

where \(F\) and \(f\) are the CDF and density of the highest competing bid.

Since \(\hat{\mu}(x)\) is the product of two independently estimated quantities, calibration errors in either factor propagate directly into the bid. Suppose the model is multiplicatively miscalibrated : \(\hat{\mu}(x) = (1+\epsilon)\,\mu(x)\) for some \(\epsilon > 0\). The bid becomes :

$$
b^{*}_{\text{miscal}} \approx v(1+\epsilon)\,\mu(x) - \frac{F(b^{*})}{f(b^{*})}
$$

The shading term is unchanged because it depends only on the market. The overbid is approximately \(v\epsilon\,\mu(x)\) per impression. At scale, this compounds : a 20% calibration error (\(\epsilon = 0.2\)) on a billion daily auctions directly erodes margin. Good AUC with bad calibration means **correct ranking but wrong prices**. We win the right impressions but pay too much.

### A Concrete Example

Consider two models scoring 10,000 impressions :

| Model | AUC | Calibration Error | Profit |
|-------|:---:|:-----------------:|-------:|
| Model A (well-calibrated) | 0.78 | 0.02 | +$12,400 |
| Model B (miscalibrated) | 0.82 | 0.15 | -$3,200 |

Model B has better discrimination but loses money because its probability
estimates are off.

## Measuring Calibration

We showed that miscalibration directly erodes margin. To control it, we need a metric that isolates calibration quality from ranking quality.

### Expected Calibration Error (ECE)

The most direct approach is to partition predictions into \(K\) bins by predicted probability. In bin \(k\), let \(n_k\) be the count, \(\bar{p}_k\) the mean prediction, and \(\bar{y}_k\) the observed frequency. The Expected Calibration Error is the weighted average gap :

$$
\text{ECE} = \sum_{k=1}^{K} \frac{n_k}{n} \lvert \bar{p}_k - \bar{y}_k \rvert
$$

If the model predicts 30% in a bin where the true conversion rate is 25%, that bin contributes \(\lvert 0.30 - 0.25 \rvert = 0.05\) to the ECE, weighted by its share of the data. A perfectly calibrated model has \(\text{ECE} = 0\).

### Reliability Diagrams

A reliability diagram is the visual counterpart of ECE : it plots \(\bar{p}_k\) against \(\bar{y}_k\) for each bin. A perfectly calibrated model lies on the diagonal. The vertical distance from each point to the diagonal is the bin's contribution to calibration error.

![Reliability diagram comparing a calibrated model (blue circles, tracking the diagonal) with an over-confident model (pink squares, systematically below the diagonal).](/images/adtech-ml/reliability-diagram.svg)

### From ECE to the Brier Score

ECE measures calibration, but it says nothing about discrimination : a model that predicts \(\bar{y}\) for every impression has perfect calibration and zero usefulness. We need a metric that captures both.

The Brier Score is the mean squared error of probabilistic predictions :

$$
\text{BS} = \frac{1}{n}\sum_{i=1}^{n}(\hat{p}_i - y_i)^2
$$

where \(\hat{p}_i \in [0,1]\) is the predicted probability and \(y_i \in \{0,1\}\) is the outcome. Lower is better. Unlike ECE, it penalises both miscalibration and poor discrimination. Its decomposition shows exactly how.

### Brier Score Decomposition

Using the same binning, add and subtract \(\bar{y}_k\) inside the squared term :

$$
(\hat{p}_i - y_i)^2 = \bigl[(\hat{p}_i - \bar{y}_k) + (\bar{y}_k - y_i)\bigr]^2
$$

Expanding :

$$
= (\hat{p}_i - \bar{y}_k)^2 + 2(\hat{p}_i - \bar{y}_k)(\bar{y}_k - y_i) + (\bar{y}_k - y_i)^2
$$

Sum over all \(i\) in bin \(k\). The cross-term vanishes because \(\bar{y}_k\) is the bin mean of \(y_i\). Sum over all bins and divide by \(n\) :

$$
\text{BS} = \underbrace{\frac{1}{n}\sum_{k=1}^{K} n_k(\bar{p}_k - \bar{y}_k)^2}_{\text{REL (calibration)}} + \underbrace{\frac{1}{n}\sum_{k=1}^{K} n_k\,\bar{y}_k(1-\bar{y}_k)}_{\text{within-bin variance}}
$$

The within-bin variance can itself be split using the same add-and-subtract trick on \(\bar{y}\). Writing \(\bar{y}\) for the overall base rate :

$$
\text{BS} = \text{REL} - \text{RES} + \text{UNC}
$$

where :

$$
\text{REL} = \frac{1}{n}\sum_{k=1}^{K} n_k(\bar{p}_k - \bar{y}_k)^2
$$

$$
\text{RES} = \frac{1}{n}\sum_{k=1}^{K} n_k(\bar{y}_k - \bar{y})^2
$$

$$
\text{UNC} = \bar{y}(1-\bar{y})
$$

This is the same structure as a bias-variance decomposition :

- **REL** (reliability) is the squared version of ECE. It measures calibration error. Lower is better.
- **RES** (resolution) measures how well the model separates groups with different base rates. Higher is better (it is subtracted). The constant model that always predicts \(\bar{y}\) has \(\text{RES} = 0\).
- **UNC** (uncertainty) is the irreducible variance of the outcome. Fixed for a given dataset.

 This decomposition reveals that a good Brier Score requires *both* calibration (low REL) *and* discrimination (high RES). In adtech, we optimise REL first because it directly affects bid accuracy.

## The Log-Likelihood of a Bernoulli Problem

In Chapter 1, we established that the DSP's objective reduces to estimating \(\mu(x) = \mathbb{E}[Y \mid X = x]\), the probability of a target event given the impression context. In practice, the target event is a click (CTR estimation) or a conversion (CVR estimation). In both cases, the outcome is binary : \(Y \in \{0,1\}\). We are therefore in a classical Bernoulli estimation problem, and the natural approach is to write its likelihood.

We model \(Y\) as a Bernoulli random variable with parameter \(\mu(x)\). The likelihood of a single observation is :

$$
P(Y = y \mid x) = \mu(x)^y (1 - \mu(x))^{1-y}
$$

For \(n\) independent observations, the log-likelihood is :

$$
\ell(\theta) = \sum_{i=1}^{n} \bigl[ y_i \log \hat{\mu}_\theta(x_i) + (1 - y_i) \log(1 - \hat{\mu}_\theta(x_i)) \bigr]
$$

Maximising \(\ell(\theta)\) is equivalent to minimising the negative log-likelihood, which is exactly the **log-loss** :

$$
\mathcal{L}(\theta) = -\frac{1}{n} \sum_{i=1}^{n} \bigl[ y_i \log \hat{\mu}_\theta(x_i) + (1 - y_i) \log(1 - \hat{\mu}_\theta(x_i)) \bigr]
$$

When taking a parametric modeling approach, we find that the natural loss function of the problem is the log-loss. Therefore, any model that estimates \(\mu(x)\) should be trained by maximising this likelihood.

### Relationship to KL Divergence

To understand what the log-loss actually measures, let us decompose it. Write \(\mu(x) = p(Y=1 \mid x)\) for the true conditional probability. The per-observation log-loss is :

$$
-\bigl[y \log \hat{\mu}(x) + (1-y)\log(1-\hat{\mu}(x))\bigr]
$$

Now add and subtract the log of the true distribution :

$$
= -\bigl[y \log \mu(x) + (1-y)\log(1-\mu(x))\bigr] + \bigl[y \log\frac{\mu(x)}{\hat{\mu}(x)} + (1-y)\log\frac{1-\mu(x)}{1-\hat{\mu}(x)}\bigr]
$$

The first term is \(H(Y \mid x)\), the entropy of the true distribution — irreducible, independent of the model. The second term is exactly the Kullback-Leibler divergence \(D_{\text{KL}}(\mu \,\|\, \hat{\mu})\). In expectation :

$$
\mathcal{L} = H(Y) + D_{\text{KL}}(\mu \,\|\, \hat{\mu})
$$

Since \(H(Y)\) is fixed by the data, the only quantity the model can reduce is \(D_{\text{KL}}\). Minimising log-loss is equivalent to minimising the divergence between the true and estimated distributions. This places our training objective within the broader framework of information theory. A connection we will not explore here, but that runs through much of what follows on calibration.

## From Logistic Regression to Calibration

### Logistic Regression : Calibrated by Construction

The simplest parametric model for this problem is logistic regression. We model \(\hat{\mu}(x) = \sigma(w^\top x + b)\), where \(\sigma(z) = 1/(1+e^{-z})\) is the sigmoid function. The parameters \((w, b)\) are obtained by maximising the log-likelihood \(\ell\), or equivalently by minimising \(\mathcal{L}\). At convergence, the gradient of \(\mathcal{L}\) vanishes. In particular, the condition \(\partial \mathcal{L}/\partial b = 0\) yields :

$$
\frac{1}{n}\sum_{i=1}^{n} \hat{\mu}(x_i) = \frac{1}{n}\sum_{i=1}^{n} y_i
$$

The mean predicted probability equals the observed frequency. This is a necessary condition for calibration, and it holds by construction whenever the model is correctly specified. However, the model is linear in the features : the decision boundary \(w^\top x + b = 0\) is a hyperplane. For problems where the true \(\mu(x)\) is a nonlinear function of \(x\), logistic regression underfits.

### Complex Models and the Calibration Gap

To capture nonlinear patterns, we can use more expressive models : gradient-boosted trees, random forests, deep networks. These models still minimise the log-loss \(\mathcal{L}\), but they no longer satisfy the simple first-order condition above. Several mechanisms break calibration :

- **Regularisation** (L2 penalty, max depth, dropout) constrains the model away from the maximum likelihood solution. The resulting \(\hat{\mu}\) minimises a penalised objective, not the likelihood itself.
- **Early stopping** halts optimisation before convergence. The gradient of \(\mathcal{L}\) has not vanished, so the calibration condition does not hold.
- **Ensembling** (bagging, boosting) averages predictions from multiple models. Even if each individual model were calibrated, the average of calibrated models is not guaranteed to be calibrated.

In practice, these models achieve better discrimination than logistic regression (higher AUC) because they capture nonlinear structure in \(\mu(x)\). But their raw outputs are typically overconfident : high predictions are too high, low predictions are too low.

### Post-Hoc Recalibration

We saw that logistic regression is calibrated but underfits and complex models fit better but are miscalibrated. We face a classic trade-off of machine learning. The standard solution is then to separate the two problems. First, train a model that captures the complexity of the underlying problem. Then, freeze its parameters and learn a monotonic mapping from its raw scores to calibrated probabilities on a held-out calibration set. For this, there a 3 well known methods.

**Platt scaling.** Parametric approach. Fit a logistic regression on the model's raw scores :

$$
\hat{p}_{\text{cal}} = \sigma(a \cdot s + b)
$$

where \(s\) is the uncalibrated score and \(a, b\) are learned. This is a two-parameter recalibration that corrects global bias and scale.

**Isotonic regression.** Non-parametric approach. Fit a monotonically non-decreasing step function mapping raw scores to calibrated probabilities. More flexible than Platt scaling but requires more data and can overfit on small calibration sets.

**Temperature scaling.** For neural networks : divide the pre-activation output \(z = w^\top h + b\) (the logit) by a learned temperature \(T > 0\) :

$$
\hat{p}_{\text{cal}} = \sigma\!\left(\frac{z}{T}\right)
$$

\(T > 1\) softens overconfident predictions ; \(T < 1\) sharpens underconfident ones. This is Platt scaling with \(a = 1/T\) and \(b = 0\).

## Key Takeaways

1. In adtech, **calibration > discrimination** : a well-calibrated
   model with lower AUC beats a miscalibrated model with higher AUC.
2. The Brier Score decomposes into calibration (REL), resolution
   (RES), and uncertainty (UNC).
3. Log-loss is the natural loss function of a Bernoulli estimation
   problem. It incentivises the model to report true probabilities.
4. Logistic regression is calibrated by construction, but underfits.
   Complex models improve discrimination at the cost of calibration.
5. Post-hoc recalibration (Platt scaling, isotonic regression,
   temperature scaling) recovers calibration on a held-out set.
