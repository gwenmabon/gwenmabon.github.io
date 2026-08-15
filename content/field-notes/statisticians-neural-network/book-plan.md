---
title: "Book Plan"
draft: true
math: true
---

## Thesis

Every DL concept has a statistical ancestor. The "innovation" is the parametrisation, not the mathematics. The reader arrives knowing stats and leaves seeing DL as a dialect of the same language.

The unifying thread is the **latent variable** \(h\) and what structure you impose on it. A neural network is a member of a very rich parametric family \(\{P_\theta, \theta \in \Theta\}\), the loss is the negative log-likelihood, backprop computes the score function \(\nabla_\theta \log \mathcal{L}\), and everything else (embeddings, attention, gates, dropout) is structure imposed on the sufficient statistic or the prior.

## The statistical toolkit

The entire book rests on first-year mathematical statistics :

- **Parametric families** \(\{P_\theta, \theta \in \Theta\}\) and **identifiability**
- **Sufficient statistics** (Fisher-Neyman factorisation theorem)
- **Maximum likelihood estimation** and its properties (consistency, asymptotic normality, efficiency via Cramer-Rao)
- **Exponential families** : natural parameters, canonical sufficient statistics, log-partition function
- **Bias-variance tradeoff** and risk decomposition
- **MAP estimation** as regularised MLE (link to priors)
- **State-space models**, Markov chains, filtering (Kalman)
- **EM algorithm** for latent variable models
- **Variational inference** as an alternative to EM when the E-step is intractable

## Where the translation is not perfect

The one place where DL genuinely goes beyond classical stat 1 : the parametric family is **non-convex** and **massively overparametrised**, so the classical asymptotics (Cramer-Rao, efficiency) don't directly apply. The implicit regularisation story (chapters 6 and 8) fills the gap. This must be acknowledged explicitly so the reader doesn't think the translation is perfect everywhere.

## Full inventory : DL notions to statistical translation

| DL concept | Statistical object |
|---|---|
| Neuron / activation | Generalised linear unit (link function + linear predictor) |
| MLP | Composed parametric family, hierarchical latent variables |
| Depth | Iterated reparametrisation of the sufficient statistic |
| Loss function (cross-entropy, MSE) | Negative log-likelihood under specific distributional assumptions |
| Softmax output | Categorical exponential family |
| Sigmoid output | Bernoulli exponential family (logistic regression) |
| Embedding layer | Learned sufficient statistic for categorical variables |
| Word2Vec / GloVe | Implicit factorisation of log co-occurrence matrix (PMI) |
| Convolution | Equivariant linear operator under translation group |
| Weight sharing | Structural prior (parameter tying = symmetry constraint) |
| Pooling | Sufficient statistic under invariance (e.g. max, mean) |
| RNN | Recursive sufficient statistic (deterministic state-space model) |
| LSTM gates | Adaptive update of temporal sufficient statistic |
| GRU | Simplified LSTM (same statistical role, fewer parameters) |
| Self-attention | Nadaraya-Watson kernel regression with learned kernel |
| Attention weights | Data-dependent categorical distribution (adaptive bandwidth) |
| Multi-head attention | Mixture of kernel estimators |
| Positional encoding | Injecting order structure into an exchangeable model |
| Transformer block | Composition : kernel regression then nonlinear projection |
| Backpropagation | Adjoint method (Pontryagin, 1956) / chain rule on computational graph |
| SGD | Robbins-Monro stochastic approximation (1951) |
| Adam / momentum | Adaptive preconditioning of the stochastic gradient |
| Learning rate schedule | Step-size control in stochastic approximation theory |
| Skip connection / ResNet | Euler discretisation of an ODE ; additive perturbation of identity |
| Neural ODE | Continuous-depth limit of ResNets (Chen et al., 2018) |
| Dropout | Monte Carlo integration over binary masks ; approximate variational inference (Gal & Ghahramani, 2016) |
| Weight decay (L2) | Gaussian prior on parameters (MAP estimation) |
| L1 regularisation | Laplace prior (MAP, sparsity) |
| Batch normalisation | Standardisation of activations ; reparametrisation that improves conditioning |
| Layer normalisation | Same, but per-example (no batch dependence) |
| Data augmentation | Encoding invariance priors into the training distribution |
| Early stopping | Implicit regularisation (equivalent to L2 in linear case) |
| GAN generator | Implicit density model (pushforward of a base measure) |
| GAN discriminator | Density ratio estimator |
| GAN training | Minimax on f-divergence (Nowozin et al., 2016) |
| VAE encoder | Amortised variational posterior |
| VAE decoder | Generative model / likelihood |
| ELBO | Variational lower bound on log-marginal likelihood |
| Normalizing flow | Diffeomorphism-based change of variables (transport of measure) |
| Diffusion model | Score matching + Langevin dynamics ; reverse-time SDE (Anderson, 1982) |
| Transfer learning | Empirical Bayes : pre-trained weights as informative prior |
| Fine-tuning | Posterior update from a warm start |
| LoRA / adapters | Low-rank perturbation of the prior mean |
| Contrastive loss (SimCLR, CLIP) | Noise-contrastive estimation (Gutmann & Hyvarinen, 2010) ; mutual information lower bound |
| Self-supervised pretext tasks | Learning sufficient statistics without labels |
| Knowledge distillation | Posterior compression / moment matching |
| Mixture of Experts | Mixture model with gating (Jacobs et al., 1991) |

## Chapter plan (9 chapters)

The logical flow follows the complexity of the latent variable \(h\) : from simple (deterministic, static) to complex (stochastic, transport). Chapters 6 and 8 are cross-cutting : they cover how you **find** and **control** the parameters.

Transfer learning fits as a section in Chapter 8 (the prior comes from pre-training). Contrastive learning fits in Chapter 9 (noise-contrastive estimation is a density ratio method, same family as GANs). MoE is a section in Chapter 1 (mixture of parametric families with gating).

### Chapter 1 : Neural Networks

*The feedforward network as a parametric family*

- The statistical problem : estimate \(P(Y \mid X)\)
- What a "model" means : a family \(\{P_\theta, \theta \in \Theta\}\)
- The neuron as a generalised linear unit (link function + linear predictor)
- Depth = composition of latent representations \(h_1, \dots, h_L\)
- The Dirac latent variable view : \(P(Y \mid X) = \int P(Y \mid h) \delta(h - f_\theta(X)) dh\)
- Why depth helps : reparametrisation factorises complex dependencies (Telgarsky, 2016 for depth separation results)
- Loss = negative log-likelihood. Cross-entropy for classification, MSE for Gaussian regression. Not a design choice : a consequence of the assumed output distribution.
- Universal approximation (Cybenko 1989, Hornik 1991) : what it says, and what it does **not** say (nothing about sample complexity or optimisation)

*"Aha" moment :* a neural network is just a very flexible parametric family. The innovation is the parametrisation, not the statistics.

### Chapter 2 : Embeddings

*Learned sufficient statistics for discrete variables*

- The problem : categorical variables with high cardinality. One-hot is a basis, but dimension = cardinality.
- Embedding = learned linear projection from one-hot to low-dimensional continuous space
- Stat interpretation : constructing a sufficient statistic for the downstream task, specific to each category
- Word2Vec as implicit matrix factorisation : Levy & Goldberg (2014) showed Skip-gram with negative sampling implicitly factorises the PMI matrix
- Connection to SVD / PCA : truncated SVD of the co-occurrence matrix gives similar representations (but not identical because of the PMI weighting)
- Why this matters : variance reduction. Projecting to \(\mathbb{R}^d\) with \(d \ll |\mathcal{X}|\) trades bias for variance in the downstream estimator.

*"Aha" moment :* Word2Vec is a matrix factorisation. The "semantic" structure in embedding space is a consequence of co-occurrence statistics, not "understanding."

### Chapter 3 : Convolutions (CNNs)

*Equivariance as structural prior*

- The problem : images have spatial structure. A fully connected layer ignores it (too many parameters, wrong inductive bias).
- Convolution = linear operator equivariant to translations. Formally : \(T_a \circ f = f \circ T_a\) for translation \(T_a\).
- Weight sharing : the same kernel applies at every location. This is a **parameter tying constraint**, equivalent to a prior that the relevant features do not depend on position.
- Pooling as sufficient statistic under invariance : max-pool / average-pool = extracting an invariant summary.
- Receptive field = range of dependencies captured. Deeper layers = longer-range dependencies (hierarchical composition).
- Group equivariance beyond translation : rotations (Cohen & Welling, 2016), scale. The general principle : if the data has a symmetry group, encode it.
- Connection to classical signal processing : convolution theorem, spectral methods.

*"Aha" moment :* a CNN does not "learn to see." It is a linear estimator constrained by translation equivariance, composed with nonlinearities. The architecture encodes a prior about spatial regularity.

### Chapter 4 : RNN and LSTM

*Dynamic sufficient statistics*

- The sequential problem : \(P(X_{1:T}) = \prod_t P(X_t \mid X_{1:t-1})\)
- The curse : conditioning on \(X_{1:t}\) has growing dimension
- RNN : compress the past into \(h_t = f_\theta(h_{t-1}, X_t)\). This is a **recursive sufficient statistic**.
- Stat parallel : deterministic state-space model. Compare to Kalman filter (linear Gaussian case), particle filter (nonlinear).
- Why vanilla RNN fails : vanishing/exploding gradients = instability of the dynamical system \(h_t\). The Jacobian \(\partial h_t / \partial h_{t-k}\) is a product of matrices ; spectral radius controls stability (Bengio et al., 1994 ; Pascanu et al., 2013).
- LSTM : structural fix. Cell state \(c_t\) has additive updates (identity + gated perturbation). This is the **same principle as ResNets** (Euler discretisation), applied to time instead of depth.
- Gates = adaptive weighting of past vs new information. Forget gate = exponential discounting with learned rate.
- GRU as simplified variant (Cho et al., 2014) : same statistical role, fewer gates.

*"Aha" moment :* the LSTM cell is a nonlinear recursive estimator of a temporal sufficient statistic with controlled forgetting. The gates are the mechanism that makes the dynamical system stable over long horizons.

### Chapter 5 : Attention and Transformers

*Kernel regression with learned kernels*

- Motivation : RNN compresses the past into a fixed-size vector. Information loss is inevitable for long sequences. What if we keep the entire sequence and let the model choose what to attend to ?
- Self-attention as Nadaraya-Watson : \(\text{Attn}(Q,K,V) = \text{softmax}(QK^\top / \sqrt{d}) V\). This is a kernel-weighted average of values, where the kernel is \(k(q, k) = \exp(q^\top k / \sqrt{d})\).
- Nadaraya-Watson (1964) : \(\hat{m}(x) = \sum_i K(x, x_i) y_i / \sum_i K(x, x_i)\). Exact same structure, but with a fixed kernel. Attention **learns** the kernel.
- Multi-head = mixture of kernel estimators, each capturing different dependency patterns.
- Positional encoding : without it, attention is permutation-invariant (treats the input as a set, not a sequence). Positional encoding breaks the exchangeability assumption.
- The \(1/\sqrt{d}\) scaling : variance control. Without it, dot products grow with dimension, softmax saturates, gradients vanish. This is **not** a trick : it is normalisation to keep the kernel well-conditioned.
- Transformer block = kernel regression (attention) + pointwise nonlinear projection (FFN). The FFN is a per-position MLP, same role as Chapter 1.
- Why Transformers replaced LSTMs : parallelism (no sequential bottleneck) + direct access to all past positions (no compression loss). The tradeoff : \(O(T^2)\) cost vs \(O(T)\) for RNN.

*"Aha" moment :* self-attention is Nadaraya-Watson with a learned kernel. The Transformer is not a fundamentally new object ; it is a composition of kernel regression and nonlinear projection.

### Chapter 6 : Backpropagation and SGD

*Optimisation as stochastic approximation*

- Backpropagation = the chain rule applied to a computational graph. Known as the **adjoint method** in optimal control (Pontryagin, 1956) and automatic differentiation in numerical analysis. Rumelhart, Hinton & Williams (1986) applied it to neural networks, but the maths predates them by decades.
- SGD = Robbins-Monro stochastic approximation (1951). Convergence conditions : \(\sum \eta_t = \infty\), \(\sum \eta_t^2 < \infty\).
- Momentum = heavy ball method (Polyak, 1964).
- Adam = adaptive preconditioning using running estimates of first and second moments. Connection to natural gradient (Amari, 1998) and Fisher information.
- Why SGD generalises : implicit regularisation. SGD favours flat minima (Keskar et al., 2017). Flat minima correspond to parameter regions where the loss is insensitive to perturbation, which correlates with generalisation (PAC-Bayes argument).
- Initialisation matters : Xavier/Glorot (2010), He (2015). The goal is to preserve the variance of activations through layers. Without proper init, the forward pass collapses or explodes before any gradient is computed.

*"Aha" moment :* SGD is not gradient descent with noise. The noise **is** the regulariser. Computer scientists named it "stochastic gradient descent" ; statisticians had called it "stochastic approximation" since 1951.

### Chapter 7 : Skip Connections and ResNets

*Euler discretisation and neural ODEs*

- The depth problem : very deep networks (50+ layers) degrade in performance, even on training data. Not overfitting ; optimisation failure.
- Skip connection : \(h_{l+1} = h_l + f_\theta(h_l)\). The network learns a **residual** (perturbation of identity).
- Why this helps : the Jacobian \(\partial h_{l+1} / \partial h_l = I + \partial f / \partial h_l\). The identity term prevents the gradient from vanishing through multiplication.
- ODE interpretation : \(h_{l+1} = h_l + f_\theta(h_l)\) is the Euler discretisation of \(dh/dt = f_\theta(h)\). A ResNet with \(L\) layers approximates a continuous-time dynamical system with step size \(\Delta t = 1\).
- Neural ODEs (Chen et al., 2018) : take the continuous limit. Replace discrete layers with an ODE solver. Depth becomes continuous. Memory cost is constant (adjoint method for gradients).
- Same principle in LSTM : the cell state update \(c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t\) is a gated identity + perturbation. ResNets and LSTMs solve the same problem (gradient stability) with the same mechanism (additive structure), in different domains (depth vs time).

*"Aha" moment :* the skip connection is not a "trick." It is a discretisation scheme. The entire ResNet is an Euler approximation of a dynamical system.

### Chapter 8 : Regularisation

*Priors, Bayes, and implicit control*

- Weight decay = L2 penalty = Gaussian prior on \(\theta\), MAP estimation. \(\lambda \|\theta\|^2\) in the loss is equivalent to \(\theta \sim \mathcal{N}(0, \frac{1}{2\lambda} I)\) in the prior.
- L1 = Laplace prior = MAP with sparsity.
- Dropout : during training, randomly zero out activations with probability \(p\). Gal & Ghahramani (2016) showed this is equivalent to approximate variational inference in a Bayesian neural network with Bernoulli-distributed weights. Prediction with dropout at test time = Monte Carlo integration over the approximate posterior.
- Batch normalisation : standardise activations per mini-batch. Effect on optimisation : reparametrises the loss surface (smoother landscape). Santurkar et al. (2018) argued the benefit is not "internal covariate shift" (the original motivation) but improved conditioning of the loss.
- Layer normalisation : same idea, per-example instead of per-batch. No batch dependence ; used in Transformers.
- Data augmentation = encoding symmetry priors into the training distribution. If we know the label is invariant to rotation, training on rotated copies is equivalent to regularising toward rotation-invariant functions.
- Early stopping = implicit L2 regularisation (in the linear case, equivalent to ridge regression with \(\lambda\) determined by stopping time).
- Transfer learning = empirical Bayes. Pre-trained weights as informative prior. Fine-tuning = posterior update from a warm start. LoRA / adapters = low-rank perturbation of the prior mean.
- The big picture : explicit regularisation (weight decay, dropout) gives you a Bayesian interpretation. Implicit regularisation (SGD noise, architecture constraints) often dominates in practice.

*"Aha" moment :* dropout is not a heuristic. It is variational inference. Weight decay is not a penalty. It is a Gaussian prior. The DL community rediscovered Bayesian ideas and gave them new names.

### Chapter 9 : Generative Models

*From latent variables to measure transport*

This is the capstone. All previous chapters used deterministic latent variables. Now we make them stochastic.

- **VAE** : introduce a stochastic latent \(z \sim q_\phi(z \mid x)\) (encoder) and a generative model \(p_\theta(x \mid z)\) (decoder). Training maximises the ELBO = \(\mathbb{E}_{q}[\log p_\theta(x \mid z)] - \text{KL}(q_\phi \| p(z))\). This is **variational inference** (Jordan et al., 1999 ; Blei et al., 2017), applied with neural network parametrisation (Kingma & Welling, 2014).
- The reparametrisation trick : \(z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon\), \(\epsilon \sim \mathcal{N}(0, I)\). This makes the ELBO differentiable w.r.t. \(\phi\). It is a change of variables to move the stochasticity outside the parameters.
- **Normalizing Flows** : \(z = g_\theta(\epsilon)\) where \(g\) is a diffeomorphism (invertible, differentiable). The density is computed by change of variables : \(\log p(z) = \log p(\epsilon) - \log |\det J_g|\). This is **transport of measure** (Villani, 2003).
- **GANs** : no explicit likelihood. The generator pushes a base measure through a network. The discriminator estimates the density ratio \(p_{\text{data}} / p_{\text{model}}\). The min-max game minimises an f-divergence between the two distributions (Nowozin et al., 2016). Statistically : implicit density estimation.
- **Diffusion models** : start from data, add noise progressively (forward SDE). Learn to reverse the process (reverse-time SDE, Anderson 1982). The training objective reduces to **score matching** (Hyvarinen, 2005) : estimate \(\nabla_x \log p(x)\). Sampling = Langevin dynamics.
- **Contrastive learning** (SimCLR, CLIP) : noise-contrastive estimation (Gutmann & Hyvarinen, 2010). The contrastive loss is a density ratio estimation problem, same family as the GAN discriminator.

*"Aha" moment :* the evolution from VAE to flows to diffusion is a progression in how we transport a simple measure (Gaussian noise) to a complex one (the data distribution). The statistical framework is measure transport ; the neural network parametrises the transport map.

## Summary of the logical arc

| Chapter | Latent variable \(h\) | Structure imposed |
|---|---|---|
| 1. Neural Networks | deterministic, static | hierarchical composition |
| 2. Embeddings | deterministic, static | projection of discrete input |
| 3. CNNs | deterministic, static | translation equivariance |
| 4. RNN / LSTM | deterministic, dynamic | recursive temporal update |
| 5. Transformers | deterministic, dynamic | global kernel regression |
| 6. Backprop / SGD | (optimisation) | stochastic approximation |
| 7. ResNets | deterministic, static | ODE discretisation |
| 8. Regularisation | (inference) | priors and approximate Bayes |
| 9. Generative Models | **stochastic** | measure transport |
