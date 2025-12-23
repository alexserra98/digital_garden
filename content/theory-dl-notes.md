---
title: Theory of Deep Learning - Lecture Notes
draft: false

---
# Optimization of a Quadratic Cost Function

## Linear Regression Model
We focus on a linear regression model:
$$y_{\theta}(x) = w^{\top}\phi_{\theta}(x^{\mu})$$ 

### Loss Functions
* **Train Loss:**
    $$\mathcal{L}(\theta) = \frac{1}{n}\sum_{\mu=1}^{n}(y^{\mu}-\hat{y}_{\theta}(x^{\mu}))^{2}$$ 
* **Test Loss (Generalization Error):**
    $$\epsilon_{g}(\theta) = \mathbb{E}_{(x,y)}[(y-\hat{y}_{\theta}(x))^{2}]$$ 

### Quadratic Form
The cost function can be expanded as a quadratic function of the weights $w$:
$$\mathcal{L}(w) = \frac{1}{2}w^{T}Aw + b^{T}w + c$$ 

Where:
* **A (Hessian/Covariance):** $A = \frac{1}{n}\Phi\Phi^{T}$. This represents the covariance structure (structure of Wishart matrix).
* **b (Cross-correlation):** This represents the cross-correlation between features and labels.

---

## Gradient Descent Dynamics

### Update Rule
We calculate the gradient descent update observation:
$$w^{t+1} = w^{t} - \eta \nabla_{w} \mathcal{L}(w)$$
$$= w^{t} - \eta (Aw^{t} + b)$$
$$= (I - \eta A)w^{t} - \eta b$$ 

### Fixed Points
We want to investigate the fixed points where $\nabla_{w} \mathcal{L}(w) = Aw + b = 0$:
$$w^{*} = -A^{-1}b$$

### Eigenvector Basis Analysis
The idea is to study the dynamics in the basis of the eigenvectors of $A$.
Let $A_{ij} = \frac{1}{n}\sum_{\mu}\phi_{i}^{\mu}\phi_{j}^{\mu}$.

The update rule in the eigenbasis becomes decoupled:
$$\tilde{w}_{i}^{t+1} = (1 - \eta \lambda_{i})\tilde{w}_{i}^{t} - \eta \tilde{b}_{i}$$

**Convergence Conditions:**
Since we need to invert $A$, we assume $\forall \lambda_i > 0$. The fixed point in this basis is:
$$w_{i}^{*} = -\frac{b_{i}}{\lambda_{i}}$$ 

The distance from the fixed point evolves as:
$$\tilde{w}_{i}^{t+1} - \tilde{w}_{i}^{*} = (1 - \eta \lambda_{i})(\tilde{w}_{i}^{t} - \tilde{w}_{i}^{*})$$ 
This resembles the recurrence relation $u(t+1) = c \cdot u(t)$.
$$\tilde{w}_{i}^{t+1} - \tilde{w}_{i}^{*} = (1 - \eta \lambda_{i})^{t}(\tilde{w}_{i}^{0} - \tilde{w}_{i}^{*})$$ 

### Scenarios based on Learning Rate ($\eta$) and Eigenvalues ($\lambda_i$)

1.  **Monotonic Convergence:**
    If $0 \le \eta \lambda_{i} < 1$, the weights $w_i^t$ converge smoothly to $w_i^*$.
    2.  **Oscillatory Convergence:**
    If $1 \le \eta \lambda_{i} \le 2$, the weights oscillate but still converge.
    3.  **Divergence:**
    If $\eta \lambda_{i} \ge 2$, the system will diverge. 

---

## Loss Dynamics and Assumptions
To ensure convergence, the learning rate must be bounded from below such that $\eta < 2/\lambda_{max}$.

Rewrite the loss using the eigenvalues $\lambda$ and the basis coordinates:
$$\mathcal{L}(w) = \frac{1}{2}\sum_{j: \lambda_{j}>0}\lambda_{j}(\tilde{w}_{j} - \tilde{w}_{j}^{*})^{2} + \text{constants}$$ 

Plugging in the time evolution:
$$\mathcal{L}(w^{t}) = \frac{1}{2}\sum_{j: \lambda_{j}>0}\lambda_{j}(\tilde{w}_{j}^{(0)} - \tilde{w}_{j}^{*})^{2}(1 - \eta \lambda_{j})^{2t}$$

### Approximations for Large $t$
For small learning rates, we can approximate $(1 - \eta \lambda_{j})^{2t} \approx e^{-2\eta \lambda_{j}t}$.
For large $t$, the loss is dominated by the smallest eigenvalue ($\lambda_{min}$):
$$\mathcal{L}(t) \approx \frac{\lambda_{min}}{2}(w_{min}^{0} - w_{min}^{*})^{2}(1 - \eta \lambda_{min})^{2t}$$ 
$$\mathcal{L}(t) \simeq C \cdot e^{-2t\eta \lambda_{min}}$$ 

The convergence speed is determined by the **Condition Number** of $A$. As the condition number grows, the leading term depends heavily on $\lambda_{min}$.

# Generalization Dynamics and Bias-Variance Trade-off

## Generalization Error

The generalization error (test loss) $\epsilon_g(w)$ can be decomposed into two main components:
$$\epsilon_{g}(w) = \underbrace{\mathbb{E}_{w}[\epsilon_{g}(w)]}_{\text{Generalization Error of } \mathbb{E}[w]} + \underbrace{\mathbb{E}_{w}[(\epsilon_{g}(w) - \mathbb{E}_{w}[\epsilon_{g}(w)])^{2}]}_{\text{Variance of Generalization Error}}$$

Assuming the loss function is convex (quadratic):
$$\epsilon_{g}(w) = \frac{1}{2}\sum_{j}\lambda_{j}(\tilde{w}_{j} - \tilde{w}_{j}^{*})^{2}$$

The overall generalization error is calculated as the expected value over the training set realization:
$$\mathbb{E}_{\{\phi^{\mu}, y^{\mu}\}}[\epsilon_{g}(w)] = \frac{1}{2}\sum_{j}\lambda_{j} \mathbb{E}_{\{\phi^{\mu}, y^{\mu}\}}[(\tilde{w}_{j} - \tilde{w}_{j}^{*})^{2}]$$

### Bias-Variance Decomposition (Assuming $\mathbb{E}[w^*]$ is optimal)

The term $\mathbb{E}[(\tilde{w}_{j} - \tilde{w}_{j}^{*})^{2}]$ is decomposed:
$$\mathbb{E}[(\tilde{w}_{j} - \tilde{w}_{j}^{*})^{2}] = \underbrace{\mathbb{E}[\tilde{w}_{j} - \mathbb{E}[\tilde{w}_{j}]]^{2}}_{\text{Variance (Training Data Specificity)}} + \underbrace{(\mathbb{E}[\tilde{w}_{j}] - \tilde{w}_{j}^{*})^{2}}_{\text{Bias (Distance from Optimal)}}$$

Substituting the time evolution of $\tilde{w}_{j}$:
$$\mathbb{E}[\tilde{w}_{j}] = \tilde{w}_{j}^{*} + (1 - \eta \lambda_{j})^{t}(\tilde{w}_{j}^{0} - \tilde{w}_{j}^{*})$$

* **Bias Term:**
    $$\text{Bias}^{2}(\tilde{w}_{j}) = \left(\mathbb{E}[\tilde{w}_{j}] - \tilde{w}_{j}^{*}\right)^{2} = (\tilde{w}_{j}^{0} - \tilde{w}_{j}^{*})^{2} (1 - \eta \lambda_{j})^{2t}$$
    The bias **decreases** exponentially with time $t$. This term dominates at small $t$.

* **Variance Term:**
    $$\text{Var}(\tilde{w}_{j}) = \mathbb{E}[\tilde{w}_{j} - \mathbb{E}[\tilde{w}_{j}]]^{2}$$
    The variance **increases** over time $t$ until it plateaus. This term is often modeled to be proportional to $\frac{1}{\lambda_j n}$ at large $t$.

---

## Teacher-Student Setting

The **Teacher-Student** model is used to analyze the generalization performance of machine learning models in a simplified, yet rigorous, setting.

### Model Setup

1.  **Teacher Model ($w^T$):** The true underlying function is defined by a "Teacher" weight vector $w^T$:
    $$y^{\mu} = (w^{T})^{\top}\phi^{\mu} + \epsilon^{\mu}$$
    where $\epsilon^{\mu}$ is **noise** (e.g., Gaussian, $\epsilon^{\mu} \sim \mathcal{N}(0, \sigma^2)$).

2.  **Student Model ($w$):** The "Student" model attempts to learn the true function by minimizing the training loss:
    $$\mathcal{L}(w) = \frac{1}{n}\sum_{\mu=1}^{n}\left((w^{\top}\phi^{\mu}) - y^{\mu}\right)^{2}$$

3.  **Optimal Solution ($w^*$):** The theoretical minimum of the generalization error (test loss) is at the **Bayes estimator** (or the solution that minimizes the true error):
    $$w_{opt} = w^{T}$$
    (In the absence of noise $\sigma=0$ or if the feature mapping $\phi$ is complete).

### Generalization Error in Teacher-Student

The generalization error can be written in terms of the distance from the Teacher's weights, $w^T$:
$$\epsilon_{g}(w) = \epsilon_{g}(w^T) + \frac{1}{2}(w - w^T)^{\top} \mathbb{E}[\Phi \Phi^{\top}] (w - w^T)$$
* **$\epsilon_{g}(w^T)$** is the minimal error due to the noise, $\sigma^2$.

### The Training Error Solution ($\hat{w}$)
The solution found by minimizing the training loss is:
$$\hat{w} = A^{-1} b$$

Where:
* $A = \frac{1}{n} \Phi \Phi^{\top}$
* $b = \frac{1}{n} \Phi Y = \frac{1}{n} \Phi (\Phi^{\top} w^{T} + \epsilon)$
    $$b = \left(\frac{1}{n} \Phi \Phi^{\top}\right) w^{T} + \frac{1}{n} \Phi \epsilon$$
    $$b = A w^{T} + \delta$$
    where $\delta = \frac{1}{n} \Phi \epsilon$ is the **noise term**.

Substituting $b$ into the solution for $\hat{w}$:
$$\hat{w} = A^{-1} (A w^{T} + \delta) = w^{T} + A^{-1} \delta$$

The Student's solution $\hat{w}$ is the sum of the **Teacher's solution** $w^T$ and a **noise component** $A^{-1} \delta$ that depends on the specific realization of the training data.

The deviation from the true solution $w^T$ is:
$$\hat{w} - w^{T} = A^{-1} \delta$$

***

# Generalization Dynamics (Continued)

## Spectrum Analysis and Convergence Profiles

We analyze the behavior of the loss function $\mathcal{L}(t)$ and the eigenvalue distribution $P(\lambda)$ of the data covariance matrix in different regimes of the sample size $n$ (denoted as $u$ in the text) versus dimension $d$.

### Case III: $n = d$ (Critical Regime)
 When the number of samples equals the dimension ($n=d$) :

* **Eigenvalue Distribution $P(\lambda)$:** The distribution has a singularity near zero .
* **Loss Dynamics:** The model can fit the data, but the training error decays very slowly .
* **Asymptotic Loss:** The asymptotic loss is defined as:
    $$\epsilon^{*} = \lim_{t\rightarrow \infty} \mathcal{L}(t)$$
    The loss approaches the noise floor $\sigma_{\epsilon}^{2}$  .

**Interpretation via ODE:**
The slow decay is due to the small eigenvalues. In the ODE formulation, modes with small $\lambda_i$ have a characteristic time $\tau_i \sim 1/\lambda_i$ that is very large, leading to "critical slowing down" in learning along those directions .

---

# Random Feature Model

## Model Definition
The Random Feature Model is a model where the features of the first layer are **fixed** (not trained)  .

* **Prediction:**
    $$\hat{y}(x) = w^{\top}\sigma(F x)$$
    Where $F$ is a fixed matrix with entries sampled from a normal distribution:
    $$F \in \mathbb{R}^{K \times d}, \quad F_{ij} \sim \mathcal{N}(0, 1)$$ .

## Motivation
* **Linear Separability:** The idea is that by random projection into a high-dimensional space, the dataset becomes linearly separable .
* **Fast Kernel:** It acts as a computationally efficient approximation of a kernel method .
    The Gram matrix is given by:
    $$\Phi = \frac{1}{n} Z^{\top} Z = \frac{1}{n} \sum_{\mu} \sigma(Fx^{\mu})\sigma(Fx^{\mu})^{\top}$$ .

---

# The Perceptron

## Definition
If the data is linearly separable, we define the **margin** $d_i(w)$ for a sample $x_i$ as:
$$d_{i}(w) = \frac{|\langle w, x_i \rangle|}{||w||}$$ .

## Perceptron Algorithm
The algorithm proceeds as follows :

1.  **Initialize:** $w_{0} = 0$  .
2.  **Iterate:** Pick an index $i \in \{1, ..., n\}$ .
    * **IF** $y_{i} \langle w_t, x_i \rangle \le 0$ (Misclassification) :
        $$w_{t+1} = w_{t} + y_{i}x_{i}$$  
    * **ELSE** (Correct classification):
        $$w_{t+1} = w_{t}$$  

## Convergence Theorem
The Perceptron algorithm makes at most $k$ mistakes (updates), bounded by:
$$k \le \frac{R^2 + \text{diam}(\mathcal{D})}{\gamma(\mathcal{D})^{2}}$$   

Where:
* **Margin $\gamma(\mathcal{D})$:** The maximum possible margin over all weight vectors.
    $$\gamma(\mathcal{D}) = \max_{||w||=1} [\min_{i} \text{dist}(x_{i}, \text{boundary})]$$  .
    It represents the smallest distance between the decision boundary and any training point for the optimal separator .
* **Diameter:** $\text{diam}(\mathcal{D}) = \max_{i} \{ ||x_{i}|| \}$ .

**Remark:** This theorem is significant because the convergence bound does not explicitly depend on the dimensionality of the input space, only on the geometric margin and the radius of the data  .

---

# Generalization Performance of Perceptron

We consider the setting where we have a training set $\mathcal{D}_n$ that is linearly separable  .

Let $w(\mathcal{D}_n)$ be the result achieved by the Perceptron algorithm on dataset $\mathcal{D}_n$ after convergence  . We evaluate the performance on a new test point $(x^*, y^*)$ .

## Probability of Misclassification
The probability that the learned weight vector misclassifies a new point is bounded by the expected number of mistakes (steps) required to learn a dataset of size $n+1$:

$$P(y^* \langle w(\mathcal{D}_n), x^* \rangle < 0) \le \frac{1}{n+1} \mathbb{E}_{\mathcal{D}_{n+1}}[k_{\text{max}}(\mathcal{D}_{n+1})]$$  

* This relates the generalization error directly to the "hardness" of the learning problem, quantified by the number of steps $k$ required for convergence  .
* If we train a classifier with a dataset of size $n$, the probability of mistake is proportional to the expected number of updates normalized by $n$  .

# Generalization of Two-Layer Neural Networks

## Teacher-Student Correlation Dynamics
We now analyze the learning dynamics in a two-layer neural network (NN) in the Teacher-Student setting. The key metrics are the **overlaps** (correlation matrices) between the Student weights and the Teacher weights.

### Network Architecture
* **Teacher:** $w^T \in \mathbb{R}^N$
    $$y^{\mu} = \sum_{i=1}^{N} w_{i}^{T} \sigma(\phi_{i}^{\top} x^{\mu}) + \eta^{\mu}$$
* **Student:** $w \in \mathbb{R}^N$
    $$\hat{y}^{\mu} = \sum_{i=1}^{N} w_{i} \sigma(\phi_{i}^{\top} x^{\mu})$$

### Overlap Matrices (Correlation)
We define the correlation matrices to track the learning progress:

1.  **Teacher-Student Overlap (Q):** Measures the similarity between the Student and Teacher weight vectors.
    $$Q_{ij}(t) = \langle w_{i}(t) w_{j}^{T} \rangle$$
    * Goal: $Q_{ij} \to \delta_{ij}$ (i.e., the Student learns the Teacher's weights).
2.  **Student-Student Overlap (M):** Measures the correlation within the Student's own weights.
    $$M_{ij}(t) = \langle w_{i}(t) w_{j}(t) \rangle$$
3.  **Teacher-Teacher Overlap (R):** Measures the correlation within the Teacher's fixed weights. (Assumed fixed for simplicity, often $R_{ij} = \delta_{ij}$).
    $$R_{ij} = \langle w_{i}^{T} w_{j}^{T} \rangle$$

## Gradient Descent Update

The update rule for the Student weight $w_i$ using gradient descent on the quadratic loss is:
$$\frac{dw_{i}}{dt} = -\frac{1}{n} \sum_{\mu=1}^{n} (w^{\top}\sigma^{\mu} - y^{\mu}) \sigma_{i}^{\mu}$$

Substitute the Teacher's output $y^{\mu} = (w^{T})^{\top}\sigma^{\mu} + \eta^{\mu}$:
$$\frac{dw_{i}}{dt} = -\frac{1}{n} \sum_{\mu=1}^{n} \left(\sum_{j} (w_{j} - w_{j}^{T})\sigma_{j}^{\mu} - \eta^{\mu}\right) \sigma_{i}^{\mu}$$
$$\frac{dw_{i}}{dt} = -\sum_{j} (w_{j} - w_{j}^{T}) \underbrace{\left(\frac{1}{n}\sum_{\mu=1}^{n} \sigma_{j}^{\mu}\sigma_{i}^{\mu}\right)}_{A_{ij}} + \underbrace{\frac{1}{n}\sum_{\mu=1}^{n} \eta^{\mu}\sigma_{i}^{\mu}}_{\delta_{i}}$$

The **effective dynamics** for the Student weights $w$ is:
$$\frac{dw}{dt} = -A(w - w^{T}) + \delta$$

Where $A$ is the **empirical feature covariance matrix** (or Gram matrix).

## Overlap Equations (Dynamics)

We take the derivative of the overlap matrix $Q_{ij} = \langle w_{i} w_{j}^{T} \rangle$ with respect to time $t$. We assume $\langle \delta w^{T} \rangle = 0$ (noise is uncorrelated with Teacher weights).
$$\frac{dQ_{ij}}{dt} = \left\langle \frac{dw_{i}}{dt} w_{j}^{T} \right\rangle$$

$$\frac{dQ_{ij}}{dt} = -\sum_{k} \left\langle A_{ik} (w_{k} - w_{k}^{T}) w_{j}^{T} \right\rangle$$

Assuming large $n$ (thermodynamic limit), the empirical covariance $A_{ik}$ converges to the expected value, $A_{ik} \to \delta_{ik}$ (i.e., $\langle \sigma_{i} \sigma_{k} \rangle = \delta_{ik}$).

The resulting mean-field ODEs for the macroscopic quantities $Q(t) = \langle w \cdot w^T \rangle$ (Teacher-Student overlap) and $M(t) = \langle w \cdot w \rangle$ (Student-Student overlap) are:

$$\frac{dQ}{dt} = -Q + R$$
$$\frac{dM}{dt} = -2(M - Q) + \text{Noise Term} + \dots$$

**Solutions for the Overlaps (in the noiseless case $\sigma=0$):**

* **Teacher-Student Overlap:**
    $$Q(t) = R + (Q(0) - R) e^{-t}$$
    $Q(t)$ converges exponentially to $R$ (Teacher-Teacher Overlap). This means the Student weights converge to the Teacher's weights.
* **Student-Student Overlap:**
    $$M(t) = Q(t) - M_{\text{noise}}(t) \cdot e^{-t}$$
    $M(t)$ also converges to $R$ as $t \to \infty$.

### Generalization Error Dynamics
The generalization error $\epsilon_g(t)$ is directly related to the distance between the Student and Teacher weights:

$$\epsilon_{g}(t) = \frac{1}{2} \langle (w - w^{T})^{\top} \Phi \Phi^{\top} (w - w^{T}) \rangle + \sigma^2$$

In the large $N$ limit (i.e., $A \approx I$), the generalization error is simplified:
$$\epsilon_{g}(t) \propto \langle (w - w^{T})^{\top} (w - w^{T}) \rangle + \sigma^2$$
$$\epsilon_{g}(t) \propto M(t) - 2Q(t) + R + \sigma^2$$

Substituting the solutions for $Q(t)$ and $M(t)$:
$$\epsilon_{g}(t) \approx \epsilon_{g}(\infty) + C \cdot e^{-2t}$$

The generalization error decays exponentially at a rate twice that of the overlaps, following the same exponential decay seen in the linear regression model.

# Symmetry Breaking and Specialization

## Symmetry Breaking Phase
Initially, the student weights are randomly initialized, and the overlap with the teacher is small.
* **Initial State:** $Q \approx 0$, $R \sim \mathcal{O}(d^{-1/2})$. The student neurons are not correlated with any specific teacher neuron.
* **Symmetry:** There is a permutation invariance; any student neuron could potentially learn any teacher neuron.

**The Breaking Point:**
As training proceeds, the system undergoes **symmetry breaking**  .
* **Transition:** $Q \to \text{Structured Matrix}$.
    The student neurons "choose" specific teacher neurons to mirror.
    $$Q \rightarrow \begin{pmatrix} 1 & d^{-1/2} \\ d^{-1/2} & 1 \end{pmatrix}$$
* **Specialization:** Each student neuron becomes correlated with only one teacher neuron  . This specialization occurs because, with sufficient data, it is more efficient to specialize different neurons rather than keeping them close to a linear model average  .

---

# Search Phase Dynamics

When the initial overlap is very small (e.g., "warm start" vs "cold start"), the network must first find the informative directions.

## Problem Setup
* **Dynamics:** We track the teacher-student overlap $R(t)$ and student-student overlap $Q(t)$ (denoted as $u$ and $q$ in some contexts here) .
    * $\dot{R}(t) = -d R(t)$
    * $\dot{Q}(t) = d Q(t)$
* **Initialization:** $R(0) = R_0$, $Q(0) = Q_0$.
* **Scaling:** We observe behavior where $R \sim \frac{1}{\sqrt{d}}$ .

## Search vs. Descent
We switch from analyzing the "Descent Phase" to the **"Search Phase"**  .
* **Student:** $\hat{y} = \sigma(w^{\top}x)$ with $w \sim \mathcal{U}(S^{d-1})$ (Unit sphere)  .
* **Teacher:** $y = \dots$ (Teacher vector $\bar{w}$).
* **Update Rule (Normalized SGD):**
    To study the search, we use a normalized update to keep the weight on the sphere:
    $$w^{t+1} = \frac{w^{t} - \eta \nabla \mathcal{L}(w)}{||w^{t} - \eta \nabla \mathcal{L}(w)||}$$  

## Geometric Dynamics of the Search
We are interested in computing the evolution of the overlap $u = w \cdot \bar{w}$ (cosine similarity with teacher) .
* **Teacher Direction:** $\bar{w}$
* **Student Direction:** $w$ starts orthogonal to $\bar{w}$ ($u \approx d^{-1/2}$) .

**Update Approximation:**
For small learning rate $\eta$:
$$w^{t+1} \approx w^{t} - \eta (v^t \cdot \bar{w}) - \frac{\eta^2}{2}||v^t||^2 w^t + \mathcal{O}(\eta^3)$$  
where $v^t = \nabla \mathcal{L}(w^t)$.

The update is composed of:
1.  **Deterministic Signal:** $\eta \nabla \mathcal{L}(w) \cdot \bar{w}$ (Drift) .
2.  **Noise:** Stochastic gradient noise (Diffusion) .

## Heuristic Derivation of the ODE
We look at the expectation of the update conditioned on the dataset $\mathcal{F}$ :
$$\mathbb{E}_{\mathcal{F}}[w^{t+1}] = w^t - \eta (\mathbb{E}[v^t] \cdot \bar{w}) - \eta^2 (\mathbb{E}[||v^t||^2]) w^t$$

Optimizing over $\eta$, we derive an ODE for the overlap $u(t)$ :
$$\dot{u}(t) \propto \frac{1}{u(t)} \frac{(\mathbb{E}[\nabla \mathcal{L} \cdot \bar{w}])^2}{\mathbb{E}[||\nabla \mathcal{L}||^2]}$$

* **Numerator (Signal):** Squared mean of the gradient projected on the teacher .
* **Denominator (Noise):** Variance of the gradient .

---

# Information Exponent

To quantify the "hardness" of the search phase, we introduce the **Information Exponent** ($k^*$)  .

## Correlation Loss and Hermite Expansion
We analyze the correlation loss:
$$\mathcal{L}(w) = 1 - \mathbb{E}[\sigma(w^{\top}x)\sigma(\bar{w}^{\top}x)]$$  

Expanding this using **Hermite polynomials** (since inputs are Gaussian) :
The dominant term in the gradient dynamics is determined by the **first non-zero coefficient** in the expansion, denoted by index $k^*$ .

## Complexity Table
The complexity of learning depends heavily on $k^*$ (the "information exponent") :

| $k^*$ | Time to Converge ($t$) | Type of Dynamics | Hardness |
| :--- | :--- | :--- | :--- |
| **1** | $t \approx d$ | Linear | Easy (Informative Gradient) |
| **2** | $t \approx d \ln d$ | Quasi-Linear | Medium |
| **$\ge 3$** | $t \approx d^{k^*}$ | Polynomial | Hard (Gradient is uninformative) |

* If $k^* \ge 3$, the problem becomes significantly harder because the gradient contains very little information about the teacher direction in the initial phase (the "Search Phase"). The system effectively has to search blindly until it falls into the basin of attraction.

# Wide 2-Layer Neural Networks

## Setting and Assumptions
We consider a setting where the input dimension $d$ is fixed ($O(1)$) while the number of neurons $N$ grows.

* **Data Generation:** $y_{i} = f(x_{i}) + z_{i}$, where $z_{i}$ is noise with $\mathbb{E}[z_{i}]=0$ and finite variance.
* **Activation Function:** $\sigma$ is sigmoidal (bounded, non-decreasing) or ReLU.
* **Prediction Model:**
    $$\phi_{\theta}(x) = \frac{1}{N}\sum_{k=1}^{N}v^{k}\sigma(\langle w^{k},x\rangle + b^{k})$$
     

## Universal Approximation Theorem
**Theorem:** For any $\epsilon > 0$, there exists a number of neurons $N_{\epsilon}$ such that:
$$\inf_{v,w,b} (\mathbb{E}_{x} [|f(x) - \phi_{\theta}(x)|]) \le \epsilon$$


In other words, a 2-layer neural network can approximate any kind of function . This happens because the network constructs sums of step functions to approximate the target . By combining ReLUs or sigmoids ($g_{\alpha} - g_{\alpha-\delta}$), one can construct "bump" functions to approximate local features .



## Barron's Theorem
This theorem provides a bound on the approximation error based on the complexity of the function .

* **Target Function:** $f(x)$ is defined via its Fourier transform $F(w)$:
    $$f(x) = \int dw e^{iw^{\top}x}F(w)$$
     
* **Input:** $P(x)$ has finite support on a ball of radius $r$ in dimension $d$.
* **Statement:** For a given $\epsilon > 0$, there exists a 2-layer network $\phi_{\theta}$ with $N$ neurons such that the error is bounded by:
    $$||f - f_{N}||_{2} < \frac{C_{f}}{\sqrt{N}}$$
      
    
    Where the constant $C_f$ depends on the spectral norm of the function:
    $$N_{\epsilon} = \frac{1}{\epsilon} \left[ 2\pi \int ||w||_{2} |F(w)| dw \right]^{2}$$
     

**Significance:** This theorem incorporates the number of neurons $N$ for an increasing complexity of the function we approximate. It shows the error decays as $1/\sqrt{N}$, avoiding the "curse of dimensionality" .

---

# Kernel Methods

## Dual Formulation of Linear Regression
Consider the standard regularized regression loss for a 2-layer network (where features $\phi(x)$ are fixed):
$$\mathcal{L}(w) = \frac{1}{2n}\sum_{i=1}^{n}[y_{i} - w^{\top}\phi(x_{i})]^{2} + \lambda w^{\top}w$$
 

Setting the gradient $\nabla_{w}\mathcal{L}(w) = 0$ leads to the solution $w = \Phi^{\top} a$, which allows us to rewrite the problem in terms of the dual variables $a$  .

The loss in terms of $a$ is:
$$\mathcal{L}(a) = \frac{1}{2}a K K a^{\top} - a K y + \frac{1}{2}y^{\top}y + \frac{\lambda}{2} a K a^{\top}$$
 

Where $K = \Phi\Phi^{\top}$ is the **Kernel Matrix** (Gram matrix) . The optimal dual coefficients are:
$$a = (K + \lambda I)^{-1}y$$
 

## Prediction
The prediction for a new point $x$ becomes a weighted sum of kernel evaluations between the training points and the new point:
$$\hat{y}(x) = K(x)^{\top} (K + \lambda I)^{-1} y$$
 

Or equivalently:
$$\hat{y}(x) = \sum_{i} [(K + \lambda I)^{-1} y]_{i} K(x, x_{i})$$
 

**Properties:**
* Allows performing linear regression in an infinite-dimensional space if we know $K$ .
* **Complexity:** Inverting the matrix requires $O(n^3)$ operations, which is expensive for large datasets .
* The kernel is fixed and not learnable from the data in this standard formulation.

---

# Random Features (RF)

## Motivation
Since the Kernel trick is computationally expensive ($O(n^3)$), we want to approximate the kernel using a lower-dimensional feature map $Z(x)$ such that:
$$Z(x)^{\top}Z(y) \approx K(x, y)$$
 

## Random Fourier Features
For shift-invariant kernels (e.g., Gaussian RBF), Bochner's theorem states the kernel is the Fourier transform of a probability distribution $p(w)$:
$$K(\delta) = \int_{\mathbb{R}^{d}} p(w) e^{iw^{\top}\delta} dw = \int_{\mathbb{R}^{d}} p(w) \cos(w^{\top}\delta) dw$$


We can approximate this integral using **Monte Carlo sampling** :
1.  Sample frequencies $w_{i} \sim p(w)$ .
2.  Construct the feature map:
    $$Z(x)^{\top}Z(y) \approx \frac{1}{N} \sum_{i=1}^{N} \cos(w_{i}^{\top}(x - y))$$
     

The prediction in the Random Feature model is:
$$y_{RF}(x) = v^{\top}\Phi_{RF}(x)$$
 

Here, we choose the random weights fixed to match the symmetry of the dataset (kernel), and only train the outer layer $v$ . This approximation is cheap and decays very fast .

---

# Connection with Neural Networks

## Taylor Expansion & The "Lazy" Regime
We can view a trained neural network through the lens of Random Features by expanding the network function around its initialization.

Consider the network:
$$\Phi_{NN}(x; w, v) = \frac{1}{\sqrt{N}}\sum_{i} (v_{i}^{0} + \epsilon v_{i}^{1}) \sigma((w_{i}^{0} + \epsilon w_{i}^{1})x)$$
 

Performing a Taylor expansion:
$$\Phi_{NN} \approx \underbrace{\frac{1}{\sqrt{N}}\sum_{i} v_{i}^{0}\sigma(w_{i}^{0}x)}_{\text{Initial State}} + \epsilon \underbrace{\sum_{i} v_{i}^{1}\sigma(w_{i}^{0}x)}_{\text{Random Feature Part}} + \epsilon \underbrace{\sum_{i} v_{i}^{0} (w_{i}^{1}x) \sigma'(w_{i}^{0}x)}_{\text{Neural Tangent Kernel Part}}$$


1.  **Random Feature (RF) Regime:** Corresponds to fixing the inner weights ($w^0$) and only training the outer weights ($v^1$). The learned parameters are outside the non-linearity .
2.  **Neural Tangent Kernel (NTK) Regime:** Corresponds to the linearization where both layers move slightly. The "features" are fixed at initialization ($v_i^0, w_i^0$) .

## Summary
* **Random Features:** The parameters inside the non-linearity are random and **fixed**. The model is linear in the trainable parameters  .
* **Neural Network (Lazy Training):** In the highly over-parametrized limit ($N \to \infty$), the weights stay close to initialization . The dynamics become approximately linear, governed by the Neural Tangent Kernel (NTK).
* This linear approximation holds true as training proceeds if $N \to \infty$.

***

