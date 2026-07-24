---
title: Introduction to the Theory of Neural Networks 3
draft: false
tags:
    - ml-theory
---
# Random Features and the Perceptron


## Random-feature model

The first-layer features are sampled once and then frozen:

$$
\hat y(x)=w^\top\sigma(Fx),
\qquad F_{ij}\sim\mathcal N(0,1).
$$

Only $w$ is learned. A high-dimensional random projection can make a finite
dataset linearly separable. The associated empirical feature covariance is

$$
\Phi=\frac1n Z^\top Z
=\frac1n\sum_{\mu=1}^n
\sigma(Fx^\mu)\sigma(Fx^\mu)^\top.
$$

This is the finite-feature version of a kernel method.

## Geometry of a linear separator

Let

$$
H_w=\{x:\langle w,x\rangle=0\}
$$

be the decision boundary. For labeled points $y_i\in\{-1,+1\}$, the signed
normalized margin is

$$
\gamma_i(w)=\frac{y_i\langle w,x_i\rangle}{\|w\|}.
$$

The dataset margin is

$$
\gamma(D,w)=\min_i\gamma_i(w),
\qquad
\gamma(D)=\max_{\|w\|=1}\gamma(D,w).
$$

Geometrically, $\gamma(D)$ is the smallest distance from the separating
hyperplane to a training point, maximized over separating hyperplanes.

## Perceptron algorithm

Initialize $w_0=0$. Repeatedly choose a sample $i$. With the unit-margin
variant used in the notebook,

$$
w_{t+1}=
\begin{cases}
w_t+y_i x_i,&y_i\langle w_t,x_i\rangle<1,\\
w_t,&\text{otherwise}.
\end{cases}
$$

For linearly separable data the perceptron makes finitely many updates.
In the standard notation, if $\|x_i\|\le R$ and there is a unit vector
$w^\star$ with $y_i\langle w^\star,x_i\rangle\ge\gamma$, then

$$
M\le\frac{R^2}{\gamma^2}.
$$

The handwritten page expresses the same dependence using dataset diameter and
margin: the number of updates scales as an inverse square in $\gamma(D)$.
Thus the theorem depends on geometry rather than explicitly on ambient
dimension.

## Online-to-batch generalization

Let

$$
D_n=\{(x_i,y_i)\}_{i=1}^n
\stackrel{\rm iid}{\sim}P_0
$$

be separable, and let $w(D_n)$ be the result of running the perceptron for
enough steps to converge. For a fresh point
$z^\star=(x^\star,y^\star)\sim P_0$, exchangeability gives the
leave-one-out/online-to-batch estimate

$$
\Pr\!\left(
y^\star\langle w(D_n),x^\star\rangle<1
\right)
\le
\frac{1}{n+1}
\mathbb E_{D_{n+1}}\!\left[M(D_{n+1})\right],
$$

where $M(D_{n+1})$ is the number of perceptron mistakes/updates on a random
ordering of the augmented sample.

Combining this with the finite mistake bound yields a test-error bound that
decays like $1/(n+1)$ for a fixed margin geometry.

## Main lesson

The notebook connects two facts:

- the perceptron converges after finitely many online errors;
- by permutation symmetry, expected online errors control the probability of
  error on a fresh example.

This is an early example of algorithm-dependent generalization analysis.


