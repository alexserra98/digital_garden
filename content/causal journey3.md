---
title: Causal Journey 3
draft: false

---
[causality]
## Markov Property, Faithfulness and Causal Minimality

It turns out that the (conditional) independences are a powerful criterion to classify mechanisms in a causal graph. Under the mild assumption that the joint distribution has a density we can define the Markov Property as follow:

**Definition 1 (Markov property)**  
Given a DAG $\mathcal{G}$ and a joint distribution $P_X$, this distribution is said to satisfy  

the *global Markov property* with respect to the DAG $\mathcal{G}$ if  

$$
\mathbf{A} \perp_{\mathcal{G}} \mathbf{B} \mid \mathbf{C} \Rightarrow \mathbf{A} \perp \mathbf{B} \mid \mathbf{C}.     \tag{1}
$$

for all disjoint vertex sets $\mathbf{A}, \mathbf{B}, \mathbf{C}$ (the symbol $\perp_{\mathcal{G}}$ denotes *d-separation*).

So we say that a $P_X$ is *markovian* w.r.t $\mathcal{G}$ if all the conditional independencies of $\mathcal{G}$ represented as d-separation are present also in the entailed distribution.
This property can be used to give a graphical sense to  **Reichenbach’s common cause priniciple**. Specifically if there is a SCM that embed two variables X, Y in a graph  that are correlated it means that there must be either direct path connecting them or a path mediated by and additional node.

Additionally:
**Proposition 2 (SCMs imply Markov property)**  
Assume that $P_X$ is induced by an SCM with graph $\mathcal{G}$. Then, $P_X$ is Markovian with respect to $\mathcal{G}$.

The opposite side of the implication of $\tag{1}$ (1) is  **Faithfulness**
**Definition 3 Faithfulness**  
Consider a distribution $P_X$ and a DAG $\mathcal{G}$.

 $P_X$ is *faithful* to the DAG $\mathcal{G}$ if  

$$
\mathbf{A} \perp \mathbf{B} \mid \mathbf{C} \Rightarrow \mathbf{A} \perp_{\mathcal{G}} \mathbf{B} \mid \mathbf{C}
$$

for all disjoint vertex sets $\mathbf{A}, \mathbf{B}, \mathbf{C}$.

Part (i) posits an implication that is the opposite of the global Markov condition.
The following is an example where there is markovianity but not faithfulness.
Let's consider the following graph with associated SCM
![[causality_3.png]]

$$
\begin{aligned}
X &:= N_X, \\
Y &:= aX + N_Y, \\
Z &:= bY + cX + N_Z,
\end{aligned}
$$

Let's say that $$ab+c=0$$ so effectively X and Z are independent. This graph is markovian because there is no way of making two node conditionally independent, although is not faithful because $$\mathbf{X} \perp \mathbf{Z} \mid \varnothing \not \Rightarrow \mathbf{X} \perp_{\mathcal{G}} \mathbf{Z} \mid \varnothing$$
We can construct an example where the opposite is true that is $P_X$ is faithful but not markovian.
If for example $$ab+c \not=0$$
But we have a $\mathcal{G}_2$ equal to  $\mathcal{G}_1$ but without the edge X ->Z then $P_X$ is faithful because we have no way of making two variables independent but it's not markovian because $$\mathbf{X} \perp_{\mathcal{G}} \mathbf{Z} \mid \mathbf{Y} \not \Rightarrow \mathbf{X} \perp \mathbf{Z} \mid \mathbf{Y}$$

As we can see this example are rather pathological and I wonder if you can construct something better, in other words if there is a "deeper" difference between markovianity and faithfulness.

A weaker(?) condition than faithfulness is causal minimality

**Definiton 4 Causal Minimality**
A distribution satisfies *causal minimality* with respect to $\mathcal{G}$ if it is Markovian with respect to $\mathcal{G}$, but not to any proper subgraph of $\mathcal{G}$.

**Proposition 6.35 (Faithfulness implies causal minimality)**  
If $P_X$ is faithful and Markovian with respect to $\mathcal{G}$, then causal minimality is satisfied.

**Proof.**  
The argument is as follows: We need to prove two arguments: $P_X$ is is markovian w.r.t $\mathcal{G}$ and it's faithful but not markovian,  is is markovian w.r.t  and it's not faithful but it is markovian. The first one is trivially wrong becuase the if $P_X$ is causally minimal it should be markovian by definition. So let's unpack the other.  If  is not causally minimal it means it's also markovian to a proper subgraph $\tilde{\mathcal{G}}$ of . Taking a subgraph means that there are two nodes that are directly connected in   but not in $\tilde{\mathcal{G}}$. If  is still markovian it means that we can be *d*-separated in $\tilde{\mathcal{G}}$ but evidently we cannot in   where they are directly connected.  The Markov condition implies the corresponding conditional independence statement in , and thus $P_X$ cannot be faithful with respect to $\mathcal{G}$. □

Bottom line a todo for the next days: I can see why having both implications in (1) makes something causally minimal but I understand why if something markovian is not given that is also faithful.
$f_A(I, a) = P(M_A(I; a)): \mathcal{I} \times \mathcal{A} \to \mathbb{R}$


$f(R; A(C)) - f(R; A(R)) \approx (A(C) - A(R) ) \cdot \frac{\partial f_A}{\partial a}|_{a=A(R)}$


