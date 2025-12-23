---
title: Causal Journey 5
draft: false

---
[[causality]]
## Intervention in Markovian Models
First of all we can define an intervention in 2 different ways.
- Given a causal graph $G$ and the entailed distribution $P^{G}_V$ we can see the effect of intervening on a $x \in V$ as constructing a new causal graph $\hat{G}$ where we removed all the incoming edges to $x$ and set it to the desired value for the intervention. 
- Alternative we can augment $G$ with a new variable $F_i$ which correspond to $do(x_i)$: 
$$ P(x_i | pa_i') = \begin{cases} P(x_i | pa_i) & \text{if } F_i = \text{idle}, \\ 0 & \text{if } F_i = \text{do}(x_i') \text{ and } x_i \neq x_i', \\ 1 & \text{if } F_i = \text{do}(x_i') \text{ and } x_i = x_i'. \end{cases} $$

### Computing the effect of interventions
We can factorise our distribution using the markovianity of model as $P(x_1, \dots, x_n) = \prod_{i} P(x_j | pa_j)$ then an intervention on x_i simply means removing the i-th factor and set x_i to a specific value.
$$
P(x_1, \dots, x_n | \hat{x}_i') =
\begin{cases}
\prod_{j \neq i} P(x_j | pa_j) & \text{if } x_i = x_i', \\
0 & \text{if } x_i \neq x_i'.
\end{cases}
$$
Notably this means the following:
$$
P(x_1, \dots, x_n | \hat{x}_i') =
\begin{cases}
\frac{P(x_1, \dots, x_n)}{P(x_i' | pa_i)} & \text{if } x_i = x_i', \\
0 & \text{if } x_i \neq x_i'.
\end{cases}
$$
Differently from the bayesian setting where we renormalise by a constant factor, here we **rescale more** those configuration where $pa_i$ predict a low probability for $x_i$. We can note  that we can partition $x_1, \dots, x_n$ as $x_i, pa_i, S_i$ where $S_i$ are the remaining nodes, then all the configuration with the same $x_i,pa_i$ are rescaled in the same way. (I find it hard to understand this at the beginning, but the point is not that conflicting configuration become more likely than other, it simply means that they are the one rescaled more. Let's say that $p(x_i^*|pa_i)$ is very low for $pa_i=z$ than the original distribution put very low  configuration to $x_i^*, pa_i, S_i$, when we intervene we force $x =x_i^*$ so we will need to increase more prob of those conflicting configuration rather than others where   $x_i', pa_i, S_i$ is already likely.)

I think that *Elements of Causal Inference* has a nice notation for this:
$$
P^{\mathcal{E};do(T=t)}(r) = \sum_z P^{\mathcal{E}}(r|z,t) P^{\mathcal{E}}(z) \neq \sum_z P^{\mathcal{E}}(r|z,t) P^{\mathcal{E}}(z|t) = P^{\mathcal{E}}(r|t).
$$
here r is the variable on which we evaluate the effect, and t the one of which we want to compute the cause. The difference from conditioning is that we reweight the conditional probability $P^{\mathcal{E}}(r|z,t)$ by $P^{\mathcal{E}}(z)$ instead of $P^{\mathcal{E}}(z|t)$ so fundamentally ignoring the effect the $t$ has on $z$. Here we can see also more clearly the meaning of the rescaling argument above. In the expression on the right let's a certain value of t and z yields high r if the probability of seeing that z given t is low this will lower the final probability of p(r|t) thus confounding the effect of t on r.

To compute the effect of intervention we need to "adjust" for some variable that are confounding the effect. In this case above we adjusted for the *direct cause* but what is going on when the graph is more complicated?
First we need to know when a certain causal effect can be extracted just from the data:
**Theorem**:
	Given a causal diagram $G$ of any Markovian model in which a subset $V$ of variables are measured, the causal effect $P(y|\hat{x})$ is identifiable whenever $\{X \cup Y \cup PA_X\} \subseteq V$, that is, whenever $X, Y$, and all parents of variables in $X$ are measured. The expression for $P(y|\hat{x})$ is then obtained by adjusting for $PA_X$ in the following way:
$$P(y|\hat{x}_i') = \sum_{pa_i} P(y|x_i', pa_i) P(pa_i)$$

In other words if you have observations about $\{X \cup Y \cup PA_X\}$ we can compute the causal effect just from the data.
**Identifiability**: Whether or not I can reconstruct causal effect from just observations about conditional probabilities of variables in the graph, and the structure of the causal graph. In other words a mechanism is identifiable if in 2 different causal models $M_1$ and $M_2$ the causal effect of two variables $x$ and $y$ is the same when the entailed distribution of the 2 models is same along with causal graph. If for example the causal effect of $x$ and $y$ is different in these 2 models it means that the conditional probabilities and the causal graph is not enough to specify it.
(**can I show an example of non-identifiable mechanism?**)
it's not always easy to decide on which variable to adjust, some adjustment might actually bias the result so we need to establish which is the right *adjustment set*? 

#### Adjustment Set and Randomised Trial
Before moving on I want to discuss what it means in statistical trials to "adjust" for a variable. Let's say that we have our treatment variable $X \in \{0,1\}$ our response variable $Y \in \{0,1\}$ and some additional variable $Z$ influencing both X and Y. To unconfound $X$ from $Y$ we can assign to each configuration a random $X$, like if $Z$ is the age of the patient, we can assign to each patient a random treatment regardless of the age thus effectively removing the edge from $Z$ to $X$ 

The first criterion is the **Back-Door** and is discussed [[Confounder | here]]. (note to add controlling back-door path is used to change $P(y|\hat{x})$ to $P(y|x)$)
The second criterion is the front door path:


### Do-Calculus
- A simplified version of the rule 3 is the following:
$$P(y \mid do(z), w) = P(y \mid w)
\quad \text{if} \quad
Y \perp\!\!\!\perp_{G_{\overline{Z}(W)}} Z \mid W$$
Ok this took me a while to understand it. At a high level it simply tells you when you can remove $do(z)$ from the conditioning set. The tricky part is the condition under which the equality holds. Here $Z(W)$ are the nodes of $Z$ which are ancestor of $W$.  The point here is why don't we ask independence in the graph $G_{\overline{Z}}$ that is why don't we ask 
$$Y \perp\!\!\!\perp_{G_{\overline{Z}}} Z \mid W$$
Well the issue is that if $Z$ is a collider  and $W$ is its descendant than conditioning on $W$ will open a new path from some parent $X$ of $Z$ and $Y$. This means that while $P(Y | do(Z), W)$ comes from a distribution where $Y$ is independent of $X$, $P(Y | W)$ has been obtained from distribution where $Y$ and $X$ are dependent.
In other words when we $do(Z)$:
$$P(Y \mid do(z), w) = \sum_{x} P(Y \mid x, do(z), w) \cdot P(x \mid do(z), w)$$

Because $X$ and $Y$ are independent here, $P(Y \mid x) = P(Y)$.

$$= \sum_{x} P(Y \mid w) \cdot P(x \mid w) = P(Y \mid w) \cdot 1 = P(Y)$$
In the other case instead:
$$P(Y \mid w) = \sum_{x} \underbrace{P(Y \mid x, w)}_{\text{Depends on } x} \cdot P(x \mid w)$$

Here, $P(Y \mid x, w)$ is NOT equal to $P(Y \mid w)$. The value of $X$ changes the probability of $Y$ .

