---
title: Causal Journey 1
draft: false

---
[[causality]]
This is the first of series of small notes I am making along the way while studying causality. The idea is not to summarise everything in details but just note down the things either impressed the most or I had found them harder to understand.
## Causal Bayesian Network
This tool is an extension of the most common bayesian graph which is DAG in which each node represent and arrow and each edge means that there is a correlation. This kind of graphical tool is pretty convenient because it simplify both sampling and making inference for otherwise complicated distribution. 
For example given the a DAG you make some observation for leaf variables and use these one to update the distribution for variable up in the hierarchy, you iterate this process for all the variables in the DAG and across many sample (this is a sort of Gibbs sampling).

Bayesian network encodes correlation between variables but not necessarily causation that is for the same joint distribution you can come up with many different bayesian graph just by swapping each edge. Let's define a causal bayesian network:

---
#### **Definition: (Causal Bayesian Network)**
Let $P(v)$ be a probability distribution on a set $V$ of variables, and let $P_X(v)$ denote the distribution resulting from the intervention $do(X = x)$ that sets a subset $X$ of variables to constants x.10 Denote by $P_*$ the set of all interventional distributions $P_X(v), X \subseteq V$, including $P(v)$, which represents no intervention (i.e., $X = \emptyset$). A DAG G is said to be a causal Bayesian network compatible with $P_*$ if and only if the following three conditions hold for every $P_X \in P_*$:

1. $P_X(v)$ is **Markov relative to** G;
    
2. $P_X(v_i) = 1$ for all $V_i \in X$ whenever $v_i$ is consistent with $X = x$;
    
3. $P_X(v_i \mid pa_i) = P(v_i \mid pa_i)$ for all $V_i \notin X$ whenever pa_i is consistent with $X = x$,
    
    i.e., each $P(v_i \mid pa_i)$ remains invariant to interventions not involving $V_i$.
    

---
Intuitively a CBN the direction of each edge means causation that is: A->B means that A is the cause of B, but the concept of "is cause of" is not crystal clear. I think the definition above formalise it beautifully. The idea is that given a certain joint distribution of a set of variables $V$ you got many different possible bayesian network. The purpose of CBN though is to make intervention, more specifically the $do(\cdot)$ intervention, so you can define the space of all possible intervention for $V$ as $P_*$ and each of these act as a constraint on the possible topology of CBN.
The $do(\cdot)$ intervention for a variables $v$ means severing all the incoming edges in the variables making it a root node. The meaning of this operation is bit subtle but I will talk about in the next posts.
Looking at the definition above I think that the first 2 points are rather intuitive the third instead is crucial, because is imposing the fact that if I make an intervention this needs to be coherent with the "true" causal model generating the observations. 
Let's say that $V=\{x_1, x_2\}$ and we know that $x_1$ is causing $x_2$ if you **only** have the observational distribution $P(v)$, you can't distinguish between $x_1 → x_2$ and $x_2 → x_1$. They belong to the same _Markov equivalence class_.
However, the definition requires compatibility with the **entire set of interventional distributions, `P*`**. This is what allows us to identify the true causal direction. Let's use your example to show why the wrong graph, $x_2 → x_1$, would fail the test if the true world is $x_1 → x_2$.
The correct causal graph is `G_true`: $x_1 → x_2$.
This true graph generates a set of distributions, `P*`, which includes:

1. **Observational `P(v)`:** $P(x_1, x_2) = P(x_2 | x_1) P(x_1)$
    
2. **Interventional $P_{x_2=k}(v)$:** When we intervene and set $x_2 = k$, we break the influence of $x_1$ on $x_2$. So, $x_1$ is unaffected and retains its original distribution, $P(x_1)$. This means under this intervention, $x_1$ and $x_2$ are independent. So, $P_{x_2=k}(x_1) = P(x_1)$ (so the 3. condition is satisfied).
    

Now, let's test the **wrong graph**, `G_wrong`: $x_2 → x_1$, against this set of distributions `P*` using the definition's conditions.

### Checking `G_wrong`: $x_2 → x_1$

For this graph, the parent of $x_1$ is $pa_1 = {x_2}$. Let's check **Condition 3.**.

This condition states that for any intervention not involving a variable, the relationship between that variable and its parents must remain invariant. Let's test this for variable $x_1$ when we perform the intervention $do(x_2 = k)$.

- The intervention set is $X = {x_2}$.
    
- The variable we are checking is $v_i = x_1$, which is not in $X$.
    
- The parents of $x_1$ in `G_wrong` are $pa_1 = {x_2}$.
    

Condition 3. requires that:

$P_{x_2​=k}​(x_1​∣pa_1)=P(x_1​∣pa_1​)$

Substituting the actual parent, we must have:

$P_{x_2​=k}​(x_1​∣x_2​)=P(x_1​∣x_2​)$

Let's evaluate both sides using the distributions from our **true world** (`G_true`):

- **Left-Hand Side (Interventional):** In the true world, when we set $x_2 = k$, $x_1$ becomes independent of $x_2$ and keeps its original distribution. Therefore, $P_{x_2=k}(x_1 | x_2) = P_{x_2=k}(x_1) = P(x_1)$.
    
- **Right-Hand Side (Observational):** This is the conditional probability from the original, non-interventional distribution. In general, $P(x_1 | x_2)$ is not equal to $P(x_1)$. They are only equal if $x_1$ and $x_2$ were independent to begin with, which would mean there was no causal arrow at all.
    

So, we have the requirement:

$P(x_1​)=P(x_1​∣x_2​)$

This is a **contradiction**! This equation only holds if `x_1` and `x_2` are independent. But we assumed a causal relationship exists.

Therefore, the graph `G_wrong`: `x_2 → x_1` is **not compatible** with the set of interventional distributions `P*` generated by the true causal world `x_1 → x_2`.

