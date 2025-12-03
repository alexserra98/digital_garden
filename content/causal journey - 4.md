# Causal Journey 4
## Confounder
[[causality]]
Given a variable $X$ that affects a response variables $Y$ the traditional statistics call a confounder $Z$ a variable that is:
- Correlated with $X$
- Correlated with $Y$ when $X=0$
- In the causal path between $X$ and $Y$
Pearl apparently have really strong opinion about this and according to his theory a variable is a confounder if it's in a non-causal path between $X$ and $Y$ or as he calls it the *back-door* path.
![[causality_4.png]]
For example look at the graph above. Here $X$ is the variable the tells us if a mother is smoking during the current pregnancy, A is the var. telling us if she have been smoking in previous pregnancy, B represents a deformation due to smoking and predisposition of the mother which is represented by D. C is the var. associated with previous miscarriages, E the var for current deformation in the fetus and Y the var associated with current miscarriage. 
Let's say that we want to test if that fact that a mom is smoking during pregnancy affect health of the child. In this case we have a direct path from $X$ to $Y$ and a *back-door* path from A to Y. In this case $B$ act as a collider which is a nodes configuration for which we can prove that A an D are independent. This is intuitive because the fact that a mother smoked during previous pregnancies doesn't correlate with the fact that she has predisposition for having miscarriages. Let's say that we control or condition on the variable $B$ this creates a spurious correlation between $A$ and $D$ because for example if we know that a child as a deformation $B$ and we know that the mother doesn't smoke than the probability of $D$ increases. This spurious correlation "opens" the *back-door* path which means enforce a selection bias on our dataset. If we want to test the effect on smoking during a pregnancy only the subsample of mothers that gave birth to children with a deformation $B$  and we check only X and Y we might not observe a big preference for smoker mothers. This is because we might be considering a lot on non-smoker mothers that have predisposition D! 

---
One of the purpose of causality is to identify independent mechanism that cause a certain phenomenon, and confounders are "mixing variables" the correlates different mechanism. This is why the $do(\cdot)$ mechanism is so important because it allows us to isolate singles mechanism in our network.

As a final note so far it seems to me that causality is a way constructing very structured priors that encodes causal knowledge but the final mechanism to do inference or to update belief seems still to be "bayesian".

