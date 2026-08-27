# Energy computation

## 1. Phase coherence computation

We want to compute the phase coherence $C_p$.

$$
C_p =
\frac{2}{N_s(N_s-1)}
\operatorname{Re}
\left\{
\sum_{k=1}^{N_s}
\sum_{j>k}
\frac{\hat{x}_k^{*}\hat{x}_j}
{|\hat{x}_k|\,|\hat{x}_j|}
\right\}
\tag{C1}
$$

In doing so, it is inconvenient to perform a multiplication for every pair of stations. So let's define

$$
\hat{x}^{\,cp}_k \equiv
\frac{\hat{x}_k}{|\hat{x}_k|}
$$

Then we can rewrite $C_p$ as

$$
C_p =
\frac{2}{N_s(N_s-1)}
\operatorname{Re}
\left\{
\sum_{k=1}^{N_s}
\left(\hat{x}^{\,cp}_k\right)^{*}
\sum_{j>k}
\hat{x}^{\,cp}_j
\right\}
\tag{C2}
$$

And let's use the symmetry of the summation with a summation over all pairs except $k=l$, and then dividing by two. 

$$
C_p =
\frac{1}{N_s(N_s-1)}
\operatorname{Re}
\left\{
\sum_{k=1}^{N_s}
\left(\hat{x}^{\,cp}_k\right)^{*}
\sum_{\substack{j=1\\j\ne k}}^{N_s}
\hat{x}^{\,cp}_j
\right\}
\tag{C3}
$$

And then let's sum over all the pairs, including $k=l$, but subtract the $k=l$ pairs seperately.  

$$
C_p =
\frac{1}{N_s(N_s-1)}
\operatorname{Re}
\left\{ \left(\sum_{k=1}^{N_s} \hat{x}^{\,cp}_k\right)^2
- \right\}
\tag{C4}
$$

This allows a further simplification.

$$
C_p =
\frac{1}{N_s(N_s-1)}
\left[
\left|
\sum_{k=1}^{N_s}
\hat{x}^{\,cp}_k
\right|^2
-
N_s
\right]
\tag{C5}
$$


Because $\left|\hat{x}^{\,cp}_k\right|=1$, this becomes

$$
C_p =
\frac{1}{N_s(N_s-1)}
\left[
\left|
\sum_{k=1}^{N_s}
\hat{x}^{\,cp}_k
\right|^2
-
N_s
\right]
\tag{C6}
$$

Finally, define the phase walkout $s$ as

$$
s =
\left|
\sum_{k=1}^{N_s}
\hat{x}^{\,cp}_k
\right|
\tag{C7}
$$
