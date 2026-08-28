# Energy computation

When we compute the coherent energies, we are averaging over station pairs.  However, averaging over pairs scales as $N_s$^2, as the number of stations squared.  It turns out that we can alternatively sum over stations and then square.  This will speed up our computations

## 1. Phase coherence computation

We want to compute the phase coherence $C_p$, averaging over all available station pairs.

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

In doing so, it is inconvenient to perform a multiplication for every pair of stations. So let's define, following equations S9-S14 in the supplementary information of Hawthorne and Ampuero (2018),

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

And let's use the symmetry of the summation with a summation over all pairs except $j=k$, and then dividing by two. 

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

And then let's sum over all the pairs, including $j=k$, but subtract the $j=k$ pairs separately.  

$$
C_p =
\frac{1}{N_s(N_s-1)}
\operatorname{Re}
\left\{
\sum_{k=1}^{N_s}
\left(\hat{x}^{\,cp}_k\right)^{*}
\sum_{j=1}^{N_s}
\hat{x}^{\,cp}_j
- \sum_{k=1}^{N_s}
\left|\hat{x}^{\,cp}_k\right|^2
\right\}
\tag{C4}
$$

This allows a further simplification.

$$
C_p =
\frac{1}{N_s(N_s-1)}
\left[\left|
\sum_{k=1}^{N_s}
\hat{x}^{\,cp}_k \right|^2
- \sum_{k=1}^{N_s}
\left|\hat{x}^{\,cp}_k\right|^2
\right]
\tag{C5}
$$

The real part operator is no longer needed, since the values inside the brackets are real.  And since $|\hat{x}^{\,cp}_k|=1$, this becomes

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

## 2. Unnormalized coherent energy computation

For the unnormalized coherent energy, we again compute the real part of the cross-spectrum between each pair of stations and average over all station pairs.  But this time there is no normalization of the cross-spectrum in the averaging.  

$$
E_u =
\frac{2}{N_s(N_s-1)}
\operatorname{Re}
\left\{
\sum_{k=1}^{N_s}
\sum_{j>k}
\hat{x}_k^{*}\hat{x}_j
\right\}
\tag{E1}
$$

We can follow the same steps as for the phase coherence to simplify this calculation over stations.

$$
E_u =
\frac{1}{N_s(N_s-1)}
\operatorname{Re}
\left\{
\sum_{k=1}^{N_s}
\hat{x}_k^{*}
\sum_{\substack{j=1\\j\ne k}}^{N_s}
\hat{x}_j
\right\}
\tag{E2}
$$

$$
E_u =
\frac{1}{N_s(N_s-1)}
\operatorname{Re}
\left\{
\sum_{k=1}^{N_s}
\hat{x}_k^{*}
\sum_{j=1}^{N_s}
\hat{x}_j
-
\sum_{k=1}^{N_s}
|\hat{x}_k|^2
\right\}
\tag{E3}
$$

$$
E_u =
\frac{1}{N_s(N_s-1)}
\left[
\left|
\sum_{k=1}^{N_s}
\hat{x}_k
\right|^2
-
\sum_{k=1}^{N_s}
|\hat{x}_k|^2
\right]
\tag{E4}
$$



## 3. Coherent energy ratio computation

Finally, we will consider the coherent energy ratio, averaged over station pairs.  

$$
E_r =
\frac{2}{N_s(N_s-1)}
\operatorname{Re}
\left\{
\sum_{k=1}^{N_s}
\sum_{j>k}
\frac{
\hat{x}_k\hat{x}_j^{*}
}{
|\hat{d}_{1k}|^{2}
|\hat{d}_{1j}|^{2}
}
\right\}
\tag{R1}
$$

Let us define

$$
\hat{x}^{\,Er}_k
\equiv
\frac{\hat{x}_k}{|\hat{d}_{1k}|^2}.
$$

The coherent energy ratio becomes

$$
E_r =
\frac{2}{N_s(N_s-1)}
\operatorname{Re}
\left\{
\sum_{k=1}^{N_s}
\sum_{j>k}
\hat{x}^{\,Er}_k
\left(\hat{x}^{\,Er}_j\right)^{*}
\right\}.
\tag{R2}
$$

And we can follow the same logic as above to simplify.

$$
E_r =
\frac{1}{N_s(N_s-1)}
\operatorname{Re}
\left\{
\sum_{k=1}^{N_s}
\hat{x}^{\,Er}_k
\sum_{\substack{j=1\\j\ne k}}^{N_s}
\left(\hat{x}^{\,Er}_j\right)^{*}
\right\}
\tag{R3}
$$

$$
E_r =
\frac{1}{N_s(N_s-1)}
\operatorname{Re}
\left\{
\sum_{k=1}^{N_s}
\hat{x}^{\,Er}_k
\sum_{j=1}^{N_s}
\left(\hat{x}^{\,Er}_j\right)^{*}
-
\sum_{k=1}^{N_s}
\left|\hat{x}^{\,Er}_k\right|^2
\right\}
\tag{R4}
$$

$$
E_r =
\frac{1}{N_s(N_s-1)}
\left[
\left|
\sum_{k=1}^{N_s}
\hat{x}^{\,Er}_k
\right|^2
-
\sum_{k=1}^{N_s}
\left|\hat{x}^{\,Er}_k\right|^2
\right]
\tag{R5}
$$
