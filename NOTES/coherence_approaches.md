# Frequency-domain coherence approaches

We want to consider several forms of coherence and coherent energy that we can compute from the frequency-domain seismograms.

We can consider computing five related frequency-domain measures for each station pair:

- Normalized phase coherence, $C_p$
- Unnormalized coherent energy, $E_u$
- Absolute unnormalized energy, $E_u^{\mathrm{abs}}$
- Coherent energy ratio, $E_r$
- Absolute energy ratio, $E_r^{\mathrm{abs}}$

## General notes

First, we may note a few frequency-domain properties about observed seismograms.  A hat denotes a Fourier transform evaluated at angular frequency $\omega$.

1. The observed waveform can be written as the source-time function multiplied by the path effect:

$$
\hat d_{jk}(\omega) = \hat s_{jk}(\omega)\,\hat g_k(\omega).
$$

   Here, $j$ denotes the earthquake and $k$ the station. $\hat s_{jk}$ is the station-dependent apparent source time function (ASTF), while $\hat g_k$ is the path or Green's-function term.

2. A target earthquake (1) can be compared with a nearby reference earthquake (2) at the same station using the unnormalized cross-spectrum:

$$
\hat x_k =
\hat d_{1k}\hat d_{2k}^{*}.
$$

3. If the two nearly co-located earthquakes share the same path effect, then:

$$
\hat x_k =
\hat s_{1k}\hat s_{2k}^{*}|\hat g_k|^2.
$$

   Consequently, the phase of $\hat x_k$ primarily reflects differences in the target earthquake's ASTF between stations, while its amplitude retains source and Green's-function power.

## Coherence $C_p$


The fully normalized inter-station phase coherence is the average cosine of the phase difference between every pair of stations:

$$
C_p =
\frac{2}{N(N-1)}
\sum_{k=1}^{N}\sum_{l=k+1}^{N}
\operatorname{Re}\left[
\frac{\hat x_{k}^{*}\hat x_{l}}
{|\hat x_{k}^{*}\hat x_{l}|}
\right].
$$

For a single station pair, with stations $k$ and $l$, 

$$
C_p =
\operatorname{Re}\left[
\frac{\hat x_{k}^{*}\hat x_{l}}
{|\hat x_{k}^{*}\hat x_{l}|}
\right].
$$


If the two earthquakes share the same Green's functions,

$$
E_p=
\frac{\operatorname{Re}\left[
\hat{s}_{1k}^{*}\hat{s}_{2k}
\hat{s}_{1l}\hat{s}_{2l}^{*}
|\hat g_k|^2|\hat g_l|^2
\right].}
{\left|
\hat{s}_{1k}\hat{s}_{2k}
\hat{s}_{1l}\hat{s}_{2l}\right|
|\hat g_k|^2|\hat g_l|^2
}
$$

At low frequencies, where the wavelength is longer than the rupture extent, the ASTFs are approximately the same across stations:

$$
\hat{s}_{1k}\approx\hat{s}_{1l}\approx\hat{s}_1,
\qquad
\hat{s}_{2k}\approx\hat{s}_{2l}\approx\hat{s}_2.
$$

Therefore,

$$
C_p = 1.
$$


At higher frequencies, if the target-event ASTFs become uncorrelated between stations,

$$
\left\langle
\hat{s}_{1k}^{*}\hat{s}_{1l}
\right\rangle\approx 0,
$$

then

$$
\left\langle C_p\right\rangle\approx 0.
$$

## Unnormalized coherent energy $E_u$

For a pair of stations $k$ and $l$, the unnormalized coherent energy is

$$
E_u =
\operatorname{Re}\left[
\hat{x}_{k}^{*}\hat{x}_{l}
\right].
$$

If the two earthquakes share the same Green's functions,

$$
E_u=
\operatorname{Re}\left[
\hat{s}_{1k}^{*}\hat{s}_{2k}
\hat{s}_{1l}\hat{s}_{2l}^{*}
|\hat g_k|^2|\hat g_l|^2
\right].
$$

At low frequencies, where the wavelength is longer than the rupture extent, the ASTFs are approximately the same across stations,

$$
E_u\propto
|\hat{s}_1|^2|\hat{s}_2|^2,
$$

with the proportionality modified by the Green's-function power,

$$
|\hat g_k|^2|\hat g_l|^2.
$$

At higher frequencies, if the target-event ASTFs become uncorrelated between stations,

$$
\left\langle
\hat{s}_{1k}^{*}\hat{s}_{1l}
\right\rangle\approx 0,
$$

then

$$
\left\langle E_u\right\rangle\approx 0.
$$

## Absolute unnormalized energy $E_u^{\mathrm{abs}}$

For a pair of stations $k$ and $l$, define

$$
E_u^{\mathrm{abs}}
=
|\hat d_{1k}|\,|\hat d_{2k}|\,
|\hat d_{1l}|\,|\hat d_{2l}|.
$$

If the earthquakes share the same Green's functions,

$$
E_u^{\mathrm{abs}}
=
|\hat{s}_{1k}|\,|\hat{s}_{2k}|\,
|\hat{s}_{1l}|\,|\hat{s}_{2l}|\,
|\hat g_k|^2|\hat g_l|^2.
$$

This provides an estimate of the product of the energy in the source time functions, weighted by the Green's-function power.



## Coherent energy ratio $E_r$

The coherent energy ratio is

$$
E_r =
\frac{
\hat d_{1k}\hat d_{2k}^{*}
\hat d_{1l}^{*}\hat d_{2l}
}{
|\hat d_{1k}|^{2}|\hat d_{1l}|^{2}
}.
$$

Using $\hat d_{jk}=\hat s_{jk}\hat g_k$, and assuming the two earthquakes share the same Green's functions,

$$
E_r =
\frac{
\hat s_{1k}\hat s_{2k}^{*}
\hat s_{1l}^{*}\hat s_{2l}
}{
|\hat s_{1k}|^{2}|\hat s_{1l}|^{2}
}.
$$

Here the Green's-functions cancel.

At low frequencies, where the source time functions are approximately the same across stations,

$$
E_r =
\frac{
|\hat s_1|^2|\hat s_2|^2
}{
|\hat s_1|^4
}
=
\frac{|\hat s_2|^2}{|\hat s_1|^2}.
$$

Thus, at low frequencies, $E_r$ measures the spectral-energy ratio of earthquake 2 relative to earthquake 1.

At higher frequencies, if the ASTFs of earthquake 1 become unrelated between stations,

$$
\left\langle
\hat s_{1k}\hat s_{1l}^{*}
\right\rangle\approx 0,
$$

then

$$
\left\langle E_r\right\rangle\approx 0.
$$

## Absolute energy ratio $E_r^{\mathrm{abs}}$

For a pair of stations $k$ and $l$, define

$$
E_r^{\mathrm{abs}}
=
\left|
\frac{
\hat d_{1k}\hat d_{2k}^{*}
\hat d_{1l}^{*}\hat d_{2l}
}{
|\hat d_{1k}|^{2}|\hat d_{1l}|^{2}
}
\right|.
$$

This simplifies to

$$
E_r^{\mathrm{abs}}
=
\frac{
|\hat d_{2k}|\,|\hat d_{2l}|
}{
|\hat d_{1k}|\,|\hat d_{1l}|
}.
$$

If the earthquakes share the same Green's functions,

$$
E_r^{\mathrm{abs}}
=
\frac{
|\hat s_{2k}|\,|\hat s_{2l}|
}{
|\hat s_{1k}|\,|\hat s_{1l}|
}.
$$

This provides an estimate of the spectral-energy ratio of earthquake 2 relative to earthquake 1, with the Green's-function power cancelled.
