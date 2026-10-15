"""
azimuth_coherence_tools.py
==========================
Station-pair phase coherence versus azimuth difference (reviewer test for the phase-coherence method).

Idea
----
After  xc = phscoh.calcxc(...); xc.calcmvout(); xc.cpfromr()  every station i has one cross-spectrum phase phi_i(f).
For a pair of stations (i, j) we use   c_ij(f) = cos(phi_i(f) - phi_j(f)).
  * c = 1 for identical phases, c ~ 0 (on average) for random phases (no 0.64 baseline of |cos(dphi/2)|).
  * The average of c over ALL pairs equals (N R^2 - 1)/(N - 1), R = |mean_i exp(i phi_i)|: the usual bias-corrected squared
    coherence, so the pair test is just a decomposition of the whole-array coherence you already have.
The pairs are grouped by |azimuth difference| (optionally also by the difference of the ray direction projected on the fault
plane) and averaged.  If the coherence profile reflects the rupture, c should fall with the separation near and above the
fall-off frequency ff, and should NOT depend on the separation well below ff.

Everything here works on the `xc` object of phscoh (attributes used: freq, xc, R, Cp, ffbest), so it runs unchanged on the
synthetic and on the real data.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

# ----------------------------------------------------------------------------------------------------------------------
# colours (sequential single-hue blue ramp, light = small separation, dark = large separation; ink for everything else)
# ----------------------------------------------------------------------------------------------------------------------
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e3e2de', '#fcfcfb'
SEQ_CMAP = LinearSegmentedColormap.from_list('seq_blue', ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'])


def bin_colors(n):
    """n ordered colours from the blue ramp (n <= 7 reads well); the lightest one still clears 2:1 on the surface."""
    return [SEQ_CMAP(v) for v in np.linspace(0.30, 1.0, max(n, 1))]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(INK2)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2)


# ----------------------------------------------------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------------------------------------------------
def _nanmean(x, axis=None):
    """np.nanmean that returns NaN (without a warning) for an all-NaN / empty slice."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(x, axis=axis)


def az_diff(a, b):
    """Smallest angle between two azimuths in degrees, 0 ... 180."""
    d = np.abs(np.asarray(a, float) - np.asarray(b, float)) % 360.0
    return np.minimum(d, 360.0 - d)


def as_vector(x):
    """Flat float array from a scalar, an array of any shape, or a list of scalars / small arrays (e.g. [array([90.]), array([90.]), ...])."""
    if isinstance(x, (list, tuple)):
        return np.concatenate([np.ravel(np.asarray(v, float)) for v in x]) if len(x) else np.zeros(0)
    return np.ravel(np.asarray(x, float))


def inplane_coords(az, takeoff=90.0, strike=0.0, dip=90.0):
    """
    Components of the unit ray vector (source -> station) in the fault plane, shape (N, 2) = (along strike, down dip).
    The apparent source time function of a station depends on THESE components (the delay of a sub-fault at position r is
    -(r . p)/c), not on the azimuth alone: two stations that are mirror images of each other across the fault plane see the
    same STF although |daz| is large.  |p_i - p_j| is therefore the physically controlling separation.
    az, strike in degrees from north; takeoff in degrees from vertically DOWN (90 = horizontal ray); dip in degrees.
    Every argument may be a scalar, an array or a list; strike, dip and takeoff may be one value for all stations or one per station.
    """
    az, takeoff, strike, dip = (as_vector(v) for v in (az, takeoff, strike, dip))
    n = max(len(az), len(takeoff), len(strike), len(dip))
    for name, v in (('az', az), ('takeoff', takeoff), ('strike', strike), ('dip', dip)):
        if len(v) not in (1, n):
            raise ValueError(f'inplane_coords: {name} has {len(v)} values, expected 1 or {n}')
    az, takeoff, strike, dip = (np.broadcast_to(v, (n,)) for v in (az, takeoff, strike, dip))
    az, takeoff, st, dp = np.radians(az), np.radians(takeoff), np.radians(strike), np.radians(dip)
    u = np.stack([np.sin(takeoff) * np.cos(az), np.sin(takeoff) * np.sin(az), -np.cos(takeoff)], axis=-1)             # ray direction (N, E, Up)
    s = np.stack([np.cos(st), np.sin(st), np.zeros(n)], axis=-1)                                                      # along strike
    dd = np.stack([np.cos(st + np.pi / 2) * np.cos(dp), np.sin(st + np.pi / 2) * np.cos(dp), -np.sin(dp)], axis=-1)    # down dip
    return np.stack([(u * s).sum(axis=-1), (u * dd).sum(axis=-1)], axis=-1)


# ----------------------------------------------------------------------------------------------------------------------
# phases from the phscoh object
# ----------------------------------------------------------------------------------------------------------------------
def _unit(z):
    a = np.abs(z)
    return np.where(a > 0, z / np.where(a > 0, a, 1.0), 0.0)


def _resultant(u):
    return np.abs(np.nanmean(u, axis=1))


def _r_estimates(A):
    """Candidate definitions of the resultant length R for cross-spectra A[f, station, taper]; name -> (R(f), variant it belongs to)."""
    K = A.shape[2]
    S = A.sum(axis=2)
    out = {'unit phasors of the taper-summed spectrum': (_resultant(_unit(S)), 'sum'),
           'amplitude-weighted, taper-summed': (np.abs(S.sum(axis=1)) / np.abs(S).sum(axis=1), 'sum')}
    if K > 1:
        out['all (station, taper) unit phasors pooled'] = (np.abs(np.nanmean(_unit(A), axis=(1, 2))), 'meanunit')
        out['amplitude-weighted, all (station, taper) pooled'] = (np.abs(A.sum(axis=(1, 2))) / np.abs(A).sum(axis=(1, 2)), 'meanunit')
        out['mean over tapers of the per-taper R'] = (np.mean([_resultant(_unit(A[:, :, k])) for k in range(K)], axis=0), 'meanunit')
    return out


def get_phasors(xc, attr=None, tol=2e-3, verbose=True):
    """
    Station phasors v[f, station] of the cross-spectrum phase, as used for the coherence, and a dict with what was chosen.

    I cannot see phscoh.py, so the choice is made by VALIDATION against xc.R.  Two things are unknown:
      (1) which attribute holds the moveout-corrected cross-spectra: every complex array of the xc object with a frequency axis is tried;
      (2) how the tapers are combined.  With tpr='multi' the arrays are [frequency, station, taper] (3 tapers).  Two readings are tried:
          'sum'      one unit phasor per station, the phase of the taper-summed cross-spectrum (the usual multitaper cross-spectrum);
          'meanunit' the mean of the unit phasors of the tapers of each station (length <= 1).  Then c_ij = mean over all taper pairs of
                     cos(phi_i,k - phi_j,l), which is what a coherence pooled over (station, taper) measures, and the length |v_i| < 1 lowers c
                     for stations whose own tapers disagree (noise).
    Each (attribute, reading) is compared with xc.R through several definitions of R (see _r_estimates); the best match below tol is used.  If nothing
    matches, the taper-summed raw cross-spectra xc.xc are used and a warning says so.  Force a choice with attr='xc' (best reading of that attribute) or
    attr='xc:sum' / attr='xc:meanunit'.
    """
    freq = np.asarray(xc.freq)
    nf = len(freq)
    cands = {}
    for name in dir(xc):
        if name.startswith('__'):
            continue
        try:
            val = getattr(xc, name)
        except Exception:
            continue
        if isinstance(val, np.ndarray) and np.iscomplexobj(val) and val.ndim in (2, 3):
            arr = val[:, :, None] if val.ndim == 2 else val
            if arr.shape[0] == nf:
                cands[name] = arr
            elif arr.shape[1] == nf:
                cands[name] = np.swapaxes(arr, 0, 1)
    if 'xc' not in cands:
        raise ValueError('xc.xc (complex cross-spectra with a frequency axis) not found - shape is ' + str(getattr(getattr(xc, 'xc', None), 'shape', None)))
    Rref = np.squeeze(np.asarray(getattr(xc, 'R', np.nan), float))
    have_R = Rref.shape == (nf,)
    variants, errs, est_err = {}, {}, {}
    import warnings
    for name, A in cands.items():
        K = A.shape[2]
        with warnings.catch_warnings(), np.errstate(all='ignore'):          # arrays such as the noise spectra contain zeros / NaN: they simply do not match xc.R
            warnings.simplefilter('ignore')
            ests = _r_estimates(A)
            for red in (('sum',) if K == 1 else ('sum', 'meanunit')):
                key = name if K == 1 else f'{name}:{red}'
                variants[key] = _unit(A.sum(axis=2)) if red == 'sum' else np.nanmean(_unit(A), axis=2)
                if have_R:
                    e = {en: float(np.nanmax(np.abs(Rv - Rref))) for en, (Rv, belongs) in ests.items() if belongs == red and np.isfinite(Rv).any()}
                    est_err[key] = e
                    errs[key] = min(e.values()) if e else np.nan
    def base(k):
        return k.split(':')[0]
    if attr is not None:
        pool = [k for k in variants if k == attr or k.startswith(attr + ':')]
        if not pool:
            raise ValueError(f'attr={attr!r}: no such array; available: {list(variants)}')
        chosen = sorted(pool, key=lambda k: (errs.get(k, 0.0) if np.isfinite(errs.get(k, 0.0)) else 9.9, k.endswith(':meanunit')))[0]
        matched = None
    else:
        good = [k for k, v in errs.items() if np.isfinite(v) and v < tol]
        if good:
            chosen = sorted(good, key=lambda k: (round(errs[k], 9), base(k) == 'xc', k.endswith(':meanunit')))[0]
            matched = True
        else:
            chosen = 'xc:sum' if 'xc:sum' in variants else 'xc'
            matched = False
    best_est = min(est_err[chosen], key=est_err[chosen].get) if chosen in est_err and est_err[chosen] else None
    info = {'attr': chosen, 'matched': matched, 'R definition': best_est, 'candidates': {k: float(v) for k, v in errs.items()},
            'n_tapers': {k: int(A.shape[2]) for k, A in cands.items()}}
    if verbose:
        print(f"phases: using xc.{chosen}" + (f"   (matches xc.R through: {best_est}, max |dR| = {errs[chosen]:.1e})" if matched else ''))
        if errs:
            print('  max |R_own - xc.R| per candidate: ' + ', '.join(f'{k}: {v:.1e}' for k, v in errs.items() if np.isfinite(v)))
        if matched is False:
            print('  WARNING: no complex array / taper reading reproduces xc.R -> taper-summed raw xc.xc used. If the events are NOT aligned on their own onsets, '
                  'the large-separation pairs will be biased low by the moveout; set attr= to the attribute that holds the corrected spectra.')
    return variants[chosen], info


# ----------------------------------------------------------------------------------------------------------------------
# pair statistics
# ----------------------------------------------------------------------------------------------------------------------
def pair_coherence(u):
    """c[f, pair] = cos(phi_i - phi_j) for all i < j, and the station indices i, j of every pair."""
    i, j = np.triu_indices(u.shape[1], k=1)
    return np.real(u[:, i] * np.conj(u[:, j])), i, j


def identity_check(u, c, xc=None):
    """The all-pairs mean of c must equal (N R^2 - s2)/(N - 1), R = |mean_i v_i|, s2 = mean_i |v_i|^2 (= 1 for unit phasors: the usual (N R^2 - 1)/(N - 1));
    sqrt of it should reproduce xc.Cp if cpfromr uses that correction (with several tapers cpfromr may use another normalisation: then only the first number matters)."""
    N = u.shape[1]
    R = _resultant(u)
    s2 = np.nanmean(np.abs(u) ** 2, axis=1)
    c_all = np.nanmean(c, axis=1)
    out = {'N': N, 'max |mean pair c - (N R^2 - s2)/(N - 1)|': float(np.nanmax(np.abs(c_all - (N * R**2 - s2) / (N - 1))))}
    Cp_own = np.sqrt(np.clip(c_all, 0, None))
    out['Cp_own'] = Cp_own
    if xc is not None and hasattr(xc, 'Cp') and np.size(xc.Cp) == len(Cp_own):
        out['max |sqrt(mean c) - xc.Cp|'] = float(np.nanmax(np.abs(Cp_own - np.squeeze(np.asarray(xc.Cp, float)))))
    return out, c_all


def _in_bin(sep, edges, b):
    """Pairs of bin b: edges[b] <= sep < edges[b+1]; the last bin also contains its upper edge."""
    return (sep >= edges[b]) & ((sep < edges[b + 1]) if b < len(edges) - 2 else (sep <= edges[b + 1]))


def bin_pairs(c, sep, edges, i, j, min_pairs=5):
    """
    Mean pair coherence per separation bin: mean[bin, f], jackknife standard error se[bin, f] (leave one STATION out, because the
    pairs share stations and are not independent), and the number of pairs n[bin].  Bins with fewer than min_pairs pairs are NaN.
    """
    edges = np.asarray(edges, float)
    nb, nf, N = len(edges) - 1, c.shape[0], int(max(i.max(), j.max())) + 1
    mean, se, n = np.full((nb, nf), np.nan), np.full((nb, nf), np.nan), np.zeros(nb, int)
    for b in range(nb):
        m = _in_bin(sep, edges, b)
        n[b] = m.sum()
        if n[b] < min_pairs:
            continue
        mean[b] = np.nanmean(c[:, m], axis=1)
        th = np.array([np.nanmean(c[:, m & (i != s) & (j != s)], axis=1) for s in range(N) if (m & (i != s) & (j != s)).sum() >= 1])
        if len(th) > 2:
            k = len(th)
            se[b] = np.sqrt((k - 1) / k * np.nansum((th - np.nanmean(th, axis=0)) ** 2, axis=0))
    return mean, se, n


def perm_test(c_band, sep_of, edges, n_stations, min_pairs=5, n_perm=5000, seed=0):
    """
    Two statistics of the band-averaged pair coherence against the pair separation, each with a one-sided permutation p-value
    (H1: coherence DEcreases with separation):
      rho   Spearman rank correlation over all pairs
      diff  mean coherence of the pairs in the FIRST bin minus the pairs in the LAST bin of `edges`  (the effect size, in coherence units)
    The null shuffles the separation labels among the STATIONS (not the pairs), which keeps the dependence between pairs that share a station.
    sep_of(idx) -> separations of all pairs when station k carries the label idx[k].
    """
    rng = np.random.default_rng(seed)
    ok = np.isfinite(c_band)
    rc = stats.rankdata(c_band[ok])
    last = len(edges) - 2

    def stat(sep):
        sep = np.asarray(sep)
        rho = np.corrcoef(stats.rankdata(sep[ok]), rc)[0, 1]
        first_m, last_m = _in_bin(sep, edges, 0) & ok, _in_bin(sep, edges, last) & ok
        diff = c_band[first_m].mean() - c_band[last_m].mean() if first_m.sum() >= min_pairs and last_m.sum() >= min_pairs else np.nan
        return rho, diff

    obs_rho, obs_diff = stat(sep_of(np.arange(n_stations)))
    if n_perm is None or n_perm <= 0:                                  # no permutations: observed statistics only, p-values NaN
        return dict(rho=obs_rho, p_rho=np.nan, diff=obs_diff, p_diff=np.nan)
    null = np.array([stat(sep_of(rng.permutation(n_stations))) for _ in range(n_perm)])
    nd = null[np.isfinite(null[:, 1]), 1]
    return dict(rho=obs_rho, p_rho=(1 + np.sum(null[:, 0] <= obs_rho)) / (n_perm + 1),
                diff=obs_diff, p_diff=(1 + np.sum(nd >= obs_diff)) / (len(nd) + 1) if np.isfinite(obs_diff) and len(nd) else np.nan)


# ----------------------------------------------------------------------------------------------------------------------
# the whole analysis for one xc object
# ----------------------------------------------------------------------------------------------------------------------
def analyse_array(xc, az, ff=None, fmax=None, daz_edges=(0, 10, 30, 60, 90, 180), bands=((0.1, 0.5), (0.5, 1.0), (1.0, 2.0)),
                  min_pairs=5, n_perm=5000, P=None, dp_edges=(0, 0.05, 0.1, 0.2, 0.4, 2.0), attr=None, seed=0, verbose=True):
    """
    xc      phscoh object after calcxc / calcmvout / cpfromr
    az      azimuth (deg) of every station, IN THE ORDER OF THE xc COLUMNS
    ff      fall-off frequency (Hz); default xc.ffbest.  bands are given in units of ff
    fmax    highest frequency used (Hz), e.g. the SNR limit of the real data
    P       optional (N, 2) in-plane ray components (inplane_coords) -> a second analysis with |p_i - p_j| as the separation
    """
    az = np.asarray(az, float)
    u, info = get_phasors(xc, attr=attr, verbose=verbose)
    N = u.shape[1]
    if len(az) != N:
        raise ValueError(f'{len(az)} azimuths but {N} stations in xc')
    freq = np.asarray(xc.freq, float)
    c, i, j = pair_coherence(u)
    ident, c_all = identity_check(u, c, xc)
    if verbose:
        print(f"identity check, N = {N} stations, {len(i)} pairs:  " + ',  '.join(f'{k} = {v:.1e}' for k, v in ident.items() if k.startswith('max')))
        if ident.get('max |sqrt(mean c) - xc.Cp|', 0.0) > 1e-3:
            print('  (the first number must be ~1e-15: it proves the pairs decompose the array coherence of the phasors used.  The second differs from 0 when cpfromr applies another '
                  'correction or weighting than the station-level phasors used here, e.g. with several tapers; that does not affect the pair analysis)')
    ff = float(xc.ffbest) if ff is None else float(ff)
    keep = np.ones(len(freq), bool) if fmax is None else freq <= fmax
    freq, c, c_all, Cp_own = freq[keep], c[keep], c_all[keep], ident['Cp_own'][keep]

    specs = {'daz': dict(label='azimuth difference |Δaz| (°)', short='Δaz', unit='°', sep=az_diff(az[i], az[j]), edges=np.asarray(daz_edges, float),
                         sep_of=lambda idx: az_diff(az[idx][i], az[idx][j]))}
    if P is not None:
        P = np.asarray(P, float)
        specs['dp'] = dict(label='in-plane ray separation |Δp|', short='Δp', unit='', sep=np.linalg.norm(P[i] - P[j], axis=1), edges=np.asarray(dp_edges, float),
                           sep_of=lambda idx: np.linalg.norm(P[idx][i] - P[idx][j], axis=1))
    for key, sp in specs.items():
        sp['mean'], sp['se'], sp['n'] = bin_pairs(c, sp['sep'], sp['edges'], i, j, min_pairs)
        rows, sp['c_band'], sp['band_bins'] = [], [], []
        for (lo, hi) in bands:
            f_lo, f_hi = lo * ff, min(hi * ff, freq.max())
            sel = (freq >= f_lo) & (freq <= f_hi)
            row = {'band': f'{lo:g}-{hi:g} ff', 'f_lo (Hz)': f_lo, 'f_hi (Hz)': f_hi, 'n_freq': int(sel.sum())}
            if sel.sum() >= 2:
                cb = _nanmean(c[sel], axis=0)
                bb = np.array([_nanmean(cb[_in_bin(sp['sep'], sp['edges'], b)]) if sp['n'][b] >= min_pairs else np.nan for b in range(len(sp['edges']) - 1)])
                good = bb[np.isfinite(bb)]
                row['c first bin'], row['c last bin'] = (good[0], good[-1]) if len(good) else (np.nan, np.nan)
                pt = perm_test(cb, sp['sep_of'], sp['edges'], N, min_pairs, n_perm, seed)
                row.update({'diff (first - last)': pt['diff'], 'p (diff)': pt['p_diff'], 'rho': pt['rho'], 'p (rho)': pt['p_rho']})
            else:
                cb, bb = np.full(c.shape[1], np.nan), np.full(len(sp['edges']) - 1, np.nan)
                row.update({'c first bin': np.nan, 'c last bin': np.nan, 'diff (first - last)': np.nan, 'p (diff)': np.nan, 'rho': np.nan, 'p (rho)': np.nan})
            sp['c_band'].append(cb)
            sp['band_bins'].append(bb)
            rows.append(row)
        sp['table'] = rows
    return dict(freq=freq, c=c, i=i, j=j, N=N, az=az, ff=ff, info=info, identity=ident, c_all=c_all, Cp_own=Cp_own, specs=specs, bands=bands, min_pairs=min_pairs)


def band_table(res, spec='daz'):
    import pandas as pd
    return pd.DataFrame(res['specs'][spec]['table']).set_index('band')


# ----------------------------------------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------------------------------------
def plot_curves(ax, res, spec='daz', xlim=None, ylabel='pair coherence  ⟨cos Δφ⟩', band_label='shading: ±1 SE of the\nleave-one-station-out jackknife'):
    sp = res['specs'][spec]
    cols = bin_colors(len(sp['edges']) - 1)
    for b in range(len(sp['edges']) - 1):
        if sp['n'][b] < res['min_pairs']:
            continue
        lab = f"{sp['edges'][b]:g}–{sp['edges'][b + 1]:g}{sp['unit']}  (n = {sp['n'][b]})"
        ax.fill_between(res['freq'], sp['mean'][b] - sp['se'][b], sp['mean'][b] + sp['se'][b], color=cols[b], alpha=0.18, lw=0)
        ax.plot(res['freq'], sp['mean'][b], color=cols[b], lw=2, label=lab)
    ax.plot(res['freq'], res['c_all'], color=INK, lw=2.5, label=f"whole array  (N = {res['N']})")
    ax.axhline(0, color=INK2, lw=0.8)
    lim = xlim if xlim else (res['freq'][0], res['freq'][-1])
    if lim[0] <= res['ff'] <= lim[1]:
        ax.axvline(res['ff'], color=INK2, lw=1.2, ls='--')
        ax.text(res['ff'], 1.04, ' ff', color=INK2, va='bottom', ha='left', fontsize=10)
    ax.set_xscale('log')
    ax.set_xlim(lim)
    ax.set_ylim(-0.5, 1.1)
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel(ylabel)
    from matplotlib.patches import Patch
    handles, labels = ax.get_legend_handles_labels()
    if band_label and np.isfinite(sp['se']).any():                    # legend entry for the shaded bands (same grey for every group)
        handles.append(Patch(facecolor=INK2, alpha=0.30, lw=0))
        labels.append(band_label)
    ax.legend(handles, labels, title='separation ' + sp['short'], loc='lower left', fontsize=8.5, title_fontsize=9, frameon=True, facecolor=SURFACE, edgecolor='none', framealpha=0.92)
    _style(ax)


def plot_band(ax, res, k, spec='daz'):
    sp = res['specs'][spec]
    cb, row = sp['c_band'][k], sp['table'][k]
    ax.scatter(sp['sep'], cb, s=14, color='#86b6ef', alpha=0.35, lw=0, label='station pairs')
    ctr = 0.5 * (sp['edges'][:-1] + sp['edges'][1:])
    bm = np.array([_nanmean(cb[_in_bin(sp['sep'], sp['edges'], b)]) if sp['n'][b] >= res['min_pairs'] else np.nan for b in range(len(ctr))])
    ax.plot(ctr, bm, 'o-', color='#0d366b', lw=2, ms=7, mec=SURFACE, mew=2, label='bin mean')
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylim(-1.05, 1.1)
    ax.set_xlabel(sp['label'])
    ax.set_ylabel('band-mean pair coherence')
    ax.set_xlim(sp['edges'][0] - 0.03 * np.ptp(sp['edges']), sp['edges'][-1] * 1.03 if spec == 'daz' else sp['edges'][-1] + 0.03 * np.ptp(sp['edges']))
    if row['n_freq'] >= 2:
        ax.set_title(f"{row['band']}  ({row['f_lo (Hz)']:.1f}–{row['f_hi (Hz)']:.1f} Hz)", fontsize=11, color=INK)
    else:
        ax.set_title(f"{row['band']}  (no frequencies below the upper limit)", fontsize=11, color=INK)
    d, pd_ = row['diff (first - last)'], row['p (diff)']
    if np.isfinite(pd_):
        txt = f"Δc (first − last bin) = {d:+.2f},  p = {pd_:.3f}\nρ = {row['rho']:+.2f},  p = {row['p (rho)']:.3f}" + ('\neffect negligible (|Δc| < 0.02)' if abs(d) < 0.02 else '')
    else:
        txt = 'band too narrow or bins too sparse'
    ax.text(0.97, 0.05, txt, transform=ax.transAxes, ha='right', va='bottom', fontsize=9, color=INK)
    _style(ax)


def plot_summary(res, spec='daz', title='', xlim=None):
    nb = len(res['bands'])
    fig = plt.figure(figsize=(5.2 * 2, 4.3 * 2), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2)
    plot_curves(fig.add_subplot(gs[0, 0]), res, spec, xlim)
    for k in range(min(nb, 3)):
        plot_band(fig.add_subplot(gs[(k + 1) // 2, (k + 1) % 2]), res, k, spec)
    fig.suptitle(title, color=INK, fontsize=13)
    fig.tight_layout()
    return fig


# diverging map for the coherence (-1 ... 1): blue = coherent, grey = 0 (nothing in common), red = opposite phase (negative coherence)
DIV_CMAP = LinearSegmentedColormap.from_list('div_blue_red', ['#8f2a2a', '#e34948', '#f2a99b', '#f0efec', '#9ec5f4', '#3987e5', '#184f95'])


def plot_heatmap(ax, res, spec='daz', step=5.0, start=0.0, vmax_sep=None, xlim=None, min_pairs=1, cbar_label='Coherence', contour=0.5):
    """Mean pair coherence as a function of frequency (x) and separation (y), diverging colours centred on 0.  `step`/`start` set the separation bins
    (start = step/2 puts the bin centres on a 5-degree station grid).  A black line follows `contour` (None = no line)."""
    from matplotlib.colors import TwoSlopeNorm
    sp = res['specs'][spec]
    top = vmax_sep if vmax_sep else sp['sep'].max()
    edges = np.arange(start, top + step, step)
    mean, _, n = bin_pairs(res['c'], sp['sep'], edges, res['i'], res['j'], min_pairs)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    pm = ax.pcolormesh(res['freq'], ctr, mean, cmap=DIV_CMAP, norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), shading='nearest')
    if contour is not None:
        ax.contour(res['freq'], ctr, np.nan_to_num(mean, nan=-9.0), levels=[contour], colors=[INK], linewidths=1.4)
    ax.axvline(res['ff'], color=INK2, lw=1.2, ls='--')
    ax.set_xscale('log')
    ax.set_xlim(xlim if xlim else (res['freq'][0], res['freq'][-1]))
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel(sp['label'])
    cb = plt.colorbar(pm, ax=ax)
    cb.set_label(cbar_label, color=INK2)
    cb.set_ticks([-1, -0.5, 0, 0.5, 1])
    cb.outline.set_visible(False)
    ax.set_facecolor(SURFACE)
    ax.text(res['ff'], 1.01, ' ff', transform=ax.get_xaxis_transform(), color=INK2, va='bottom', ha='left', fontsize=10)
    return pm


def plot_coherence(res, spec='daz', title='', xlim=None, figsize=(7.4, 4.7)):
    """FIGURE 1: pair coherence against frequency, one curve per group of station pairs (grouped by separation), jackknife band, whole array in black."""
    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    plot_curves(ax, res, spec, xlim, ylabel='Coherence')
    if title:
        ax.set_title(title, color=INK, fontsize=12, pad=14)
    fig.tight_layout()
    return fig


def plot_heatmap_figure(res, spec='daz', step=5.0, start=2.5, title='', xlim=None, min_pairs=1, figsize=(7.8, 4.7)):
    """FIGURE 2: coherence as a function of frequency and azimuth difference (|Δaz|, or |Δp| with spec='dp')."""
    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    plot_heatmap(ax, res, spec, step=step, start=start, xlim=xlim, min_pairs=min_pairs)
    if title:
        ax.set_title(title, color=INK, fontsize=12, pad=24)
    ax.text(0.012, 0.975, 'black line: coherence = 0.5', transform=ax.transAxes, ha='left', va='top', fontsize=8.5, color=INK,
            bbox=dict(boxstyle='round,pad=0.25', fc=SURFACE, ec='none', alpha=0.95))
    fig.tight_layout()
    return fig


def plot_pooled(results, spec='daz', k=1, ax=None):
    """Band k (index into bands) coherence per separation bin, one thin line per event + the mean over events."""
    ax = ax or plt.subplots(figsize=(5.5, 4.3), facecolor=SURFACE)[1]
    first = next(iter(results.values()))['specs'][spec]
    ctr = 0.5 * (first['edges'][:-1] + first['edges'][1:])
    allv = []
    for name, r in results.items():
        v = r['specs'][spec]['band_bins'][k]
        allv.append(v)
        ax.plot(ctr, v, 'o-', color='#6da7ec', lw=1.2, ms=4, alpha=0.8)
        ax.annotate(str(name), (ctr[np.isfinite(v)][-1] if np.isfinite(v).any() else ctr[-1], v[np.isfinite(v)][-1] if np.isfinite(v).any() else 0), fontsize=7, color=INK2, xytext=(3, 0), textcoords='offset points')
    mv = _nanmean(np.array(allv), axis=0)
    if np.isfinite(mv).any():
        ax.plot(ctr, mv, 'o-', color='#0d366b', lw=2.5, ms=7, mec=SURFACE, mew=2, label='mean over events')
    else:
        ax.text(0.5, 0.5, 'band empty (above FMAX)', transform=ax.transAxes, ha='center', color=INK2)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel(first['label'])
    ax.set_ylabel('band-mean pair coherence')
    ax.set_title(f"band {first['table'][k]['band']}", fontsize=11, color=INK)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False)
    _style(ax)
    return ax


def results_table(results, spec='daz'):
    """One row per event and band: Spearman rho, permutation p, coherence of the closest-separation bin and of the widest bin."""
    import pandas as pd
    rows = []
    for name, r in results.items():
        for row in r['specs'][spec]['table']:
            rows.append({'event': name, 'N stations': r['N'], 'ff (Hz)': round(r['ff'], 2), **{k: v for k, v in row.items() if k in ('band', 'c first bin', 'c last bin', 'diff (first - last)', 'p (diff)', 'rho', 'p (rho)')}})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------------------
# helpers for real data (obspy streams + phscoh); obspy / phscoh are only needed when these are called
# ----------------------------------------------------------------------------------------------------------------------
def ensure_azimuths(st1, inv=None, eq=None, lat=None, lon=None):
    """Azimuth (deg, source -> station) of every trace of st1 in stream order. Uses tr.stats.azimuth if present, else computes it from the inventory and the event."""
    out = []
    for tr in st1:
        if 'azimuth' not in tr.stats:
            from obspy.geodetics import gps2dist_azimuth
            ev_lat = lat if lat is not None else eq.origins[0].latitude
            ev_lon = lon if lon is not None else eq.origins[0].longitude
            sla = slo = None
            try:
                c = inv.get_coordinates(tr.get_id())
                sla, slo = c['latitude'], c['longitude']
            except Exception:
                for net in inv:
                    if net.code == tr.stats.network:
                        for sta in net:
                            if sta.code == tr.stats.station:
                                sla, slo = sta.latitude, sta.longitude
            if sla is None:
                raise ValueError(f'no coordinates for {tr.get_id()}')
            tr.stats.azimuth = gps2dist_azimuth(ev_lat, ev_lon, sla, slo)[1]
        out.append(float(tr.stats.azimuth))
    return np.array(out)


def _xc_array(xcobj):
    """xc.xc as a 3-D array [frequency, station, taper]."""
    A = np.asarray(xcobj.xc)
    return A[:, :, None] if A.ndim == 2 else A


def _fresh(st):
    """Deep copy of a stream (or list of traces): calcxc may modify its input in place, so every test call gets its own copy."""
    import obspy
    return obspy.Stream([tr.copy() for tr in st])


def _rel_err(x, y):
    n = np.linalg.norm(np.asarray(x).ravel())
    return float(np.linalg.norm((np.asarray(x) - np.asarray(y)).ravel()) / n) if n > 0 else np.inf


def check_station_order(phscoh, st1, st2, rtol=1e-6, verbose=True, **xc_kw):
    """
    True if column k of xc.xc belongs to st1[k] (and xc has one column per trace).  Test: calcxc is run twice on copies of the streams, once with st1 reversed; if the output follows the order of st1,
    the second result is the first one with its station axis flipped.  If this is False the columns are NOT in the order of st1: the reason is printed, and
    resolve_station_order() finds the actual order.
    """
    a = _xc_array(phscoh.calcxc(_fresh(st1), _fresh(st2), **xc_kw))
    b = _xc_array(phscoh.calcxc(_fresh(list(st1)[::-1]), _fresh(st2), **xc_kw))
    if a.shape[1] != len(st1):
        if verbose:
            print(f'station order check FAILED: xc has {a.shape[1]} columns but st1 has {len(st1)} traces (stations without a usable reference or data are dropped inside phscoh), '
                  'so the columns are not station k of st1')
        return False
    if a.shape != b.shape:
        if verbose:
            print(f'station order check: the two runs give different shapes {a.shape} and {b.shape}')
        return False
    atol = 1e-9 * np.nanmax(np.abs(a))
    if np.allclose(a, b[:, ::-1], rtol=rtol, atol=atol):
        return True
    if verbose:
        if np.allclose(a, b, rtol=rtol, atol=atol):
            why = 'the columns do not depend on the order of the input: phscoh orders the stations itself (sorted by name or by something else)'
        else:
            E = np.array([[_rel_err(a[:, k], b[:, j]) for j in range(b.shape[1])] for k in range(a.shape[1])])
            same_set = bool(np.all(E.min(axis=1) < 1e-3))
            why = ('the two runs contain the same spectra in a different order' if same_set else
                   'the two runs give different spectra for the same stations (calcxc changes its input or uses random numbers)') + f'; worst best-match relative error {E.min(axis=1).max():.1e}'
        print('station order check FAILED: ' + why)
    return False


def resolve_station_order(phscoh, st1, st2, tol=1e-3, **xc_kw):
    """
    order[k] = index in st1 of the station that is in column k of the xc built from st1 and st2 (xc may have fewer columns than st1 has traces: stations that
    phscoh drops simply do not appear in order).  Use  az_columns = az[order].
    Found by leaving every station out in turn: calcxc is run on st1 without trace i, and the column of the full run that has no counterpart in that run belongs to
    station i (no missing column: station i is not in xc).  Needs N + 1 calcxc runs, each on copies.  Raises if the assignment is not one-to-one and exact.
    """
    full = _xc_array(phscoh.calcxc(_fresh(st1), _fresh(st2), **xc_kw))
    M, traces = full.shape[1], list(st1)
    owner = [None] * M
    for i in range(len(traces)):
        sub = _xc_array(phscoh.calcxc(_fresh(traces[:i] + traces[i + 1:]), _fresh(st2), **xc_kw))
        E = np.array([[_rel_err(full[:, k, :], sub[:, j, :]) for j in range(sub.shape[1])] for k in range(M)]).reshape(M, sub.shape[1])
        missing = [k for k in range(M) if E[k].size == 0 or E[k].min() >= tol]
        if len(missing) > 1 or (len(missing) == 1 and owner[missing[0]] is not None):
            raise RuntimeError(f'leaving out station {i} ({traces[i].get_id()}) changes {len(missing)} columns: calcxc uses array-wide information '
                               '(e.g. a common window or reference), so the column order cannot be resolved this way')
        if len(missing) == 1:
            owner[missing[0]] = i
    if any(o is None for o in owner):
        raise RuntimeError(f'columns {[k for k, o in enumerate(owner) if o is None]} of xc could not be assigned to a station')
    return np.array(owner)


def ff_from_cp(freq, Cp, level=0.5):
    """Fall-off frequency as the first crossing of Cp = level (linear interpolation between the two samples around the crossing).  A plain, transparent
    stand-in for phscoh's pickffreq: it does not use its smoothing / cutoff / freqmin logic, so it can differ from the catalogue value."""
    Cp = np.squeeze(np.asarray(Cp, float))
    freq = np.asarray(freq, float)
    below = np.where(Cp < level)[0]
    if len(below) == 0:
        return float(freq[-1])
    k = below[0]
    if k == 0:
        return float(freq[0])
    f0, f1, c0, c1 = freq[k - 1], freq[k], Cp[k - 1], Cp[k]
    return float(f0 + (level - c0) * (f1 - f0) / (c1 - c0))


def safe_pickffreq(xc, verbose=True):
    """
    xc.pickffreq(), but if it raises (for instance phscoh's  ffall[k] = np.interp([0.5], ...)  fails with recent NumPy because a length-1 array is assigned to
    a scalar slot) xc.ffbest is set from the first crossing of Cp = 0.5 instead and a message says so (once per session).  Returns 'phscoh' or 'fallback'.
    The azimuth analysis only uses ff to define the frequency bands, so this is enough for it; for the paper's ff values fix phscoh.py or pass FF = catalogue value.
    """
    try:
        xc.pickffreq()
        return 'phscoh'
    except Exception as e:
        xc.ffbest = ff_from_cp(xc.freq, xc.Cp)
        if verbose and not getattr(safe_pickffreq, '_warned', False):
            print(f'NOTE: xc.pickffreq() failed ({type(e).__name__}: {e}); using the first crossing of Cp = 0.5 as ff instead (here {xc.ffbest:.2f} Hz). '
                  'Probable cause: np.interp([0.5], ...) assigned to a scalar element in phscoh.pickffreq under a recent NumPy -> change [0.5] to 0.5 there.')
            safe_pickffreq._warned = True
        return 'fallback'


def save_result(res, path):
    """Compact JSON of what the pooled figure and the tables need (the full result holds functions and cannot be pickled)."""
    import json
    keep = {'N': res['N'], 'ff': res['ff'], 'bands': [list(b) for b in res['bands']], 'min_pairs': res['min_pairs'],
            'specs': {k: {'label': sp['label'], 'short': sp['short'], 'unit': sp['unit'], 'edges': sp['edges'].tolist(), 'n': sp['n'].tolist(),
                          'band_bins': [list(map(float, b)) for b in sp['band_bins']], 'table': sp['table']} for k, sp in res['specs'].items()}}
    with open(path, 'w') as fh:
        json.dump(keep, fh, default=float, indent=1)


def load_result(path):
    import json
    with open(path) as fh:
        d = json.load(fh)
    for sp in d['specs'].values():
        sp['edges'], sp['n'] = np.array(sp['edges'], float), np.array(sp['n'], int)
        sp['band_bins'] = [np.array(b, float) for b in sp['band_bins']]
    return d


# ----------------------------------------------------------------------------------------------------------------------
# synthetic data on disk: save the phscoh result once, read it back later without testappstf / phscoh
# ----------------------------------------------------------------------------------------------------------------------
class SyntheticXC:
    """Minimal stand-in for the phscoh object after calcxc / calcmvout / cpfromr / pickffreq: freq, xc (and xcdata ...), R, Cp, ffbest."""
    def __init__(self, **arrays):
        for k, v in arrays.items():
            setattr(self, k, v)


def save_synthetic(path, xc, az, takeoff, P, meta=None):
    """
    Write ONE synthetic realisation to  path  (.npz, no pickle).  Stored: every complex cross-spectrum array of `xc` with a frequency axis (xc, xcdata ...;
    the noise arrays xcn* are left out), R, Cp, ffbest, freq; the station azimuths `az` and take-off angles in the order of the xc columns; the in-plane ray
    components P; and `meta` (a dict of plain values: earthquake type, radius, seed, strike, dip, settings ...).
    """
    import json, os
    freq = np.asarray(xc.freq, float)
    nf = len(freq)
    out = {'freq': freq, 'az': np.asarray(az, float), 'takeoff': np.broadcast_to(np.asarray(takeoff, float), np.shape(az)).copy(), 'P': np.asarray(P, float),
           'ffbest': np.asarray(float(xc.ffbest))}
    names = []
    for name in dir(xc):
        if name.startswith('__') or name.startswith('xcn'):
            continue
        try:
            val = getattr(xc, name)
        except Exception:
            continue
        if isinstance(val, np.ndarray) and np.iscomplexobj(val) and val.ndim in (2, 3) and nf in val.shape[:2]:
            out['arr_' + name] = val
            names.append(name)
    if 'xc' not in names:
        raise ValueError('xc.xc with a frequency axis not found - nothing to save')
    for name in ('R', 'Cp'):
        if hasattr(xc, name):
            out[name] = np.squeeze(np.asarray(getattr(xc, name), float))
    m = dict(meta or {}, arrays=names)
    out['meta'] = np.array(json.dumps(m, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    np.savez_compressed(path, **out)
    return path


def load_synthetic(path):
    """Read one file written by save_synthetic.  Returns dict(name, xc, az, takeoff, P, meta); `xc` works with get_phasors / analyse_array like the phscoh object."""
    import json, os
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z['meta']))
        arrays = {k[4:]: z[k] for k in z.files if k.startswith('arr_')}
        extra = {k: z[k] for k in ('R', 'Cp') if k in z.files}
        xc = SyntheticXC(freq=z['freq'], ffbest=float(z['ffbest']), **arrays, **extra)
        az, tk, P = z['az'], z['takeoff'], z['P']
    name = meta.get('name') or os.path.splitext(os.path.basename(path))[0]
    return dict(name=name, xc=xc, az=az, takeoff=tk, P=P, meta=meta, path=path)


def load_synthetic_folder(folder, pattern='*.npz'):
    """All files of `folder` written by save_synthetic, as an ordered dict  name -> load_synthetic(...)  (sorted by file name)."""
    import glob, os
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    if not files:
        raise FileNotFoundError(f'no {pattern} files in {os.path.abspath(folder)!r}: generate them first (OPTION A) or copy the saved files there')
    out = {}
    for f in files:
        d = load_synthetic(f)
        out[d['name']] = d
    return out


# ----------------------------------------------------------------------------------------------------------------------
# the ideal analytic case (no phscoh / testappstf needed): same attributes as a phscoh object
# ----------------------------------------------------------------------------------------------------------------------
def fit_moveout(X, freq, P, f_fit, tmax, n=81):
    """
    Emulates the moveout step of the real pipeline for the analytic case: finds the time shift vector t = (t_strike, t_dip) (seconds) that makes the
    phases of the stations most coherent below f_fit, assuming the shift of station i is  t . p_i  (p_i = in-plane ray components, see inplane_coords).
    A hypocentre that is not at the centroid of the slip adds exactly such a term to every cross-spectrum.  Returns the corrected spectra and t.
    """
    sel = freq <= f_fit
    u = _unit(X[sel])
    g = np.linspace(-tmax, tmax, n)
    best, tbest = -1.0, (0.0, 0.0)
    for tx in g:
        for tz in (g if np.ptp(P[:, 1]) > 1e-9 else [0.0]):
            corr = u * np.exp(2j * np.pi * freq[sel][:, None] * (tx * P[:, 0][None, :] + tz * P[:, 1][None, :]))
            r = np.mean(np.abs(corr.mean(axis=1)))
            if r > best:
                best, tbest = r, (tx, tz)
    Xc = X * np.exp(2j * np.pi * freq[:, None] * (tbest[0] * P[:, 0][None, :] + tbest[1] * P[:, 1][None, :]))
    return Xc, tbest


class IdealXC:
    """Minimal stand-in for the phscoh object: freq, xc (nf, N, 1), R, Cp, ffbest."""
    def __init__(self, freq, X, X_raw=None, moveout=None):
        self.freq, self.xc = freq, X[:, :, None]
        self.xc_raw = (X if X_raw is None else X_raw)[:, :, None]       # before the moveout fit
        self.moveout = moveout
        u = _unit(X)
        N = X.shape[1]
        self.R = _resultant(u)
        self.Cp = np.sqrt(np.clip((N * self.R**2 - 1) / (N - 1), 0, None))
        below = np.where(self.Cp < 0.5)[0]
        self.ffbest = float(freq[below[0]]) if len(below) else float(freq[-1])     # first frequency where the whole-array Cp drops below 0.5


def ideal_xc(radius=100.0, az_deg=None, vp=6000.0, vr_over_vp=0.52, pf=1.1, slip='crack', nx=41, ref_radius=0.0,
             hypo=None, takeoff=90.0, freq=None, moveout='fit'):
    """
    Ideal kinematic model, analytic and noise free:
      * circular rupture of radius `radius` (m) on a VERTICAL fault (strike along x), rupture speed vr_over_vp * vp from the hypocentre `hypo` (default: half a radius from the centre along strike)
      * slip = crack-like sqrt(1-(r/a)^2) ('crack') or uniform ('uniform')
      * stations on a dense, even azimuth grid (default every 5 deg, 0...180), all with the same take-off angle (90 = horizontal ray)
      * target and reference events share the hypocentre and the Green's functions (no location offset), no noise
      * both are aligned on their own onsets (the pick of the first arrival), exactly as mk1='t0', mk2='t0' does; the leftover station-dependent shift
        (hypocentre != centroid) is then removed by a moveout fit below 0.5 ff (moveout='fit'; 'none' keeps the raw phases)
      * reference = point source (ref_radius = 0) or a smaller crack
    The cross-spectrum at station i is  S_target,i(f) conj(S_ref,i(f)).  Returns (IdealXC, az_deg, ff_theory = pf vp / (2 radius)).
    """
    az_deg = np.arange(0.0, 180.1, 5.0) if az_deg is None else np.asarray(az_deg, float)
    hypo = (-0.5 * radius, 0.0) if hypo is None else hypo       # off-centre hypocentre (as in the random hypocentres of your synthetics); a hypocentre at the
                                                                # centre makes the source mirror-symmetric, so stations on opposite sides would look identical
    ff_theory = pf * vp / (2 * radius)
    freq = np.linspace(0.02 * ff_theory, 4.0 * ff_theory, 240) if freq is None else np.asarray(freq, float)
    P = inplane_coords(az_deg, takeoff, strike=0.0, dip=90.0)                   # (N, 2): along strike, down dip

    def spectrum(a):
        g = np.linspace(-a, a, nx)
        x, z = np.meshgrid(g, g)
        x, z = x.ravel(), z.ravel()
        r = np.hypot(x, z)
        m = r <= a
        x, z, r = x[m], z[m], r[m]
        s = np.sqrt(np.clip(1 - (r / a) ** 2, 0, None)) if slip == 'crack' else np.ones_like(r)
        dx, dz = x - hypo[0], z - hypo[1]
        trup = np.hypot(dx, dz) / (vr_over_vp * vp)                                   # rupture time from the hypocentre
        T = trup[None, :] - (dx[None, :] * P[:, 0:1] + dz[None, :] * P[:, 1:2]) / vp  # (N, K) arrival time relative to the onset
        return np.array([(s[None, :] * np.exp(-2j * np.pi * f * T)).sum(axis=1) for f in freq])   # (nf, N)

    S = spectrum(radius)
    if ref_radius > 0:
        S = S * np.conj(spectrum(ref_radius))
    if moveout == 'fit':       # the hypocentre is not at the centroid -> remove the station-dependent shift below 0.5 ff, as the moveout step of the pipeline does
        Sc, t = fit_moveout(S, freq, P, 0.5 * ff_theory, tmax=2.0 * radius / vp)
        return IdealXC(freq, Sc, X_raw=S, moveout=t), az_deg, ff_theory
    return IdealXC(freq, S), az_deg, ff_theory
