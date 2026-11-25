import obspy
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import time
from scipy.stats import norm, skew
from matplotlib.lines import Line2D

import phscoh as phscoh

# This function runs section 6 of the notebook: the phase coherence of all stations, the bootstrap, and the figure.
# 1. phase coherence of all stations, and the fall-off frequency
# 2. bootstrap: the phase coherence is calculated again many times, each time with a random subset of the stations, without replacement
# 3. figure with four panels: displacement amplitude of the target earthquake, displacement amplitude of the reference earthquake,
#    phase coherence with the bootstrap interval, histogram of the bootstrap fall-off frequency

def format_time(seconds):
    """Time in hours:minutes:seconds."""
    seconds = int(round(seconds))
    return '{:d}:{:02d}:{:02d}'.format(seconds // 3600, (seconds % 3600) // 60, seconds % 60)


def show_step(i, n, text, t_start):
    """Print a progress bar, the step that is running now, and the time since the start."""
    n_bar = 30
    n_fill = int(n_bar * (i - 1) / n)
    print('\n[' + '#' * n_fill + '-' * (n_bar - n_fill) + '] step {:d} of {:d}, time {}'.format(i, n, format_time(time.time() - t_start)))
    print(text)


def show_end(name, t_start):
    """Print the full progress bar and the final time."""
    print('\n[' + '#' * 30 + '] ' + name + ' finished, final time {}'.format(format_time(time.time() - t_start)))


def read_seismograms(filename):
    """Read the seismograms in the MSEED format, and the picks, the SNR and the magnitude saved by data_processing (same name, _picks.csv)."""
    st = obspy.read(filename, format='MSEED')

    picks = {}
    with open(os.path.splitext(filename)[0] + '_picks.csv') as f:
        for row in csv.DictReader(f):
            picks[row['id']] = row

    for tr in st:
        for k in ['t1', 'sta_lta_time', 't2', 'SNR', 'xcmax', 'mag']:
            if picks[tr.get_id()][k] != '':
                tr.stats[k] = float(picks[tr.get_id()][k])

    return st


def phase_coherence_bootstrap (fdir, file_1, file_2, trange=[0, 1.7], n_boot=1000, frac_stations=0.8, ci_level=95, seed=42):

    """ Function to calculate the phase coherence and its bootstrap interval, and to plot them with the displacement amplitude

    Perameter
    _________

    fdir          = folder of the final seismograms saved by data_processing
    file_1        = file name of the final target seismograms
    file_2        = file name of the final reference seismograms
    trange        = time window segment to extract for spectral analysis in second, the same for all stations and all resamples
    n_boot        = number of bootstrap resamples
    frac_stations = fraction of stations kept in each resample, drawn without replacement
    ci_level      = confidence level, in percent
    seed          = seed of the random draws, None for fresh draws

    Return
    ______

    xc           = phase coherence of all stations, phscoh object
    boot_ff_arr  = fall-off frequency of the bootstrap resamples

    """


    #####################################################
    ########## read the final seismograms ###############
    #####################################################

    t_start = time.time()
    show_step(1, 7, '1. Read the final seismograms', t_start)
    st1_use = read_seismograms(os.path.join(fdir, file_1))
    st_2 = read_seismograms(os.path.join(fdir, file_2))
    print('  target   :', os.path.join(fdir, file_1), '(' + str(len(st1_use)) + ' traces)')
    print('  reference:', os.path.join(fdir, file_2), '(' + str(len(st_2)) + ' traces)')

    # magnitudes of the target and the reference earthquake, saved by data_processing
    mag_1 = st1_use[0].stats.mag
    mag_2 = st_2[0].stats.mag


    #####################################################
    ########## bootstrap confidence interval ############
    #####################################################

    # Repeats phscoh.calcxc() -> calcmvout() -> cpfromr() many times, each
    # time keeping a random subset of frac_stations (default 80%) of the
    # stations in st1_use, without replacement. The spread of the
    # resulting Cp(f) curves gives a bootstrap confidence interval that
    # reflects station-selection uncertainty.

    rng = np.random.default_rng(seed)   # fixed seed -> reproducible draws every run (pass None for fresh draws)

    n_stations = len(st1_use)
    n_sub = max(2, int(round(frac_stations * n_stations)))   # need >=2 stations for a meaningful coherence
    print(f'Bootstrapping {n_boot} times, using {n_sub} of {n_stations} stations each time (no replacement)')

    # ----- point estimate using all stations -----
    show_step(2, 7, '2.1. Phase coherence of all stations: cross-spectra of the target and the reference earthquake (calcxc)', t_start)
    xc = phscoh.calcxc(st1_use, st_2, trange=trange, mk1='sta_lta_time', mk2='t2', nsint=2, tpr='cosine')
    show_step(3, 7, '2.2. Phase coherence of all stations: phase coherence (calcmvout, cpfromr)', t_start)
    xc.calcmvout()
    xc.cpfromr()
    show_step(4, 7, '2.3. Phase coherence of all stations: fall-off frequency (pickffreq)', t_start)
    xc.pickffreq()
    ref_freq = np.asarray(xc.freq)
    n_freq = ref_freq.shape[0]
    print(f'Reference run: {n_freq} frequency points, fall-off frequency = {xc.ffbest:.3f} Hz')

    show_step(5, 7, '3. Bootstrap', t_start)
    boot_Cp = []
    boot_ff = []
    n_skipped_shape = 0

    for i in range(n_boot):

        # random subset of the usable target-earthquake stations
        idx = rng.choice(n_stations, size=n_sub, replace=False)
        st1_boot = obspy.Stream([st1_use[int(j)] for j in idx])

        try:
            xc_b = phscoh.calcxc(st1_boot, st_2, trange=trange, mk1='sta_lta_time', mk2='t2', nsint=2, tpr='cosine')
            xc_b.calcmvout()
            xc_b.cpfromr()

            freq_b = np.asarray(xc_b.freq)
            Cp_b_raw = np.asarray(xc_b.Cp)
            if freq_b.shape != Cp_b_raw.shape:
                # freq and Cp disagree with each other within this single resample
                # -- a genuinely broken result, skip it
                n_skipped_shape += 1
                continue

            # interpolate onto the reference grid rather than requiring an exact
            # length match -- a resample's native grid can still legitimately
            # differ in range/length even with trange now fixed above
            order = np.argsort(freq_b)
            Cp_b = np.interp(ref_freq, freq_b[order], Cp_b_raw[order], left=np.nan, right=np.nan)
            boot_Cp.append(Cp_b)

            try:
                xc_b.pickffreq()
                boot_ff.append(xc_b.ffbest)
            except Exception:
                pass   # fall-off frequency picking occasionally fails on a small subset; skip it

        except Exception:
            # occasionally a subset gives too few matched stations / a degenerate spectrum; skip it
            continue

        if (i + 1) % 100 == 0:
            print(f'  {i + 1}/{n_boot} resamples done, time {format_time(time.time() - t_start)}')

    boot_Cp = np.array(boot_Cp)   # shape: (n_success, n_freq)
    print(f'{boot_Cp.shape[0]} of {n_boot} bootstrap resamples succeeded '
          f'({n_skipped_shape} skipped due to a mismatched frequency grid)')

    if boot_Cp.shape[0] < 2:
        print('Not enough successful resamples to build a confidence interval -- try increasing n_boot, '
              'or check that phscoh.calcxc works reliably on smaller station subsets.')
        show_end('Section 2 (phase coherence and bootstrap)', t_start)
        return xc, np.array(boot_ff)

    # ----- confidence interval on Cp(f), by percentile -----
    lo_pct = (100 - ci_level) / 2
    hi_pct = 100 - lo_pct
    Cp_lo = np.nanpercentile(boot_Cp, lo_pct, axis=0)   # nan-aware: interpolation can leave edge NaNs
    Cp_hi = np.nanpercentile(boot_Cp, hi_pct, axis=0)

    # ----- fall-off frequency: percentile CI, mean/median/std -----
    have_ff = len(boot_ff) > 0
    if have_ff:
        boot_ff_arr = np.array(boot_ff)
        ff_lo, ff_hi = np.percentile(boot_ff_arr, [lo_pct, hi_pct])
        ff_mean = np.mean(boot_ff_arr)
        ff_median = np.median(boot_ff_arr)
        ff_std = np.std(boot_ff_arr, ddof=1)
        ff_2std_lo, ff_2std_hi = ff_mean - 2 * ff_std, ff_mean + 2 * ff_std

        print(f'Fall-off frequency: best estimate (all stations) = {xc.ffbest:.3f} Hz')
        print(f'  bootstrap (n={len(boot_ff_arr)}): mean={ff_mean:.3f} Hz, median={ff_median:.3f} Hz, std={ff_std:.3f} Hz')
        print(f'  {ci_level}% CI (percentile) = [{ff_lo:.3f}, {ff_hi:.3f}] Hz')
        print(f'  mean +/- 2*std            = [{ff_2std_lo:.3f}, {ff_2std_hi:.3f}] Hz')
    else:
        ff_lo = ff_hi = None
        boot_ff_arr = np.array(boot_ff)
        print('Could not build a fall-off frequency confidence interval (pickffreq failed on every resample)')


    #####################################################
    ########## displacement amplitude ###################
    #####################################################

    show_step(6, 7, '4. Displacement amplitude of the target and the reference earthquake (signal and noise)', t_start)

    # phscoh.calcxc saved the amplitudes of the signal (xc.amp) and of the noise (xc.ampn) of every station,
    # in the same windows and with the same taper as the phase coherence
    # xc.amp  has dimensions: frequency, station, taper, 2 earthquakes (0: target, 1: reference)
    # xc.ampn has dimensions: frequency, station, 2 earthquakes, noise intervals
    sig_1, noi_1 = xc.amp[:, :, 0, 0], xc.ampn[:, :, 0, 0]     # signal and noise, target earthquake
    sig_2, noi_2 = xc.amp[:, :, 0, 1], xc.ampn[:, :, 1, 0]     # signal and noise, reference earthquake

    # one colour per station, in the order of xc.ns: the first half of the 'jet' colour map (dark blue to light green)
    names = [ns.split('.')[1] for ns in xc.ns]
    col_sta = [plt.cm.jet(0.5 * k / max(1, len(names) - 1)) for k in range(len(names))]


    #####################################################
    ########## figure with four panels ##################
    #####################################################

    show_step(7, 7, '5. Plot the four panels', t_start)

    with plt.rc_context({'font.size': 20}):
        fig = plt.figure(figsize=(14, 11), constrained_layout=True)
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 0.22, 1])
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_leg = fig.add_subplot(gs[1, :]);  ax_leg.axis('off')      # legend row, shared by A and B
        ax_c = fig.add_subplot(gs[2, 0])
        ax_d = fig.add_subplot(gs[2, 1])
        fig.get_layout_engine().set(hspace=0.04, h_pad=0.15)

        def panel_label(ax, letter, title):
            ax.set_title(letter, loc='left', fontsize=20, fontweight=None)
            ax.set_title(title, loc='center')

        def plot_spectra(ax, sig, noi):
            for k in range(len(names)):
                ax.loglog(ref_freq, sig[:, k], linewidth=2, color=col_sta[k])
                ax.loglog(ref_freq, noi[:, k], linewidth=1, color=col_sta[k], linestyle='--')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Displacement amplitude (m/Hz)')
            ax.set_xlim(min(ref_freq), max(ref_freq))
            ax.grid(True, which='both', ls='--')
            ax.minorticks_on()

        # A: target earthquake, signal (solid) and noise (dashed)
        plot_spectra(ax_a, sig_1, noi_1)
        panel_label(ax_a, '(a)', f'Target earthquake ($M_w$ {mag_1:.2f})')

        # B: reference earthquake, signal (solid) and noise (dashed)
        plot_spectra(ax_b, sig_2, noi_2)
        panel_label(ax_b, '(b)', f'Reference earthquake ($M_w$ {mag_2:.2f})')

        # shared legend for A and B: station colours, solid = signal, dashed = noise
        handles = [Line2D([0], [0], color=col_sta[k], lw=2, label=names[k]) for k in range(len(names))]
        handles += [Line2D([0], [0], color='k', lw=2, label='Signal'),
                    Line2D([0], [0], color='k', lw=1, ls='--', label='Noise')]
        ax_leg.legend(handles=handles, loc='center', ncol=int(np.ceil(len(handles) / 5)), fontsize=14, frameon=True, edgecolor='black')

        # C: phase coherence with the bootstrap interval, the fall-off frequency is printed in the legend
        ax_c.fill_between(ref_freq, Cp_lo, Cp_hi, color='mediumblue', alpha=0.2,
                          label=f'{ci_level:g}% CI ({int(round(frac_stations * 100))}% of stations, n={boot_Cp.shape[0]})')
        ax_c.plot(ref_freq, xc.Cp, linewidth=2.0, color='mediumblue', label='Coherence (all stations)')
        ax_c.axvline(xc.ffbest, color='black', linestyle='--', label=f'$f_f$ = {xc.ffbest:.2f} Hz (no bootstrap, all stations)')
        if have_ff:
            ax_c.axvline(ff_mean, color='darkorange', linestyle='--', linewidth=1.5,
                         label='$f_f$ bootstrap mean (%d%% stations)' % round(frac_stations * 100))
            ax_c.axvspan(ff_lo, ff_hi, color='grey', alpha=0.2, label=f'$f_f$ {ci_level:g}% CI')
        ax_c.set_xscale('log')
        ax_c.set_xlim(min(ref_freq), max(ref_freq))
        ax_c.set_xlabel('Frequency (Hz)')
        ax_c.set_ylabel('Coherence $C_p$')
        ax_c.grid(True, which='both', ls='--')
        ax_c.legend(fontsize=12, loc='lower left')
        panel_label(ax_c, '(c)', 'Phase coherence')

        # D: histogram of the fall-off frequency of the bootstrap resamples
        if have_ff:
            counts, edges, _ = ax_d.hist(boot_ff_arr, bins=20, color='mediumblue', alpha=0.6, edgecolor='black')
            bin_w = edges[1] - edges[0]

            # normal curve with the bootstrap mean and std, scaled from probability density to histogram counts
            xg = np.linspace(edges[0], edges[-1], 400)
            ax_d.plot(xg, norm.pdf(xg, ff_mean, ff_std) * len(boot_ff_arr) * bin_w, color='crimson', linewidth=2.5,
                      label='Normal fit (mean, std)')

            ax_d.axvline(xc.ffbest, color='black', linestyle='--', linewidth=2, label='$f_f$ (no bootstrap, all stations)')
            ax_d.axvline(ff_mean, color='darkorange', linestyle='-', linewidth=1.5,
                         label='$f_f$ bootstrap mean (%d%% stations)' % round(frac_stations * 100))
            ax_d.axvspan(ff_lo, ff_hi, color='grey', alpha=0.25, label=f'{ci_level:g}% CI (percentile)')
            ax_d.set_xlabel('Fall-off frequency (Hz)')
            ax_d.set_ylabel('Count')
            ax_d.legend(fontsize=12, loc='lower left')

            stats_txt = (f'n = {len(boot_ff_arr)}\n'
                         f'mean = {ff_mean:.3f} Hz\n'
                         f'median = {ff_median:.3f} Hz\n'
                         f'std = {ff_std:.3f} Hz\n'
                         f'skewness = {skew(boot_ff_arr):.2f}\n'          # 0 for a normal distribution
                         f'{ci_level:g}% CI = [{ff_lo:.3f}, {ff_hi:.3f}]')
            ax_d.text(0.98, 0.97, stats_txt, transform=ax_d.transAxes, fontsize=12, va='top', ha='right',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax_d.text(0.5, 0.5, 'No successful\nfall-off frequency picks', ha='center', va='center',
                      transform=ax_d.transAxes)
            ax_d.set_axis_off()
        panel_label(ax_d, '(d)', 'Fall-off frequency histogram')

        #fig.savefig('38564031_phase_coherence_bootstrap_four_panel.png', dpi=600, bbox_inches='tight')
        plt.show()

    print('Fall-off frequency is:')
    print (xc.ffbest)


    show_end('Section 2 (phase coherence and bootstrap)', t_start)

    return xc, boot_ff_arr
