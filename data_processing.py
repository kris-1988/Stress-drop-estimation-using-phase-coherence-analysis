import obspy
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.signal.trigger import recursive_sta_lta, trigger_onset

import signal_preprocessing
import taup_predicted_arrivals1
import cross_correlation_function_w_filter
from taup_predicted_arrivals1 import predict_arrivals

# This function runs sections 1, 2, 4 and 5 of the notebook in one step, until one step before the phase coherence.
# 1. target earthquake: catalog, seismograms, pre-processing, onset detection (ray tracing, STA/LTA), SNR
# 2. reference earthquake: the same steps
# 4. onset pick time correction by cross-correlation
# 5. final signal selection, and the plot of the signal comparison
# The final seismograms are saved in the MSEED format, and are read again by the phase coherence step.

def plot_waveforms(st_target, st_reference, filename, title):
    """Generates a 6-column waveform comparison plot with a global legend."""
    n_traces = len(st_target)
    cols = 6
    rows = int(np.ceil(n_traces / cols))

    # 1. Use layout='constrained' to automatically manage margins and spacing
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows), layout='constrained')
    plt.rc('font', size=20)

    # Flatten axes array for easy iteration
    axes = np.atleast_1d(axes).flatten()

    for i, tr1 in enumerate(st_target):
        ax = axes[i]
        tr2 = st_reference.select(id=tr1.get_id())

        if not len(tr2):
            continue

        # Time alignment and normalization
        t_target = tr1.times() - tr2[0].stats.t2
        d_target = tr1.data / np.max(np.abs(tr1.data))
        t_ref = tr2[0].times() - tr2[0].stats.t2
        d_ref = tr2[0].data / np.max(np.abs(tr2[0].data))

        # Plot waveforms with generalized labels for the global legend
        ax.plot(t_target, d_target, color='red', linewidth=2.0, label='target')
        ax.plot(t_ref, d_ref, color='black', linewidth=1.0, label='reference')

        ymin, ymax = -2.2, 2.2

        # Plot onset detection line at 0 (blue)
        ax.vlines(0, ymin, ymax, color='blue', linewidth=2, linestyle='--', label='final onset detection')

        # Mark the cross-correlation duration
        ax.axvspan(-2.0, 2.0, color='grey', alpha=0.2, label='cross-correlation duration')

        # Add station name inside the subplot, upper right
        ax.text(0.95, 0.95, tr1.stats.station, transform=ax.transAxes,
                ha='right', va='top', fontsize=20, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

        # Only add labels to edge plots to keep it clean
        if i % cols == 0:
            ax.set_ylabel('Normalised \n amplitude')
        if i >= n_traces - cols:
            ax.set_xlabel('Time (s)')

        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(ymin, ymax)

    # Hide any remaining unused subplots in the grid
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Constrained layout will automatically position the title correctly
    fig.suptitle(title, fontsize=20, fontweight='bold')

    # Grab handles and labels from the first populated subplot to build the global legend
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break

    # Put the legend outside of the plot area
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=4, fontsize=20)

    #fig.savefig(filename, bbox_inches='tight')
    plt.show()
    plt.close(fig)


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


def save_seismograms(st, filename, mag):
    """Save the seismograms in the MSEED format, and the picks, the SNR and the magnitude in a text file (same name, _picks.csv)."""
    st.write(filename, format='MSEED')

    keys = ['t1', 'sta_lta_time', 't2', 'SNR', 'xcmax']
    filename_picks = os.path.splitext(filename)[0] + '_picks.csv'
    with open(filename_picks, 'w') as f:
        f.write('id,' + ','.join(keys) + ',mag\n')
        for tr in st:
            values = [repr(float(tr.stats[k])) if k in tr.stats else '' for k in keys]
            f.write(tr.get_id() + ',' + ','.join(values) + ',' + repr(float(mag)) + '\n')

    print('  seismograms:', filename, '(' + str(len(st)) + ' traces)')
    print('  picks      :', filename_picks)


def data_processing (event_time, event_time_2, fdir, file_1, file_2, SNR1=10, SNR2=2, XCMIN=0.7):

    """ Function to process the seismograms of the target and the reference earthquake until one step before the phase coherence

    Perameter
    _________

    event_time   = time of the target earthquake, obspy UTCDateTime
    event_time_2 = time of the reference earthquake, obspy UTCDateTime
    fdir         = folder to save the final seismograms
    file_1       = file name of the final target seismograms
    file_2       = file name of the final reference seismograms
    SNR1         = minimum SNR of the target earthquake
    SNR2         = minimum SNR of the reference earthquake
    XCMIN        = minimum cross-correlation coefficient

    Return
    ______

    st1_use = final target seismograms, obspy stream
    st_2    = final reference seismograms, obspy stream

    #the seismograms are saved in the MSEED format (file_1, file_2)
    #MSEED does not keep the picks (t1, sta_lta_time, t2), the SNR and the cross-correlation coefficient of tr.stats,
    #they are saved in a text file with the same name and _picks.csv (file_1 and file_2 without the extension), one line for each trace

    """


    # start time, and the folder of the final seismograms (it is made if it does not exist)
    t_start = time.time()
    os.makedirs(fdir, exist_ok=True)

    #####################################################
    ########## 1. target earthquake #####################
    #####################################################

    # 1.1. Select a target earthquake from a catalog
    show_step(1, 16, '1.1. Target earthquake: select from the catalog', t_start)
    # grab the earthquake from the catalog
    # may want to specify location or magnitude as well, or id number
    c = Client('SCEDC')
    cat = c.get_events(starttime=event_time -5, endtime=event_time + 5, minmagnitude=1)
    eq = cat[0]

    # extract some information about the earthquake
    print('We are analysing an earthquake at location ')
    print(eq.origins[0].longitude,eq.origins[0].latitude,eq.origins[0].depth)
    print('\nwith magnitude ',eq.magnitudes[0].mag)

    # re-pick event time
    event_time=eq.origins[0].time
    print('at time',event_time)


    # 1.2. Read seismogram of target earthquake
    show_step(2, 16, '1.2. Target earthquake: download the seismograms', t_start)
    # identify available stations
    c = Client('SCEDC')
    inv = c.get_stations(network= 'CI',longitude=eq.origins[0].longitude,
                         latitude=eq.origins[0].latitude,
                         maxradius=155./111.,channel='BHZ',level='channel')

    # make a list of data to download
    starttime=event_time - 60
    endtime= event_time +180
    bulk=[]
    for nw in inv:
        for stn in nw:
            for chn in stn:
                bulk.append((nw.code,stn.code,"*",chn.code,starttime,endtime))

    # and get all the data
    st=c.get_waveforms_bulk(bulk,attach_response=True)

    print('We have data from {:d} stations'.format(len(st)))
    print(st)

    # Save waveform
    #st.write(os.path.join(fdir, "38564031-2019-07-14T06:06:39.280000Z.MSEED"), format="MSEED")

    # create data inventory
    # select from the inventory to show which stations actually have data
    invh=obspy.Inventory()
    for tr in st:
        invh=invh+inv.select(station=tr.stats.station,network=tr.stats.network,
                             channel=tr.stats.channel)


    # 1.3. Target seismogram pre-processing
    show_step(3, 16, '1.3. Target earthquake: pre-process the seismograms', t_start)
    st=signal_preprocessing.signal_preprocessing_1(st)

    print('Number of channels with good data after pre-processing:')
    print(len(st))


    # 1.4.a. Predict theoretical arrival time
    show_step(4, 16, '1.4.a. Target earthquake: predict the theoretical P arrival time (ray tracing)', t_start)
    eqloc=[eq.origins[0].longitude,eq.origins[0].latitude,eq.origins[0].depth/1000]
    tp = predict_arrivals (eqloc,invh)


    for tr in st:
        delta_t = event_time - tr.stats.starttime
        tr.stats.t1 = tp['.'.join([tr.stats.network,tr.stats.station])] + delta_t
        #print(tr.stats.t1)

    #tr.stats.t1 is predicted theoretical traveltime


    # 1.4.b. Refine onset with STA/LTA
    print('Theoretical arrival time is predicted at', len(tp), 'stations')
    show_step(5, 16, '1.4.b. Target earthquake: refine the onset with STA/LTA', t_start)
    df= st[0].stats.sampling_rate

    sig_shorts = []

    # create a set of potentially usable traces
    st_i=st
    st =obspy.Stream()

    for ss in st_i:
        ttt=ss.stats.t1

        # Now we slice the signal n seconds before theoretical travel time
        # and m second after theoretical travel time
        sig_short = ss.slice(ss.stats.starttime + ttt - 5, ss.stats.starttime +ttt + 5)
        sig_shorts.append(sig_short)

        cft = recursive_sta_lta(sig_short.data, int(.5 * df), int(4.5 * df))
        on_of = trigger_onset(cft, 1.5, 0.5)
        if len(on_of)>= 1 :  #avoid no detection (update: 2025.10.20)
            on_of_1= on_of[0,[0]]
            on_of_1= float(on_of_1[0]/df)
            #print(on_of_1)

            ss.stats.sta_lta_time=  ss.stats.t1  + on_of_1 -5 #Put on-off time in the obspy stream object, -6 because we trim signal 6 sec before t1
            #print (ss.stats.sta_lta_time)

            # add this trace to the "good" stream
            st.append(ss)
        else:
            print('Discarding ',tr.get_id())


    # 1.5. Compute Signal to Noise Ratio (SNR)
    print('STA/LTA onset is detected at', len(st), 'of', len(st_i), 'stations')
    show_step(6, 16, '1.5. Target earthquake: compute the signal to noise ratio (SNR)', t_start)
    # change so that you don't have to use t1, just grab it from the stream
    for tr in st:
        t_snr=tr.stats.t1
        Sig = tr.slice (tr.stats.starttime + t_snr, tr.stats.starttime + t_snr + 2) #SIGNAL
        Noi = tr.slice (tr.stats.starttime + t_snr -3, tr.stats.starttime + t_snr -1) #NOISE
        SNR = (np.average(np.power(Sig,2)))/(np.average(np.power(Noi,2)))
        tr.stats.SNR = SNR
        #print('SNR:' + tr.stats.station)
        #print (SNR)


    #####################################################
    ########## 2. reference earthquake ##################
    #####################################################

    # 2.1. Select a reference earthquake from a catalog
    print('SNR is computed for', len(st), 'traces')
    show_step(7, 16, '2.1. Reference earthquake: select from the catalog', t_start)
    # grab the earthquake from the catalog
    # may want to specify location or magnitude as well, or id number
    c = Client('SCEDC')
    cat_2 = c.get_events(starttime=event_time_2 -5, endtime=event_time_2 + 5, minmagnitude=1)
    eq_2 = cat_2[0]

    # extract some information about the earthquake
    print('We are analysing an earthquake at location ')
    print(eq_2.origins[0].longitude,eq_2.origins[0].latitude,eq_2.origins[0].depth)
    print('\nwith magnitude ',eq_2.magnitudes[0].mag)

    # re-pick event time
    event_time_2=eq_2.origins[0].time
    print('at time',event_time_2)


    # 2.2. Read seismogram of reference earthquake
    show_step(8, 16, '2.2. Reference earthquake: download the seismograms', t_start)
    # identify available stations
    c = Client('SCEDC')
    inv_2 = c.get_stations(network= 'CI',longitude=eq_2.origins[0].longitude,
                         latitude=eq_2.origins[0].latitude,
                         maxradius=155/111.,channel='BHZ',level='channel')

    # make a list of data to download
    starttime_2=event_time_2 - 60
    endtime_2= event_time_2 +180
    bulk_2=[]
    for nw in inv_2:
        for stn in nw:
            for chn in stn:
                bulk_2.append((nw.code,stn.code,"*",chn.code,starttime_2,endtime_2))

    # and get all the data
    st_2=c.get_waveforms_bulk(bulk_2,attach_response=True)

    print('We have data from {:d} stations'.format(len(st_2)))
    print(st_2)

    #st_2.write(os.path.join(fdir, '38564191-2019-07-14T06:53:33.090000Z.MSEED'), format='MSEED')

    # select from the inventory to show which stations are actually used
    invh_2=obspy.Inventory()
    for tr in st_2:
        invh_2=invh_2+inv_2.select(station=tr.stats.station,network=tr.stats.network,
                             channel=tr.stats.channel)


    # 2.3. Reference seismogram pre-processing
    show_step(9, 16, '2.3. Reference earthquake: pre-process the seismograms', t_start)
    st_2=signal_preprocessing.signal_preprocessing_1(st_2)

    print('Number of channels with good data after pre-processing:')
    print(len(st_2))


    # 2.4.a. Predict theoretical arrival time
    show_step(10, 16, '2.4.a. Reference earthquake: predict the theoretical P arrival time (ray tracing)', t_start)
    # To predict the onset of P-waves by theoretical arrival time
    # This onset information will used to mark the time to slice the signal in the next steps
    eqloc_2=[eq_2.origins[0].longitude,eq_2.origins[0].latitude,eq_2.origins[0].depth/1000]
    tp_2 = predict_arrivals (eqloc_2,invh_2)


    for tr2 in st_2:
        delta_t_2 = event_time_2 - tr2.stats.starttime
        tr2.stats.t1 = tp_2['.'.join([tr2.stats.network,tr2.stats.station])] + delta_t_2
        #print(tr2.stats.t1)
        #print(type(tr2.stats.t1))


    # 2.4.b. Refine onset with STA/LTA
    print('Theoretical arrival time is predicted at', len(tp_2), 'stations')
    show_step(11, 16, '2.4.b. Reference earthquake: refine the onset with STA/LTA', t_start)
    df= st_2[0].stats.sampling_rate

    #put parameter here to slice signal

    sig_shorts2 = []

    # create a set of potentially usable traces
    st_2i=st_2
    st_2=obspy.Stream()

    for ss in st_2i:
        ttt=ss.stats.t1

        # Now we slice the signal n seconds before theoretical travel time
        # and m second after theoretical travel time
        sig_short2 = ss.slice(ss.stats.starttime + ttt - 5, ss.stats.starttime +ttt + 5)
        sig_shorts.append(sig_short2)

        cft = recursive_sta_lta(sig_short2.data, int(.5 * df), int(4.5 * df))
        on_of = trigger_onset(cft,1.5, 0.5)
        #print(on_of)
        if len(on_of)>= 1 :  #avoid no detection (update: 2025.10.20)
            on_of_1= on_of[0,[0]]
            on_of_1= float(on_of_1[0]/df)
            #print(on_of_1)
            on_of_1= on_of[0,[0]]
        #print(on_of_1)

            on_of_1= float(on_of_1[0]/df)
        #print(on_of_1)

            ss.stats.sta_lta_time=  ss.stats.t1 -5 + on_of_1 #Put on-off time in the obspy stream object, -5 because we trim signal 5 sec before t1
        #print (type(ss.stats.sta_lta_time))

            # add this trace to the "good" stream
            st_2.append(ss)
        else:
            print('Discarding ',tr.get_id())


    # 2.5. Signal to Noise Ratio (SNR)
    print('STA/LTA onset is detected at', len(st_2), 'of', len(st_2i), 'stations')
    show_step(12, 16, '2.5. Reference earthquake: compute the signal to noise ratio (SNR)', t_start)
    # save the SNR on the stream

    # change so that you don't have to use t1, just grab it from the stream
    for tr in st_2:
        t_snr=tr.stats.t1
        Sig = tr.slice (tr.stats.starttime + t_snr, tr.stats.starttime + t_snr + 2) #SIGNAL
        Noi = tr.slice (tr.stats.starttime + t_snr -3, tr.stats.starttime + t_snr -1) #NOISE
        SNR = (np.average(np.power(Sig,2)))/(np.average(np.power(Noi,2)))
        tr.stats.SNR = SNR

        #print('SNR:' + tr.stats.station)
        #print (SNR)


    #####################################################
    ########## 4. onset pick time correction ############
    #####################################################

    print('SNR is computed for', len(st_2), 'traces')
    show_step(13, 16, '4. Onset pick time correction by cross-correlation', t_start)

    for tr1 in st:
        # copy and normalize
        tr1c=tr1.copy() #trace 1 for cross-correlation
        tr1c.data=tr1c.data/np.max(np.abs(tr1c.data)) # trace 1  cc normalization

        # grab the trace for the reference earthquake
        # with the same ID
        tr2=st_2.select(id=tr1.get_id())
        if len(tr2)> 0:
            # if data exists, copy and normalize
            tr2=tr2[0]
            tr2c=tr2.copy() #trace 2 for cross-correlation
            tr2c.data=tr2c.data/np.max(np.abs(tr2c.data)) #trace 2 cc normalization

            if tr2.stats.SNR > 5:


                # do the xc by using the cross_correlation_function_w_filter


                x_c = cross_correlation_function_w_filter.calc_xc(UTCDateTime(tr1c.stats.starttime + tr1c.stats.sta_lta_time), tr1,
                                UTCDateTime(tr2c.stats.starttime + tr2c.stats.sta_lta_time), tr2, 2, 2, 1)

                # save xc in the original streams
                tr1.stats.xcmax=x_c[1]
                tr2.stats.t2=tr2.stats.sta_lta_time+x_c[0]
                print('t2 is:')
                print((tr2.stats.t2))
                print(x_c)


    #####################################################
    ########## 5. final signal selection ################
    #####################################################

    print('Cross-correlation is calculated at', sum(1 for tr in st if 'xcmax' in tr.stats), 'stations')
    show_step(14, 16, '5.1. Final signal selection by the SNR and cross-correlation criteria', t_start)

    # 5.1. Impose the criteria
    # decide what to pass, don't want all station
    st1_use = obspy.Stream()

    # consider each trace that we might include
    for tr1 in st:
        tr2=st_2.select(id=tr1.get_id())


        # make sure the trace exists in the reference earthquake
        if len(tr2):
            tr2=tr2[0]

            # check some properties
            if tr2.stats.SNR>SNR2 and tr1.stats.SNR>SNR1:
                if 'xcmax' in tr1.stats.keys():
                    if tr1.stats.xcmax > XCMIN:
                        st1_use.append(tr1)

    print ('The signal passed those criteria are:')
    print(st1_use)
    print(len(st1_use), 'of', len(st), 'stations pass: SNR target >', SNR1, ', SNR reference >', SNR2, ', cross-correlation >', XCMIN)


    # 5.2. Plot the signal comparison
    show_step(15, 16, '5.2. Plot the signal comparison', t_start)
    # This plot is to compare two waveform (target and reference earthquake) at the recorded stations.
    # The signals are filtered (0.1-3.0 Hz) only for this plot, the saved seismograms are not filtered.
    st1_use_filtered = st1_use.copy().filter('bandpass', freqmin=0.1, freqmax=3.0)
    st_2_filtered = st_2.copy().filter('bandpass', freqmin=0.1, freqmax=3.0)

    plt.rc('font', size=20)

    plot_waveforms(
        st_target=st1_use_filtered,
        st_reference=st_2_filtered,
        filename='38564031_filtered_signal_comparison.png',
        title='Filtered Signals (0.1-3.0 Hz)'
    )


    #####################################################
    ########## save the final seismograms ###############
    #####################################################

    show_step(16, 16, '5.3. Save the final seismograms', t_start)
    # MSEED does not keep the picks, the SNR and the cross-correlation coefficient, they are saved in a text file (_picks.csv) with the magnitude
    save_seismograms(st1_use, os.path.join(fdir, file_1), eq.magnitudes[0].mag)
    save_seismograms(st_2, os.path.join(fdir, file_2), eq_2.magnitudes[0].mag)
    show_end('Section 1 (signal processing)', t_start)


    return st1_use, st_2
