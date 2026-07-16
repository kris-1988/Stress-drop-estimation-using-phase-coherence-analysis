import obspy
import numpy as np

# This Python function, calc_xc, calculates the Normalized Cross-Correlation between two seismic waveforms (tr1 and tr2) around specific arrival time marks (pick1 and pick2).
# The inputs are:
# tr1 (ObsPy Trace object): The first seismic waveform.
# tr2 (ObsPy Trace object): The second seismic waveform.
# pick1 (ObsPy UTCDateTime): The estimated arrival time of the wave on tr1.
# pick2 (ObsPy UTCDateTime): The estimated arrival time of the wave on tr2.
# before_t_pick (float/int): How many seconds before the pick time the window should start.
# after_t_pick (float/int): How many seconds after the pick time the window should end.
# max_shift (float/int): The maximum amount of time shift which the algorithm searches for the alignment between two waveforms.

def calc_xc(pick1, tr1, pick2, tr2, before_t_pick, after_t_pick, max_shift, filter=None, filter_option=None):

    # bandpass filter
    data1=tr1.copy()
    if filter is not None:
        data1.filter(filter=filter)
    data2=tr2.copy()
    
    # grab data for the template trace
    data1=data1.trim(starttime=pick1-before_t_pick,endtime=pick1+after_t_pick).data
    # grab data for the second trace
    data2=data2.trim(starttime=pick2-before_t_pick-max_shift,endtime=pick2+after_t_pick+max_shift).data

    # normalize the template data
    # LHS of denominator
    data1=data1/np.sum(np.power(data1,2))**0.5

    # cross correlate
    # numerator
    xc=np.correlate(data2,data1)

    # normalizations of the second trace
    # RHS of denominator

    # length of template trace
    N=data1.size

    # cumulative sum of power in second trace
    data2=np.cumsum(np.power(data2,2))
    data2=np.append(0,data2)

    # determine steps in cumulative sum in each interval
    data2=data2[N:]-data2[0:-N]
    data2=np.power(data2,0.5)

    # normalize
    xc=np.divide(xc,data2)

    # note the pick time and the maximum xc value
    imax=np.argmax(xc)
    tshf=tr1.stats.delta*imax-max_shift
    xc=[tshf,xc[imax]]

    return xc
 
