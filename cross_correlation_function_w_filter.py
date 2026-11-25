import obspy
import numpy as np

def calc_xc(pick1, tr1, pick2, tr2, before_t_pick, after_t_pick, max_shift, filter_type='lowpass', 
           freq_min=0.1, freq_max=3.0):  #filter_type = None, if filter is not considered

    # bandpass filter
    data1=tr1.copy()
    data2=tr2.copy()

    # apply filter to both the data1 and data2
    if filter_type =='bandpass':
        data1.filter('bandpass', freqmin = freq_min, freqmax = freq_max)
        data2.filter('bandpass', freqmin = freq_min, freqmax = freq_max)
    elif filter_type =='lowpass':
        data1.filter('lowpass', freq = freq_max)
        data2.filter('lowpass', freq = freq_max)
    
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
 
