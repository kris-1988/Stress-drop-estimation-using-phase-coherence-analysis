import obspy 
import numpy as np
from obspy import read
from obspy import UTCDateTime
import matplotlib.pyplot as plt

# This function is a standard data-cleaning pipeline for earthquake data (seismograms) using the ObsPy library. 
# It includes demean, tapering, filtering, remove instrument response, and merge and fill the gaps.
# sig = signal
# flm = filter
def signal_preprocessing_1 (sig, flm= [0.8,18.0]):
    #[0.8,18.0]
    #[0.8,10.0]
    #[0.8,8.0]
    #
    """ Function to apply signal preprocessing to earthquake seismorgam

    Perameter
    _________

    sig = seismogram data, obspy stream of waveform to process
    flm = frequency band to filter

    Return
    ______

    sig = pre-processed data

    #detrend: remove the effect of a trend from the trace
    #tapper: decaying to zero near the end of each window, to minimize the effect 
        of the discontinuity between the beginnng and the end of time series
    #filter: band-passed filter
    
    """
    

    # Apply detrend 
    sig.detrend('demean')

    # Apply tapper, max_percentage : maximum tapper percentage, max_length = max. tapper length in second
    sig.taper(max_percentage = 0.5, max_length = 100 ,type = 'cosine')
    
    # Bandpass filter
    sig.filter('bandpass', freqmin = flm[0], freqmax = flm[1])

    # Remove instrument response
    sig.remove_response()

    # Merge overlap or gap data and fill the gap with certain value
    sig.merge(method=1, fill_value=0)

    # Make input data type as float
    for tr in sig:
        tr.data=tr.data.astype(float)
    
 
    return sig
