import numpy as np
import obspy
import scipy
import os,glob,math,copy
import general
import obspy
import spectrum
from scipy.interpolate import splrep,splev
#from scipy.windows import hann

def calcxc(st1,st2=None,trange=None,mk1='t0',mk2=None,
           nsint=None,fmax=None,tpr='multi',dfres=None):
    """
    compute cross-spectra for a pair of waveforms
    :param      st1:  first set of waveforms
    :param      st2:  second set of waveforms 
                         (default: same as st1)
    :param   trange:  time range to consider, 
                         in seconds relative to mk1/mk2 picks
                         (default: [0.,3.])
    :param      mk1:  time picks to use as reference zero for st1 
                         (default: 't0')
    :param      mk2:  time picks to use as reference for st1 
                         (default: mk1)
    :param    nsint:  list of time shifts of intervals with noise, 
                         relative to earthquakes
                         (default: None--no noise interval)
                         entering a single integer creates
                            that number of noise intervals, 
                            input is as in defnoiseint
    :param     fmax:  maximum frequency to calculate cross-spectra for
                         (default: 80% of Nyquist)
    :param      tpr:  how to taper the data
                         'multi': multi-taper (default)
                         'slepian': a single Slepian taper
                         'cosine': flat with cosine on 10% of each end
                         'halfslepian': constant, then half a single Slepian taper
    :param    dfres:  desired frequency resolution used for multi-taper
                         (default: best allowed by interval)
    :return  xc:  an xcross instance, with at least values
    :            xc:  cross-correlations for each station 
                         dimensions 0: frequency, 1: station, 2: taper
    :           amp:  amplitudes of earthquake intervals
                         dimensions 0: frequency, 1: station,  
                                    2: taper, 3: 2 earthquakes
    :           xcn:  cross-correlations for each station, 
                         with noise added, averaged over tapers
                         dimensions 0: frequency, 1: station, 
                                    2: noise intervals
    :          ampn:  amplitudes in noise intervals
                         dimensions 0: frequency, 1: station, 
                                    2: earthquakes, 3: noise intervals
    :          freq:  frequencies in Hz
    :           nsc:  array of 'network.station.component' lists
    :           tlm:  time range used relative to picks
    :    starttime1:  reference times for event 1
    :    starttime2:  reference times for event 2
    :          dtim:  time spacing in seconds
    :          fmax:  maximum frequency used
    :         obsft:  saved FT for each observation
                         dimensions 0: frequency, 1: station,  
                                    2: taper, 3: 2 earthquakes
    :         dfres:  frequency resolution used
    :           mk1:  string indicating pick used for earthquake 1
    :           mk2:  string indicating pick used for earthquake 2
    """

    #---------SET THE DEFAULT PARAMETERS---------------------
    
    # time range
    if trange is None:
        trange=np.array([0.,3.])
    trange=np.atleast_1d(trange).copy()

    # default is no noise interval
    if nsint is None:
        nsint = []
    elif isinstance(nsint,int):
        # for an integer, pick some intervals
        nsint=defnoiseint(trange=trange,N=nsint,allshf=0.)

    # keep values from first dataset as default
    if st2 is None:
        st2 = st1
    if mk2 is None:
        mk2 = mk1
    
    # time spacing: needs to be the same for all waveforms
    dt1 = [tr.stats.delta for tr in st1]
    dt2 = [tr.stats.delta for tr in st2]
    dt = min(min(dt1),min(dt2))
    nr = int(math.floor(math.log10(abs(dt))))
    dt = round(dt,8-nr-1)

    # maximum frequency used in calculations
    if fmax is None:
        fmax = 0.4/dt

    # number of points in extracted intervals
    N = int(np.round(np.diff(trange)[0]/dt))
    trange[1]=trange[0]+N*dt

    # number of points in FT
    Nf = 2**math.ceil(math.log(N,2))
    Nf = N

    #----------SET UP THE TAPERING AND PICK FREQUENCIES-----------

    # desired frequency resolution
    if dfres is None:
        # use best if not given
        dfres = 1./np.diff(trange)[0]
    else:
        # check it's in range
        dfres=np.maximum(dfres,1./np.diff(trange)[0])

    # initialize frequencies
    freq=np.fft.rfftfreq(Nf,d=dt)
    ix=np.logical_and(freq>0,freq<=fmax)
    freq = freq[ix]
    Nfs = len(freq)

    if tpr.lower()=='multi':
        # decide on the tapers' concentration
        NW = dfres / (1./np.diff(trange)[0]) * 2

        # compute tapers
        [U,V] = spectrum.mtm.dpss(N,NW)

        # just select some?
        ii = V>=0.95
        U = U[:,ii]
        V = V[ii]

        Nt = len(V)

    elif tpr.lower()=='slepian':

        # decide on the tapers' concentration
        NW = dfres / (1./np.diff(trange)[0]) * 2

        # compute tapers
        [U,V] = spectrum.mtm.dpss(N,NW)
        U = U[:,0:1]
        V = V[0:1]
        Nt = len(V)

    elif tpr.lower()=='cosine':

        tr = obspy.Trace()
        tr.data = np.ones(N,dtype=float)
        tr.taper(side='both',max_percentage=0.1,type='cosine')
        U = tr.data.reshape([N,1])
        V = np.ones(1,dtype=float)
        Nt = len(V)

    elif tpr.lower()=='halfslepian':

        # decide on the tapers' concentration
        NW = dfres / (1./np.diff(trange)[0]) * 2

        # compute tapers
        [U,V] = spectrum.mtm.dpss(N,NW)
        U = U[:,0:1]
        V = V[0:1]
        Nt = len(V)

        # set the first half to zero
        U[0:U.shape[0]/2,:] = np.max(U)

        # and taper up to the first value in the first 1/10th
        [U1,V1] = spectrum.mtm.dpss(N/5,NW)
        scl = np.max(U)/np.max(U1)
        scl = (float(U1.shape[0])/float(U.shape[0]))**0.5
        U1 = U1[0:U1.shape[0]/2,0]*scl
        U[0:U1.size,0] = U1

    else:
        error('Not a valid tapering option')


    #-------IDENTIFY FREQUENCIES TO COMPARE, INITIALIZE OUTPUT-----------

    # make a list of all available networks/stations/channels
    nsc=np.intersect1d(np.array([tr.stats.network+'.'+tr.stats.station+'.'+
                                 tr.stats.channel for tr in st1]),
                       np.array([tr.stats.network+'.'+tr.stats.station+'.'+
                                 tr.stats.channel for tr in st2]))

    # initialize cross-spectra
    xc = np.zeros([len(freq),len(nsc),Nt],complex)

    # initialize cross-spectra with added noise
    xcn = np.zeros([len(freq),len(nsc),len(nsint)],complex)
    
    # initialize power in main window
    amp = np.zeros([len(freq),len(nsc),Nt,2],float)

    # initialize power in earlier windows
    ampn = np.zeros([len(freq),len(nsc),2,len(nsint)],float)

    # initialize Fourier-transformed observations
    obsft = np.zeros([len(freq),len(nsc),Nt,2],complex)

    # keep track of start times
    starttime1=np.ndarray([len(nsc)],dtype=float)
    starttime2=starttime1.copy()

    # keep track of the station locations
    stloc=np.ndarray([len(nsc),3])*float('nan')
    
    #------CALCULATE THE CROSS-SPECTRA AND AMPLITUDES---------------

    # go through the traces in the first stream
    for m in range(0,len(nsc)):
        nw,st,ch=nsc[m].split('.')
    
        # check for matching traces
        sti1=st1.select(station=st,network=nw,channel=ch).copy().merge()
        sti2=st2.select(station=st,network=nw,channel=ch).copy().merge()

        # to record station locations
        try:
            stloc[m,0]=sti1[0].stats.sac['stlo']
            stloc[m,1]=sti1[0].stats.sac['stla']
            stloc[m,2]=sti1[0].stats.sac['stel']
        except:
            try:
                stloc[m,0]=sti2[0].stats.sac['stlo']
                stloc[m,1]=sti2[0].stats.sac['stla']
                stloc[m,2]=sti2[0].stats.sac['stel']
            except:
                pass

        # identify picks
        pk1,pk2=[],[]
        for tr in sti1:
            if mk1 in tr.stats:
                pk1=tr.stats[mk1]
        for tr in sti2:
            if mk2 in tr.stats:
                pk2=tr.stats[mk2]

    
        # times to grab
        pk1=sti1[0].stats.starttime+pk1
        pk2=sti2[0].stats.starttime+pk2
        
        # save start times
        starttime1[m]=pk1.timestamp
        starttime2[m]=pk2.timestamp

        # grab the data of interest,
        # buffer the end in case of sampling problems
        data1=sti1.copy().trim(pk1+trange[0],pk1+trange[1]+dt,pad=True,
                               fill_value=0,nearest_sample=True)
        data1=data1.merge()[0]
        data2=sti2.copy().trim(pk2+trange[0],pk2+trange[1]+dt,pad=True,
                               fill_value=0,nearest_sample=True)
        data2=data2.merge()[0]

        # for accuracy, we'll need precise arrivals
        tm1,tm2=data1.stats.starttime,data2.stats.starttime
        dt1,dt2=tm1-pk1,tm2-pk2
        phsshf=2*math.pi*(dt1-dt2)*freq
        phsshf=np.exp(1j*phsshf)

        # just the values
        # indices in case they're different lengths---usually by one sample
        data1=data1.data[0:N]
        data2=data2.data[0:N]
        
        # to keep track of power
        amp1,amp2=np.zeros(Nfs),np.zeros(Nfs)

        for k in range(0,Nt):
            # times the taper
            data1i=np.multiply(U[:,k],data1)
            data2i=np.multiply(U[:,k],data2)

            # fft
            data1i = np.fft.rfft(data1i,Nf)
            data2i = np.fft.rfft(data2i,Nf)

            # save for later
            obsft[:,m,k,0] = data1i[ix]
            obsft[:,m,k,1] = data2i[ix]

            # cross-correlate and save
            xci=np.multiply(data1i.conj()[ix],data2i[ix])
            xc[:,m,k]=np.multiply(xci,phsshf)

            # keep track of amplitudes
            amp[:,m,k,0]=np.real(np.multiply(data1i[ix],data1i.conj()[ix]))
            amp[:,m,k,1]=np.real(np.multiply(data2i[ix],data2i.conj()[ix]))

        for n in range(0,len(nsint)):

            # shift the relevant picks
            pk1i,pk2i=tm1+nsint[n][0],tm2+nsint[n][0]

            # grab the data, buffer at the end to deal 
            # with incorrect sampling
            datan1=sti1.copy().trim(pk1i+trange[0],pk1i+trange[1]+dt,
                                    pad=True,
                                    fill_value=0,nearest_sample=True)
            datan1=datan1.merge()[0].data[0:N]
            datan2=sti2.copy().trim(pk2i+trange[0],pk2i+trange[1]+dt,
                                    pad=True,
                                    fill_value=0,nearest_sample=True)
            datan2=datan2.merge()[0].data[0:N]

            # initialize x-c and amplitudes---will average over tapers
            xci = np.zeros(Nfs)
            amp1,amp2=np.zeros(Nfs),np.zeros(Nfs)

            for k in range(0,Nt):
                # for amplitude tracking
                data1i=np.multiply(U[:,k],datan1)
                data2i=np.multiply(U[:,k],datan2)
                data1i = np.fft.rfft(data1i,Nf)
                data2i = np.fft.rfft(data2i,Nf)
                amp1=amp1+np.real(np.multiply(data1i[ix],data1i.conj()[ix]))
                amp2=amp2+np.real(np.multiply(data2i[ix],data2i.conj()[ix]))

                # and for the noise in the cross-correlation

                # times the taper
                data1i=np.multiply(U[:,k],data1+datan1)
                data2i=np.multiply(U[:,k],data2+datan2)
                
                # fft
                data1i = np.fft.rfft(data1i,Nf)
                data2i = np.fft.rfft(data2i,Nf)
                
                # cross-correlation
                xci=xci+np.multiply(data1i.conj()[ix],data2i[ix])

            # save
            xci = np.multiply(xci,phsshf)
            xcn[:,m,n] = xci/Nt
            ampn[:,m,0,n]=amp1/Nt
            ampn[:,m,1,n]=amp2/Nt

    #-----------------COLLECT FOR OUTPUT--------------------------------

    xc=xcross({'xc':xc,'xcn':xcn,'amp':amp,'ampn':ampn,'freq':freq,
               'nsc':nsc,'tlm':trange,'starttime1':starttime1,
               'starttime2':starttime2,'dtim':dt,'fmax':fmax,
               'obsft':obsft,'dfres':dfres,
               'mk1':mk1,'mk2':mk2,'stloc':stloc})

    return xc

def calcxctim(st1,st2=None,trange=None,mk1='t6',mk2=None,
              nsint=None,fmax=None,tpr='multi',dfres=None,twin=[-2,2]):
    """
    compute cross-spectra for a pair of waveforms, 
    It's like calcxc, but with a time domain approach: 
    take a portion of st1, cross-correlate with the full 
    waveform of st2, then taper and compute the spectra of that 
    cross-correlation.  This is useful if the data are very noisy.
    The cross-correlation concentrates the signal in a short
    period of interest, and then you can extract and analyse that signal.
    :param      st1:  first set of waveforms
    :param      st2:  second set of waveforms 
                         (default: same as st1)
    :param   trange:  time range to consider, 
                         in seconds relative to mk1/mk2 picks
                         (default: [0.,3.])
    :param      mk1:  time picks to use as reference zero for st1 
                         (default: 't6')
    :param      mk2:  time picks to use as reference for st1 
                         (default: mk1)
    :param    nsint:  time shifts of intervals with noise, 
                         relative to earthquakes
                         (default: None--no noise interval)
                       entering a single integer creates
                         that number of noise intervals, 
                          as input to defnoiseint
    :param     fmax:  maximum frequency to calculate cross-spectra for
                         (default: 80% of Nyquist)
    :param      tpr:  how to taper the data
                         'multi': multi-taper (default)
                         'slepian': a single Slepian taper
                         'cosine': flat with cosine on 10% of each end
                         'halfslepian': constant, then half a single Slepian taper
    :param    dfres:  desired frequency resolution used for multi-taper
                         (default: best allowed by interval)
    :param     twin:  portion of x-c to examine
    :return      xc:  an xcross instance, with at least values
    :            xc:  cross-correlations for each station 
                         dimensions 0: frequency, 1: station, 2: taper
    :           amp:  amplitudes of earthquake intervals
                         dimensions 0: frequency, 1: station,  
                                    2: taper, 3: 2 earthquakes
    :           xcn:  cross-correlations for each station, 
                         with noise added, averaged over tapers
                         dimensions 0: frequency, 1: station, 
                                    2: noise intervals
    :          ampn:  amplitudes in noise intervals
                         dimensions 0: frequency, 1: station, 
                                    2: earthquakes, 3: noise intervals
    :          freq:  frequencies in Hz
    :           nsc:  array of 'network.station.component' lists
    :           tlm:  time range used relative to picks
    :    starttime1:  reference times for event 1
    :    starttime2:  reference times for event 2
    :          dtim:  time spacing in seconds
    :          fmax:  maximum frequency used
    :         dfres:  frequency resolution used
    :           mk1:  string indicating pick used for earthquake 1
    :           mk2:  string indicating pick used for earthquake 2
    :          twin:  portion of x-c extracted
    """

    #---------SET THE DEFAULT PARAMETERS---------------------
    
    # time range
    if trange is None:
        trange=np.array([0.,3.])
    trange=np.atleast_1d(trange).copy()

    # default is no noise interval
    if nsint is None:
        nsint = []
    elif isinstance(nsint,int):
        # for an integer, pick some intervals
        nsint=defnoiseint(trange=trange,N=nsint,allshf=0.)

    # keep values from first dataset as default
    if st2 is None:
        st2 = st1
    if mk2 is None:
        mk2 = mk1
    
    # time spacing: needs to be the same for all waveforms
    dt1 = [tr.stats.delta for tr in st1]
    dt2 = [tr.stats.delta for tr in st2]
    dt = min(min(dt1),min(dt2))
    nr = int(math.floor(math.log10(abs(dt))))
    dt = round(dt,8-nr-1)

    # maximum frequency used in calculations
    if fmax is None:
        fmax = 0.4/dt

    # number of points in extracted intervals
    N = int(np.round(np.diff(trange)[0]/dt))


    #----------SET UP THE TAPERING AND PICK FREQUENCIES-----------

    # number of points in window
    Nwin = int(np.round(np.diff(twin)[0]/dt))
    Nwin = Nwin + 1 - (Nwin % 2)

    # number of points in FT of x-c
    Nf = Nwin*2

    # desired frequency resolution
    if dfres is None:
        # use best if not given
        dfres = 1./np.diff(twin)[0]
    else:
        # check it's in range
        dfres=np.maximum(dfres,1./np.diff(twin)[0])

    # initialize frequencies
    freq=np.fft.rfftfreq(Nf,d=dt)
    ix=np.logical_and(freq>0,freq<=fmax)
    freq = freq[ix]
    Nfs = len(freq)

    if tpr.lower()=='multi':
        # decide on the tapers' concentration
        NW = dfres / (1./np.diff(twin)[0]) * 2

        # compute tapers
        [U,V] = spectrum.mtm.dpss(Nwin,NW)

        # just select some?
        ii = V>=0.95
        U = U[:,ii]
        V = V[ii]

        Nt = len(V)

    elif tpr.lower()=='slepian':

        # decide on the tapers' concentration
        NW = dfres / (1./np.diff(twin)[0]) * 2

        # compute tapers
        [U,V] = spectrum.mtm.dpss(Nwin,NW)
        U = U[:,0:1]
        V = V[0:1]
        Nt = len(V)

    elif tpr.lower()=='cosine':

        tr = obspy.Trace()
        tr.data = np.ones(Nwin,dtype=float)
        tr.taper(side='both',max_percentage=0.1,type='hann')
        U = tr.data.reshape([Nwin,1])
        V = np.ones(1,dtype=float)
        Nt = len(V)

    elif tpr.lower()=='none':

        U = np.ones([Nwin,1],dtype=float)
        V = np.ones(1,dtype=float)
        Nt = len(V)

    else:
        error('Not a valid tapering option')


    #-------IDENTIFY FREQUENCIES TO COMPARE, INITIALIZE OUTPUT-----------

    # make a list of all available networks/stations/channels
    nsc=np.intersect1d(np.array([tr.stats.network+'.'+tr.stats.station+'.'+
                                 tr.stats.channel for tr in st1]),
                       np.array([tr.stats.network+'.'+tr.stats.station+'.'+
                                 tr.stats.channel for tr in st2]))

    # initialize cross-spectra
    xc = np.zeros([len(freq),len(nsc),Nt],complex)

    # initialize cross-spectra with added noise
    xcn = np.zeros([len(freq),len(nsc),len(nsint)],complex)
    
    # initialize power in main window
    amp = np.zeros([len(freq),len(nsc),Nt,2],float)

    # initialize power in earlier windows
    ampn = np.zeros([len(freq),len(nsc),2,len(nsint)],float)

    # keep track of start times
    starttime1=np.ndarray([len(nsc)],dtype=float)
    starttime2=starttime1.copy()

    # keep track of the station locations
    stloc=np.ndarray([len(nsc),3])*float('nan')
    
    #------CALCULATE THE CROSS-SPECTRA AND AMPLITUDES---------------

    # go through the traces in the first stream
    for m in range(0,len(nsc)):
        nw,st,ch=nsc[m].split('.')
    
        # check for matching traces
        sti1=st1.select(station=st,network=nw,channel=ch).copy().merge()
        sti2=st2.select(station=st,network=nw,channel=ch).copy().merge()

        # to record station locations
        try:
            stloc[m,0]=sti1[0].stats.sac['stlo']
            stloc[m,1]=sti1[0].stats.sac['stla']
            stloc[m,2]=sti1[0].stats.sac['stel']
        except:
            try:
                stloc[m,0]=sti2[0].stats.sac['stlo']
                stloc[m,1]=sti2[0].stats.sac['stla']
                stloc[m,2]=sti2[0].stats.sac['stel']
            except:
                pass

        # identify picks
        pk1,pk2=[],[]
        for tr in sti1:
            if mk1 in tr.stats:
                pk1=tr.stats[mk1]
        for tr in sti2:
            if mk2 in tr.stats:
                pk2=tr.stats[mk2]

        # times to grab
        pk1=sti1[0].stats.starttime+pk1
        pk2=sti2[0].stats.starttime+pk2
        
        # save start times
        starttime1[m]=pk1.timestamp
        starttime2[m]=pk2.timestamp

        # grab the data of interest for the template,
        # buffer the end in case of sampling problems
        data1=sti1.copy().trim(pk1+trange[0],pk1+trange[1]+dt,pad=True,
                               fill_value=0,nearest_sample=True)
        data1=data1.merge()[0]
        data1.taper(side='both',max_percentage=0.1,type='hann')

        data2=sti2.copy().trim(pk2+trange[0]-np.diff(trange)[0],
                               pk2+trange[1]+dt+np.diff(trange)[0],pad=False,
                               fill_value=0,nearest_sample=True)
        data2=data2.merge()[0]

        # grab the data of interest for the template,
        # buffer the end in case of sampling problems
        bata1=sti1.copy().trim(pk1+trange[0]-np.diff(trange)[0],
                               pk1+trange[1]+dt+np.diff(trange)[0],pad=False,
                               fill_value=0,nearest_sample=True)
        bata1=bata1.merge()[0]

        bata2=sti2.copy().trim(pk2+trange[0],
                               pk2+trange[1]+dt,pad=True,
                               fill_value=0,nearest_sample=True)
        bata2=bata2.merge()[0]
        bata2.taper(side='both',max_percentage=0.1,type='hann')

        #-------------------------------------------------
        # cross-correlate the two
        Nfi=np.maximum(data1.stats.npts,data2.stats.npts)*2
        data1i=np.fft.rfft(data1.data,n=Nfi)
        data2i=np.fft.rfft(data2.data,n=Nfi)
        xci=np.multiply(data2i,data1i.conj())
        xci=np.fft.irfft(xci)

        # compute the times relative to zero
        tm1,tm2=data1.stats.starttime,data2.stats.starttime
        dt1,dt2=tm1-pk1,tm2-pk2
        
        # time when the two are aligned
        tal=dt1-dt2
        ial=int(np.round(tal/dt))
        iwin=np.arange(int(np.round(twin[0]/dt)),
                       int(np.round(twin[1]/dt))).astype(int)
        iwin=int((Nwin-1)/2)
        iwin=np.arange(-iwin,iwin+0.5).astype(int)
        xci=xci[(iwin+ial) % xci.size]

        # for accuracy, we'll need precise arrivals
        phsshf=2*math.pi*(tal-ial*dt)*freq
        phsshf=np.exp(1j*phsshf)

        #--------------------------------------------------------
        # cross-correlate the template with itself
        data1i=np.fft.rfft(data1.data,n=Nfi)
        data2i=np.fft.rfft(bata1.data,n=Nfi)
        amp1=np.multiply(data2i,data1i.conj())
        amp1=np.fft.irfft(amp1)

        # compute the times relative to zero
        tm1,tm2=data1.stats.starttime,bata1.stats.starttime
        dt1,dt2=tm1-pk1,tm2-pk1
        
        # time when the two are aligned
        tal=dt1-dt2
        ial=int(np.round(tal/dt))
        amp1=amp1[(iwin+ial) % amp1.size]

        #--------------------------------------------------------
        # cross-correlate the 2nd event with itself
        data1i=np.fft.rfft(bata2.data,n=Nfi)
        data2i=np.fft.rfft(data2.data,n=Nfi)
        amp2=np.multiply(data2i,data1i.conj())
        amp2=np.fft.irfft(amp2)

        # compute the times relative to zero
        tm1,tm2=bata2.stats.starttime,data2.stats.starttime
        dt1,dt2=tm1-pk2,tm2-pk2
        
        # time when the two are aligned
        tal=dt1-dt2
        ial=int(np.round(tal/dt))
        amp2=amp2[(iwin+ial) % amp2.size]

        # de-mean?
        #xci=xci-np.mean(xci)
        
        # taper
        Ntap=U.shape[1]
        xci=np.multiply(xci.reshape([xci.size,1]),U)
        amp1=np.multiply(amp1.reshape([amp1.size,1]),U)
        amp2=np.multiply(amp2.reshape([amp2.size,1]),U)

        # center the zero
        xci=np.roll(np.append(xci,np.zeros([Nwin,Ntap]),axis=0),-int((Nwin-1)/2))
        amp1=np.roll(np.append(amp1,np.zeros([Nwin,Ntap]),axis=0),-int((Nwin-1)/2))
        amp2=np.roll(np.append(amp2,np.zeros([Nwin,Ntap]),axis=0),-int((Nwin-1)/2))

        # compute the FT
        xci=np.fft.rfft(xci,n=Nf,axis=0)
        xci=xci[ix,:]

        amp1=np.fft.rfft(amp1,n=Nf,axis=0)
        amp1=amp1[ix,:]

        amp2=np.fft.rfft(amp2,n=Nf,axis=0)
        amp2=amp2[ix,:]
        
        # save x-c
        xc[:,m,:]=np.multiply(xci,phsshf.reshape([freq.size,1]))

        # keep track of amplitudes
        amp[:,m,:,0]=np.abs(amp1)
        amp[:,m,:,1]=np.abs(amp2)

        for n in range(0,len(nsint)):

            # NOISE INTERVALS NOT IMPLEMENTED

            # shift the relevant picks
            pk1i,pk2i=tm1+nsint[n][0],tm2+nsint[n][0]

            xcn[:,m,n]=np.mean(xc[:,m,:],axis=1)


    #-----------------COLLECT FOR OUTPUT--------------------------------

    xc=xcross({'xc':xc,'xcn':xcn,'amp':amp,'ampn':ampn,'freq':freq,
               'nsc':nsc,'tlm':trange,'starttime1':starttime1,
               'starttime2':starttime2,'dtim':dt,'fmax':fmax,
               'dfres':dfres,'mk1':mk1,'mk2':mk2,'stloc':stloc,
               'twin':twin})

    return xc


class xcross:
    # an object class to contain and process the cross spectra
    
    # set up with a dictionary
    def __init__(self,dct={},pr=None):

        # set all values from the input dictionary
        try:
            for key in dct:
                setattr(self,key,dct[key])
        except:
            dct = dct.__dict__
            for key in dct:
                setattr(self,key,dct[key])

        # set default selection parameters
        self.xcmin=0.9
        self.sratmin=0.9
        self.flmxc=np.array([1.,5.])
        self.sratflm=[[1.,5.],[5.,10.],[10.,20.]]

        # default parameters for noise calculations
        self.Rtrue=np.array([0.4,0.6,0.8,1])
        self.adjfrc=np.array([0.15,0.5,0.85])

        # set number of tapers
        if self.xc.ndim>=3:
            self.Ntap=self.xc.shape[2]
        else:
            self.Ntap=1
        self.tix = np.arange(0,self.Ntap)

        # set number of frequencies
        self.Nf = self.xc.shape[0]

        # set number of stations
        if self.xc.ndim>=2:
            self.Nobs=self.xc.shape[1]
        else:
            self.Nobs=1


        # reshape the inputs
        self.xc=self.xc.reshape([self.Nf,self.Nobs,self.Ntap])
        self.amp=self.amp.reshape([self.Nf,self.Nobs,self.Ntap,2])

        # if the input included a pair, grab that info
        if pr is not None:
            # time range
            self.tlm=np.array([pr.tlm1,pr.tlm2])
            self.dtim=pr.delta
            self.fmax=np.max(self.freq)
            self.dfres=pr.dfres
            self.mk1=pr.arv
            self.mk2=pr.arv
            if not 'starttime1' in self.__dict__.keys():
                self.starttime1=pr.starttime1
            if not 'starttime2' in self.__dict__.keys():
                self.starttime2=pr.starttime2


        #-------COPY ALL THE DATA TO STORE BY OBSERVING COMPONENT---------
        self.xcdata = self.xc

        # amplitudes
        try:
            self.ampdata=self.amp
        except:
            self.ampdata=np.ones([self.Nf,self.Nobs,2],dtype=float)
            self.ampdata.fill(np.nan)

        # cross-correlations with noise
        try:
            self.xcndata = self.xcn
        except:
            self.xcndata = np.ones([self.Nf,self.Nobs,0],dtype=float)
            self.xcndata.fill(np.nan)

        # amplitude of noise intervals
        try:
            self.ampndata = self.ampn
        except:
            self.ampndata = np.ones([self.Nf,self.Nobs,2,0],dtype=float)
            self.ampndata.fill(np.nan)

        # average over stations, not components
        self.avtype = 'station'
        self.avebygroup()

        # default stations used 
        self.igrp=np.arange(0,self.Ngrp)

    def rfromcp(self):
        """
        compute the value of cp that would be expected given R
        """

        # number of stations
        N = float(len(self.igrp))
        N = max(N,1.)

        self.rfcp = (self.Cp*(N-1)+1)/N
        self.rfcp = np.power(self.rfcp,0.5)

    def cpfromr(self):
        """
        compute the value of R that would be expected given Ec/Et
        """

        # number of stations
        N = float(len(self.igrp))
        N = max(N,1.)

        if N>1:
            self.Cp = 1./(N-1)*(np.power(self.R,2)*N-1.)
            self.Cplim = 1./(N-1)*(np.power(self.Rlim,2)*N-1.)
        else:
            self.Cp = np.ndarray(self.R.size,dtype=float)
            self.Cp.fill(np.nan)

            self.Cplim = np.ndarray(self.Rlim.size,dtype=float)
            self.Cplim.fill(np.nan)

    def fitcp(self,fmin=None,fmax=None,nnode=None):
        """
        fit a spline curve to the coherence profile
        :param      fmin:  minimum frequency for range
        :param      fmax:  maximum frequency for range
        :param     nnode:  number of nodes within falloff

        """
        
        if fmin is not None:
            self.fminspl = fmin
        elif not 'fminspl' in self.__dict__.keys():
            self.fminspl = 2

        if fmax is not None:
            self.fmaxspl = fmax
        elif not 'fmaxspl' in self.__dict__.keys():
            self.fmaxspl = np.max(self.freq)

        if nnode is not None:
            self.nnodespl = nnode
        elif not 'nnodespl' in self.__dict__.keys():
            self.nnodespl = 4

                    
        # fit a spline curve to the coherence
        self.tck,trash,self.prd=fitcurve(self.freq,self.Cp,fmin=self.fminspl,
                                         fmax=self.fmaxspl,nnode=self.nnodespl)

    def smoothcp(self,smstd=1.1):
        """
        smooth the coherence profiles
        """

        if smstd is not None:
            self.smstd = smstd
            
        self.Cp,trash=general.logsmooth(self.freq,self.Cp,fstd=self.smstd,
                                              logy=False,subsamp=False)

    def calcazdt(self):
        """
        calculate the average difference in time relative to a
        randomly chosen value
        sets
        :self.aztscl : the set of normalized travel times for all pairs 
                           of stations used in igrp
        :self.aztave : the median normalized travel time
        """

        if self.igrp.size<2:
            # if there's no data
            self.aztave = float('nan')
            self.aztscl = np.array([],dtype=float)
        else:

            # calculate change in travel time for varying location
            # along and across fault
            xyz,dprod=self.calcdirs()
            
            # and need to difference
            if self.avtype=='individual':
                # just grab the values to use
                dprod=dprod[:,self.igrp]
            elif self.avtype=='station':
                # find the station for each component
                ns=np.array(['.'.join(nsci.split('.')[0:2]) for nsci in self.nsc])
                ii=[np.where(ns==nsi)[0][0] for nsi in self.nsa[self.igrp]]
                # grab one per station
                dprod=dprod[:,np.array(ii)]

            # now consider all possible pairs
            i1,i2=np.meshgrid(np.arange(0,self.igrp.size),
                              np.arange(0,self.igrp.size))
            iok=i2>i1
            i1,i2=i1[iok],i2[iok]
            
            # get the relative timing
            dprod=dprod[:,i2]-dprod[:,i1]
            dprod=np.sum(np.power(dprod,2),axis=0)
            self.aztscl=np.power(dprod,0.5)/2**0.5
            self.aztave=np.median(self.aztscl)
        
    def calcdirs(self):
        """
        calculate the dot products between all takeoff directions
        :return    xyz: normalized vectors for the takeoff direction 
                             of each observation
        :return  dprod: dot product of all normalized vectors
        """

        # to north
        xN = np.cos(np.pi/180*self.ststrike)

        # to east
        xE = np.sin(np.pi/180*self.ststrike)

        # horizontal
        xH = np.sin(np.pi/180*self.sttakeang)

        # down
        xD = np.cos(np.pi/180*self.sttakeang)

        # E, N, D
        xyz = np.ndarray([3,len(self.sttakeang)])
        xyz[0,:] = np.multiply(xE,xH)
        xyz[1,:] = np.multiply(xN,xH)
        xyz[2,:] = xD

        # direction along fault strike
        eqstk=np.array([np.sin(np.pi/180*self.eqstrike),
                        np.cos(np.pi/180*self.eqstrike),0])

        # direction along fault dip
        eqdip=np.array([np.sin(np.pi/180*(self.eqstrike+90)),
                        np.cos(np.pi/180*(self.eqstrike+90))])
        eqdip=np.array([np.cos(np.pi/180*self.eqdip)*eqdip[0],
                        np.cos(np.pi/180*self.eqdip)*eqdip[1],
                        np.sin(np.pi/180*self.eqdip)])

        # calculate dot product between all vectors
        dprod = np.vstack([np.dot(xyz.T,eqstk),np.dot(xyz.T,eqdip)])

        return xyz,dprod
        
        
    def pickffreq(self,Rtcut=0.6,frc=None,freqmin=None,cpcutoff=None,ncspl=35):
        """
        estimate falloff frequencies
        :param    ncspl: also compute up to ncspl spline fits for the bootstrapped coherences
                            (default: False)
        """

        if cpcutoff is not None:
            self.cpcutoff = cpcutoff
        elif not 'cpcutoff' in self.__dict__.keys():
            self.cpcutoff = 0.5

        # calculate percentages
        self.calcrprc(rprc=frc)

        # can't be below some value
        if freqmin is None:
            freqmin=2./np.diff(self.tlm)[0]
        if len(self.igrp)>=2:
            # for the best estimate
            self.ffbest = pickffreq(self.Cp,self.freq,self.cpcutoff,freqmin)[0]

            # and the others
            self.ffrprc = pickffreq(self.Cplim,self.freq,self.cpcutoff,freqmin)

            # min, max, and median
            imin = np.argmin(self.rprc)
            self.ffmin = self.ffrprc[imin]

            imax = np.argmax(self.rprc)
            self.ffmax = self.ffrprc[imax]

            imed = np.argmin(np.abs(0.5-self.rprc))
            self.ffmed = self.ffrprc[imed]

            # number of stations and phase coherence
            N = float(len(self.igrp))
            N = max(N,1.)

            # also get histogram falloff frequencies
            if N>1:
                Cprngs = 1./(N-1)*(np.power(self.Rrng,2)*N-1.)
                self.ffhist = pickffreq(Cprngs,self.freq,self.cpcutoff,freqmin)
            else:
                self.ffhist = np.ones(Rrngs.shape[1])*float('nan')

            # and extract values
            self.ffhist.sort()
            ix=(self.ffhist.size*self.rprc).astype(int)
            self.ffminh = self.ffhist[ix[imin]]
            self.ffmaxh = self.ffhist[ix[imax]]
            self.ffmedh = self.ffhist[ix[imed]]

            # fit Cp
            self.fitcp()

            # and get intersections
            df = np.diff(self.freq)[0]
            ix=np.where(self.prd<self.cpcutoff)[0]
            if ix.size:
                self.ffspl=self.freq[ix[0]]-df/2.
            else:
                self.ffspl=self.freq[-1]

                
            # fit a spline curve to the additional ranges
            nch = np.minimum(self.Rrng.shape[1],ncspl)
            if N>1:
                self.ffshist = np.ndarray(nch,dtype=float)
                ich=np.random.permutation(self.Rrng.shape[1])[0:nch]
                ich.sort()
                for k in range(0,nch):
                    tck,trash,prd=fitcurve(self.freq,Cprngs[:,ich[k]],fmin=self.fminspl,
                                           fmax=self.fmaxspl,nnode=self.nnodespl)

                    ix=np.where(prd<self.cpcutoff)[0]
                    if ix.size:
                        self.ffshist[k]=self.freq[ix[0]]-df/2.
                    else:
                        self.ffshist[k]=self.freq[-1]
                        
            else:
                self.ffshist = np.ones(nch)*float('nan')
            
        else:
            self.ffbest=float('nan')
            self.ffmin=float('nan')
            self.ffmax=float('nan')
            self.ffmed=float('nan')
            self.ffminh=float('nan')
            self.ffmaxh=float('nan')
            self.ffmedh=float('nan')


    def pickffreqold(self,Rtcut=0.6,frc=[0.15,0.5,0.85],freqmin=None):
        """
        estimate falloff frequencies
        """
        self.calcsigfrc()
        if freqmin is None:
            freqmin=2./np.diff(self.tlm)[0]
        if len(self.igrp)>=2:
            self.ffbest,self.ffmin,self.ffmax = \
                pickffreq(self.R,self.S[:,self.igrp],
                          freq=self.freq,freqmin=freqmin,
                          Rtcut=Rtcut,frc=frc,Ntap=self.Ntap)
        else:
            self.ffbest=float('nan')
            self.ffmin=float('nan')
            self.ffmax=float('nan')


    def avebygroup(self,avtype=None):
        """
        average over groups rather than separating by observations
        NOTE THAT THIS RESETS igrp
        :param      avtype:   averaging type, one of
                               'station','individual'
                               (default: self.avtype)
        """

        # save averaging type if necessary
        if avtype is not None:
            self.avtype=avtype

        if self.avtype == 'station':
            # average over stations
            self.avebystat()
        if self.avtype == 'individual':
            # just copy the original
            self.xc = self.xcdata
            self.xcn = self.xcndata
            self.amp = self.ampdata
            self.ampn = self.ampndata

        # set number of stations
        if self.xc.ndim>=2:
            self.Ngrp=self.xc.shape[1]
        else:
            self.Ngrp=1

        # reset the stations or observations to use
        self.igrp = np.arange(0,self.Ngrp)


    def adjtiming(self):
        """
        fit a linear curve to the frequency-domain phases to 
        allow for variations in relative timing
        """

        if not "timeshift" in self.__dict__.keys():
            self.timeshift = np.zeros(self.xc.shape[1],dtype=float)

        # compute average cross-correlation phases
        phs = np.mean(self.xc,axis=2)
        phs = np.angle(phs)/(2*np.pi)

        # weights, start even per log frequency
        dfreq = (self.freq[1]-self.freq[0])/2.
        wgts=np.log(self.freq+dfreq)-np.log(self.freq-dfreq)

        wgts=np.multiply(wgts,np.exp(-np.power(self.freq/15,2)))
        #wgts=np.multiply(wgts,np.power(self.freq,-1))
        
        wgts=wgts.reshape([wgts.size,1])
        
        M = self.freq.reshape(wgts.shape)
        M = np.multiply(M,wgts)

        b = np.multiply(phs,wgts)

        # solve for preferred time shifts
        from numpy.linalg import lstsq
        X,resid,rank,s = lstsq(M,b)

        # adjust the x-c
        phsa = np.multiply(self.freq.reshape([self.freq.size,1]),
                           X.reshape([1,X.size]))
        phsa = np.exp(-1j*2*np.pi*phsa)
        phsa = phsa.reshape([self.freq.size,X.size,1])
        self.xc = np.multiply(self.xc,phsa)

        # add this time shift to previous
        self.timeshift=self.timeshift+X.flatten()
        

    def avebystat(self):
        """
        average all values over stations
        """

        # all the computed values
        nsi = np.array([nsci.split('.') for nsci in self.nsc])
        nsi = nsi.reshape([len(self.nsc),3])
        self.nsa=np.array([nsi[k,0]+'.'+nsi[k,1] for 
                           k in range(0,nsi.shape[0])])
        self.cmps=nsi[:,2]
        
        # just the unique stations
        self.ns = np.unique(self.nsa)
        
        # need to re-organize xc, xcn, amp, ampn

        # create them all
        shp=list(self.xc.shape)
        shp[1]=len(self.ns)
        self.xc=np.zeros(shp,dtype=complex)

        shp=list(self.xcn.shape)
        shp[1]=len(self.ns)
        self.xcn=np.zeros(shp,dtype=complex)

        shp=list(self.amp.shape)
        shp[1]=len(self.ns)
        self.amp=np.zeros(shp,dtype=float)

        shp=list(self.ampn.shape)
        shp[1]=len(self.ns)
        self.ampn=np.zeros(shp,dtype=float)

        # sum contributions
        for k in range(0,len(self.ns)):
            ii, = np.where(self.nsa==self.ns[k])
            for m in ii:
                self.xc[:,k,:]=self.xc[:,k,:]+self.xcdata[:,m,:]
                self.xcn[:,k,:]=self.xcn[:,k,:]+self.xcndata[:,m,:]
                self.amp[:,k,:,:]=self.amp[:,k,:,:]+self.ampdata[:,m,:,:]
                self.ampn[:,k,:,:]=self.ampn[:,k,:,:]+self.ampndata[:,m,:,:]

    def calcdirxc(self):
        """
        calculate cross-correlation at each frequency per station
        """

        # average energy over tapers
        xci=np.power(np.mean(np.power(np.abs(self.xc[:,:,self.tix]),2),axis=2),0.5)

        # average coherent energy over tapers and normalize
        self.xcbystat = np.divide(np.mean(self.xc[:,:,self.tix],axis=2),xci)
        

        # normalize by the averaged amplitude and
        # also average over stations
        self.xcdir = np.mean(np.divide(self.xcbystat[:,self.igrp],
                                       np.abs(self.xcbystat[:,self.igrp])),
                             axis=1)

    def pickobs(self,xcmin=None,flmxc=None,sratmin=None,sratflm=None):
        """
        select observations according to x-c and signal fraction
        :param   xcmin:  minimum frequency-averaged phase coherence for the 
                    direct cross-correlation (default: None---no min)
        :param   flmxc:  frequency limits to test for direct coherence
                     (default: [1,5])
        :param sratmin:  minimum signal ratio at each station
                     can be a list---one per frequency band
                     (default: None---no min)
        :param sratflm:  list of frequency bands to test for noise level
                     (default: [[1,5],[5,10],[10,20]]
        """

        # set specified values
        if xcmin is not None:
            self.xcmin=xcmin
        if flmxc is not None:
            self.flmxc=flmxc
        if sratmin is not None:
            self.sratmin=sratmin
        if sratflm is not None:
            self.sratflm=sratflm

        if self.sratmin > 0.:
            # needs to be a list
            if isinstance(self.sratflm[0],float) or \
                    isinstance(self.sratflm[0],int):
                self.sratflm=[self.sratflm]
        
        # repeat signal fraction to preferred size
        sratmin=np.atleast_1d(self.sratmin)
        if len(sratmin)==1:
            sratmin=np.repeat(sratmin,len(self.sratflm))
            
        # check for direct phase coherence
        iok = np.ones(self.Ngrp,dtype=bool)
        if self.xcmin>0.:
            ix=np.logical_and(self.freq>=self.flmxc[0],
                              self.freq<=self.flmxc[1])
            mnxc=np.divide(np.real(self.xc[ix,:,:]),
                           np.abs(self.xc[ix,:,:]))
            mnxc=np.mean(mnxc,axis=2)
            mnxc=np.mean(mnxc,axis=0)
            iok = np.logical_and(iok,mnxc>self.xcmin)

        # check for signal ratio
        if np.max(sratmin) > 0.:
            for m in range(0,len(self.sratflm)):
                ix=np.logical_and(self.freq>=self.sratflm[m][0],
                                  self.freq<self.sratflm[m][1])
                try:
                    mnsrat1=np.mean(self.srat1[ix,:],axis=0)
                    mnsrat2=np.mean(self.srat2[ix,:],axis=0)
                except:
                    mnsrat1=np.mean(self.srat[ix,:,0],axis=0)
                    mnsrat2=np.mean(self.srat[ix,:,1],axis=0)
                iok = np.logical_and(iok,mnsrat1>sratmin[m])
                iok = np.logical_and(iok,mnsrat2>sratmin[m])

        # collect values
        self.igrp,=np.where(iok)

    def calcpc(self,igrp=None):
        """
        to calculate moveout
        """
        if igrp is not None:
            self.igrp=igrp

        self.Pca,self.Pc,self.Pcrng,self.Pta,self.Pt,self.Pda,self.Pd=\
           calcPc(self.xc,self.amp,igrp=self.igrp,tix=self.tix)

    def subpnamp(self,igrp=None):
        """
        to calculate the noise power and subtract it,
        using amplitudes
        """

        if igrp is not None:
            self.igrp=igrp

        # divide noise by the template
        Pn=np.mean(self.ampn[:,:,0,:],axis=2)
        Pn=np.divide(Pn,np.mean(self.amp[:,:,:,0],axis=2))

        # average over all
        Ns = self.ampn.shape[1]
        if Ns>=1:
            self.Pna=np.mean(Pn,axis=1)
        else:
            self.Pna=np.ndarray(self.Nf,dtype=float)*float('nan')

        # average over subset
        Ng = len(self.igrp)
        if Ng>=1:
            self.Pn=np.mean(Pn[:,self.igrp],axis=1)
        else:
            self.Pn=np.ndarray(self.Nf,dtype=float)*float('nan')


        # subtract from total
        self.Pl=self.Pt-self.Pn
        self.Pla=self.Pta-self.Pna

    def subpnxc(self,igrp=None):
        """
        to calculate the noise power and subtract it,
        using cross-correlations
        """

        if igrp is not None:
            self.igrp=igrp

        # subtract original to isolate the noise
        shp=np.array([self.Nf,self.xcn.shape[1],1])
        xc = self.xcn-np.mean(self.xc,axis=2).reshape(shp)
        xc = np.abs(xc)

        # divide noise by the template
        xc = np.divide(xc,np.mean(self.amp[:,:,:,0],axis=2).reshape(shp))

        # square and average over noise intervals
        xc = np.power(xc,2)
        xc = np.mean(xc,axis=2)

        # average over all
        Ns = self.ampn.shape[1]
        if Ns>=1:
            self.Pna=np.mean(xc,axis=1)
        else:
            self.Pna=np.ndarray(self.Nf,dtype=float)*float('nan')

        # average over subset
        Ng = len(self.igrp)
        if Ng>=1:
            self.Pn=np.mean(xc[:,self.igrp],axis=1)
        else:
            self.Pn=np.ndarray(self.Nf,dtype=float)*float('nan')


        # subtract from total
        self.Pl=self.Pt-self.Pn
        self.Pla=self.Pta-self.Pna

    def calcmvout(self,igrp=None):
        """
        to calculate moveout
        """
        if igrp is not None:
            self.igrp=igrp

        self.Ra,self.R,self.Rrng=calcmvout(self.xc,self.amp,
                                           igrp=self.igrp,tix=self.tix)

        # calculate percentages
        self.calcrprc()

        # also calculate Cp
        self.cpfromr()

    def calcrprc(self,rprc=None):
        """
        calculate a range of possible phase walkout values
        """

        # the percentages to calculate
        if rprc is not None:
            self.rprc = rprc
        else:
            try:
                rprc = self.rprc
            except:
                self.rprc = np.array([0.5,0.15,0.85])

        # copy before sorting
        Rrngs = self.Rrng.copy()
        Rrngs.sort(axis=1)

        # extract the relevant indices
        ix = np.atleast_1d(self.rprc)*Rrngs.shape[1]
        ix = ix.astype(int)
        ix = np.minimum(ix,Rrngs.shape[1])

        # number of stations and phase coherence
        N = float(len(self.igrp))
        N = max(N,1.)

        # the limits
        self.Rlim = Rrngs[:,ix]

        if N>1:
            self.Cplim = 1./(N-1)*(np.power(self.Rlim,2)*N-1.)
        else:
            self.Cplim = self.Rlim.copy()

    def calcenfrc(self,igrp=None):
        """
        to calculate coherent and incoherent energy
        """
        if igrp is not None:
            self.igrp=igrp

        # calculate
        self.Ec,self.Et=\
            calcenfrc(self.xc,self.amp,self.ampn,igrp=self.igrp)

        # fraction
        self.Ecfrc = np.divide(self.Ec,self.Et)

    def calcavesigfrc(self):
        """
        average S over the relevant stations
        """

        self.Sav = np.mean(self.S[:,self.igrp],axis=1)

    def calcsigfrc(self):
        """
        compute event-averaged signal fraction
        """
        try:
            self.S = np.multiply(self.srat1,self.srat2)
        except:
            self.S = np.multiply(self.srat[:,:,0],self.srat[:,:,1])

        self.Sav = np.mean(self.S[:,self.igrp],axis=1)

    def dropunused(self):
        """
        delete events and stations that aren't used in the calculations
        to save memory before saving to a file
        """

        if self.avtype != 'individual':
            print('Are you sure---the groupings are set to '+
                  self.avtype)
            print('Nothing done')
        else:
            # extract the relevant portions of the data
            self.xcdata=self.xcdata[:,self.igrp,:]
            self.ampdata=self.ampdata[:,self.igrp,:,:]
            self.xcndata=self.xcndata[:,self.igrp,:]
            self.ampndata=self.ampndata[:,self.igrp,:,:]

            # stations, number of observations
            self.Nobs=self.igrp.size
            self.nsc=self.nsc[self.igrp]

            # all the computed values
            nsi = np.array([nsci.split('.') for nsci in self.nsc])
            nsi = nsi.reshape([len(self.nsc),3])
            self.ns=np.array([nsi[k,0]+'.'+nsi[k,1] for 
                              k in range(0,nsi.shape[0])])
            self.ns = np.unique(self.ns)

            # replace with all the indices
            self.igrp = np.arange(0,self.Nobs)

            # copy to current data
            self.avebygroup()

    def signalfraction(self,sg=None):
        """
        compute the estimated signal ratios
        :param     sg:  signal amplitude, if to be replaced
        """

        # average signal plus noise over tapers
        Nf,Ns,Nt,Ne=self.amp.shape
        sgn = np.mean(self.amp,axis=2).reshape([Nf,Ns,Ne])

        # average noise
        if self.ampn.ndim>3:
            ns = np.mean(self.ampn,axis=3).reshape([Nf,Ns,Ne])
        else:
            ns = self.ampn

        # noise to signal plus noise
        nrat = np.real(np.divide(ns,sgn))

        # don't let it be more than 1
        nrat = np.minimum(nrat,np.array([1.]))

        # signal to noise
        if sg is None:
            # just one minus noise ratio
            srat = 1.-nrat
        else:
            # if the signal is given
            srat = np.divide(sg,sgn).astype(float)
            srat = np.minimum(srat,np.array([1.]))
            
        # recorded amplitudes were squared
        nrat = np.power(nrat,.5)
        srat = np.power(srat,.5)

        # save
        self.nrat = nrat
        self.srat = srat

        # also need to save signal ratio averaged over events
        self.S=np.multiply(self.srat[:,:,0],self.srat[:,:,1])

def calcmvout(xc,amp,igrp=None,tix=None):
    """
    :param     xc:  cross-correlations, or other values
    :param    amp:  amplitudes of individual components 
                       (not currently used)
    :param   igrp:  extra station pairs to calculate, if desired
    :param    tix:  indices of the tapers to use
    :return     r:  averaged radii, over all stations
    :return    ri:  averaged radii, for the specified station pairs
    """

    # copy to avoid overwriting
    xc = copy.copy(xc)

    # FIRST AVERAGE OVER TAPERS TO REDUCE NOISE
    if xc.ndim>=3:
        # pick which tapers to consider
        Nt=xc.shape[2]
        if tix is None:
            tix=np.arange(0,Nt)

        # just some
        xc = xc[:,:,tix]
        Nt = xc.shape[2]

        # to keep for uncertainties
        xct = xc

        # and average
        xc = np.mean(xc,axis=2)
    else:
        # just one taper to use
        Nt=1
        if tix is None:
            tix=np.arange(0,Nt)
    
    # number of stations
    if xc.ndim>=2:
        Ns=xc.shape[1]
    else:
        Ns=1

    # default is to use all stations
    if igrp is None:
        igrp = np.arange(0,Ns)

    # number of frequencies
    Nf=xc.shape[0]

    # normalize
    xc = np.divide(xc,np.abs(xc))

    # to a convenient shape
    xc = xc.reshape([Nf,Ns])

    # average for specified station groups
    ri=np.abs(np.mean(xc[:,igrp],axis=1))

    # and for everything
    r=np.abs(np.mean(xc,axis=1))

    if Nt > 1:
        # if there are multiple tapers, bootstrap them
        Nu = 1000
        ru = np.ndarray([Nf,Nu],dtype=float)
        xct = xct[:,igrp,:]
        Nsi = len(igrp)
        for k in range(0,Nu):
            ii = np.random.choice(Nt,Nt)
            xci = np.mean(xct[:,:,ii],axis=2)
            xci = np.divide(xci,np.abs(xci))
            ru[:,k] = np.abs(np.mean(xci,axis=1))
    else:
        # just repeat a number of times
        Nu = 100
        ru =np.repeat(ri.reshape([ri.size,1]),Nu,axis=1)



    # return radii
    return r,ri,ru

def calcPc(xc,amp,igrp=None,tix=None):
    """
    :param     xc:  cross-correlations, or other values
    :param    amp:  amplitudes of individual components 
                       (not currently used)
    :param   igrp:  extra station pairs to calculate, if desired
    :param    tix:  indices of the tapers to use
    :return     r:  averaged radii, over all stations
    :return    ri:  averaged radii, for the specified station pairs
    """

    # copy to avoid overwriting
    xc = copy.copy(xc)

    # FIRST AVERAGE OVER TAPERS TO REDUCE NOISE
    if xc.ndim>=3:
        # pick which tapers to consider
        Nt=xc.shape[2]
        if tix is None:
            tix=np.arange(0,Nt)

        # just some
        xc = xc[:,:,tix]
        amp = amp[:,:,tix,0]
        Nt = xc.shape[2]

        # to keep for uncertainties
        xct = xc
        ampt = amp

        # and average
        xc = np.mean(xc,axis=2)
        amp = np.mean(amp,axis=2)
    else:
        # just one taper to use
        Nt=1
        if tix is None:
            tix=np.arange(0,Nt)
        amp=amp[:,:,0,0]
    
    # number of stations
    if xc.ndim>=2:
        Ns=xc.shape[1]
    else:
        Ns=1

    # default is to use all stations
    if igrp is None:
        igrp = np.arange(0,Ns)

    # number of frequencies
    Nf=xc.shape[0]

    # normalize by the template amplitude
    xc = np.divide(xc,amp)

    # to a convenient shape
    xc = xc.reshape([Nf,Ns])

    # calculate directly coherent power
    Ng = len(igrp)    
    if Ng>=1:
        pdi=np.mean(np.real(xc[:,igrp]),axis=1)
        pdi=np.multiply(np.power(pdi,2),np.sign(pdi))
    else:
        pdi=np.ndarray(xc.shape[0],dtype=float)*float('nan')

    if Ns>=1:
        pd=np.mean(np.real(xc),axis=1)
        pd=np.multiply(np.power(pd,2),np.sign(pd))
    else:
        pd=np.ndarray(xc.shape[0],dtype=float)*float('nan')

    # average Pc for specified station groups
    if Ng>1:
        ri1=np.power(np.abs(np.sum(xc[:,igrp],axis=1)),2)
        pti=np.sum(np.power(np.abs(xc[:,igrp]),2),axis=1)
        pci=1/Ng/(Ng-1)*(ri1-pti)
        pti=pti/Ng
    else:
        pci=np.ndarray(xc.shape[0],dtype=float)*float('nan')
        pti=np.ndarray(xc.shape[0],dtype=float)*float('nan')

    # and for everything
    if Ns>1:
        r1=np.power(np.abs(np.sum(xc,axis=1)),2)
        pt=np.sum(np.power(np.abs(xc),2),axis=1)
        pc=1/Ns/(Ns-1)*(r1-pt)
        pt=pt/Ns
    else:
        pc=np.ndarray(xc.shape[0],dtype=float)*float('nan')
        pt=np.ndarray(xc.shape[0],dtype=float)*float('nan')
        
    if Nt > 1 and Ng > 1:
        # if there are multiple tapers, bootstrap them
        Nu = 1000
        pcu = np.ndarray([Nf,Nu],dtype=float)
        xct = xct[:,igrp,:]
        ampt = ampt[:,igrp,:]
        Nsi = len(igrp)
        for k in range(0,Nu):
            ii = np.random.choice(Nt,Nt)
            xci = np.mean(xct[:,:,ii],axis=2)
            ampi = np.mean(ampt[:,:,ii],axis=2)
            xci = np.divide(xci,ampi)
            r1=np.power(np.abs(np.sum(xci,axis=1)),2)
            r2=np.sum(np.power(np.abs(xci),2),axis=1)
            pcu[:,k]=1/Ng/(Ng-1)*(r1-r2)
    else:
        # just repeat a number of times
        Nu = 100
        pcu =np.repeat(pc.reshape([pc.size,1]),Nu,axis=1)

    # return radii
    return pc,pci,pcu,pt,pti,pd,pdi


def calcmvoutold(xc,amp=None,igrp=None):
    """
    :param     xc:  cross-correlations, or other values
    :param    amp:  not used
    :param   igrp:  extra station pairs to calculate, if desired
    :return     r:  averaged radii, over all stations
    :return    ri:  averaged radii, for the specified station pairs
    """

    # copy to avoid overwriting
    xc = copy.copy(xc)

    # FIRST AVERAGE OVER TAPERS TO REDUCE NOISE
    if xc.ndim>=3:
        Nt=xc.shape[2]
        xc = np.mean(xc,axis=2)
        Nt=1
    else:
        Nt=1

    # number of stations
    if xc.ndim>=2:
        Ns=xc.shape[1]
    else:
        Ns=1

    # default is to use all stations
    if igrp is None:
        igrp = []

    # number of frequencies
    Nf=xc.shape[0]

    # normalize
    xc = np.divide(xc,np.abs(xc))
    
    # to a convenient shape
    xc = xc.reshape([Nf,Ns,Nt])

    # average for specified station groups
    ri = np.ndarray([Nf,len(igrp)])
    for k in range(0,len(igrp)):
        # average over stations
        rri=np.abs(np.mean(xc[:,igrp[k],:],axis=1)).reshape([Nf,Nt])
        ri[:,k]=np.mean(rri,axis=1)
    ri=ri.flatten()

    # and for everything
    r=np.abs(np.mean(xc,axis=1)).reshape([Nf,Nt])
    r=r.flatten()

    # return radii
    return r,ri


def defnoiseint(trange=None,N=3,allshf=0.):
    """
    pick some noise intervals, spaced before trange
    :param    trange: time range
    :param         N: number of values (default: 3)
    :param    allshf: time to shift everything (default: 0.)
    :return noiseint: list of noise intervals
    """

    if trange is None:
        trange = [0.,1.]
    trange = np.atleast_1d(trange)
    
    # spacing between start times
    sp = np.diff(trange)[0]*1.5

    # initialize
    noiseint = []
    
    for k in range(0,N):
        noiseint.append(-sp*(k+1.)+allshf+trange)

    return noiseint


def adjcoh(S,Cptrue=[0.4,0.6,0.8,1],frc=0.5,Ntap=1):
    """
    :param        S:  signal fractions
                        number frequencies x number of stations
                        repeated for all stations if one number is given
    :param   Cptrue:  true coherence values to calculate for
                        (default: [0.4,0.6,0.8,1])
    :param      frc:  location within the distribution to calculate for
                        (default: 0.5---just the median)
    :param     Ntap:  number of tapers to average over (default: 1)
    :return   Cpadj:  expected adjusted phase coherence values
    :return  Cptrue:  true coherence values used in calculations
    :return     frc:  location within the distribution to calculate for
    :return    Cpit:  one estimate of the phase coherence for each frequency
    """

    # fractions to evaluate
    frc=np.atleast_1d(frc)
    Nfr=frc.size

    # number of values to estimate
    Cptrue=np.atleast_1d(Cptrue)
    Ntrue=Cptrue.size

    # number of frequencies
    Nf = S.shape[0]

    # initialize
    Cpadj = np.ndarray([Nf,Ntrue,Nfr],dtype=float)

    # initialize iteration
    Cpit = np.ndarray([Nf,Ntrue],dtype=float)

    if S.size:
        for k in range(0,Nf):
            # iterate over frequencies
            for m in range(0,Ntrue):
                # iterate over coherent fractions
                Cpadj[k,m,:],nper,bns,vls = \
                    exprad(S=S[k,:],Cp=Cptrue[m],frc=frc,Ntap=Ntap)

                # pick one sample
                Cpit[k,m] = vls[np.random.choice(vls.size,1)[0]]
    else:
        Cpadj = np.ma.masked_array(Radj,mask=True)
    
    return Cpadj,Cptrue,frc,Cpit
    
def pickffreq(Cp,freq,cutoff,freqmin):
    """
    :param       Cp:  observed phase coherence or walkout
    :param     freq:  frequencies
    :param   cutoff:  the cutoff value of interest
    :param  freqmin:  minimum allowable frequencies
    :return   ffall:  falloff frequency 
                 (or frequencies if multiply columns in Cp)
    """


    # organize as a matrix
    Cp = np.atleast_1d(Cp)
    if Cp.ndim<2:
        Cp = Cp.reshape([Cp.size,1])

    # identify values smaller than cutoff
    sml = Cp<cutoff
    
    # but not smaller than minimum frequency
    fsml = freq<freqmin
    sml[fsml,:] = False

    # how many below are too small
    sml = np.cumsum(sml,axis=0)
    sml = np.sum(sml==0,axis=0)

    # but want the index before that
    sml = np.maximum(sml-1,0).astype(int)
    ffall = np.ndarray(len(sml),dtype=float)
    for k in range(0,len(sml)):
        if sml[k]==0:
            ffall[k] = freq[sml[k]]
        else:
            ffall[k]=np.interp([0.5],np.flipud(Cp[sml[k]:(sml[k]+2),k]),
                               np.flipud(np.log(freq[sml[k]:(sml[k]+2)])))
            ffall[k]=np.exp(ffall[k])

    return ffall

def pickffreqold(R,S,freq=None,freqmin=0.,Rtcut=0.6,
              frc=[0.15,0.5,0.85],Ntap=1):
    """
    :param        R:  observed phase walkout
    :param        S:  signal fractions
                            ? x number of stations
    :param     freq:  frequencies
    :param  freqmin:  minimum allowable frequencies
    :param    Rtcut:  true coherent fraction cutoff (default: 0.6)
    :param      frc:  location within the distribution to calculate for
                            [max,median,min] frequencies
                            (default: 0.5---just the median)
    :param     Ntap:  number of tapers to average over (default: 1)
    :return  ffbest:  best estimate of falloff frequency---using median
    :return   ffmin:  low estimate of falloff frequency---using frc[2]
    :return   ffmax:  high estimate of falloff frequency---using frc[0]
    """
    

    # calculate adjusted phase walkouts
    Radj,Rtrue,frc,Rit=adjcoh(S,Rtrue=Rtcut,frc=frc,Ntap=Ntap)

    # default frequencies
    if freq is None:
        freq=np.arange(0.,S.shape[0])

    # only within some range
    iok = freq>=freqmin

    # want two in a row
    iok2,= np.where(iok)
    iok2 = np.minimum(iok2+1,R.shape[0]-1)
    freq2 = freq[iok2]

    freq = freq[iok]
    Nf=freq.size

    # best---observed is smaller than median
    ffbest=np.logical_and(R[iok]<Radj[iok,0,1],R[iok2]<Radj[iok2,0,1])
    ffbest,=np.where(ffbest)
    ffbest=np.min(np.append(ffbest,Nf-1))
    ffbest=(freq[ffbest]+freq2[ffbest])/2.
    
    # min---observed is smaller than 85th percentile
    ffmin=np.logical_and(R[iok]<Radj[iok,0,2],R[iok2]<Radj[iok2,0,2])
    ffmin,=np.where(ffmin)
    ffmin=np.min(np.append(ffmin,Nf-1))
    ffmin=(freq[ffmin]+freq2[ffmin])/2.

    # max---observed is smaller than 15th percentile
    ffmax=np.logical_and(R[iok]<Radj[iok,0,0],R[iok2]<Radj[iok2,0,0])
    ffmax,=np.where(ffmax)
    ffmax=np.min(np.append(ffmax,Nf-1))
    ffmax=(freq[ffmax]+freq2[ffmax])/2.

    return ffbest,ffmin,ffmax

def exprad(S=None,Cp=None,wgt=None,frc=None,Ntap=1):
    """
    get expected amplitude for specified fractions
    :param     S: the (amplitude) signal fraction for each station,
                    repeated if only one value given
                    (default: 0.7)
    :param    Cp: the phase coherence
                    repeated if only one value given
                    (default: 0.9)
    :param   wgt: weights for each station (default: simple average)
    :param   frc: fractions within the distributions to separate
                    (default: [0.15,0.5,0.85])
    :param  Ntap:  number of independent realizations to average over
                    (default: 1)
    :return  cpi: phase coherence at the specified fractions
    :return nper: number per bin
    :return  bns: bin edges
    :return   cp: the full distribution of estimated coherence values
    """

    # make some values up if not given
    if frc is None:
        frc = np.array([0.15,0.5,0.85])
    if S is None:
        S = 0.7
    if Cp is None:
        Cp = 0.9
    S = np.atleast_1d(S)
    Cp = np.atleast_1d(Cp)

    # number of stations
    Ns = max(len(S),len(Cp))
      
    # convert Cp to R
    R = np.power(((Ns-1)*Cp+1)/Ns,0.5)
    
    # default even weighting
    if wgt is None:
        wgt = np.ones(Ns)/Ns
    else:
        wgt = np.atleast_1d(wgt)
        Ns = max(Ns,len(wgt))

    # repeat to the preferred size
    if len(S)<Ns:
        S = S.repeat(Ns/len(S))
    if len(wgt)<Ns:
        wgt = wgt.repeat(Ns/len(wgt))
    if len(R)<Ns:
        R = R.repeat(Ns/len(R))

    # generate the random values
    S=S.reshape([Ns,1])
    R=R.reshape([Ns,1])
    wgt=wgt.reshape([Ns,1])

    # number of values to consider
    N = 1000

    # coherent
    I=np.multiply(R,S)
    cht=np.ones([1,N])
    cht=np.multiply(I.reshape([Ns,1]),cht)

    # incoherent
    I=np.power(1-np.power(R,2),0.5)
    I=np.multiply(I,S)
    # best to use a full normal distribution
    rnd=(np.random.randn(Ns*N)+1j*np.random.randn(Ns*N))/2**0.5
    #rnd=np.exp(np.random.rand(Ns*N)*(1j*2*math.pi))
    rnd=np.multiply(I.reshape([Ns,1]),rnd.reshape([Ns,N]))

    # noise
    I=np.power(1-np.power(S,2),0.5)
    I=I/Ntap**0.5
    #ns=np.exp(np.random.rand(Ns*N)*(1j*2*math.pi))
    ns=(np.random.randn(Ns*N)+1j*np.random.randn(Ns*N))/2**0.5
    ns=np.multiply(I.reshape([Ns,1]),ns.reshape([Ns,N]))
    

    # add
    r = cht + rnd + ns

    # normalize
    r = np.divide(r,np.abs(r))
    
    # weight
    r = np.multiply(wgt.reshape([Ns,1]),r)

    # sum over stations
    r = np.sum(r,axis=0)
    r = np.minimum(np.abs(r),1)

    # to cp
    cp = 1/(Ns-1)*(np.power(r,2)*Ns-1)

    # sort
    cp = np.sort(cp)

    # bin
    bns = np.linspace(-1.,1.,101)
    nper,trash = np.histogram(cp,bins=bns)

    # percentages
    ix = np.floor(frc*N).astype(int)
    cpi = cp[ix]

    return cpi,nper,bns,cp



def fitcurve(freq,Cp,fmin=1,fmax=None,nnode=4):
    """
    :param     freq: freuencies of data
    :param       Cp: data to match
    :param     fmin: minimum smallest frequency to allow
    :param     fmax: maximum largest frequency to allow
    """

    # organize as a matrix
    Cp = np.atleast_1d(Cp)
    if Cp.ndim<2:
        Cp = Cp.reshape([Cp.size,1])

    if fmax is None:
        fmax = np.max(freq)

    # try a range of frequencies
    minrat = 1.5
    f1 = np.linspace(np.log(fmin),np.log(fmax),20)
    f2 = f1+np.log(4)
    f1=np.hstack([f1])
    f2=np.hstack([f1+np.log(4)])
    # f2 = np.linspace(np.log(fmin*minrat),np.log(fmax*minrat),30)
    # f1,f2=np.meshgrid(f1,f2)
    # iok=f1<f2-np.log(minrat)
    # f1,f2=f1[iok],f2[iok]
    
    #f1=np.array([0.828044])
    #f2=np.array([5.0106])

    # weight by 1/frequency to make things even in log space
    wgts=np.divide(1,freq)
    dfreq=freq[1]-freq[0]
    wgts=np.log(np.divide(freq+dfreq/2,freq-dfreq/2))
    rwgts=np.max(wgts)*10*np.ones(10,dtype=float)

    # track misfits
    msft = np.ndarray([f1.size,Cp.shape[1]],dtype=float)

    tcks = []
    
    # go through each one
    lfreq = np.log(freq)
    for k in np.arange(0,f1.size):
        f1i,f2i=f1[k],f2[k]

        # which frequencies to match       
        ii=np.logical_and(lfreq>f1i,lfreq<f2i)

        # the data
        df=f2i-f1i
        toadd=np.linspace(0.01,2,rwgts.size)*df
        x=np.hstack([f1i-np.flipud(toadd),lfreq[ii],f2i+toadd])
        w=np.ones(x.size,dtype=float)
        scl=np.sum(wgts[ii])/np.sum(rwgts)*100
        w=np.hstack([rwgts*scl,wgts[ii],rwgts*scl])
        
        if np.sum(ii)>=2:
            # pick nodes with enough spacing
            nknot=np.minimum(nnode,np.sum(ii))+1
            nper=np.array([0,0,0])
            while np.sum(nper<2) and nknot>0:
                nknot=nknot-1
                t=np.linspace(f1i,f2i,nknot+2)
                ix=np.searchsorted(t,x)
                nper=np.bincount(ix,minlength=t.size+1)

            for m in range(0,Cp.shape[1]):
                # estimate fit
                y=np.hstack([np.ones(rwgts.size),Cp[ii,m],np.zeros(rwgts.size)])
                tck=splrep(x,y,w,k=3,t=t)
                tcks.append(tck)

                # predict for all frequencies
                prd=splev(lfreq,tck,ext=0)

                # misfit
                msft[k,m]=np.dot(wgts,np.abs(Cp[:,m]-prd))
                

        else:
            for m in range(0,Cp.shape[1]):
                # prediction if no events in middle range
                prd = np.zeros(Cp.size,dtype=float)
                prd[lfreq<=f1i] = 1.
                prd[ii]=(f2i-lfreq[ii])/(f2i-f1i)
                tcks.append([])
                
                # misfit
                msft[k,m]=np.dot(wgts,np.abs(Cp[:,m]-prd))
                
    for m in range(0,Cp.shape[1]):
        # minimum misfit
        imin=np.argmin(msft[:,m])
        tck = tcks[imin]
        f1i,f2i=f1[imin],f2[imin]
        if tck:
            # predict for all frequencies
            prd=splev(lfreq,tck,ext=0)
        else:
            prd = np.zeros(Cp.size,dtype=float)
            prd[lfreq<=f1i] = 1.
            prd[ii]=(f2i-lfreq[ii])/(f2i-f1i)

    return tck,freq,prd
