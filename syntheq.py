import numpy as np
import math
import scipy
from scipy import interpolate
from scipy import ndimage
import obspy

class seq:
    """
    a synthetic earthquake, 
    with slip distribution, timing, and 
    functions to create synthetic seismograms

    PROPERTIES OF THE SLIP DISTRIBUTION
    :a        :  the half-length along the x-axis, in meters
    :b        :  the half-length along the y-axis, in meters
    :dx       :  spacing of slip points along the fault, in m
    :rndedg   :  the [amplitude,# of nodes] of a perturbation of
                    a circular rupture edge
    :slipdist :  the type of slip distribution, usually a string with
            the form a-b, where a is
        elliptical  :  an elliptical slip distribution
        rectangular :  a uniform rectangular distribution
        circtapered :  a circular (or rndedg-modified) uniform slip 
                         tapered to zero in the final 10%
            and b could be empty or include
        fractal     :  generate a slip distribution over a wide area
                          whose power falls off as wavenumber^(-4);
                          shift that distribution so that 90% of the
                          values within the earthquake are positive;
                          and multiply by the distribution from part a
    :sdrop    :  stress drop, in Pa
    :shmod    :  shear modulus, in Pa
    :dsm      : distance over which to smooth linearly in Boatwright models 

    OUTPUT SLIP DISTRIBUTION
    :slip     :  the slip distribution
    :x        :  x-values along slip distribution
    :y        :  y-values along slip distribution
    :xx       :  grid of x-values along slip distribution
    :yy       :  grid of y-values along slip distribution

    PROPERTIES OF RUPTURE
    :rho      :  overshoot parameter for Boatwright models
    :rtime    :  rise time in seconds
    :ruptev   :  the specified rupture evolution
        boatwright-m:  follows the Boatwright-Madariaga model from 
                        Boatwright, BSSA, 1980
        boatwright-d:  follows the decelerating Boatwright model from 
                        Boatwright, BSSA, 1980
        circrise:      starts at a specified point and ruptures out 
                        at a constant speed from there
                        the start point need not be the center
        haskell:       not actually implemented---just returns rectangular 
                        slip distribution
    :slipfun  :  the slip function at each point
        yoffe       :  a Yoffe function
        yoffereg    :  a regularized Yoffe function
    :vrupt    :  rupture velocity, in m/s
    :xyst     :  location of starting point [x,y] in meters
    :u        :  deceleration parameter, for Boatwright models

    OUTPUT PROPERTIES OF RUPTURE
    :sliprise :  slip as a function of time for an individual point
    :tslip    :  timing of slip for an individual point

    LOCATION AND FAULT ORIENTATION
    :depth    :  depth in m
    :dip      :  dip in degrees down from horizontal, to right of strike
    :strike   :  fault strike in degrees clockwise from N
    :xyloc    :  location of the fault in [E,N] in 
    :llloc    :  location of the fault center in [lon,lat]

    PROPERTIES OF OBSERVATIONS
    :vprop    :  assumed wave propagation velocity, in m/s
    :stxyloc  :  [E,N] locations of the stations
    :stllloc  :  [lon,lat] locations of stations
    :ststrike :  station azimuths in degrees
    :sttakeang:  takeoff angles to stations, in degrees from down
    :stdist   :  distances to stations
    :stvprop  :  set of wave propagation velocities per station, in m/s

    OUTPUT PROPERTIES OF OBSERVATIONS
    :ttrav    :  set of wave travel times from each point to stations, in s
                   relative to the center of the fault
    :tmom     :  times of the source time functions
    :cmom     :  the actual source time function (moment rate)
    :amom     :  the set of apparent source time functions

    FUNCTIONS FOR MODIFIYING THE SLIP DISTRIBUTION
    makeslipdist: to create the slip distribution
    mincircle: find the circle containing most of the slip

    FUNCTIONS FOR SETTING UP THE RUPTURE
    randstart: to randomize starting location

    FUNCTIONS FOR GENERATING THE APPARENT SOURCE TIME FUNCTIONS
    calcslipfun: calculates the rise time function
    calcappstf: calls one of
    -calcappstfconv (for rise time)
    -calcappstfsnap (for Boatwright)
    cursnapshot: snapshot of slip at a given time
    istf: astf for one point

    FUNCTIONS FOR GENERATING SEISMOGRAMS
    calcttrav: source-station travel times
    rupture_start: calculate when the rupture should reach each point
    initfakegf: make up some Green's functions
    calcfakeobs: convolve to create Green's functions

    FUNCTIONS FOR SETTING UP OBSERVATIONS
    initobs: initialize the station locations

    FUNCTIONS FOR RETURNING THE STFS, GFS, AND OBSERVATIONS
    astf: return the astfs
    astfwaveforms: return the astfs as obspy waveforms
    cstf: return the source time function
    tstf: return the timing of the stfs
    calcsdur: find the duration containing 90% of the moment

    gfwaveforms: return the Green's functions as waveforms
    
    obswaveforms: return the observations as waveforms
    """
    
    # set up with defaults
    def __init__(self,ruptev='boatwright-m',a=200.,b=None,
                 vrupt=None,sdrop=3.e6,
                 shmod=3.e10,xyst=None,u=2.,rho=1.5,
                 dsm=None,slipdist=None,
                 vprop=6000.,rndedg=[0.,0.]):
        """
        :param    ruptev: the type of rupture
        :param         a: the half-length along the x-axis, in meters
        :param         b: the half-length along the y-axis, in meters
        :param     vrupt: the rupture velocity, in m/s
        :param     sdrop: stress drop, in Pa
        :param     shmod: shear modulus, in Pa
        :param      xyst: location of starting point [x,y] in meters
        :param         u: deceleration parameter, for Boatwright models
        :param       rho: overshoot parameter for Boatwright models
        :param       dsm: distance over which to smooth linearly in 
                              Boatwright models 
        :param  slipdist: the type of slip distribution,
        :param     vprop: assumed wave propagation velocity, in m/s
        :param    rndedg: the [amplitude,# of nodes] of a perturbation of
                              a circular rupture edge
        """

        # type of slip model
        self.ruptev = ruptev

        #-------INITIALIZE OBSERVATION PROPERTIES-----------------
        self.vprop = vprop
        self.stxyloc = None
        self.stllloc = None
        self.ststrike = None
        self.sttakeang = None
        self.stdist = None
        self.stvprop = None
        self.ttrav = None

        #----------GEOMETRY PROPERTIES--------------------
        # along-strike half-length
        self.a = a
        # half-length perpendicular to strike
        if b is None:
            b = a
        self.b = b

        # spacing for slip distribution
        self.dx = min(self.a,self.b)/500.

        # variations in the radius (amplitude, value)
        self.rndedg=[.5,10]
        self.rndedg=rndedg

        # the type of slip distribution
        if slipdist is None:
            if self.ruptev in ['boatwright-m','boatwright-d','elliptical','circrise']:
                slipdist = 'elliptical-unsmoothed'
            elif self.ruptev in ['haskell']:
                slipdist = 'rectangular'
        self.slipdist=slipdist

        #---------STRESS DROP AND AMOUNT OF SLIP----------
        # stress drop
        self.sdrop = sdrop
        # shear modulus
        self.shmod = shmod
        # overshoot parameter
        self.rho = rho
        # maximum slip
        self.dmax = self.a * self.sdrop / self.shmod * self.rho

        #----------RUPTURE TIMING AND STARTING LOCATION----
        # rupture velocity
        if vrupt is None:
            self.vrupt = self.vprop*0.8/3**0.5
        else:
            self.vrupt = vrupt
        # starting location
        if xyst is None:
            xyst = np.array([0.,0.])
        self.xyst = xyst

        #----------SLIP EVOLUTION-----------------------------------
        # rise time
        self.rtime = self.a/self.vrupt/5.
        # type of slip function
        self.slipfun = 'yoffereg'
        # deceleration parameter
        self.u = u
        # spatial smoothing
        if dsm is None:
            dsm = min(self.a,self.b)/100.
        self.dsm = dsm

        #-----keep track of slip distribution parameters-----------
        self.mincirclefrc = None
        self.mincirclerad = None
        self.mincirclexy = None

        #-------FAULT PROPERTIES-----------------------------------
        self.strike = 0.
        self.dip = 90.
        self.depth = 5000.
        self.xyloc = np.array([0.,0.])
        self.llloc = np.array([0.,0.])


        # delete unrelated parameters
        if self.ruptev == 'boatwright-d':
            self.xyst = np.array([0,0])
            self.b=self.a
            self.rho=float('nan')
        if self.ruptev == 'boatwright-m':
            self.xyst = np.array([0,0])
            self.b=self.a
            self.u=float('nan')
        elif self.ruptev == 'haskell':
            self.u=float('nan')
            self.rho=float('nan')

        #----------initialize the slip distribution-----------
        self.makeslipdist()

        #----------initialize the observation locations-------
        self.initobs(justloc=True)


    #----------------------------------------------------------------------
    #----------FUNCTIONS FOR MODIFYING THE SLIP DISTRIBUTION---------------

    def makeslipdist(self,dx=None):
        """
        initialize the slip distribution
        :param   dx:  grid spacing
        """
        if dx is None:
            dx = self.dx

        self.slip,self.x,self.y,self.xx,self.yy = \
            makeslipdist(a=self.a,b=self.b,dx=dx,slipdist=self.slipdist,
                         dmax=self.dmax,rndedg=self.rndedg)

    def calccentroid(self):
        """
        compute the centroid
        """
        
        x=np.dot(self.slip.flatten(),self.xx.flatten())/ \
           np.sum(self.slip.flatten())
        y=np.dot(self.slip.flatten(),self.yy.flatten())/ \
           np.sum(self.slip.flatten())

        self.xycent=np.array([x,y])

    def addpatch(self,loc=None,rd=None,amp=None):
        """
        :param     loc: location (default: center)
        :param      rd: patch radius (default: 0.05*a)
        :param     amp: amplitude of peak (default: max of slip)
        """

        if amp is None:
            amp = np.max(self.slipdist)
        if loc is None:
            loc = np.array([0.,0])
        if rd is None:
            rd = self.a*0.05

        slip=np.zeros(self.slip.shape,dtype=float)
        dst=np.power(self.xx-loc[0],2)+np.power(self.yy-loc[1],2)
        ii=dst<rd**2
        slip[ii]=np.power(1-dst[ii]/rd**2,0.5)*amp

        self.slip=self.slip+slip

    def mincircle(self,frc=0.8):
        """
        smallest circle with fraction frc of slip
        :param     frc:  fraction of moment
        :return    rad:  circle radius
        :return     xy:  center of circles
        """
        
        if self.mincirclefrc!=frc:
            # calculate if not done before
            self.mincirclerad,self.mincirclexy = \
                mincircle(self.xx,self.yy,self.slip,frc)
            self.mincirclefrc=0.8

        # return
        rad=self.mincirclerad
        xy=self.mincirclexy

        return rad,xy

    #----------------------------------------------------------------------
    #----------FUNCTIONS FOR SETTING UP THE RUPTURE------------------------

    def randstart(self,frc=0.2):
        """
        randomize the starting location
        :param      frc:  allowable distances from the origin [0,0], as a
                           fraction of the minimum rupture half-length
        sets self.xyst
        """

        # a normalized location
        xyst = randstart(frc=frc)

        # radius to scale
        rd = np.minimum(self.a,self.b)
        self.xyst = xyst*rd

    #----------------------------------------------------------------------
    #----------FUNCTIONS FOR GENERATING THE ASTFS--------------------------

    def calcslipfun(self,tm=None,dtim=None):
        """
        calculate the slip at a point through time, 
        normalized to the total slip at this point
        :param   tm:   times, if there are specified points to 
                             to calculate for (default: -1 - 8 times rtime
        :param dtim:   time spacing (default: 0.005*rtime)
        sets sliprise and tslip
        """

        # timing
        if tm is None:
            if dtim is None:
                dtim = self.rtime *0.005

            # specified start and end times
            if self.slipfun=='yoffe':
                tmin = -0.2*self.rtime
                tmax = 1.2*self.rtime
            if self.slipfun=='yoffereg':
                tmin = -0.2*self.rtime
                tmax = 1.5*self.rtime
            else:
                tmin = -0.5*self.rtime
                tmax = 5.*self.rtime

            self.tslip = np.arange(tmin,tmax,dtim)

        # initialize
        self.sliprise = np.zeros(self.tslip.shape,dtype=float)

        if self.slipfun=='yoffe':
            # yoffe function
            ii = np.logical_and(self.tslip>0.,self.tslip<self.rtime)
            self.sliprise[ii] = \
                np.power(np.divide(self.rtime-self.tslip[ii],
                                   self.tslip[ii]),0.5)
        elif self.slipfun=='yoffereg':
            # modified yoffe function
            # smoothing time
            ts = self.rtime / 8.
            tr = self.rtime

            C1 = np.zeros(self.tslip.shape,dtype=float)
            C2 = np.zeros(self.tslip.shape,dtype=float)
            C3 = np.zeros(self.tslip.shape,dtype=float)
            C4 = np.zeros(self.tslip.shape,dtype=float)
            C5 = np.zeros(self.tslip.shape,dtype=float)
            C6 = np.zeros(self.tslip.shape,dtype=float)

            # compute C1 in the relevant interval
            ii = np.logical_and(self.tslip>=0,self.tslip<tr)
            t = self.tslip[ii]
            C1[ii]=np.multiply(0.5*t+0.25*tr,
                               np.power(np.multiply(t,tr-t),0.5)) + \
                   np.multiply(t*tr-tr**2,
                               np.arcsin(np.power(t/tr,0.5))) - \
                   0.75*tr**2*np.arctan(np.power(np.divide(tr-t,t),0.5))
                             

            # compute C2 in the relevant interval
            ii = np.logical_and(self.tslip>=0,self.tslip<2.*ts)
            t = self.tslip[ii]
            C2[ii] = 3./8.*math.pi*tr**2

            # compute C3 in the relevant interval
            ii = np.logical_and(self.tslip>=ts,self.tslip<tr+ts)
            t = self.tslip[ii]
            C3[ii] = np.multiply(ts-t-0.5*tr,np.power(np.multiply(t-ts,tr-t+ts),0.5)) \
                     +2.*tr*np.multiply(tr-t+ts,np.arcsin(np.power((t-ts)/tr,0.5))) \
                     +1.5*tr**2*np.arctan(np.power(np.divide(tr-t+ts,t-ts),0.5))

            # compute C4 in the relevant interval
            # NOTE THAT THIS CHANGES EQUATION A18 OF TINTI ET AL, 2005
            # IT REPLACES tr(tr+t-2ts) with tr(-tr+t-2ts)
            ii = np.logical_and(self.tslip>=2*ts,self.tslip<tr+2*ts)
            t = self.tslip[ii]
            C4[ii] = np.multiply(-ts+0.5*t+0.25*tr,
                                  np.power(np.multiply(t-2*ts,tr-t+2*ts),0.5)) \
                     +tr*np.multiply(-tr+t-2*ts,
                                     np.arcsin(np.power((t-2*ts)/tr,0.5))) \
                     -0.75*tr**2*np.arctan(np.power(np.divide(tr-t+2*ts,t-2*ts),0.5))

            # compute C5 in the relevant interval
            ii = np.logical_and(self.tslip>=tr,self.tslip<tr+ts)
            t = self.tslip[ii]
            C5[ii] = math.pi/2.*tr*(t-tr)

            # compute C6 in the relevant interval
            ii = np.logical_and(self.tslip>=tr+ts,self.tslip<tr+2*ts)
            t = self.tslip[ii]
            C6[ii] = math.pi/2.*tr*(2*ts-t+tr)


            t = self.tslip

            # and go through the relevant intervals
            ii = np.logical_and(t>=0,t<ts)
            self.sliprise[ii]=C1[ii]+C2[ii]

            ii = np.logical_and(t>=ts,t<2*ts)
            self.sliprise[ii]=C1[ii]-C2[ii]+C3[ii]

            ii = np.logical_and(t>=2*ts,t<tr)
            self.sliprise[ii]=C1[ii]+C3[ii]+C4[ii]

            ii = np.logical_and(t>=tr,t<tr+ts)
            self.sliprise[ii]=C5[ii]+C3[ii]+C4[ii]

            ii = np.logical_and(t>=tr+ts,t<tr+2*ts)
            self.sliprise[ii]=C4[ii]+C6[ii]

            # absolute scaling
            K = 2./math.pi/tr/ts**2
            self.sliprise = K*self.sliprise


    def calcappstf(self,dtim=None,dcentroid=False):
        """
        calculate apparent source time function 
        method depends on type of rise time
        :param  dtim: time sampling
        """

        # reference location
        if dcentroid:
            self.calccentroid()
            xyref=self.xycent
        else:
            xyref=np.array([0.,0.])

        # recalculate travel times
        self.calcttrav(xyref=xyref)

        if 'boatwright' in self.ruptev:
            # boatwright slip function requires calculation at each time step
            self.calcappstfsnap(dtim=dtim)
        else:
            # rise time model can be computed through convolution
            self.calcappstfconv(dtim=dtim)

    def calcappstfsnap(self,dtim=None):
        """
        calculate apparent source time functions by summing over snapshots
        :param  dtim: time sampling
        """

        # initialize timing 
        tm = 2. * max(self.a,self.b) / self.vrupt
        tm = np.array([-.1*tm,1.1*tm])
        
        # preferred timing
        if dtim is None:
            dtim = (tm[1]-tm[0])/2000.
            dtim2 = (self.dx/self.vprop)/5.
            dtim = np.minimum(dtim,dtim2)

        # number of resampling values
        nsamp = 3.
        dtim = dtim/nsamp
        
        # and for travel times
        tt = np.array([np.min(self.ttrav),np.max(self.ttrav)])
        tt[0]=min(tt[0],0.)
        tt[1]=max(tt[1],0.)
        tt = tt+np.array([-1.,1])*3.*dtim
        
        # want records at these times---record them
        tmi = tm + tt
        tmikeep = tmi.copy()

        # if the final observations go from tmi[0] to tmi[1]
        # they could come from times in tm between
        tm=np.array([tmi[0]-tt[1],tmi[1]-tt[0]])

        # and buffer the output for calculation
        tmi=np.arange(tm[0],tm[1],dtim)
        tm=np.arange(tm[0]+tt[0],tm[1]+tt[1],dtim)

        # initialize source time function
        astf = np.zeros([len(tm),self.Nstat])
        cstf = np.zeros([len(tm)])

        # identify locations with nonzero slip
        ii,=np.where(self.slip.flatten())

        # time differences and relative differences
        Nt = self.ttrav.shape[0]*self.ttrav.shape[1]
        tdf = self.ttrav.reshape([Nt,self.Nstat])
        tdf = tdf[ii,:]

        # find the corresponding times after
        iaf=np.searchsorted(tm,tmi[0]+tdf.flatten())
        iaf=iaf.reshape([len(ii),self.Nstat])
        # and before
        ibf=iaf-1

        # weighting for each of these
        waf=tm[iaf]-tdf-tmi[0]
        wbf=tdf+tmi[0]-tm[ibf]
        wtot = waf + wbf
        waf = np.divide(wtot-waf,wtot)
        wbf = np.divide(wtot-wbf,wtot)

        # and for the unshifted approach
        jaf=np.searchsorted(tm,tmi[0])
        jbf=jaf-1
        wjaf=1.-(tm[jaf]-tmi[0])/dtim
        wjbf=1.-wjaf

        # go through the times and get snapshots of slip
        for k in range(0,len(tmi)):
            print(k)
            tim = tmi[k]

            # a snapshot of slip 
            cursnap = self.cursnapshot(tmi[k])

            # just the values of interest
            cursnap = cursnap.flatten()[ii]

            for m in range(0,self.Nstat):
                astf[:,m]=astf[:,m]+\
                    np.bincount(ibf[:,m],np.multiply(wbf[:,m],cursnap),
                                len(tm))
                astf[:,m]=astf[:,m]+\
                    np.bincount(iaf[:,m],np.multiply(waf[:,m],cursnap),
                                len(tm))

            # and add the average
            cstf[jbf]=cstf[jbf]+np.sum(cursnap)*wjbf
            cstf[jaf]=cstf[jaf]+np.sum(cursnap)*wjaf

            # go to next value
            ibf,iaf=ibf+1,iaf+1
            jbf,jaf=jbf+1,jaf+1

        # only keep some results
        iok=np.logical_and(tm>=tmikeep[0],tm<=tmikeep[1])
        tm = tm[iok]
        astf = astf[iok,:]
        cstf = cstf[iok]

        # smooth and resample
        for k in range(0,astf.shape[1]):
            astf[:,k]=scipy.ndimage.filters.gaussian_filter1d(astf[:,k],
                                                              nsamp)
        cstf = scipy.ndimage.filters.gaussian_filter1d(cstf,nsamp)
        iok = np.arange(0,len(tm),int(nsamp))
        tm = tm[iok]
        astf = astf[iok,:]
        cstf = cstf[iok]

        # assign here
        self.tmom = tm
        self.amom = astf
        self.cmom = cstf

        # and go ahead and calculate some fake observations
        self.calcfakeobs()

    def calcappstfconv(self,dtim=None):
        """
        calculate apparent source time functions by convolution
        :param  dtim: time sampling
        """

        # initialize timing 
        tm = 2. * max(self.a,self.b) / self.vrupt
        tm = np.array([-.1*tm,1.1*tm])
        
        # preferred timing
        if dtim is None:
            dtim = (tm[1]-tm[0])/2000.
            dtim2 = (self.dx/self.vprop)/5.
            dtim2 = np.minimum(self.a,self.b)/self.vprop/200.
            dtim = np.minimum(dtim,dtim2)

        # total shift
        self.rupture_start()
        rshp = np.append(self.ruptst.shape,1)
        totshf = self.ttrav + \
            self.ruptst.reshape(rshp)
        
        # add one with zero time shift for the average
        totshf = np.append(totshf,self.ruptst.reshape(rshp),axis=2)

        # and for travel times
        tt = np.array([np.min(totshf),np.max(totshf)])
        tt[0]=min(tt[0],0.)
        tt[1]=max(tt[1],0.)
        tt = tt+np.array([-1.,1])*3.*dtim
        
        # want records at these times---record them
        tmi = tm + tt
        tmikeep = tmi.copy()

        # if the final observations go from tmi[0] to tmi[1]
        # they could come from times in tm between
        tm=np.array([tmi[0]-tt[1],tmi[1]-tt[0]])

        # and buffer the output for calculation
        tm=np.arange(tm[0]+tt[0],tm[1]+tt[1],dtim)
        # only consider one initial time
        tmi=np.atleast_1d([0.])

        # initialize source time function
        astf = np.zeros([len(tm),self.Nstat+1])
        cstf = np.zeros([len(tm)])

        # identify locations with nonzero slip
        ii,=np.where(self.slip.flatten())

        # time differences and relative differences
        Nt = totshf.shape[0]*totshf.shape[1]

        # time spacing
        tsp = np.diff(tm)

        # just use the one snapshof of slip
        cursnap = self.slip

        # how much to downsample
        # tdfx = np.median(np.median(np.median(np.abs(np.diff(totshf,
        #                                              axis=0)),axis=0),axis=0))
        # tdfy = np.median(np.median(np.median(np.abs(np.diff(totshf,
        #                                              axis=1)),axis=1),axis=0))
        # tdfd = np.minimum(self.a,self.b)/self.vprop/150
        # Nup = np.mean(np.array([tdfx,tdfy])/tdfd)
        Nup = 1.
        if Nup >1.5:
            print('May be under-resolved: '+str(Nup))
            # new values
            Nx,Ny=int(self.x.size*Nup),int(self.y.size*Nup)
            x=np.linspace(self.x[0],self.x[-1],Nx)
            y=np.linspace(self.y[0],self.y[-1],Ny)
            
            # downsample slip
            f=scipy.interpolate.RectBivariateSpline(self.x,self.y,cursnap)
            cursnap=f(x,y)

            # downsample timing
            totshf2=np.ndarray([Nx,Ny,totshf.shape[2]])
            for k in range(0,totshf.shape[2]):
                f=scipy.interpolate.RectBivariateSpline(self.x,self.y,totshf[:,:,k])
                totshf2[:,:,k]=f(x,y)
            totshf=totshf2
            print('Resampled')
        elif Nup <0.25:
            print('May be over-resolved: '+str(Nup))
            nsm=int(np.floor(0.7/Nup))
            
            # new values
            ix=np.arange(0,self.x.size,nsm)
            iy=np.arange(0,self.y.size,nsm)
            x,y=self.x[ix],self.y[ix]
            
            # downsample slip
            cursnap=cursnap[ix,:]
            cursnap=cursnap[:,iy]

            # downsample timing
            totshf=totshf[ix,:,:]
            totshf=totshf[:,iy,:]
            print('Resampled')

        # to upsample
        Ng = int(np.ceil(cursnap.size*2/3.e6))
        igrp = np.linspace(cursnap.shape[0],Ng+1).astype(int)
        igrp = np.unique(igrp)
        Ng = len(igrp)-1
        ara = self.dx**2 / (self.xx.shape[0]/cursnap.shape[0]) / \
              (self.xx.shape[1]/cursnap.shape[1])

        for kg in range(0,Ng):
            # portion of the slip distribution
            svl = cursnap[igrp[kg]:igrp[kg+1],:]
            
            # only interested in nonzero values
            ii = np.abs(svl)>np.max(cursnap)*1.e-8
            svl = svl[ii]

            # times in this range
            for m  in range(0,self.Nstat+1):
                # zoom in to the slip times
                tshf = totshf[igrp[kg]:igrp[kg+1],:,m]
                
                # and just the nonzero values
                tshf = tshf[ii]

                # find the intervals and the amplitudes for adding
                iaf = np.searchsorted(tm,tmi[0]+tshf)
                ibf = iaf-1
                waf = np.divide(tshf+tmi[0]-tm[ibf],tsp[ibf])

                # add to source time functions
                astf[:,m]=astf[:,m]+\
                           np.bincount(ibf,np.multiply(1-waf,svl),
                                       len(tm))
                astf[:,m]=astf[:,m]+\
                           np.bincount(iaf,np.multiply(waf,svl),
                                       len(tm))

        # calculate the slip rate function at each point
        self.calcslipfun(dtim=dtim)

        #-------REPLACE LATER--------------
        # self.sliprise=np.zeros(self.sliprise.shape,dtype=float)
        # iz=np.argmin(np.abs(self.tslip))
        # self.sliprise[iz]=1.
        
        # convolve
        Nmax = self.sliprise.size + astf.shape[0]
        Nf = int(2**np.ceil(np.log2(Nmax*2)))

        v1 = np.fft.rfft(astf,n=Nf,axis=0)
        v2 = np.fft.rfft(self.sliprise,n=Nf)
        v1 = np.multiply(v1,v2.reshape([v2.size,1]))
        v1 = np.fft.irfft(v1,axis=0)

        # extract the time with data and scale to total potency
        astf = v1[0:Nmax,0:-1]*ara

        # and for the average moment rate
        cstf = v1[0:Nmax,-1]*ara
        
        # timing
        tstf = (np.arange(0.,Nmax+1)-0.5)*dtim+tm[0]+self.tslip[0]

        # assign here
        self.tmom = tstf
        self.amom = np.append(np.zeros([1,astf.shape[1]],dtype=float),
                              np.cumsum(astf*dtim,axis=0),axis=0)
        self.cmom = np.append(np.array([0.]),np.cumsum(cstf*dtim))

        # and go ahead and calculate some fake observations
        self.calcfakeobs()


    def cursnapshot(self,tm=None):
        """
        grab the  current spatial distribution fo slip
        :param       tm:   time of interest
        :return cursnap:   current snapshot
        """

        if tm is None:
            tm = self.a/self.vrupt/4.
    
        if self.ruptev == 'boatwright-m':
            cursnap=snap_bwm(self.xx,self.yy,tm=tm,a=self.a,vrupt=self.vrupt,
                             sdrop=self.sdrop/self.shmod,ro=self.rho,
                             vprop=self.vprop,slip=self.slip)
        elif self.ruptev == 'circrise':
            cursnap=snap_circrise(self.xx,self.yy,tm=tm,slip=self.slip,
                                  vrupt=self.vrupt,rtime=self.rtime,
                                  xyst=self.xyst)
                                  

        return cursnap

    def istf(self,x=None,y=0.,tm=None,rvel=False):
        """
        generate the slip rate function at a specific point,
        relative to rupture start time
        :param    x:  x-location---along strike (default: a/2)
        :param    y:  y-location---along dip (default: 0)
        :param   tm:  times to calculate (default: 2000 spaced points)
        :param rvel:  return slip rate instead of slip (default: False)
        """
        # default location
        if x is None:
            x = self.a/2.

        if self.ruptev == 'boatwright-m':
            if tm is None:
                tm = 2. * self.a / self.vrupt
                tm = np.linspace(-.1*tm,1.1*tm,2000)
            stf,tim = istf_bwm(x=x,y=y,tm=tm,a=self.a,vrupt=self.vrupt,
                               ro=self.rho,sdrop=self.sdrop/self.shmod,
                               rvel=rvel,dsm=self.dsm)
        elif self.ruptev == 'boatwright-d':
            if tm is None:
                tm = 2. * self.a / self.vrupt
                tm = np.linspace(-.1*tm,1.1*tm,2000)
            stf,tim = istf_bwd(x=x,y=y,tm=tm,a=self.a,vrupt=self.vrupt,
                               u=self.u,sdrop=self.sdrop/self.shmod,rvel=rvel,
                               dsm=self.dsm)
        elif self.ruptev == 'circrise':
            # find the rupture start time at this point
            try:
                f = scipy.interpolate.interp2d(self.y,self.x,self.ruptst)
            except:
                self.rupture_start()
                f = scipy.interpolate.interp2d(self.y,self.x,self.ruptst)
            tdf = f(y,x)[0]

            # find the total slip at this point
            try:
                f = scipy.interpolate.interp2d(self.y,self.x,self.slip)
            except:
                self.makeslipdist()
                f = scipy.interpolate.interp2d(self.y,self.x,self.slip)
            stot = f(y,x)[0]

            # the slip rate function
            try:
                stf,tim=self.sliprise,self.tslip
            except:
                self.calcslipfun()
                stf,tim=self.sliprise,self.tslip
            tim = tim + tdf
            stf = stf * stot

        return stf,tim

    #------------------------------------------------------------------
    #---------FUNCTIONS FOR GENERATING SEISMOGRAMS---------------------

    def calcttrav(self,xyref=[0.,0.]):
        """
        compute and save travel times for each location and station
        """

        # initialize on regular and shifted grid
        self.ttrav = np.ndarray([len(self.x),len(self.y),self.Nstat])
        self.ttravs = np.ndarray([len(self.x),len(self.y),self.Nstat])

        # recompute the grid
        self.xxs = self.xx.copy()
        self.yys = self.yy.copy()
        for k in np.arange(0,self.yys.shape[0],2):
            self.yys[k,:]=self.yys[k,:]+self.dx/2.
            
        # compute travel times
        for k in range(0,self.Nstat):
            self.ttrav[:,:,k]=ttrav(self.xx,self.yy,
                                    takang=self.sttakeang[k],
                                    stkost=self.ststrike[k]-self.strike,
                                    dip=self.dip,vprop=self.stvprop[k],
                                    xyref=xyref)
            self.ttravs[:,:,k]=ttrav(self.xxs,self.yys,
                                     takang=self.sttakeang[k],
                                     stkost=self.ststrike[k]-self.strike,
                                     dip=self.dip,vprop=self.stvprop[k],
                                     xyref=xyref)

    def rupture_start(self):
        """
        calculate time at which each point starts slipping
        sets ruptst
        """
        
        self.ruptst = rupture_start(xx=self.xx,yy=self.yy,
                                    xyst=self.xyst,vrupt=self.vrupt)


    def initfakegf(self,dtim=None,tdec=3.,tlen=20.):
        """
        compute some fake Green's functions
        :param   dtim: time spacing (default:fraction of event)
        :param   tdec: decay timescale in s (default: 3)
        :param   tlen: length of signal in s (default: 20)
        """

        if dtim is None:
            dtim = min(self.a,self.b)/self.vrupt/500.
        tlen=float(tlen)
        tdec=float(tdec)
            
        # initialize timing and values
        self.gftlen = tlen
        self.gftdec = tdec
        self.gftim = np.arange(-tlen/10,tlen*.9,dtim)
        self.gf = np.ndarray([len(self.gftim),self.Nstat]);

        for k in range(0,self.Nstat):
            self.gf[:,k],self.tgf=fakegf(tdec,tlen,self.gftim,
                                         stout=False)

    def calcfakeobs(self):
        """
        create some fake observations by convolving apparent stfs
        with Green's functions
        sets obs,obstim
        """

        # check that there's data
        try:
            Nok = self.gf.size
        except:
            self.initfakegf()

        # first observation
        obsi,self.obstim = \
            conv(self.astf()[:,0],self.gf[:,0],
                 self.tstf(),self.gftim)
        self.obs = np.ndarray([obsi.size,self.Nstat])
        self.obs[:,0]=obsi
        
        for k in range(0,self.Nstat):
            self.obs[:,k],self.obstim = \
                conv(self.astf()[:,k],self.gf[:,k],
                     self.tstf(),self.gftim)

    #---------------------------------------------------------------
    #-------FUNCTIONS FOR SETTING UP THE OBSERVATIONS---------------
        
    def initobs(self,strike=None,takeang=None,xyloc=None,llloc=None,
                dist=None,vprop=None,dtim=None,justloc=False):
        """
        initialize a set of observation points
        parameters listed first take priority over those listed later
        :param  strike: strike in degrees clockwise from north
        :param takeang: take-off angle in degrees (down=0)
        :param   xyloc: x,y locations
        :param   llloc: lon-lat locations
        :param    dist: distance from earthquake in km
        :param   vprop: propagation velocity of seismic wave  
                           (default: self.vprop.)
        :param    dtim: time spacing for the fake GF 
                           (default: fraction of rise time)
        :param justloc: just the locations, not fake GF (default: False)
        """
        
        if vprop is None:
            vprop = self.vprop

        # decide how to set parameters
        toset = None
        if (strike is not None) and (takeang is not None):
            toset = 'strike_takeang'
        elif xyloc is not None:
            toset = 'xyloc'
        elif llloc is not None:
            toset = 'llloc'
        elif (strike is not None) and (dist is not None):
            toset = 'strike_dist'
        elif strike is not None:
            takeang = self.sttakeang
            dist = self.stdist
            if takeang is not None:
                toset = 'strike_takeang'
            elif dist is not None:
                toset = 'strike_dist'
        elif takeang is not None:
            strike = self.ststrike
            if strike is not None:
                toset = 'strike_takeang'
        elif dist is not None:
            strike = self.ststrike
            if strike is not None:
                toset = 'strike_dist'
        else:
            strike = np.array([0,20,40,90])
            takeang = 90.
            toset = 'strike_takeang'
            
        # set parameters
        if toset is 'strike_takeang':
            # initialize
            if not isinstance(strike,np.ndarray):
                strike = np.array(strike)
            if not isinstance(takeang,np.ndarray):
                takeang = np.array(takeang)
            Nstat = max(strike.size,takeang.size)

            # repeat as necessary
            takeang = takeang.repeat(Nstat/takeang.size)
            strike = strike.repeat(Nstat/strike.size)
            
        # set any unset parameters to zero
        if dist is None:
            dist = np.ones(Nstat)
            xyloc = np.ones([3,Nstat])
            llloc = np.ones([3,Nstat])
            
        if not isinstance(vprop,np.ndarray):
            vprop = np.array(vprop)
        vprop = vprop.repeat(Nstat/vprop.size)

        # set the relevant values
        self.ststrike = strike
        self.sttakeang = takeang
        self.stxyloc = xyloc
        self.stllloc = llloc
        self.stdist = dist
        self.stvprop = vprop
        self.Nstat = Nstat
            
        # go ahead and calculate the travel times
        self.calcttrav()

        # initialize some Green's functions
        if not justloc:
            self.initfakegf(dtim=dtim)

    #---------------------------------------------------------------
    #-------FUNCTIONS FOR RETURNING THE STFS, GFS, AND OBS-------------

    def tstf(self):
        """
        :return tstf:  times for apparent source time function in moment rate
        """

        tstf = (self.tmom[0:-1]+self.tmom[1:])/2.
        return tstf

    def astf(self,istat=None):
        """
        :return astf:  moment rate apparent source time functions
        """

        if istat is None:
            istat = np.arange(0,self.Nstat,1)
            
        # time for apparent source time function
        astf = np.diff(self.amom[:,istat],n=1,axis=0)

        # divide for timing
        dtim = np.diff(self.tmom,n=1,axis=0)
        dtim = np.ndarray([len(dtim),1],buffer=dtim)
        astf = np.divide(astf,np.tile(dtim,[1,len(istat)]))

        return astf

    def cstf(self):
        """
        :return cstf:  moment rate source time functions
        """

        # time for apparent source time function
        cstf = np.diff(self.cmom,n=1,axis=0)

        # divide for timing
        dtim = np.diff(self.tmom,n=1,axis=0)
        cstf = np.divide(cstf,dtim)

        return cstf


    def calcsdur(self,prc=0.95):
        """
        :return   dur: duration that contains central 95% of the moment
        """
        
        # cumulative source time function
        cummom = self.cmom
        cummom = cummom/cummom[-1]

        # find percentiles
        prc = np.array([-1.,1.])*0.5*prc+0.5
        vl = np.interp(prc,cummom,self.tmom)

        # and difference
        dur = vl[1]-vl[0]

        return dur

    #------------FOR OUTPUT AS OBSPY WAVEFORMS------------------------------

    def astfwaveforms(self,marker='t0'):
        """
        return the apparent source time functions, 
        but as obspy stream functions
        :param marker: marker to use to mark time zero (default: 't0')
        :return    st: the relevant traces
        """

        # apparent source time functions
        tstf = self.tstf()
        astf = self.astf()
        
        st = obspy.Stream()
        dtim=np.median(np.diff(tstf))

        for k in range(0,self.Nstat):
            # get central time of the apparent source time functions
            tcent = np.dot(astf[:,k],tstf)/np.sum(astf[:,k])

            tr = obspy.Trace()
            tr.data = astf[:,k]
            tr.stats.delta = dtim
            tr.stats[marker] = tcent-tstf[0]
            tr.stats.station  = 'S'+str(k+1)
            st.append(tr)

        # also the average
        tr = obspy.Trace()
        cstf = self.cstf()
        tcent = np.dot(cstf,tstf)/np.sum(cstf)
        tr.data = cstf 
        tr.stats.delta = dtim
        tr.stats[marker] = tcent-tstf[0]
        tr.stats.station  = 'STF'
        st.append(tr)

        return st

    def obswaveforms(self,marker='t0'):
        """
        return the fake observations
        but as obspy stream functions
        :param marker: marker to use to mark time zero (default: 't0')
        :return    st: the relevant traces
        """

        # apparent source time functions
        tstf = self.tstf()
        astf = self.astf()
        
        st = obspy.Stream()
        dtim=np.median(np.diff(self.obstim))

        for k in range(0,self.Nstat):
            # get central time of the apparent source time functions
            tcent = np.dot(astf[:,k],tstf)/np.sum(astf[:,k])

            # create trace
            tr = obspy.Trace()
            tr.data = self.obs[:,k]
            tr.stats.delta = dtim
            tr.stats[marker] = tcent-self.obstim[0]
            tr.stats.station  = 'S'+str(k+1)
            st.append(tr)

        return st

    def gfwaveforms(self,marker='t0'):
        """
        return the Green's functions
        but as obspy stream functions
        :param marker: marker to use to mark time zero (default: 't0')
        :return    st: the relevant traces
        """
        
        st = obspy.Stream()
        dtim=np.median(np.diff(self.gftim))

        for k in range(0,self.Nstat):
            tr = obspy.Trace()
            tr.data = self.gf[:,k]
            tr.stats.delta = dtim
            tr.stats[marker] = -self.gftim[0]
            tr.stats.station  = 'S'+str(k+1)
            st.append(tr)

        return st


#----SLIP DISTRIBUTIONS---FINAL AND SNAPSHOTS----------------------


def makeslipdist(a=None,b=None,dx=None,slipdist='elliptical',dmax=None,
                 rndedg=[.5,10,0.95]):
    """
    :param      a:  half-distance along x-axis
    :param      b:  half-distance along y-axis
    :param     dx:  spacing
    :param    slipdist:  type of distribution
                      usually a-b, where a is
                       'elliptical','circtapered','rectangular', and
                      b is optional but could be
                       'fractal'
    :param   dmax:  maximum slip
    :param rndedg:  [amplitude to perturb edge, 
                     degree of spline to use for perturbing edge,
                     include edges up to ? %ile]
    :return  slip:  slip values
    :return     x:  x-values
    :return     y:  y-values
    """

    if a is None:
        a=300.
    if b is None:
        b=a

    rndedg=np.atleast_1d(rndedg).flatten()
    if rndedg.size==1:
        rndedg=np.append(rndedg,10)
    if rndedg.size==2:
        rndedg=np.append(rndedg,0.8)

    #--------BOUNDARIES IF NECESSARY--------------------------------

    scl = 0.1
    # define the boundaries of the slip distribution
    # if there will be a perturbation of the edges
    if rndedg[0]:

        # create spline with some number of knots
        N=int(rndedg[1])
        thet1=np.arange(0,N+1)*2.*math.pi/float(N)
        thet1=thet1-math.pi
        rmin,rmax=-float('inf'),float('inf')
        # consider only such that radius is more than 0
        np.random.seed()
        while rmin<0.2:
            r=np.random.randn(N)*rndedg[0]+1.
            r=np.append(r,r[0])
            tck = scipy.interpolate.splrep(thet1,r,per=1,k=3)

            # to check and 
            thet=np.linspace(-math.pi,math.pi,2000)
            vl = scipy.interpolate.splev(thet,tck)
            rmin,rmax = np.min(vl),np.max(vl)

        rmin=0.
        while rmin<0.2:
            rha = np.array([],dtype=float)
            while rha.size<N:
                rh=(np.random.randn(N-rha.size)*rndedg[0]/4.+1.)*(1.-scl)
                rha=np.append(rha,rh[np.logical_and(rh>0.,rh<1.)])
            rins=np.append(rha,rha[0])
            rins=np.multiply(rins,r)
            tckins = scipy.interpolate.splrep(thet1,rins,per=1,k=3)
            vl = scipy.interpolate.splev(thet,tckins)
            rmin,rmax = np.min(vl),np.max(vl)
    else:
        rmax=1.

    #---------DEFINE SPACING, MAKE X-Y GRID------------------------

    # default distribution, slip maximum, spacing
    if slipdist is None:
        slipdist = 'elliptical'
    if dmax is None:
        dmax = min(a,b)*3e6/3e10
    if dx is None:
        dx=min(a,b)/2000.

    if isinstance(dx,list):
        # if the input is already the grid
        xx = dx[0]
        yy = dx[1]
        x = xx[:,0]
        y = yy[0,:]
    else:
        # create axes
        if 'fractedge' in slipdist:
            x=int(rmax*3*a/dx+5)
            y=int(rmax*3*b/dx+5)
        else:
            x=int(1.5*rmax*a/dx+5)
            y=int(1.5*rmax*b/dx+5)

        x=dx*np.arange(-x,x).astype(float)
        y=dx*np.arange(-y,y).astype(float)

        # values
        xx,yy=np.meshgrid(x,y,indexing='ij')
        
    #---------INITIAL SETUP, WITHOUT VARIABILITY ADDED----------------

    if 'elliptical' in slipdist:
        # an elliptical slip distribution
        R=np.power(xx/a,2)+np.power(yy/b,2)

        if rndedg[0]:
            # get angle for each point and rescale radii
            thet = np.angle(xx+1j*yy)
            rmax = scipy.interpolate.splev(thet,tck)
            R = np.divide(R,rmax)

        # elliptical scaled distribution
        slip = np.zeros(xx.shape)            
        slip[R<1]=np.power(1-R[R<1],0.5)

        if not 'unsmooothed' in slipdist:
            # smooth the taper more
            scl=np.minimum(a,b)/dx*0.03
            scl=np.minimum(a,b)/dx*0.2
            slip=ndimage.gaussian_filter(slip,sigma=(scl,scl),order=0.,mode='nearest')


    if 'circtapered' in slipdist:
        # circular, tapered at the edges
        R=np.power(np.power(xx/a,2)+np.power(yy/b,2),0.5)

        if rndedg[0]:
            # get angle for each point and rescale radii
            thet = np.angle(xx+1j*yy)
            rmin = scipy.interpolate.splev(thet,tckins)
            rmax = scipy.interpolate.splev(thet,tck)
            rmax = np.maximum(rmin,rmax)
            R2 = np.divide(R,rmin)*(1-scl)
            ibig = R2>(1-scl)
            R2[ibig]=np.divide(R[ibig]-rmin[ibig],rmax[ibig]-rmin[ibig])*scl+(1-scl)
            R = R2

        # one inside
        slip = np.zeros(xx.shape)
        slip[R<1]=1.

        # taper over some range
        ii = np.logical_and(R>(1.-scl),R<1.)
        R = R[ii]
        R = (R-(1.-scl))/scl
        slip[ii] = np.cos(R*math.pi)/2.+0.5

        # smooth the taper more
        scl=np.minimum(a,b)/dx*0.05
        slip=ndimage.gaussian_filter(slip,sigma=(scl,scl),order=0.,mode='nearest')


    elif 'rectangular' in slipdist:
        # rectangular
        slip=np.abs(xx)<a
        slip=np.logical_and(slip,np.abs(yy)<b)
        slip=slip.astype(float)

    #----------ADD VARIATIONS-------------------------------------

    if 'fractal' in slipdist:

        # dimension sizes
        N1,N2=xx.shape[0],xx.shape[1]

        # create a dummy distribution to get a random phase
        np.random.seed()
        slipf = np.random.rand(N1*N2).reshape([N1,N2])
        slipf = np.fft.fft2(slipf)

        # frequencies
        freqx = np.fft.fftfreq(N1,dx).reshape([N1,1])
        freqy = np.fft.fftfreq(N2,dx).reshape([1,N2])
        freq=np.power(freqx,2)+np.power(freqy,2)
        freq=np.power(freq,0.5)

        # amplitude scaling
        amp = freq.copy()
        amp[amp==0]=float('inf')
        amp = np.power(amp,-2)

        # multiply by phase
        slipf = np.divide(slipf,np.abs(slipf))
        slipf = np.multiply(slipf,amp)

        # back to space domain
        slipf = np.real(np.fft.ifft2(slipf)).astype(float)

        # just part
        slipf = slipf[0:xx.shape[0],0:xx.shape[1]]
        
        # scale standard deviation and set mean to 0
        slipf = slipf-np.mean(slipf)
        scl = 1.0/np.std(slipf.flatten())
        slipf = scl*slipf 

        # shift so that 90% of the values are positive
        # within region with nonzero slip
        ii = slip!=0.
        vls = np.sort(slipf[ii])
        ii = int(len(vls)*0.1)
        
        if 'shiftmean' in slipdist:
            slipf = slipf - vls[ii]
        else:
            slipf[slipf<0.]=0.

        # smooth a bit
        slipf=ndimage.gaussian_filter(slipf,sigma=(2,2),order=0.,mode='nearest')

        # multiply slip by taper
        slip = np.multiply(slip,slipf)


        
    # scale to specified maximum
    slip = slip*dmax

    # collect for output
    return slip,x,y,xx,yy

def randstart(frc=0.2):
    """
    pick a random location within a circle with radius frc
    :param      frc:  radius of allowed circle
    :return    xyst:  picked location [x,y] 
    """

    # for a range
    frc=np.atleast_1d(frc)
    if frc.size==1:
        frc=np.append([0.],frc)
    
    # initial guess
    xyst = (np.random.rand(2)-0.5)*frc[1]*2
    
    # keep trying until it's inside the center
    rd2 = np.sum(np.power(xyst,2))
    while rd2>frc[1]**2 or rd2<frc[0]**2:
        xyst = (np.random.rand(2)-0.5)*frc*2
        rd2 = np.sum(np.power(xyst,2))
        
    return xyst

#----------TO GET SNAPSHOTS OF SLIP-------------------------

def snap_circrise(xx,yy,tm,slip,vrupt,rtime,xyst):
    """
    :param        x:  grid of x-locationsn
    :param        y:  grid of y-locations
    :param       tm:  times relative to start time 
    :param    vrupt:  initial propagation velocity (default: 3000)
    :param     xyst:  starting location
    :return cursnap:  snapshot of current slip distribution
    """

    # distance from the central point
    dst = np.power(xx-xyst[0],2)+np.power(yy-xyst[1],2)
    dst = np.power(dst,0.5)

    # current snapshot
    cursnap = np.zeros(xx.shape)

    # anything that's already finished
    idone = dst<=(tm-rtime)*vrupt
    cursnap[idone]=slip[idone]

    # anything in between
    imid=np.logical_and(dst<=tm*vrupt,~idone)
    
    # time since front arrived,
    # as a fraction of the rise time
    tsin = (tm-dst[imid]/vrupt)/rtime
    cursnap[imid]=np.multiply(tsin,slip[imid])
    
    # return snapshot
    return cursnap

def snap_bwm(xx,yy,tm=None,a=200.,vrupt=3000.,sdrop=None,ro=1.5,vprop=6000.,slip=None):
    """
    a spatial snapshot of slip in the boatwright-madariaga model
    :param     x:  grid of x-locations
    :param     y:  grid of y-locations
    :param    tm:  times relative to start time (default: 3000 points that span range)
    :param     a:  maximum radius (default: 200)
    :param vrupt:  initial propagation velocity (default: 3000)
    :param sdrop:  initial strain drop (stress drop/shear modulus, default: 1e-4)
    :param    ro:  overshoot parameter (default: 1.5)
    :param vprop:  P-wave velocity (default: 6000)
    :param  rvel:  return velocity instead of slip (default: false)
    :param  slip:  a final slip distribution, if it will differ from the 
                    default (default: not used)
    :return cursnap:  current snapshot
    """
    
    if sdrop is None:
        sdrop = 3.e6/3.e10

    # radius
    r = np.power(xx,2)+np.power(yy,2)
    r = np.power(r,0.5)
    r = r.reshape(np.prod(r.shape))

    # timing
    ts = a/vrupt + (a-r)/vprop
    th = ( (ro*a)**2 + (1.-ro**2)*r**2 ) / (vrupt**2 * ts)
    u = np.divide(th,th-ts)

    # a reference slip rate
    u0 = sdrop * vrupt

    # initialize snapshot
    cursnap = np.zeros(r.shape)

    # constant stress drop range
    ii = np.logical_and(tm>r/vrupt,tm<=ts)
    cursnap[ii] = u0 * (tm**2-np.power(r[ii],2)/vrupt**2)**0.5

    # healing decelerating range
    ii = np.logical_and(tm>ts,tm<th)
    cursnap[ii]=u0*np.power(tm**2 - 
                            np.multiply(u[ii],np.power(tm-ts[ii],2)) - 
                            np.power(r[ii],2)/vrupt**2,0.5)
    
    # at the end?
    ii = tm>=th
    cursnap[ii] = u0*a/vrupt*ro * np.power(1-np.power(r[ii],2)/a**2,0.5)

    # values within the radius
    ii = r<a

    if slip is not None:
        # expected final distribution
        finsnap = np.zeros(r.shape)
        finsnap=u0*a/vrupt*ro*np.power(1-np.power(r[ii],2)/a**2,0.5)

        # ratio and scale
        rt = np.divide(slip.flatten()[ii],finsnap)
        cursnap[ii] = np.multiply(cursnap[ii],rt)
        
    # nothing outside
    cursnap[~ii] = 0.

    # back to original shape
    cursnap = cursnap.reshape(xx.shape)


    return cursnap



def mincircle(xx,yy,slip,frc=0.8):
    """
    find the smallest circle containing a fraction of the slip
    :param     xx:  grid of x-values
    :param     yy:  grid of y-values
    :param   slip:  slip distribution
    :param    frc:  fraction of slip desired (default: 0.8)
    :return   rad:  radius
    :return    xy:  center of the circle
    """

    dx = np.diff(xx[:,0])[0]
    
    # limits
    r2=np.max(xx.flatten())-np.min(xx.flatten())
    r2=np.maximum(r2,np.max(yy.flatten())-np.min(yy.flatten()))
    r2=r2*2

    r1 = dx

    # estimate the plausible range

    # grid points that sum to some fraction
    cslip=np.flipud(np.sort(slip.flatten()))
    cslip=np.cumsum(cslip)
    imin,=np.where(cslip>frc*cslip[-1])
    imin = imin[0]

    # convert to an initial area and radius
    dx = np.median(np.diff(xx[:,0]))
    area = float(imin) * dx**2
    r = (area / math.pi)**0.5

    # to buffer before convolution
    nbuf = int(r*0.3/dx)
    sbuf = np.zeros(np.array(slip.shape)+nbuf*2,dtype=float)
    sbuf[nbuf:-nbuf,nbuf:-nbuf]=slip

    # another x-y grid
    x=np.arange(0,xx.shape[0]).astype(float)*dx
    y=np.arange(0,yy.shape[1]).astype(float)*dx
    x=x-np.mean(x)
    y=y-np.mean(y)
    
    # values
    xxi,yyi=np.meshgrid(x,y,indexing='ij')
    rd = np.power(xxi,2)+np.power(yyi,2)
    rd = np.power(rd,0.5)

    # desired total
    tdes = np.sum(slip)*frc

    xx=xx.flatten()
    yy=yy.flatten()

    # and for indexing
    xi=np.arange(0,2*nbuf+1).astype(float)*dx
    yi=np.arange(0,2*nbuf+1).astype(float)*dx
    xi=xi-np.mean(xi)
    yi=yi-np.mean(yi)
    xxi,yyi=np.meshgrid(xi,yi,indexing='ij')
    xxi,yyi=xxi.flatten(),yyi.flatten()

    # but need to be outside the range
    stot = float('inf')
    r1 = r
    r2 = r
    xy = np.array([0.,0])

    # while stot > tdes:
    #     # for the lower bound
    #     r1 = r1*0.95

    #     print('r1 '+str(r1))
        
    #     # filter and convolve
    #     flt = (rd<=r1).astype(float)
    #     #vl = scipy.signal.convolve2d(slip,flt,mode='same')
    #     vl = scipy.signal.convolve2d(sbuf,flt,mode='valid')

    #     # maximum
    #     imax=np.argmax(vl)
    #     stot=vl.flatten()[imax]
    #     xy = np.array([xxi[imax],yyi[imax]])


    # stot = -float('inf')
    # r2 = r
    # while stot < tdes:
    #     # for the lower bound
    #     r2 = r2*1.05

    #     print('r2 '+str(r2))
        
    #     # filter and convolve
    #     flt = (rd<=r2).astype(float)
    #     #vl = scipy.signal.convolve2d(slip,flt,mode='same')
    #     vl = scipy.signal.convolve2d(sbuf,flt,mode='valid')

    #     # maximum
    #     imax=np.argmax(vl)
    #     stot=vl.flatten()[imax]
    #     xy = np.array([xxi[imax],yyi[imax]])

    # while r2/r1 > 1.02:
    #     r = (r1+r2)/2.
        
    #     print('r '+str(r))

    #     # filter and convolve
    #     flt = (rd<=r).astype(float)
    #     #vl = scipy.signal.convolve2d(slip,flt,mode='same')
    #     vl = scipy.signal.convolve2d(sbuf,flt,mode='valid')

    #     # maximum
    #     imax=np.argmax(vl)
    #     stot=vl.flatten()[imax]
    #     xy = np.array([xxi[imax],yyi[imax]])

    #     if stot>tdes:
    #         r2=r
    #     else:
    #         r1=r
            
    rad = (r1+r2)/2.

    return rad,xy


#--------POINTWISE SOURCE TIME FUNCTIONS------------------


def istf_bwd(x=0.,y=0.,tm=None,a=200.,vrupt=3000.,u=5.,sdrop=None,rvel=False,dsm=0.):
    """
    the source time function of one point in a Boatwright deceleration model
    :param     x:  x-location (default: 0)
    :param     y:  y-location (default: 0)
    :param    tm:  times relative to start time (default: 3000 points that span range)
    :param     a:  maximum radius (default: 200)
    :param vrupt:  initial propagation velocity (default: 3000)
    :param     u:  deceleration parameter  (default: 5)
    :param sdrop:  initial strain drop (stress drop/shear modulus, default: 1e-4)
    :param  rvel:  return velocity instead of slip (default: False)
    :param   dsm:  distance over which to smooth linearly from the front (default: 0)
    :return  stf:  source time function at this point
    :return   tm:  times of stf
    """

    if sdrop is None:
        sdrop = 3.e6/3.e10
    if u is None:
        u=2.

    # timing
    th = a/vrupt * (u/(u-1.))**(.5)
    ts = a/vrupt * ((u-1.)/u)**(.5)

    # a reference slip rate
    u0 = sdrop * vrupt

    if tm is None:
        tm = np.linspace(-th*0.02,th*1.02,3000)

    # radius
    r = (x**2+y**2)**0.5

    # add these values
    if dsm>0:
        tlm = r/vrupt + np.array([0,dsm/vrupt])
        tm = np.concatenate([tm,tlm])
    
    # initialize
    stf = np.zeros(len(tm))

    # constant stress drop range
    ii = np.logical_and(tm>r/vrupt,tm<=ts)
    stf[ii] = u0 * (tm[ii]**2-r**2/vrupt**2)**0.5

    # decelerating range
    ii = np.logical_and(tm>ts,tm<th)
    stf[ii] = u0 * (tm[ii]**2 - u*(tm[ii]-ts)**2 - r**2/vrupt**2)**0.5
    
    # at the end
    ii = tm>=th
    stf[ii] = u0*a/vrupt * (1-r**2/a**2)**0.5

    # values in the early range
    if dsm>0.:
        # interpolate in range
        ii = np.logical_and(tm>tlm[0],tm<tlm[1])
        stf[ii] = np.interp(tm[ii],tlm,stf[-2:])

        # delete the added values
        tm = tm[:-2]
        stf = stf[:-2]

    # to velocity?
    if rvel:
        # difference and divide for timing
        stf = np.diff(stf,n=1,axis=0)
        dtim = np.diff(tm,n=1,axis=0)
        stf = np.divide(stf,dtim)
        
        # average times
        tm = (tm[0:-1]+tm[1:])/2.

    return stf,tm



def istf_haskell(x=0.,y=0.,tm=None,a=200.,vrupt=3000.,sdrop=None,ro=1.5,
                 vprop=6000.,rvel=False):
    """
    the source time function of one point in a Haskell model
    :param     x:  x-location: in propagation direction (default: 0)
    :param     y:  y-location (default: 0)
    :param    tm:  times relative to start time (default: 3000 points that span range)
    :param vrupt:  initial propagation velocity (default: 3000)
    :param rtime:  rise time---duration of slip (default: 0.01)
    :param sdrop:  initial strain drop (stress drop/shear modulus, default: 1e-4)
    :return  stf:  source time function at this point
    :return   tm:  times of stf
    :return rvel:  return velocity instead of slip (default: false)
    """
    

def istf_bwm(x=0.,y=0.,tm=None,a=200.,vrupt=3000.,sdrop=None,ro=1.5,vprop=6000.,rvel=False,dsm=0.):
    """
    the source time function of one point in a Boatwright Madariaga model
    :param     x:  x-location (default: 0)
    :param     y:  y-location (default: 0)
    :param    tm:  times relative to start time 
                    (default: 3000 points that span range)
    :param     a:  maximum radius (default: 200)
    :param vrupt:  initial propagation velocity (default: 3000)
    :param sdrop:  initial strain drop (stress drop/shear modulus, 
                                        default: 1e-4)
    :param    ro:  overshoot parameter (default: 1.5)
    :param vprop:  P-wave velocity (default: 6000)
    :param  rvel:  return velocity instead of slip (default: false)
    :parm    dsm:  distance over smoothing (default: 0)
    :return  stf:  source time function at this point
    :return   tm:  times of stf
    """

    if sdrop is None:
        sdrop = 3.e6/3.e10

    # radius
    r = (x**2+y**2)**0.5

    # timing
    ts = a/vrupt + (a-r)/vprop
    th = ( (ro*a)**2 + (1.-ro**2)*r**2 ) / (vrupt**2 * ts)
    u = th / (th-ts)

    # a reference slip rate
    u0 = sdrop * vrupt

    if tm is None:
        tm = np.linspace(-th*0.02,th*1.02,3000)

    # add these values
    if dsm>0:
        tlm = r/vrupt + np.array([0,dsm/vrupt])
        tm = np.concatenate([tm,tlm])
    
    # initialize
    stf = np.zeros(len(tm))

    # constant stress drop range
    ii = np.logical_and(tm>r/vrupt,tm<=ts)
    stf[ii] = u0 * (tm[ii]**2-r**2/vrupt**2)**0.5

    # healing decelerating range
    ii = np.logical_and(tm>ts,tm<th)
    stf[ii] = u0 * (tm[ii]**2 - u*(tm[ii]-ts)**2 - r**2/vrupt**2)**0.5
    
    # at the end
    ii = tm>=th
    stf[ii] = u0*a/vrupt*ro * (1-r**2/a**2)**0.5

    # values in the early range
    if dsm>0.:
        # interpolate in range
        ii = np.logical_and(tm>tlm[0],tm<tlm[1])
        stf[ii] = np.interp(tm[ii],tlm,stf[-2:])

        # delete the added values
        tm = tm[:-2]
        stf = stf[:-2]

    # to velocity?
    if rvel:
        # difference and divide for timing
        stf = np.diff(stf,n=1,axis=0)
        dtim = np.diff(tm,n=1,axis=0)
        stf = np.divide(stf,dtim)
        
        # average times
        tm = (tm[:-1]+tm[1:])/2.

    return stf,tm


#-------CALCULATE RUPTURE START TIMES---------------------------

def rupture_start(xx,yy,xyst,vrupt):        
    """
    :param      xx:  grid of x-locations
    :param      yy:  grid of y-locations
    :param    xyst:  starting location the rupture
    :param   vrupt:  rupture velocity (assumed uniform)
    :return ruptst:  rupture start time at each point
    """

    # distance from the starting point
    ruptst = np.power(xx-xyst[0],2)+np.power(yy-xyst[1],2)
    ruptst = np.power(ruptst,0.5)

    # divided by rupture propagation rate
    ruptst = ruptst / vrupt

    return ruptst
    

#------CALCULATE TRAVEL TIMES----------------------------

def ttrav(xx,uu,takang=None,stkost=None,dip=None,vprop=None,xyref=None):
    """
    :param     xx:  values along x-values---along strike
    :param     uu:  values along y axis---along dip
    :param takang:  take-off angle in degrees: 0=down, 180=up
    :param stkost:  offset of observed strike from positive x-axis
             in degrees, positive=station clockwise from fault strike
    :param    dip:  dip in degrees, y positive down-dip (default: 90)
    :param  vprop:  propagation velocities in m/s (default: 6000)
    :param  xyref:  reference location (default [0,0])
    :param  vrupt:  rupture velocity in m/s
    :param  rtime:  rise time
    :return ttrav:  travel times for each point
                     x in first dimension, y in second
    """

    if takang is None:
        # default horizontal
        takang = 90.
    if stkost is None:
        # default same direction
        stkost = 0.
    if dip is None:
        # default vertical
        dip = 90.
    if vprop is None:
        # p-wave velocity
        vprop = 6000.
    if xyref is None:
        xyref = np.array([0,0])

    # to radians
    takang=takang*math.pi/180.
    stkost=stkost*math.pi/180.
    dip=dip*math.pi/180.

    # subtract the reference
    xx = xx-xyref[0]
    uu = uu-xyref[1]

    # compute xyz locations of each point
    # down positive
    #[xx,uu] = np.meshgrid(x,y,indexing='ij')
    yy = uu*math.cos(dip)
    zz = uu*math.sin(dip)
    
    # direction of propagation
    xyz = [math.cos(stkost)*math.sin(takang),
           math.sin(stkost)*math.sin(takang),
           math.cos(takang)]

    # variation in travel time relative to central point
    ttrav=(xx*xyz[0]+yy*xyz[1]+zz*xyz[2])/(-vprop)

    return ttrav



#-----GREEN'S FUNCTIONS----------------------


def fakegf(tdec=3,tlen=10,tim=None,stout=False):
    """
    :param   tdec:  decay timescale
    :param   tlen:  max length
    :param    tim:  times of calculations
    :param  stout:  return as stream output
    :return    gf:  decaying white noise
    :return   tim:  times
    """
    tdec=float(tdec)
    tlen=float(tlen)
    if tim is None:
        # generate times
        dtim=tdec/1000
        tim=[math.floor(-0.5*tlen/dtim),math.ceil(tlen*1.5/dtim)]
        tim=np.arange(tim[0],tim[1],1)*dtim
    else:
        dtim=np.median(np.diff(tim))

    # amplitude
    gf=np.exp(-(tim/tdec)**2)
    gf[tim<0]=0

    # also allow onset
    tons = tdec/6.
    gfs = np.ones(gf.size,dtype=float)
    ii = np.logical_and(tim>=0,tim<=tons)
    gfs[ii] = (1-np.cos(tim[ii]/tons*np.pi))/2.
    gfs[tim<0]=0
    gf = np.multiply(gf,gfs)
    
    # random values
    gf=np.multiply(gf,np.random.randn(len(gf)))

    # create a trace
    st=obspy.Trace()
    st.data=gf
    st.stats.delta=np.median(np.diff(tim))
    st.stats.t6=-tim[0]

    # smoothing below 80% of Nyquist
    st.filter('lowpass',freq=0.4/dtim,zerophase=True)
    gf = st.data

    if stout:
        gf=obspy.Stream(st)

    return gf,tim

def conv(vl1,vl2,tim1=None,tim2=None,pk1='t6',pk2='t6'):
    """
    :param    vl1:   first set of values, or stream object
    :param    vl2:   second set of values, or stream object
    :param   tim1:   first set of times
    :param   tim2:   second set of times
    """
    if isinstance(vl1,list):
        vl1=np.array(vl1)
    if isinstance(vl2,list):
        vl2=np.array(vl2)
    if isinstance(vl1,np.ndarray) and isinstance(vl2,np.ndarray):
        # if they're both arrays
        if tim1 is None:
            tim1=np.arange(0,len(vl1))
        if tim2 is None:
            tim2=np.arange(0,len(vl2))

        # if they have different timing
        dtim1,dtim2=np.median(np.diff(tim1)),np.median(np.diff(tim2))
        dtim = np.minimum(dtim1,dtim2)
        if np.abs((dtim1-dtim)/dtim)>1.e-5:
            tim1i=np.arange(tim1[0],tim1[-1],dtim)
            vl1=np.interp(tim1i,tim1,vl1)
            tim1=tim1i
        if np.abs((dtim2-dtim)/dtim)>1.e-5:
            tim2i=np.arange(tim2[0],tim2[-1],dtim)
            vl2=np.interp(tim2i,tim2,vl2)
            tim2=tim2i
            
        # for FT
        N=len(vl1)+len(vl2)
        Nft=2**math.ceil(math.log(N,2))
        vl1=np.fft.rfft(vl1,Nft)
        vl2=np.fft.rfft(vl2,Nft)
        # convolve
        data=np.multiply(vl1,vl2)
        # inverse
        data=np.fft.irfft(data)
        # select portion
        data=np.real(data[0:N])
        # output times
        tim=np.median(np.diff(tim1))
        tim=np.arange(0,N)*tim
        tim=tim+tim1[0]+tim2[0]
    elif isinstance(vl2,obspy.Trace):
        # format for output
        data=vl2.copy()
        # extract portions
        tim2=vl2.times()-vl2.stats[pk1]
        vl2=vl2.data
        datai,tim=conv(vl1,vl2,tim1,tim2,pk1,pk2)
        # for output
        if isinstance(datai,np.ndarray):
            data.data=datai
            data.stats[pk2]=-tim[0]
        else:
            data=datai
    elif isinstance(vl1,obspy.Trace):
        # format for output
        data=vl1.copy()
        # extract portions
        tim1=vl1.times()-vl1.stats[pk1]
        vl1=vl1.data
        datai,tim=conv(vl1,vl2,tim1,tim2,pk1,pk2)
        # for output
        if isinstance(datai,np.ndarray):
            data.data=datai
            data.stats[pk1]=-tim[0]
        else:
            data=datai
    elif isinstance(vl1,obspy.Stream) and isinstance(vl2,obspy.Stream) and len(vl1)==len(vl2) and len(vl1)>1:
        # go through matching values
        data=obspy.Stream()
        tim=None
        for k in range(0,len(vl1)):
            tr1,tr2=vl1[k],vl2[k]
            datai,tim=conv(tr1,tr2,pk1=pk1,pk2=pk2)
            data.append(datai)
    elif isinstance(vl1,obspy.Stream):
        data=obspy.Stream()
        tim=None
        for tr in vl1:
            datai,tim=conv(tr,vl2,tim2=tim2,pk1=pk1,pk2=pk2)
            if isinstance(datai,obspy.Trace):
                data.append(datai)
            else:
                data=data+datai

    elif isinstance(vl2,obspy.Stream):
        data=obspy.Stream()
        tim=None
        for tr in vl2:
            datai,tim=conv(vl1,tr,tim1,pk1=pk1,pk2=pk2)
            if isinstance(datai,obspy.Trace):
                data.append(datai)
            else:
                data=data+datai

    return data,tim


#--------ADDING NOISE---------------------------------

def addnoise(st,mk='t0',trange=None,nrat=0.3333,pdec=0.,flma=None):
    """
    add noise to waveforms
    :param       st: waveforms
    :param       mk: marker used as reference (default: 't0')
    :param   trange: time range to use as signal estimate
                     (default: [0,1])
    :param     nrat: noise amplitude as a fraction of the signal
                     (default: 0.3333)
    :param     pdec: parameter for power law decay of noise amplitude
                     (default: 0---white noise, -1=flicker, etc)
    :param     flma: frequency limits (default: 3 to 30 times 1/diff(trange))
    """

    if trange is None:
        trange = [0.,1.]
    trange = np.atleast_1d(trange)
    if flma is None:
        flma = 1/np.diff(trange)[0]*np.array([3,30])
    if trange is None:
        trange = np.array([0,1])

    # copy stream to modify
    stn = st.copy()
    
    for tr in stn:
        # for each trace, identify portions of interest
        tget = tr.stats[mk]+trange
        ii=np.logical_and(tr.times()>=tget[0],tr.times()<=tget[1])
        

        if 1 == 0:
            # standard deviations
            amp = np.std(tr.data[ii])*nrat

            tr.data = tr.data + \
                np.random.normal(loc=0.,scale=amp,size=len(tr.data))
        else:
            # Fourier transform data
            Ndat = np.sum(ii)
            fdata = np.fft.rfft(tr.data[ii])
            freq = np.fft.rfftfreq(np.sum(ii),tr.stats.delta)
            iok = np.logical_and(freq>=flma[0],freq<=flma[1])
            amp = np.mean(np.power(np.abs(fdata[iok]),2))

            # for all the data
            freqa = np.fft.rfftfreq(len(tr.data),tr.stats.delta)

            if pdec=='same':
                # same profile as the other
                fadd=np.interp(freqa,freq,np.abs(fdata)).astype(complex)
            else:
                # amplitude spectra
                fadd = np.power(np.abs(freqa),pdec).astype(complex)

            # scale average in this range
            iok = np.logical_and(freqa>=flma[0],freqa<=flma[1])
            amps = np.mean(np.power(np.abs(fadd[iok]),2))
            fadd = (amp / amps)**0.5 * nrat * fadd

            # phases
            ii, = np.where(freqa>0.)
            phs=np.exp(np.random.rand(len(ii))*(1j*2.*math.pi))

            fadd[ii]=np.multiply(fadd[ii],phs)
            fadd[0]=0.

            # to time domain
            fadd = np.fft.irfft(fadd,len(tr.data))
            

            # and add
            tr.data = tr.data + fadd / (float(Ndat)/len(fadd))**.5
            
    
    return stn
