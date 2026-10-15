import copy
import spectrum
import general
import seisproc
#import joshazimuth
import syntheq 
import numpy as np
import os
import graphical
import phscoh
import math
import obspy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import interp1d

#--------------ALL CALCULATIONS----------------------------------

def allsynth(eqtyp='bwm',seqs=None,seqs2=None):

    print('Synthetics for '+eqtyp)

    prt = True

    # initialize synthetics
    if seqs is None:
        rad = np.array([50.,150,500])
        seqs,seqs2=appstfcalc(N=91,eqtyp=eqtyp,rad=rad)

    # plot apparent stf
    plotappstf(seqs=seqs,eqtyp=eqtyp,prt=prt)
    
    # including Green's functions
    plotcohgf(seqs=seqs,seqs2=seqs2,prt=prt,eqtyp=eqtyp,Nmax=8)

    # processing illustration
    plotprocexamp(seqs=seqs,prt=True,eqtyp=eqtyp)

    # test noise
    rad = np.array([100.,400])
    seqspair,trash=appstfcalc(N=10,eqtyp=eqtyp,rad=rad)
    noisetest(seqs=seqspair,prt=prt,eqtyp=eqtyp)

#--------TO CALCULATE THE APPARENT SOURCE TIME FUNCTIONS------------------

def plotall():
    
    typs=['het','bwm','circ']
    vrupts = [0.1,0.8*3**0.5,0.8]
    for eqtyp in typs:
        for vrupt in vrupts:
            lbl=eqtyp+'_'+'{:0.2f}'.format(vrupt)
            lbl=lbl.replace('.','p')
            print(lbl)

            seqs,seqs2=appstfcalc(eqtyp=eqtyp,vrupts=vrupt)

            plotappstf(seqs=seqs,eqtyp=lbl,prt=True)
            
            plotgfcoh(seqs=seqs,seqs2=seqs2,eqtyp=lbl,prt=True)

def appstfcalcvrupt(vrupts=[0.1,0.3,1.],rad=300.,secrad=0.):
    vrupts=np.atleast_1d(vrupts)
    seqs,seqs2=appstfcalc(N=91,eqtyp='het',secrad=secrad,rad=[rad],
                          justsetup=True)
    seqs[0].vrupt = seqs[0].vprop*vrupts[0]
    for k in range(1,vrupts.size):
        seqs.append(copy.copy(seqs[0]))
        seqs[k].vrupt = seqs[k].vprop*vrupts[k]
    if secrad>0:
        seqs2[0].vrupt = seqs2[0].vprop*vrupts[0]
        for k in range(1,vrupts.size):
            seqs2.append(copy.copy(seqs2[0]))
        seqs2[k].vrupt = seqs2[k].vprop*vrupts[k]
        setupseqs(seqs+seqs2)
    else:        
        # and setup
        setupseqs(seqs)

    return seqs,seqs2

def appstfcalcrtime(rtime=[0.05,0.3,2.]):
    rtime=np.atleast_1d(rtime)
    seqs,seqs2=appstfcalc(N=91,eqtyp='het',secrad=0.,rad=[300.],
                          justsetup=True)
    seqs[0].rtime = seqs[0].a/seqs[0].vprop*rtime[0]
    for k in range(1,rtime.size):
        seqs.append(copy.copy(seqs[0]))
        seqs[k].rtime = seqs[k].a/seqs[k].vrupt*rtime[k]

    # and setup
    setupseqs(seqs)

    return seqs


def introfig(seq=None,prt=True):
    """
    example fig for intro
    :param    seq:  earthquake (created if not given)
    :param    prt:  print the figure?
    """

    if seq is None:
        # synthetic
        seq=syntheq.seq(a=200.,ruptev='circrise',
                        slipdist='elliptical',
                        xyst=[-100.,0])

        msp=np.max(seq.slip)
        seq.addpatch(loc=[100,0],rd=0.02*seq.a,amp=np.max(seq.slip)*30)
        seq.addpatch(loc=[-100,0],rd=0.02*seq.a,amp=msp*40)
        
        # timing for everything
        dtim = 0.001 #default
        dtim = 1.0

        # initialize observation points
        seq.initobs(strike=[0,90,180.],takeang=90.)

        # and some fake Green's functions
        seq.initfakegf(dtim=dtim,tdec=1.,tlen=20.)

        # calculate apparent source time functions
        seq.calcappstf(dtim=dtim)


    # initialize plots
    plt.close()
    f = plt.figure(figsize=(10,8))
    gs,p=gridspec.GridSpec(3,1,width_ratios=[1]),[]
    gs.update(left=0.1,right=0.52)
    gs.update(bottom=0.07,top=0.97)
    gs.update(hspace=0.1,wspace=0.1)
    for gsi in gs:
        p.append(plt.subplot(gsi))
    p=np.array(p)

    gs2,p2=gridspec.GridSpec(1,1,width_ratios=[1]),[]
    gs2.update(left=0.55,right=0.97)
    gs2.update(bottom=0.1,top=0.92)
    gs2.update(hspace=0.1,wspace=0.1)
    for gsi in gs2:
        p2.append(plt.subplot(gsi))
    p2=np.array(p2)
    p2=p2[0]



    # source time functions
    st = seq.astf()
    st = st / np.max(st.flatten())
    tst = seq.tstf()

    # move to frequency domain
    dtim=np.median(np.diff(tst))
    Nf = int(10./dtim)
    fst = np.fft.fft(st,n=Nf,axis=0)

    freq=np.fft.fftfreq(Nf,dtim)
    ii=np.where(freq>0)
    ii=ii[np.arange(0,len(ii),Nf)] 
    freq = freq[ii]
    fst = fst[ii,:]
    fst = fst / np.max(fst.flatten())

    # phase
    phs = np.angle(fst)
    amp = np.abs(fst)

    stk = seq.ststrike
    cls = ['red','black','blue']
    cls = ['r','k','b']

    for k in range(0,3):
        p[0].plot(tst,st[:,k],color=cls[k])
        p[1].plot(freq,amp[:,k],color=cls[k])
        
        phsi = phs[:,k]
        ix, = np.where(phsi[1:]-phsi[0:-1]>2)
        ix = ix + 1
        vl = np.ones(ix.shape)*float('nan')
        vls = np.vstack([phsi[ix+1]-2*np.pi,vl,phsi[ix]+2*np.pi])
        ixs = np.vstack([ix,ix,ix])
        freqs = np.vstack([freq[ix+1],ix,freq[ix]])

        phsi=np.insert(phsi,ixs.flatten(),vls.flatten())
        freqi=np.insert(freq,ixs.flatten(),freqs.flatten())
        #phsi=np.insert(phsi,ix,vl)
        #freqi=np.insert(freq,ix,vl)

        p[2].plot(freqi,phsi*180/math.pi,color=cls[k])

    cls = ['b','k','r']
    cf1 = seq.vrupt / seq.a / 2.
    cf2 = seq.vprop / seq.a / 2.

    p[1].set_xlim([1.,100])
    p[0].set_ylim([0,1.01])
    p[0].set_xlim(seq.calcsdur()*np.array([-.5,2]))
    p[1].set_ylim([1.e-4,1.5])
    p[2].set_xlim([1.,100])
    p[1].set_xscale('log')
    p[1].set_yscale('log')
    p[2].set_xscale('log')
    p[2].set_ylim([-180,180])
    p[2].set_xlabel('frequency (Hz)')
    p[0].set_xlabel('time (s))')
    p[2].set_yticks([-180,0,180])
    p[1].set_xticklabels('')

    p[2].set_ylabel('apparent source time\nfunction phase')
    p[1].set_ylabel('apparent source time\nfunction amplitude')
    p[0].set_ylabel('apparent source\ntime function')
        

    for k in range(1,3):
        ylm=p[k].get_ylim()
        p[k].plot([cf1,cf1],ylm,color='k',linestyle='--')
        p[k].plot([cf2,cf2],ylm,color='k',linestyle='--')

    for ph in p[1:3]:
        xlm=ph.get_xlim()
        ph.axvspan(xmin=xlm[0],xmax=cf2,color='lightgray',zorder=0,alpha=0.8)

        
    ht1=p[1].text(cf1,.01,'$V_{rupt}/D$',
                  backgroundcolor='w',fontsize=12,
                  horizontalalignment='center')
    ht2=p[1].text(cf2,.001,'$V_{p}/D$',
                  backgroundcolor='w',fontsize=12,
                  horizontalalignment='center')

    thet=np.linspace(0,2*math.pi,7000)
    x,y=np.cos(thet),np.sin(thet)
    xy=np.ndarray([len(thet),2])
    xy[:,0],xy[:,1]=x,y
    ply = matplotlib.path.Path(xy)
    xyi=np.ndarray([len(thet),2])

    tstep=np.linspace(.1,.9,5)*2
    for tst in tstep:
        yi=-1.+y*tst
        xi=x*tst
        xyi[:,0],xyi[:,1]=xi,yi
        iok=ply.contains_points(xyi)
        p2.plot(xi[iok],yi[iok],color='k',linewidth=0.5)
        
    p2.plot(x,y,color='k',linewidth=1.5)
    
    amp = 6
    x = np.array([0,1.4,0])*amp
    y = np.array([-1,0,1])*amp
    for k in range(0,3):
        p2.plot(x[k],y[k],marker='^',linestyle='none',
                markersize=12,color=cls[k])
    xlm=np.array([-.6,1.5])*amp
    ylm=np.array([-1.1,1.1])*amp
    p2.set_xlim(xlm)
    p2.set_ylim(ylm)
    p2.set_xticks([])
    p2.set_yticks([])
    p2.set_aspect(np.diff(ylm)[0]/np.diff(xlm)[0])
    
    # locations
    y1,y2=-.5,.5

    # times
    t1,t2=-(y2-y1)*2,0

    x=np.linspace(0,amp-2,7000)
    xm=3.
    vl=np.exp(-np.power(x-xm,2)/.5**2)

    ons = np.array([1.,1.])
    pm1 = np.array([-.3,1.])
    pm2 = np.array([-.3,2.])
    lw = 1.5

    p2.plot([-30,30],[y1,y1],color='k',linestyle='--',
            linewidth=0.5)
    p2.plot([-30,30],[y2,y2],color='k',linestyle='--',
            linewidth=0.5)

    p2.plot(x+1.8-t1,vl*.5+y1,color=cls[1],linewidth=lw)
    p2.plot(x+1.8-t2,vl*.5+y2,color=cls[1],linewidth=lw)

    p2.plot(ons*(xm+1.8-t1),pm2+y1,color=cls[1],linewidth=0.5)
    p2.plot(ons*(xm+1.8-t2),pm1+y2,color=cls[1],linewidth=0.5)

    p2.plot(vl*.5-2.5,x+y1-t1,color=cls[2],linewidth=lw)
    p2.plot(vl*.5-1.5,x+y2-t2,color=cls[2],linewidth=lw)

    p2.plot(pm2-2.5,ons*(xm+y1-t1),color=cls[2],linewidth=0.5)
    p2.plot(pm1-1.5,ons*(xm+y2-t2),color=cls[2],linewidth=0.5)

    p2.plot(vl*.5-2.5,-x+y1+t1,color=cls[0],linewidth=lw)
    p2.plot(vl*.5-1.5,-x+y2+t2,color=cls[0],linewidth=lw)

    p2.plot(pm2-2.5,ons*(-xm+y1+t1),color=cls[0],linewidth=0.5)
    p2.plot(pm1-1.5,ons*(-xm+y2+t2),color=cls[0],linewidth=0.5)

    p2.plot([0,0],[y1,y2],marker='o',color='w',
            linestyle='none',markersize=10,
            markeredgecolor='w')

    ht1=p2.text(0,y1,'1',horizontalalignment='center',
                verticalalignment='center',
                backgroundcolor='none',fontsize='medium')
    ht2=p2.text(0,y2,'2',horizontalalignment='center',
                verticalalignment='center',
                backgroundcolor='none',fontsize='medium')

    arrowprops={'arrowstyle':'<->','shrinkA':0.0,'shrinkB':0.0}
    ht=p2.annotate('',xy=(1.12,y1),xytext=(1.1,y2),arrowprops=arrowprops)

    ht2=p2.text(1.25,0,'d',horizontalalignment='left',
                verticalalignment='center',
                backgroundcolor='none',fontsize='small')

    ht=p2.annotate('',xy=(-.8,-xm+y1+t1),xytext=(-.9,-xm+y2+t2),
                   arrowprops=arrowprops,color=cls[0])    
    ht=p2.annotate('',xy=(-.8,xm+y1-t1),xytext=(-.9,xm+y2-t2),
                   arrowprops=arrowprops,color=cls[2])    
    ht=p2.annotate('',xy=(xm+1.8-t1,y2+.8),
                   xytext=(xm+1.8-t2,y2+.8),
                   arrowprops=arrowprops,color=cls[2])    
    p2.plot(ons*(xm+1.8-t1),pm1+y1,color=cls[1],linewidth=0.5)
    p2.plot(ons*(xm+1.8-t2),pm1+y2,color=cls[1],linewidth=0.5)


    lb = '$d/V_{rupt}+d/V_p$'
    ht2=p2.text(-.7,-xm+(t1+t2)/2,lb,horizontalalignment='left',
                verticalalignment='center',
                backgroundcolor='none',fontsize='small')
    lb = '$d/V_{rupt}-d/V_p$'
    ht2=p2.text(-.7,xm-(t1+t2)/2,lb,horizontalalignment='left',
                verticalalignment='center',
                backgroundcolor='none',fontsize='small')
    lb = '$d/V_{rupt}$'
    ht2=p2.text(xm+1.8-(t1+t2)/2,y2+.8,lb,horizontalalignment='center',
                verticalalignment='bottom',
                backgroundcolor='none',fontsize='small')

    plbl=np.append(p,p2)
    graphical.cornerlabels(plbl,loc='ul',fontsize='small')

    if prt:
        fname= 'PCintrofig'
        fname=os.path.join(os.environ['FIGURES'],fname+'.pdf')
        pp=PdfPages(fname)
        pp.savefig(f)
        pp.close()
        plt.close()
    else:
        plt.show()

    


    return seq

def appstfcalcdirec(drs=[0.,0.5,1.]):
    drs=np.atleast_1d(drs)
    seqs,seqs2=appstfcalc(N=91,eqtyp='het',secrad=0.,rad=[300.],
             justsetup=True)
    seqs[0].xyst=np.array([seqs[0].a*drs[0],0.])
    for k in range(1,drs.size):
        seqs.append(copy.copy(seqs[0]))
        seqs[k].xyst=np.array([seqs[k].a*drs[k],0.])
    
    # and setup
    setupseqs(seqs)

    return seqs


def appstfcalc(justsetup=False,N=181,eqtyp='het',rad=None,
               secrad=0.7,shstat=0.,vrupts=None,stk=None,vprop=6000):
    """
    initialize some earthquakes and observations, 
    and calculate apparent source time functions
    :param         N:   number of stations
    :param     eqtyp:   type of slip model
    :param       rad:   radii
    :param    secrad:   a second radius relative to first (default: 1.)
                           if zero no second set is computed
    :param    shstat:   shift station azimuth randomly by +-shstat
    :param    vrupts:   any specified rupture velocities, as a fraction of 
                         the wave propagation velocity
    :return     seqs:   a list of the synthetic earthquakes
    :return    seqs2:   a list of more synthetic earthquakes
                         with the same Green's functions
    """

    if 'bwm' in eqtyp:
        typ = 'boatwright-m'
        slipdist = 'elliptical-unsmoothed'
        rndedg = [0.,0]
        rndst = False
    elif 'hetb' in eqtyp:
        typ = 'boatwright-m'
        slipdist = 'elliptical-fractal'
        rndedg = [0.,0]
        rndst = False
    elif 'hetnoshf' in eqtyp:
        typ = 'circrise'
        slipdist = 'elliptical-fractal'
        print(slipdist)
        rndedg = [0.2,7]
        rndst = True
    elif 'het' in eqtyp:
        typ = 'circrise'
        slipdist = 'elliptical-fractal-shiftmean'
        rndedg = [0.2,7]
        rndst = True
    elif 'circ' in eqtyp:
        typ = 'circrise'
        slipdist = 'elliptical'
        rndedg = [0.2,7]
        rndst = True
        rndedg = [0.,0]
        rndst = False

    if stk is None:
        # pick a set of azimuths and distances
        stk=np.linspace(0,90,N)+(np.random.rand(N)-0.5)*10
        if 'shift' in eqtyp:
            stk=np.linspace(0.,180.,N)
        else:
            stk=np.linspace(0.,90.,N)
        stk=np.linspace(0.,180.,N)
        stk=stk+(np.random.rand(N)-0.5)*2.*float(shstat)
        tkg=np.ones(stk.size,dtype=float)*90.
    else:
        stk=np.atleast_1d(stk)
        if stk.ndim>1:
            stk,tkg=stk[:,0],stk[:,1]
        else:
            tkg=np.ones(stk.size,dtype=float)*90.
        N = stk.size

    # sizes in meters
    if rad is None:
        rad = np.array([50,150,500])
    Nd=len(rad)
    rad = np.atleast_1d(rad)

    # change the rupture speed
    vrupt = 0.8*6000./3**.5
    if 'slow' in eqtyp:
        vrupt = vrupt*0.5
    elif 'fast' in eqtyp:
        vrupt = vrupt*2.

    # if the rupture velocities were specified
    if vrupts is not None:
        vrupt = vprop * vrupts
        if isinstance(vrupt,float):
            vrupt = vrupt * np.ones(rad.size)
    else: 
        vrupt = vrupt * np.ones(rad.size)

    # initialize earthquakes
    seqs,seqs2=[],[]
    xyst=np.array([0.,0.])
    xyst2=xyst.copy()
    for k in range(0,len(rad)):
        d,vrupti = rad[k],vrupt[k]
        if 'shift' in eqtyp:
            xyst=np.array([-d,0.])
            xyst2=xyst.copy()
        elif rndst:
            xyst=syntheq.randstart(0.2)*d
            xyst2=syntheq.randstart(0.2)*d
        seqs.append(syntheq.seq(a=d,ruptev=typ,slipdist=slipdist,xyst=xyst,
                                vrupt=vrupti,vprop=vprop,rndedg=rndedg))
        if secrad:
            seqs2.append(syntheq.seq(a=d*secrad,ruptev=typ,slipdist=slipdist,
                                     vrupt=vrupti,vprop=vprop,rndedg=rndedg,
                                     xyst=xyst2))

    # initialize observation points
    for seq in seqs+seqs2:
        seq.makeslipdist()
        seq.initobs(strike=stk,takeang=tkg)

    if not justsetup:
        setupseqs(seqs+seqs2)

    return seqs,seqs2


def setupseqs(seqs,tgf=1.,dtim=None):
    """
    :param      tgf: decay timescale for Green's function
    :param     dtim: time spacing, if known
    """

    # copy the Green's function to all values and compute
    vprop=np.array([seq.vprop for seq in seqs])
    vrupt=np.array([seq.vprop for seq in seqs])
    rad=np.array([np.minimum(seq.a,seq.b) for seq in seqs])

    # timing for everything
    dtim = np.min(rad)/np.maximum(np.max(vrupt),np.max(vprop))/50.

    # and some fake Green's functions---same for all
    seqs[0].initfakegf(dtim=dtim,tdec=1.,tlen=6.)
    for seq in seqs:
        seq.gf = seqs[0].gf.copy()
        seq.gftim = seqs[0].gftim.copy()

    # calculate apparent source time functions
    for seq in seqs:
        seq.calcappstf(dtim=dtim)

def plotappstfvrupts(seqs=None,eqtyp='het',prt=True):

    if seqs is None:
        seqs,trash=appstfcalcvrupt()

    p = plotappstf(seqs=seqs,eqtyp='test',prt=False)
    for k in range(0,len(seqs)):
        p[k,0].set_title('$V_r / V_p = $ {:0.1g}'.format(seqs[k].vrupt/
                                                         seqs[k].vprop))

    f = p[0,0].figure

    if prt:
        fname= 'PCappstf_vrupts'+'-'+eqtyp
        graphical.printfigure(fname,f,pngback=True)
        plt.close()
    else:
        plt.show()

def plotappstfrtime(seqs=None,eqtyp='het',prt=True):

    if seqs is None:
        seqs=appstfcalcrtime()

    p = plotappstf(seqs=seqs,eqtyp='test',prt=False)
    for k in range(0,len(seqs)):
        vl = seqs[k].rtime / (seqs[k].a/seqs[k].vrupt)
        p[k,0].set_title('$t_r / (R / V_r) = $ {:0.1g}'.format(vl))

    f = p[0,0].figure

    if prt:
        fname= 'PCappstf_rtimes'+'-'+eqtyp
        graphical.printfigure(fname,f,pngback=True)
        plt.close()
    else:
        plt.show()

def plotappstfdirec(seqs=None,eqtyp='het',prt=True):

    if seqs is None:
        seqs=appstfcalcrtime()

    p = plotappstf(seqs=seqs,eqtyp='test',prt=False)
    for k in range(0,len(seqs)):
        vl = seqs[k].xyst[0] / seqs[k].a
        ll = r'$x_{start}$'
        p[k,0].set_title(ll+'$ / R = $ {:0.1g}'.format(vl))

    f = p[0,0].figure

    if prt:
        fname= 'PCappstf_direc'+'-'+eqtyp
        graphical.printfigure(fname,f,pngback=True)
        plt.close()
    else:
        plt.show()






#-------THE CODES TO TEST THE EFFECTIVENESS-----------------------------

def plotappstf(seqs=None,eqtyp='bwm',prt=True):
    """
    plot coherence just for some apparent source time functions
    :param      seqs:  earthquakes
    :param     eqtyp:  earthquake type---for labelling
    :param       prt:  print the figure?
    :return        p:  the set of plots
    """
    
    # initialize earthquakes
    if seqs is None:
        seqs = appstfcalc(N=91,eqtyp=eqtyp)
        seqs = seqs[1:]

    # number of earthquakes
    Nd = len(seqs)

    # number of stations
    Na=seqs[0].Nstat
    N=min(8,Na)
    iplt=np.linspace(0.,180.,N+1).astype(float)
    iplt=iplt[0:-1]+np.random.rand(N)*np.diff(iplt)[0]
    iplt=np.searchsorted(seqs[0].ststrike,iplt)
    iplt=np.minimum(iplt,Na-1)
    iplt.sort()
    #iplt=np.arange(0,N)
    N = len(iplt)

    # initialize plots
    plt.close()
    f = plt.figure(figsize=(12,9.))
    gs,p=gridspec.GridSpec(Nd,3,width_ratios=[1,1,1],
                           height_ratios=np.ones(Nd)),[]
    gs.update(wspace=0.3,hspace=0.25)
    gs.update(left=0.05,right=0.99)
    gs.update(bottom=0.06,top=0.92)
    for gsi in gs:
        p.append(plt.subplot(gsi))
    p=np.array(p).reshape([Nd,3])

    ons=np.ones(2)
    for k in range(0,Nd):
        ttl = "{0:0.0f}".format(seqs[k].a*2)
        rad,xy=seqs[k].mincircle(0.8)
        ttl2 = "{0:0.0f}".format(rad*2)
        p[k,0].set_title(ttl+' m diameter, 80% in '+ttl2+' m',
                         fontsize='medium')
        p[k,0].set_title(ttl+' m diameter',
                         fontsize='medium')
        p[k,2].plot(seqs[0].vprop/seqs[k].a*ons/2,[-1.,2.],
                    color='lightgray',linewidth=2)

    # colors
    cls = ['blue','green','cyan','navy','gray','yellow','orange','red']

    # normalization
    nml = 0.
    for k in range(0,Nd):
        nml=max(nml,np.max(seqs[k].astf().flatten()))

    h,hf=[],[]
    lm = np.array([0.,0.])
    for k in range(0,Nd): # Nd
        # this earthquake
        seq = seqs[k]

        # grab the apparent source time functions and filter
        mk = 't0'
        sti = seq.astfwaveforms(marker=mk)
        # tmn = np.mean([tr.stats.t0 for tr in sti])
        # for tr in sti:
        #     tr.stats.t0 = tmn
        
        st = obspy.Stream()
        for ix in iplt:
            st.append(sti[ix])
            #stf = st.copy().filter('bandpass',freqmax=200.,freqmin=30.,
            #                   corners=2,zerophase=True)
            #stf = st.copy().filter('bandpass',freqmax=500.,freqmin=30.,
            #                   corners=2,zerophase=True)
        stf=st.copy().filter('bandpass',freqmax=10.,freqmin=5.,
                             corners=2,zerophase=True)
        nml1=1./np.mean([np.max(tr.data) for tr in st])
        nml2=1./np.mean([np.max(tr.data) for tr in stf])*0.2

        # move to frequency domain
        ost = seq.astf()

        # cumulative amplitude
        scum = np.sum(ost,axis=1)
        scum = np.cumsum(scum)
        scum = scum / np.max(scum)

        ilm = np.searchsorted(scum,np.array([0.005,0.995]))
        Ndur = ilm[1]-ilm[0]
        ilm[0] = max(ilm[0]-Ndur*2,0)
        ilm[1] = min(ilm[1]+Ndur*2,scum.shape[0])
        #ilm = np.array([0,ost.shape[0]])
        ilm = np.arange(np.maximum(0,ilm[0]),np.minimum(ost.shape[0],ilm[1]))
        Nf = (ilm[-1]-ilm[0])*5

        dtim=np.median(np.diff(seq.tstf()))
        osti = ost[ilm,:]
        # apply a taper
        tpr = False
        if tpr:
            import spectrum
            [U,V] = spectrum.mtm.dpss(osti.shape[0],40)
            osti = np.multiply(osti,U[:,0:1])

        fst = np.fft.fft(osti,n=Nf,axis=0)
        freq=np.fft.fftfreq(Nf,dtim)
        ii,=np.where(np.logical_and(freq>0,freq<=500))
        freq = freq[ii]
        fst = fst[ii,:]
        
        # shift the phases for the average arrival time
        twgt=np.dot(seq.astf().T,seq.tstf().reshape([seq.tstf().size,1]))
        twgt=np.divide(twgt,np.sum(seq.astf(),axis=0))
        tshf=np.mean(twgt)-seq.tstf()[ilm[0]]
        tshf = np.exp(1j*2*np.pi*freq*tshf)
        fst = np.multiply(fst,tshf.reshape([tshf.size,1]))


        # extract phase
        phs = np.angle(fst)

        #import code
        #code.interact(local=locals())


        # just some
        fst = fst[:,iplt]
        #import code
        #code.interact(local=locals())

        amps = np.abs(fst)
        ampss,fs=general.logsmooth(freq,amps,fstd=1.3,logy=True,subsamp=False)
        ampok = amps>0.5*ampss
        ampok = np.sum(ampok,axis=1)>0.6*ampok.shape[1]

        # normalize
        fst=np.divide(fst,np.abs(fst))
        fstm=np.abs(np.mean(fst,axis=1))

        # and convert to Cp
        Cp = 1./(N-1.)*(N*np.power(fstm,2)-1.)
        Cpok = np.ma.masked_array(Cp,mask=~ampok)
        #Cpok = np.ma.masked_array(Cp,mask=False)

        for m in range(0,N):
            # plot in the time domain
            tm=st[m].times()-st[m].stats[mk]
            cl = cls[m%len(cls)]
            hh=p[k,0].plot(tm,st[m].data*nml1,color=cl,linestyle='--')
            h.append(hh)
            hh=p[k,0].plot(tm,stf[m].data*nml2,color=cl,linestyle='-')
            hf.append(hh)
            lm[0] = np.min(np.append(st[m].data*nml1,lm[0]))
            lm[0] = np.min(np.append(stf[m].data*nml2,lm[0]))
            lm[1] = np.max(np.append(st[m].data*nml1,lm[1]))
            lm[1] = np.max(np.append(stf[m].data*nml2,lm[1]))

        tlm=seq.calcsdur(0.9)
        p[k,0].set_xlim(np.array([-1.,1.])*2*tlm)
        p[k,0].set_ylim([-.5,1.4])

        # limit the plotted frequency range
        fmax=seq.vprop/seq.a/2.*15
        ixf=freq<=fmax

        p[k,2].plot(freq[ixf],Cp[ixf],color='dimgray',linewidth=0.5,linestyle='--')
        p[k,2].plot(freq[ixf],Cpok[ixf],color='black',linewidth=2)            
        p[k,2].set_xscale('log')
        p[k,2].set_xlim([1,300.])
        p[k,2].plot(p[k,2].get_xlim(),[0,0],color='gray',linestyle='--',
                    zorder=1)
        p[k,2].set_ylim([-.52,1.02]) 

        ps=p[k,0].get_position()
        yrat=f.get_figheight()*ps.height/(f.get_figwidth()*ps.width)
        x0,y0=ps.x1-ps.width*.3,ps.y1-ps.height*.3/yrat
        pc=plt.axes([x0,y0,(ps.x1-x0)*0.9,(ps.y1-y0)*0.9])
        slip = np.flipud(seq.slip.copy())
        slip=np.ma.masked_array(slip,slip==0.)
        # here up is N, down is to the right
        xlm,ylm=general.minmax(seq.x),general.minmax(seq.y)
        hslip=pc.imshow(slip,zorder=3,extent=np.append(ylm,xlm))
        pc.plot(seq.xyst[1],seq.xyst[0],color='k',marker='*',
                markersize=7,zorder=4)
        cmap=matplotlib.cm.get_cmap('Reds')
        cmap.set_bad('w')
        hslip.set_cmap(cmap)
        pc.set_xticks([])
        pc.set_yticks([])
        pc.set_axis_off()
        pc.set_zorder(3)
        p[k,0].get_xaxis().set_zorder(20)

        phsi = phs.copy()*180./np.pi
        fsamp = freq.copy()
        flm = [fsamp[0],fsamp[1]]

        # relative to a reference
        iref = np.argmin(np.abs(seq.ststrike-90.))
        phsi=phsi-phsi[:,iref:(iref+1)]

        # mod 360
        phsi = phsi % 360
        phsi[phsi>180.]=phsi[phsi>180.]-360.

        # plot phases
        stklm=np.array([seq.ststrike[0],seq.ststrike[-1]])
        dstk = np.median(np.diff(seq.ststrike))
        stklm=stklm+np.array([1.,-1])*dstk/2
        extent=[flm[0],flm[1],stklm[0],stklm[1]]
        #vls=p[k,1].imshow(phsi.transpose(),extent=extent,aspect='auto')
        vls=p[k,1].pcolormesh(fsamp,seq.ststrike,phsi.transpose(),
                              vmin=-180,vmax=180,cmap='hsv')
        p[k,1].set_xscale('log')
        p[k,1].set_xlim([1,300.])
        #p[k,1].set_xlim(flm)
        #vls.set_clim([-180,180])
        #vls.set_cmap('hsv')

    ps = p[0,1].get_position()
    wd = ps.width*0.1
    psc=[ps.x0+wd,ps.y1+.01,ps.width-2*wd,.02]

    cbs = f.add_axes(psc)
    cb = f.colorbar(vls,cax=cbs,orientation='horizontal',
                    ticklocation='top',ticks=[-180,0,180])
    cbs.tick_params(axis='x',labelsize=9)
    cb.set_label('phase',fontsize=9)


    # lm = lm*1.05/nml
    # for k in range(0,Nd):
    #     p[k,0].set_ylim(lm)

    for ph in p.flatten():
        ph.xaxis.set_tick_params(labelsize='small')
    for ph in p[:,1]:
        ph.set_yticks([0,45,90,135,180])
        ph.set_yticklabels(['0','','90','','180'])
    for ph in p[:,2]:
        ph.set_yticks([0,.5,1.])
    for ph in p[:,0]:
        ph.set_yticks([0,.5,1.])
    graphical.delticklabels(p[:,1:],axs='x')

    slbl=[]
    for m in range(0,N):
        ttl = "{0:0.0f}".format(seqs[0].ststrike[iplt[m]])
        slbl.append(ttl)

    ilb = int((Nd-1)/2)
    p[ilb,0].set_ylabel('apparent source time function',
                         fontsize='large')
    p[ilb,2].set_ylabel('coherent phase moveout R',
                         fontsize='large')
    p[ilb,2].set_ylabel('phase coherence $C_p$',
                         fontsize='large')
    p[ilb,1].set_ylabel(r'station azimuth ($^{\circ}$)',
                         fontsize='large')
    p[Nd-1,0].set_xlabel('time (s)',fontsize='medium')
    p[Nd-1,2].set_xlabel('frequency (Hz)',fontsize='medium')
    p[Nd-1,1].set_xlabel('frequency (Hz)',fontsize='medium')

    h = np.array(h).reshape([Nd,N])
    hf = np.array(hf).reshape([Nd,N])

    lg=p[0,0].legend(hf[Nd-1,:],slbl,loc='lower left',
                     fontsize='x-small')
    ps=p[0,0].get_position()
    ax = p[0,0].add_artist(lg)
    ax.set_bbox_to_anchor((ps.x1,ps.y0),transform=plt.gcf().transFigure)
    lg2=p[0,0].legend(np.array([h[0,0],hf[0,0]]),['original','>30 Hz'],
                      loc='lower right',fontsize='x-small')
    lg.set_title('station\nazimuth')
    plt.setp(lg.get_title(),fontsize='x-small')
    #plt.setp(lg.get_title(),horizontalalignment='center')
    lg.borderaxespad=0.
    lg2.borderaxespad=0.

    graphical.cornerlabels(p.flatten(),loc='ul',fontsize='small')

    if prt:
        fname= 'PCappstf'+'-'+eqtyp
        graphical.printfigure(fname,f,pngback=True)
        plt.close()
    else:
        plt.show()


    return p

def plotgfcoh(seqs=None,seqs2=None,eqtyp='bwm',prt=True):
    """
    plot coherence just for some apparent source time functions
    :param      seqs:  earthquakes
    :param     eqtyp:  earthquake type---for labelling
    :param       prt:  print the figure?
    """
    
    # initialize earthquakes
    if seqs is None:
        seqs = appstfcalc(N=91,eqtyp=eqtyp)
        seqs = seqs[1:]
    if seqs2 is None:
        seqs2 = appstfcalc(N=91,eqtyp=eqtyp)
        seqs2 = seqs2[1:]
        

    # number of earthquakes
    Nd = len(seqs)

    # number of stations
    Na=seqs[0].Nstat
    N=min(15,Na)
    iplt=np.linspace(0.,180.,N+1).astype(float)
    iplt=iplt[0:-1]+np.random.rand(N)*np.diff(iplt)[0]
    iplt=np.searchsorted(seqs[0].ststrike,iplt)
    #iplt=np.searchsorted(seqs[0].ststrike,np.array([0.,20,40,60,90]))
    iplt=np.minimum(iplt,Na-1)
    #iplt =np.random.choice(Na,N,replace=False)
    iplt=np.unique(iplt)
    iplt.sort()
#    iplt=np.arange(0,N,1)
    #iplt=np.append(np.arange(0,N/3),np.arange(0,N/3)+int(Na/2))
    #iplt=(iplt+Na/4).astype(int)
    #    iplt=np.linspace(0.,Na-1,N).astype(int)
    N = len(iplt)

    st,stn=[],[]
    sto,stno=[],[]
    for k in range(0,len(seqs)):
        # waveforms for each earthquake
        sti=seqs[k].obswaveforms()
        stj=seqs2[k].obswaveforms()
        stia,stja=obspy.Stream(),obspy.Stream()
        for kk in iplt:
            stia=stia+sti[kk]
            stja=stja+stj[kk]
        st.append(stia)
        sto.append(stja)

    # downsample
    nsamp=int(np.round(1./300./st[0][0].stats.delta))
    nsamp=min(nsamp,10)
    for sti in (st+stn+sto+stno):
        sti.decimate(factor=nsamp)


    # initialize plots
    plt.close()
    f = plt.figure(figsize=(12,9.))
    gs,p=gridspec.GridSpec(Nd,3,width_ratios=[1,1,1],
                           height_ratios=np.ones(Nd)),[]
    gs.update(wspace=0.17,hspace=0.15)
    gs.update(left=0.07,right=0.99)
    gs.update(bottom=0.06,top=0.92)
    for gsi in gs:
        p.append(plt.subplot(gsi))
    p=np.array(p).reshape([Nd,3])

    ons=np.ones(2)
    for k in range(0,Nd):
        ttl = "{0:0.0f}".format(seqs[k].a*2)
        rad,xy=seqs[k].mincircle(0.8)
        ttl2 = "{0:0.0f}".format(rad*2)
        p[k,0].set_title(ttl+' m diameter, 80% in '+ttl2+' m',
                         fontsize='medium')
        p[k,0].set_title(ttl+' m diameter',
                         fontsize='medium')
        p[k,2].plot(seqs[0].vprop/seqs[k].a*ons/2,[-1.,2.],
                    color='lightgray',linewidth=2)


    # colors
    cls = graphical.colors(len(iplt))

    # normalization
    nml = 0.
    for k in range(0,Nd):
        nml=max(nml,np.max(seqs[k].astf().flatten()))

    h,hf=[],[]
    lm = np.array([0.,0.])
    awd = np.median(np.diff(seqs[0].ststrike[iplt]))/5.
    for k in range(0,Nd): # Nd
        # this earthquake
        seq = seqs[k]

        # plot the waveforms

        for ks in range(0,len(iplt)):
            stk=seq.ststrike[iplt[ks]]
            tr=sto[k][ks]
            dplt=tr.data/np.max(np.abs(tr.data))*awd*1.4
            p[k,0].plot(tr.times()-tr.stats.t0,dplt+stk,
                        color='black',linewidth=1.5)

            tr=st[k][ks]
            dplt=tr.data/np.max(np.abs(tr.data))*awd*1.4
            p[k,0].plot(tr.times()-tr.stats.t0,dplt+stk,
                        color=cls[ks],linewidth=1)
        
        #p[k,0].set_ylim([-.5,len(iplt)-0.5])
        p[k,0].set_xlim([-.5,3])


        # get cross-correlation
        trange=[-.1,1]
        xc=phscoh.calcxc(st[k],sto[k],trange,mk1='t0',mk2='t0',
                         nsint=None,dfres=2.)    
        xc.calcdirxc()
        xc.calcmvout()
        xc.cpfromr()
        xc.pickffreq()

        # move to frequency domain
        ost = np.mean(xc.xc,axis=2)

        # extract phase
        phs = np.angle(ost)

        # subtract reference phase
        iref=np.argmin(np.abs(seq.ststrike[iplt]-90))
        pref = phs[:,iref]
        pref = pref.reshape([len(xc.freq),1])
        phsi = phs-pref

        # mod 360
        phsi = (phsi*180./np.pi) % 360
        phsi[phsi>180.]=phsi[phsi>180.]-360.
        fsamp=xc.freq.copy()

        Nstat=phsi.shape[1]
        
        for mm in range(0,Nstat):
            stk=seq.ststrike[iplt[mm]]
            stkp = stk+np.array([-1.,1])*360/Nstat/10
            ix = np.array([mm,mm])
            vls=p[k,1].pcolormesh(fsamp,stkp,phsi[:,ix].T,
                                  vmin=-180,vmax=180,
                                  cmap='hsv')
        p[k,1].set_xscale('log')
        p[k,1].set_xlim([1,200.])
        p[k,1].set_xlim([0,300.])

        for k2 in range(0,k):
            xc2=phscoh.calcxc(st[k],sto[k2],trange,mk1='t0',mk2='t0',nsint=None)    
            xc2.calcmvout()
            xc2.cpfromr()

            # limit the plotted frequency range
            fmax=seq.vprop/seq.a/2.*15
            ixf=xc2.freq<=fmax

            h1,=p[k,2].plot(xc2.freq[ixf],xc2.Cp[ixf],
                            color='gray',linewidth=1.5,linestyle='--')

        #p[k,2].plot(xc.ffspl,0.5,color='k',marker='*',linestyle='none')

        # limit the plotted frequency range
        fmax=seq.vprop/seq.a/2.*15
        ixf=xc.freq<=fmax

        p[k,2].plot(xc.freq[ixf],xc.Cp[ixf],color='k',linewidth=1.5)
        p[k,2].set_xscale('log')
        p[k,2].set_xlim([1,300.])
        p[k,1].set_xlim([1,300.])
        h2,=p[k,2].plot(p[k,2].get_xlim(),[0,0],color='black',linestyle='-',
                        zorder=1,markersize=7)

        p[k,2].plot(xc.ffbest,0.5,color='k',marker='*',linestyle='none')

    if len(seqs)>1:
        smlr = '{:0.0f}'.format((1-seqs2[0].a/seqs[0].a)*100)
        if len(seqs)>2:
            lms=general.minmax([1-seqi.a/seqs[-1].a for seqi in seqs[0:-1]])
            smlr2 = '{:0.0f} - {:0.0f}'.format(lms[0]*100,lms[1]*100)
        else:
            smlr2 = '{:0.0f}'.format((1-seqs[0].a/seqs[1].a)*100)
        clbl=['2$^{nd}$ earthquake '+smlr+'% smaller',
              '2$^{nd}$ earthquake '+smlr2+'% smaller']
        lg = p[k,2].legend([h2,h1],clbl,loc='center',
                           bbox_to_anchor=(0.5,1.15))

    awd=awd*1.5

    for ph in p[:,1]:
        ps=ph.get_position()
        ps.x0=ps.x0-0.03
        ps.x1=ps.x1-0.03
        ph.set_position(ps)
    for ph in p.flatten():
        ph.xaxis.set_tick_params(labelsize='small')
    for ph in p[:,0:2].flatten():
        ph.set_yticks([0,45,90,135,180])
        ph.set_yticklabels(['0','','90','','180'])
        ph.set_ylim([-awd,180+awd])
    for ph in p[:,0]:
        x=[trange[0],trange[1],trange[1],trange[0],trange[0]]
        y=[-awd,-awd,180+awd,180+awd,-awd]
        ply = Polygon(np.vstack([x,y]).transpose())
        ply.set_edgecolor('none')
        ply.set_color('lightgray')
        ply.set_alpha(0.7)
        ph.add_patch(ply)
    for ph in p[:,2]:
        ph.set_yticks([-.5,0,.5,1.])
        ph.set_ylim([-.53,1.03])
    graphical.delticklabels(p,axs='x')
    graphical.delticklabels(p[:,0:2],axs='y')

    ps = p[0,1].get_position()
    wd = ps.width*0.1
    psc=[ps.x0+wd,ps.y1+.01,ps.width-2*wd,.02]

    cbs = f.add_axes(psc)
    cb = f.colorbar(vls,cax=cbs,orientation='horizontal',
                    ticklocation='top',ticks=[-180,0,180])
    cbs.tick_params(axis='x',labelsize=9)
    cb.set_label('relative phase',fontsize=9)


    slbl=[]
    for m in range(0,N):
        ttl = "{0:0.0f}".format(seqs[0].ststrike[iplt[m]])
        slbl.append(ttl)

    ilb = int((Nd-1)/2)
    p[ilb,2].set_ylabel('coherent phase moveout R',
                         fontsize='large')
    p[ilb,2].set_ylabel('phase coherence $C_p$',
                         fontsize='large')
    p[ilb,0].set_ylabel(r'station azimuth ($^{\circ}$)',
                         fontsize='large')
    p[Nd-1,0].set_xlabel('time (s)',fontsize='medium')
    p[Nd-1,2].set_xlabel('frequency (Hz)',fontsize='medium')
    p[Nd-1,1].set_xlabel('frequency (Hz)',fontsize='medium')


    graphical.cornerlabels(p.flatten(),loc='ul',fontsize='small')

    if prt:
        fname= 'PCgfcoh'+'-'+eqtyp
        graphical.printfigure(fname,f)
        plt.close()
    else:
        plt.show()


def plotcohgf(seqs=None,seqs2=None,prt=True,eqtyp='unknown',Nmax=10):
    """
    plot coherence just for some apparent source time functions
    :param      seqs:  earthquakes
    :param     seqs2:  second set of earthquakes
    :param     eqtyp:  type of synthetic---for labelling
    :param       prt:  print to a file?
    :param      Nmax:  maximum number of stations
    """

    # a directory to write to
    fdir=os.path.join(os.environ['DATA'],'STRESSDROPS','TESTDATA')
    
    # initialize earthquakes
    if seqs is None:
        seqs = appstfcalc()
    if seqs2 is None:
        seqs2 = seqs

    # number of earthquakes
    Nd = len(seqs)

    # number of stations
    N=seqs[0].Nstat
    if Nmax<N:
        ix=np.linspace(0.,N-1.,Nmax)
        ix=np.round(ix).astype(int)
        ix=np.unique(ix)
        ix =np.random.choice(N,Nmax,replace=False)
        ix.sort()
    else:
        ix=np.arange(0,N)
    N = len(ix)

    st,stn=[],[]
    sto,stno=[],[]
    for k in range(0,len(seqs)):
        # waveforms for each earthquake
        sti=seqs[k].obswaveforms()
        stj=seqs2[k].obswaveforms()
        stia,stja=obspy.Stream(),obspy.Stream()
        for kk in ix:
            stia=stia+sti[kk]
            stja=stja+stj[kk]
        st.append(stia)
        sto.append(stja)

    # downsample
    nsamp=int(np.round(1./300./st[0][0].stats.delta))
    nsamp=min(nsamp,10)
    for sti in (st+stn+sto+stno):
        sti.decimate(factor=nsamp)


    for k in range(0,len(seqs)):
        # add noise
        stn.append(syntheq.addnoise(st[k],trange=[0.,2.],nrat=0.05,
                                    pdec=-1.,flma=[1.,3.]))
        stno.append(syntheq.addnoise(sto[k],trange=[0.,2.],nrat=0.05,
                                     pdec=-1.,flma=[1.,3.]))

    # time range
    trange=np.array([-0.,2.])

    # first highpass
    hp = 3./np.diff(trange)[0]
    for sti in (st+stn+sto+stno):
        sti.detrend()
        sti.filter('highpass',freq=hp)

    # initialize plots
    f = plt.figure(figsize=(10,8))
    gs,p=gridspec.GridSpec(Nd,3,width_ratios=[1,1,1]),[]
    gs.update(left=0.1,right=0.95)
    gs.update(bottom=0.1,top=0.92)
    gs.update(hspace=0.25,wspace=0.25)
    for gsi in gs:
        p.append(plt.subplot(gsi))
    p=np.array(p).reshape([Nd,3])

    # colors
    col = graphical.colors(Nd)
    col2 = graphical.colors(Nd,True)

    for k in range(0,Nd):
        # for each of the larger earthquakes
        st1 = st[k]
        st1n = stn[k]
        rad = np.round(seqs[k].a)
        
        for m in range(0,k+1):
            # for each reference earthquake
            if m==k:
                # only use the second set of earthquakes
                # for the common size
                st2 = sto[m]
                st2n = stno[m]
            else:
                st2 = st[m]
                st2n = stn[m]

            # get cross-correlation
            xc=phscoh.calcxc(st1,st2,trange,mk1='t0',mk2='t0',nsint=None)    

            # direct coherence
            xc.calcdirxc()
            xcm = xc.xcdir

            # and coherence
            xc.calcmvout()
            xc.cpfromr()
            xc.pickffreq()
            
            r = xc.R
            Cp = xc.Cp
            freq = xc.freq

            # get cross-correlation
            xcg=phscoh.calcxc(st1n,st2n,trange,mk1='t0',mk2='t0',nsint=None)
            xcg.calcdirxc()
            xcg.calcmvout()
            xcg.cpfromr()

            # and coherence
            rn=xcg.R
            Cpn=xcg.Cp
            freqn=xcg.freq

            # to smooth for comparison
            nsamp = 5
            nsamp = 2./np.median(np.diff(freq))
            rf = gaussian_filter1d(r,nsamp,0)
            freqf = gaussian_filter1d(freq,nsamp,0)
            rnf = gaussian_filter1d(rn,nsamp,0)
            freqnf = gaussian_filter1d(freqn,nsamp,0)

            # amplitudes
            amp = np.mean(xc.amp[:,0,:,0],axis=1)
            #amp = np.mean(np.mean(xc.amp[:,:,:,0],axis=1),axis=1)
            ampf = gaussian_filter1d(amp,nsamp,0)
            ampn = np.mean(xcg.amp[:,0,:,0],axis=1)
            #            ampn = np.mean(np.mean(xc.amp[:,:,:,0],axis=1),axis=1)
            ampnf = gaussian_filter1d(ampn,nsamp,0)

            if m==k:
                # plot amplitudes
                xlm = np.array([1.,100.])
                i1=np.logical_and(freq>=xlm[0],freq<=xlm[1])
                i2=np.logical_and(freqf>=xlm[0],freqf<=xlm[1])
                mx=np.max(amp[i1])
                mx=np.max(np.append(mx,ampn[i1]))
                mx=np.max(np.append(mx,ampnf[i2]))
                mx=np.max(np.append(mx,ampf[i2]))
                p[k,0].plot(freqf,ampnf/mx,color='gray')
                p[k,0].plot(freqf,ampf/mx,color=col[m])
                p[k,0].set_yscale('log')
                p[k,0].set_ylim([10.**-10,1.5])
                p[k,1].plot(xc.ffspl,0.5,color='k',linestyle='none',marker='*')
                print(xc.ffspl)
#                 ons=np.ones(2)
#                 for ph in p[k,:]:
#                     ph.plot(seqs[0].vprop/seqs[k].a*ons/2,[-1.,2.],
#                             color='lightgray',linewidth=1.5)


            # plot
            p[k,1].plot(freq,xcm,color=col[m],linestyle='-.')
            p[k,1].plot(freq,Cp,color=col[m],linestyle='-')
            p[k,1].set_ylim([-.04,1.04])

            rad = seqs[k].a
            ttl = "{0:0.0f}".format(rad)
            rad,xy=seqs[k].mincircle(0.8)
            ttl2 = "{0:0.0f}".format(rad)
            p[k,0].set_title(ttl+' m radius, 80% in '+ttl2+' m',
                             fontsize='medium')

            p[k,2].plot(freqn,Cpn,color=col[m],linestyle='-')
            p[k,2].set_ylim([-.54,1.04])

            for nn in range(0,3):
                p[k,nn].set_xlim([2.,200])
                p[k,nn].set_xscale('log')
                p[Nd-1,nn].set_xlabel('frequency (Hz)',fontsize='small')

            ps = p[k,0].get_position()
            wd = ps.width*0.6
            ht = ps.height*0.4
            psc=[ps.x0,ps.y0,wd,ht]
            ph = f.add_axes(psc)
            nn=0
            tm = st1n[nn].times()-st1n[nn].stats.t0
            ph.plot(tm,st1n[nn].data,color='gray')
            tm = st1[nn].times()-st1[nn].stats.t0
            ph.plot(tm,st1[nn].data,color='k')
            ph.set_xlim([trange[0]-1.,trange[1]+2.])
            ylm = np.max(abs(np.append(st1[nn].data,st1[nn].data)))
            ph.set_ylim(np.array([-1,1.])*ylm*1.03)
            ph.xaxis.tick_top()
            ph.xaxis.set_label_position('top')
            ph.set_xlabel('time (s)',fontsize='xx-small')
            ph.xaxis.set_tick_params(labelsize='xx-small')
            ph.set_yticks([])
            
    for ph in p.flatten():
        ph.xaxis.set_tick_params(labelsize='small')
        ph.yaxis.set_tick_params(labelsize='small')
        ph.set_xlim([2,100.])

    for ph in p[:,1:].flatten():
        ph.set_yticks([-.5,0,0.5,1])
        ph.set_ylim([-.54,1.04])
        ph.plot([freq[0],freq[-1]],[0.,0.],color='k',linestyle='--',
                zorder=1,linewidth=0.5)

    for ph in p[:,0].flatten():
        ph.set_yticks(np.power(10.,np.array([-6,-3,0])))
        ph.set_ylim(np.power(10.,np.array([-6,0.3])))

    graphical.delticklabels(p,axs='x')
    graphical.delticklabels(p[:,0:1])
    graphical.cornerlabels(p.flatten(),loc='ul',fontsize='x-small')

    #graphical.delticklabels(p[:,1:])
    ilb=(Nd-1)/2
    p[ilb,0].set_ylabel('station-averaged power spectra',
                         fontsize='medium')
    p[ilb,1].set_ylabel('$C_p$ from synthetic data',fontsize='medium')
    p[ilb,2].set_ylabel('$C_p$ from synthetic data with noise',fontsize='medium')

    if prt:
        fname= 'PCcohgf_'+eqtyp
        fname=os.path.join(os.environ['FIGURES'],fname+'.pdf')
        pp=PdfPages(fname)
        pp.savefig(f)
        pp.close()
        plt.close()
    else:
        plt.show()


def noisetest(seqs=None,prt=True,eqtyp='hetb'):
    """
    plot coherence just for some apparent source time functions
    :param      seqs:  earthquakes
    :param       prt:  print to a file?
    :param     eqtyp:  type of event
    """
    
    # initialize earthquakes
    if seqs is None:
        seqs = appstfcalc(N=5,rad=[50,200.],eqtyp=eqtyp)
    seq1=seqs[0]
    seq2=seqs[1]

    # number of stations
    N=seq1.Nstat

    # waveforms for each earthquake
    st1=seq1.obswaveforms()
    st2=seq2.obswaveforms()

    # buffer both
    st1 = seisproc.bufferwaveforms(st1,tbf=20.,taf=20.)
    st2 = seisproc.bufferwaveforms(st2,tbf=20.,taf=20.)

    # downsample
    nsamp=int(np.round(1./300./st1[0].stats.delta))
    nsamp=min(nsamp,16)
    (st1+st2).decimate(factor=nsamp)

    # time range
    trange=np.array([-0.5,2.5])

    # for initial high-pass
    hp = 3./np.diff(trange)[0]

    # highpass filter before final cross-correlation
    st1i=st1.copy().filter('highpass',freq=hp)
    st2i=st2.copy().filter('highpass',freq=hp)

    # get cross-correlation
    xc=phscoh.calcxc(st1i,st2i,trange,mk1='t0',mk2='t0',nsint=None,tpr='multi')

    # and coherence
    xc.calcmvout()
    xc.cpfromr()
    xc.pickffreq()
    Cp=xc.Cp

    # amplitudes
    amp1 = np.mean(np.mean(xc.amp[:,:,:,0],axis=2),axis=1)
    amp1f = gaussian_filter1d(amp1,nsamp,0)
    amp2 = np.mean(np.mean(xc.amp[:,:,:,1],axis=2),axis=1)
    amp2f = gaussian_filter1d(amp2,nsamp,0)

    # initialize
    Nt = 50
    Cpn = np.ndarray([len(Cp),Nt])
    amp1n = np.ndarray([len(Cp),Nt])
    amp2n = np.ndarray([len(Cp),Nt])
    Ss = np.ndarray([len(Cp),seq1.Nstat,Nt])
    Ntap = xc.Ntap
    xcs = np.ndarray([len(Cp),seq1.Nstat,Ntap,Nt],
                     dtype=complex)
    ns = np.ndarray([len(Cp),seq1.Nstat,2,Nt])


    for k in range(0,Nt):
        # add noise
        st1n=syntheq.addnoise(st1,trange=[0.,2.],nrat=0.5,
                              pdec=0.,flma=[1.,3.])
        st2n=syntheq.addnoise(st2,trange=[0.,2.],nrat=0.4,
                              pdec=0.,flma=[1.,3.])

        # filter the noise
        (st1n+st2n).filter('highpass',freq=hp)

        # get cross-correlation of noisy signals
        xcg=phscoh.calcxc(st1n,st2n,trange,mk1='t0',mk2='t0',nsint=1,tpr='multi')

        # amplitudes
        amp1n[:,k]=np.mean(np.mean(xcg.amp[:,:,:,0],axis=2),axis=1)
        amp2n[:,k]=np.mean(np.mean(xcg.amp[:,:,:,1],axis=2),axis=1)

        # and coherence
        xcg.calcmvout()
        Cpn[:,k]=xcg.Cp

        # save x-c for some reason
        xcs[:,:,:,k]=xcg.xc

        # compute signal fractions
        xcg.signalfraction(sg=np.mean(xc.amp,axis=2))
        Ss[:,:,k] = xcg.S

        # get noise from another interval
        xci=phscoh.calcxc(st1n,st2n,trange-10.,mk1='t0',
                          mk2='t0',nsint=None)
        ns[:,:,:,k] = np.mean(xci.amp,axis=2)


    amp1n=np.mean(amp1n,axis=1)
    amp2n=np.mean(amp2n,axis=1)

    # save frequencies
    freq = xc.freq

    # to smooth for comparison
    nsamp = 2./np.median(np.diff(freq))
    Cpf = gaussian_filter1d(Cp,nsamp,0)
    Cpnf = gaussian_filter1d(Cpn,nsamp,0)
    freqf = gaussian_filter1d(freq,nsamp,0)

    # frequencies to check
    fcheck = np.array([3.,5.,6.,7.,10.,13.,17.])
    Nf = fcheck.size

    # initialize plots
    plt.close()
    f = plt.figure(figsize=(10,8))
    gs,p=gridspec.GridSpec(2,1,width_ratios=[1]),[]
    gs.update(left=0.1,right=0.55)
    gs.update(bottom=0.1,top=0.92)
    gs.update(hspace=0.1,wspace=0.1)
    for gsi in gs:
        p.append(plt.subplot(gsi))
    p=np.array(p)

    gs2,p2=gridspec.GridSpec(Nf,1,width_ratios=[1]),[]
    gs2.update(left=0.63,right=0.97)
    gs2.update(bottom=0.1,top=0.92)
    gs2.update(hspace=0.1,wspace=0.1)
    for gsi in gs2:
        p2.append(plt.subplot(gsi))
    p2=np.array(p2)

    col = ['blue','red','yellow','green','black']

    xlm=np.array([1.,20.])
    for ph in p:
        ph.set_xlim(xlm)
        ph.set_xscale('log')
    
    # to normalized power spectra
    iok=np.logical_and(freq>=xlm[0],freq<=xlm[1])
    nm1,nm2=np.max(amp1[iok]),np.max(amp2[iok])

    #p[1].plot(freq,amp1n,color='gray')
    #p[1].plot(freq,amp2n,color='gray')
    ns1=np.mean(np.mean(ns,axis=3)[:,:,0],axis=1)
    ns2=np.mean(np.mean(ns,axis=3)[:,:,1],axis=1)
    hn1,=p[1].plot(freq,ns1/nm1,color='darkgreen',
                   linestyle='--')
    hn2,=p[1].plot(freq,ns2/nm2,color='saddlebrown',
                   linestyle='--')
    ha1,=p[1].plot(freq,amp1/nm1,color='darkgreen')
    ha2,=p[1].plot(freq,amp2/nm2,color='saddlebrown')
    p[1].set_yscale('log')
    
    ymin=np.hstack([ns1[iok]/nm1,ns2[iok]/nm2,
                    amp1[iok]/nm1,amp2[iok]/nm2])
    ymax,ymin=np.max(ymin),np.min(ymin)
    p[1].set_ylim([ymin/4.,ymax*2.])

    p[1].set_ylabel('station-averaged power spectra')

    lbls=['earthquakes','added noise']
    lg1=p[1].legend([ha1,hn1],lbls,loc='lower left',
                    fontsize='small')

    # limits of walkout values obtained
    Cp1=int(np.float(Nt*0.15))
    Cpm=int(np.float(Nt*0.5))
    Cp2=int(np.float(Nt*0.85))
    Cpn=np.sort(Cpn,axis=1)
    Cp1,Cp2,Cpm = Cpn[:,Cp1],Cpn[:,Cp2],Cpn[:,Cpm]

    # bins for plotting
    bns = np.linspace(-1.,1.,101)
    bnx = (bns[0:-1]+bns[1:])/2.

    # frequency indices
    ifcheck = np.searchsorted(freq,fcheck)

    # keep track of maximum
    npermax = 0.

    for k in range(0,Nf):
        # to illustrate frequencies of interest
        p[0].plot(np.ones(2)*fcheck[k],np.array([-1.,2.]),
                  color='lightgray',linewidth=2)
        p[0].text(fcheck[k],.02,chr(k+ord('c')),
                  backgroundcolor='w',fontsize='small',
                  horizontalalignment='center')
                  
        # bin
        nper,trash = np.histogram(Cpn[ifcheck[k],:],bins=bns)
        nper = nper.astype(float)

        # normalize to a probability density
        nper=nper/np.sum(nper)
        nper=nper/np.median(np.diff(bns))

        # maximum value
        npermax=np.max(np.append(nper,npermax))

        x,y=graphical.baroutlinevals(bnx,nper)
        p2[k].plot(x,y,color='darkblue')
        p2[k].set_xlim([-.25,1.04])

        # create polygon
        iok=np.logical_and(bnx>=Cp1[ifcheck[k]],
                           bnx<=Cp2[ifcheck[k]])
        if ~np.sum(iok):
            iok = general.closest(bnx,Cp1[ifcheck[k]])
        x,y=graphical.baroutlinevals(bnx[iok],nper[iok],wzeros=True)
        ply = Polygon(np.vstack([x,y]).transpose())
        ply.set_edgecolor('none')
        ply.set_color('lightblue')
        ply.set_alpha(1)
        p2[k].add_patch(ply)


    # to shade a region
    x1,y1=graphical.baroutlinevals(freq,Cp1)
    x2,y2=graphical.baroutlinevals(freq,Cp2)
    x2,y2=np.flipud(x2),np.flipud(y2)
    x,y=np.append(x1,x2),np.append(y1,y2)

    # create polygon
    ply = Polygon(np.vstack([x,y]).transpose())
    ply.set_edgecolor('none')
    ply.set_color('lightblue')
    ply.set_alpha(1)
    p[0].add_patch(ply)

    # original values
    horig,=p[0].plot(freq,Cp,color='k')
    p[0].set_ylabel('$C_p$')
    p[0].set_ylim([-.24,1.04])
    p[-1].set_xlabel('frequency (Hz)',fontsize='small')
    p2[-1].set_xlabel('$C_p$')

    # median observed
    x,y=graphical.baroutlinevals(freq,Cpm)
    hobs,=p[0].plot(x,y,color='darkblue')

    # initialize values for expected walkout
    xcg.signalfraction(np.mean(xc.amp,axis=2))
    frc = np.array([.15,.5,.85])
    Cplm,nperp,bnsp,trash = \
        phscoh.exprad(S=xcg.S[0,:],Cp=1.,
                      wgt=None,frc=None,Ntap=Ntap)

    # # estimate signal to noise ratio, again
    # ns = np.mean(ns,axis=3)
    # sg = np.abs(xc['amp'])
    # S = np.divide(sg,ns+sg)
    # S = np.power(S,0.5)
    # S = np.multiply(S[:,:,0],S[:,:,1])
    
    # alternatively, from comparing original with combined
    S = np.median(Ss,axis=2)

    # compute expected range of walkout values
    nperp=np.ndarray([nperp.size,freq.size])
    Cplm=np.ndarray([frc.size,freq.size])
    for k in range(0,freq.size):
        Cplm[:,k],nperp[:,k],bnsp,trash =  \
          phscoh.exprad(S=S[k,:],Cp=Cp[k],wgt=None,frc=frc,
                        Ntap=Ntap)

    # for binning
    bnx = (bnsp[0:-1]+bnsp[1:])/2.

    # normalize to a probability density
    nml=np.sum(nperp,axis=0)
    nml=nml*np.median(np.diff(bnsp))
    nml=nml.reshape([1,freq.size])
    nperp=np.divide(nperp,nml)

    for k in range(0,Nf):
        # original
        p2[k].plot(Cp[ifcheck[k]]*np.ones(2),[-2.,500.],
                   color='k')

        # histogram of predicted values
        x,y=graphical.baroutlinevals(bnx,nperp[:,ifcheck[k]])
        p2[k].plot(x,y,color='darkred')

        # in central 70%
        iok=np.logical_and(bnx>=Cplm[0,ifcheck[k]],
                           bnx<=Cplm[2,ifcheck[k]])
        if ~np.sum(iok):
            iok = general.closest(bnx,Cplm[0,ifcheck[k]])

        x,y=graphical.baroutlinevals(bnx[iok],nperp[iok,ifcheck[k]],
                                     wzeros=True)

        # maximum value
        npermax=np.max(np.append(nperp[:,ifcheck[k]],npermax))

        # create polygon
        ply = Polygon(np.vstack([x,y]).transpose())
        ply.set_edgecolor('none')
        ply.set_color('coral')
        ply.set_alpha(0.2)
        p2[k].add_patch(ply)

    # plot limits
    npermax=npermax*1.05
    ylb = np.round(npermax*0.8)
    for ph in p2:
        ph.set_ylim([0.,npermax])
        ph.set_yticks([0.,ylb])
    ylb=int(ifcheck.size/2)
    p2[ylb].set_ylabel('observation density')


    # plot signal fraction?
    #p[0].plot(freq,np.mean(S,axis=1),color='navy')

    # shade a region corresponding to expected values
    x1,y1=graphical.baroutlinevals(freq,Cplm[0,:])
    x2,y2=graphical.baroutlinevals(freq,Cplm[2,:])
    x2,y2=np.flipud(x2),np.flipud(y2)
    x,y=np.append(x1,x2),np.append(y1,y2)

    # create polygon
    ply = Polygon(np.vstack([x,y]).transpose())
    ply.set_edgecolor('none')
    ply.set_color('coral')
    ply.set_alpha(0.2)
    p[0].add_patch(ply)
    #import code
    #code.interact(local=locals())

    x,y=graphical.baroutlinevals(freq,Cplm[1,:])
    hpred,=p[0].plot(x,y,linestyle='-',color='darkred')

    plbl=np.append(p.flatten(),p2)
    graphical.delticklabels(p2.reshape([Nf,1]))
    graphical.delticklabels(p.reshape([2,1]))
    graphical.cornerlabels(plbl,loc='ul',fontsize='small')

    lbls = ['noise-free\n original',
            'observed\nwith noise',
            'predicted\nwith noise']
    lg0=p[0].legend([horig,hobs,hpred],lbls,
                    loc='lower left',fontsize='small')
    

    if prt:
        fname= 'PCnoisetest_'+eqtyp
        fname=os.path.join(os.environ['FIGURES'],fname+'.pdf')
        pp=PdfPages(fname)
        pp.savefig(f)
        pp.close()
        plt.close()
    else:
        plt.show()


def cohwnstat():

    Nstat = np.array([3,5,8,50])

    coh = np.linspace(0.,1.,30)
    N = 10000
    
    prc = np.array([0.5,0.15,0.85])


    Rkp = np.ndarray([len(prc),len(Nstat),len(coh)])
    Cpkp = np.ndarray([len(prc),len(Nstat),len(coh)])

    icoh = -1
    for ch in coh:
        icoh += 1
        ich = (1.-ch)**0.5

        
        # get random values

        # use a normal distribution?
        # ich = ich/2.**0.5
        # vl=np.random.randn(np.max(Nstat)*N)*ich
        # vl=vl+np.random.randn(np.max(Nstat)*N)*ich*1j

        # or not?
        vl=np.random.rand(np.max(Nstat)*N)*2.*math.pi
        vl=np.exp(1j*vl)
        vl=vl*ich

        # add the coherent values
        vl=vl+ch**0.5*np.ones(np.max(Nstat)*N)
        
        # normalize
        vl = np.divide(vl,np.abs(vl))


        #vl=vl.reshape([np.max(Nstat),N])


        istat = -1
        for Ns in Nstat:
            istat += 1
            iget = np.random.choice(np.max(Nstat),Ns)
            
            ii=int(vl.size/Ns)
            iprc = (ii*prc).astype(int)
            vli =vl[0:(ii*Ns)].reshape([Ns,ii]) 

            # compute walkout
            R = np.mean(vli,axis=0)
            R = np.abs(R)

            # and coherence
            Cp = 1./(Ns-1.)*(np.power(R,2)*Ns-1)

            # sort values
            R.sort()
            Cp.sort()

            # save percentiles
            Rkp[:,istat,icoh] = R[iprc]
            Cpkp[:,istat,icoh] = Cp[iprc]
            

    plt.close()
    f = plt.figure(figsize=(11,10.))
    gs,p=gridspec.GridSpec(1,2,width_ratios=[1,1.2]),[]
    gs.update(wspace=0.1,hspace=0.1)
    gs.update(left=0.08,right=0.98)
    gs.update(bottom=0.06,top=0.92)
    for gsi in gs:
        p.append(plt.subplot(gsi))
    p=np.array(p)
    pm=np.array(p).reshape([1,2])

    a = np.power(coh,0.5)
    b = np.power(1-coh,0.5)

    vrn = np.power(np.multiply(a,b),2)*2 
    vrn = vrn + np.power(b,4)
    vrn = vrn / 2

    cpp = np.power(a,4)
    cpp = np.divide(cpp,cpp+vrn)
    cpp = np.power(cpp,0.5)

    # create polygon
    xvl,yvl=0.6,0.64
    xvl=0.
    x=np.array([xvl,1,1,xvl,xvl])
    y=np.array([yvl,yvl,1,1,yvl])
    ply = Polygon(np.vstack([x,y]).transpose())
    ply.set_edgecolor('none')
    ply.set_color('lightgray')
    ply.set_alpha(1)
    p[1].add_patch(ply)


    # p[0].plot([0,1],[0,1],color='gray')
    # p[1].plot([0,1],[0,1],color='gray')
    # p[0].plot(coh,cpp,color='gray')
    # p[1].plot(coh,cpp,color='gray')
    cols = graphical.colors(len(Nstat))
    h,lbls=[],[]
    for istat in range(0,len(Nstat)):
        hh,=p[0].plot(coh,Rkp[0,istat,:],color=cols[istat],
                      label=str(Nstat[istat]))
        h.append(hh)
        lbls.append(str(Nstat[istat]))
        p[1].plot(coh,Cpkp[0,istat,:],color=cols[istat])

        for ip in range(1,len(iprc)):
            p[0].plot(coh,Rkp[ip,istat,:],color=cols[istat],
                      linestyle='--')
            p[1].plot(coh,Cpkp[ip,istat,:],color=cols[istat],
                      linestyle='--')
    lg=plt.legend(h,lbls,fontsize='small',loc='lower right')
    lg.set_title('number of stations')

    p[0].set_xlabel('coherent fraction')
    p[1].set_xlabel('coherent fraction')
    p[0].set_ylabel('phase walkout $R$')
    p[1].set_ylabel('phase coherence $C_p$')

    for ph in p:
        ph.set_aspect('equal')
        ph.set_xlim([-.25,1.])
    p[0].set_ylim([-0.25,1.03])
    p[1].set_ylim([-1.03,1.03])


    imax = np.argmax(Nstat)
    for k in range(0,Cpkp.shape[2]):
        print(str(coh[k])+','+str(Cpkp[0,imax,k]))
    
    graphical.cornerlabels(p,loc='lr',fontsize='small')
    graphical.printfigure('PCcohwnstat',f)
    
def cohmap(coh=None,Nstat=50):
    """
    :param     coh:  coherent fractions
    :param   Nstat:  number of stations
    :return   Cpkp:  set of median coherence values
    """
    
    Nstat = np.atleast_1d(Nstat)

    if coh is None:
        coh = np.linspace(0.,1.,51)
    else:
        coh = np.atleast_1d(coh)
    N = 5000
    
    prc = np.array([0.5,0.15,0.85])

    Rkp = np.ndarray([len(prc),len(Nstat),len(coh)])
    Cpkp = np.ndarray([len(prc),len(Nstat),len(coh)])

    icoh = -1
    for ch in coh:
        icoh += 1
        ich = (1.-ch)**0.5

        
        # get random values

        # use a normal distribution?
        # ich = ich/2.**0.5
        # vl=np.random.randn(np.max(Nstat)*N)*ich
        # vl=vl+np.random.randn(np.max(Nstat)*N)*ich*1j

        # or not?
        vl=np.random.rand(np.max(Nstat)*N)*2.*math.pi
        vl=np.exp(1j*vl)
        vl=vl*ich

        # add the coherent values
        vl=vl+ch**0.5*np.ones(np.max(Nstat)*N)
        
        # normalize
        vl = np.divide(vl,np.abs(vl))


        #vl=vl.reshape([np.max(Nstat),N])


        istat = -1
        for Ns in Nstat:
            istat += 1
            iget = np.random.choice(np.max(Nstat),Ns)
            
            ii=int(vl.size/Ns)
            iprc = (ii*prc).astype(int)
            vli =vl[0:(ii*Ns)].reshape([Ns,ii]) 

            # compute walkout
            R = np.mean(vli,axis=0)
            R = np.abs(R)

            # and coherence
            Cp = 1./(Ns-1.)*(np.power(R,2)*Ns-1)

            # sort values
            R.sort()
            Cp.sort()

            # save percentiles
            Rkp[:,istat,icoh] = R[iprc]
            Cpkp[:,istat,icoh] = Cp[iprc]

    imax = np.argmax(Nstat)            

    fdir=os.path.join(os.environ['PYFILES'],'PhaseCoherence')
    fname=os.path.join(fdir,'cohmap_'+str(Nstat[imax]))
    fl=open(fname,'w')
    for k in range(0,Cpkp.shape[2]):
        print(str(coh[k])+','+str(Cpkp[0,imax,k]))
        fl.write(str(coh[k])+','+str(Cpkp[0,imax,k])+'\n')
    fl.close()

    Cpkp=Cpkp[0,imax,:]

    return Cpkp

def testtaperr():
    
    coh = np.linspace(0.,0.8,4)
    coh = np.array([0.4,0.6,0.75,0.9])
    Nstat = 8

    import spectrum
    # window length
    wlen = 0.8
    dtim = 1./200
    N = int(round(wlen/dtim))
    Nf = N
    dfr = 1./wlen*4.
    freq = np.fft.rfftfreq(Nf,d=dtim)
    Nfh = len(freq)
    imid = np.argmin(np.abs(freq-50))

    # create tapers
    # decide on the tapers' concentration
    NW = dfr / (1./wlen) * 2

    # compute tapers
    [U,V] = spectrum.mtm.dpss(N,NW)
    U = U.reshape([N,1,U.shape[1]])
    Nt = len(V)

    U = U[:,:,0:-8]
    V = V[0:-8]
    Nt = len(V)

    print(str(V.size)+' tapers')
    

    prc = np.array([0.68,0.95,0.997])
    prc = np.array([0.2,0.3,0.4,0.5,0.6,0.68,0.95])
    prc = np.hstack([[0.5],0.5-prc/2,0.5+prc/2])
    prc = np.arange(0.05,1.,0.05)
    prc.sort()
    Ncheck = 600
    iprc = (prc*Ncheck).astype(int)

    # to keep track of values in range
    Nlarge = np.zeros([Nfh,len(prc),len(coh)]) 
    Nlarges = np.zeros([Nfh,len(prc),len(coh)]) 
    Nlargeb = np.zeros([Nfh,len(prc),len(coh)]) 

    Ntry = 100

    Ctrue = np.zeros([Ntry,len(coh)]) 
    Cbest = np.zeros([Ntry,len(coh)]) 
    Cpred = np.zeros([Ntry,len(prc),len(coh)]) 

    sfrc = 0.9
    ns = (1-sfrc)/(Nt-1)*4.
    ns = (1-sfrc**0.5)*4/(Nt-1)**0.5
    sfrcr = (1-sfrc)/sfrc
    #ns = 1-(1-4*sfrcr/Nt)**0.5
    ns = 1-1./(1+4*sfrcr/Nt+6*sfrcr**2/Nt)**0.5
    #ns = 1-1./(1+4*sfrcr/(Nt*3/4)+6*sfrcr**2/(Nt*3/4))**0.5


    isave = np.argmin(np.abs(coh-0.8))
    
    for n in range(0,len(coh)):
        # how much coherent energy
        pcoh = coh[n]

        for m in range(0,Ntry):
            
            # create some source time functions
            ts = np.arange(0,N).astype(float)*dtim
            tsdur = 0.01
            stf = np.exp(-(ts-0.1*N*dtim)**2/tsdur**2)
            stf1 = np.multiply(np.random.randn(N),stf).reshape([N,1])
            stf2 = np.multiply(np.random.randn(N),stf).reshape([N,1])

            # add a random component to the stf
            rstf1 = np.random.randn(N*Nstat).reshape([N,Nstat])
            rstf2 = np.random.randn(N*Nstat).reshape([N,Nstat])
            rstf1 = np.multiply(rstf1,stf.reshape([stf.size,1]))
            rstf2 = np.multiply(rstf2,stf.reshape([stf.size,1]))

            stf1 = stf1*pcoh**0.5 + rstf1*(1-pcoh)**0.5
            stf2 = stf2*pcoh**0.5 + rstf2*(1-pcoh)**0.5

            fstf1 = np.fft.rfft(stf1,2*N,axis=0)
            fstf2 = np.fft.rfft(stf2,2*N,axis=0)
            
            # and convolve with Green's functions
            Ng = Nstat
            vl1 = np.ndarray([N,Ng,1])
            vl2 = np.ndarray([N,Ng,1])
            
            tm = np.arange(0,N).astype(float)*dtim
            tm = tm - np.mean(tm)
            
            for k in range(0,Ng):
                gf = np.random.randn(N-int(N*0.3))
                fgf = np.fft.rfft(gf,2*N)
                vl1[:,k,0] = np.fft.irfft(np.multiply(fgf,fstf1[:,k]))[0:N]
                vl2[:,k,0] = np.fft.irfft(np.multiply(fgf,fstf2[:,k]))[0:N]


            # calculate a true coherence, with no tapering or noise
            f1 = np.fft.rfft(vl1,Nf,axis=0)
            f2 = np.fft.rfft(vl2,Nf,axis=0)

            # cross-correlate
            xc = np.multiply(f1,f2.conj())
            xc = xc[:,:,0]
            xc = np.divide(xc,np.abs(xc))
            R = np.abs(np.mean(xc,axis=1))
            Cptrue=1./(Nstat-1)*(Nstat*np.power(R,2)-1)

            # add noise
            st = np.std(vl1.flatten())
            rnd = np.random.randn(vl1.size).reshape(vl1.shape)*st
            vl1 = sfrc**0.5 * vl1 + (1-sfrc)**0.5 * rnd
            rnd = np.random.randn(vl1.size).reshape(vl1.shape)*st
            vl2 = sfrc**0.5 * vl2 + (1-sfrc)**0.5 * rnd

            # multiply by the tapers and get coefficients
            f1 = np.multiply(U,vl1)
            f2 = np.multiply(U,vl2)
            f1 = np.fft.rfft(f1,Nf,axis=0)
            f2 = np.fft.rfft(f2,Nf,axis=0)

            # cross-correlate
            xc = np.multiply(f1,f2.conj())

            # average over tapers
            xcm = np.mean(xc,axis=2)
            xcm = xcm.reshape([xcm.shape[0],xcm.shape[1],1])
            # rotate to this phase
            phs = np.divide(xcm,np.abs(xcm))

            # calculate uncertainties in these directions
            xch = np.multiply(xc,phs.conj())
            xcr = np.std(np.real(xch),axis=2)/(Nt-1)**0.5
            xci = np.std(np.imag(xch),axis=2)/(Nt-1)**0.5
            phs = phs.reshape(xcr.shape)

            # average over tapers
            xca = np.mean(xc,axis=2)
            xcm = np.multiply(xca,phs.conj())

            # variation in amplitude
            ampfin=np.power(np.abs(xca),2)
            ampor=np.mean(np.power(np.abs(xc),2),axis=2)
            ampred=np.divide(ampor-ampfin,ampfin)/(Nt-1.)
            Cred=1.-2.*np.mean(np.power(ampred,0.5),axis=1)

            # and percentiles
            Rprc = np.ndarray([Nfh,Ncheck],dtype=float)
            for ku in range(0,Ncheck):
                xch=xcm+np.multiply(np.random.randn(Nfh*Nstat).reshape([Nfh,Nstat]),xcr)
                xch=xch+np.multiply(np.random.randn(Nfh*Nstat).reshape([Nfh,Nstat]),xci)*1j
                xch=np.multiply(xch,phs)

                # bootstrap over tapers?
                ii = np.random.choice(Nt,Nt)
                #ii = np.random.permutation(Nt)[0:Nt*3/4]
                xch = np.mean(xc[:,:,ii],axis=2)

                # compute walkout
                xch=np.divide(xch,np.abs(xch))
                Rprc[:,ku]=np.abs(np.mean(xch,axis=1))
            Cpprc=1./(Nstat-1)*(Nstat*np.power(Rprc,2)-1)

            #Cpprc = np.divide(Cpprc,Cred.reshape([Cred.size,1]))

            # average over tapers
            xc = np.mean(xc,axis=2)
            xc = np.divide(xc,np.abs(xc))
            R = np.abs(np.mean(xc,axis=1))
            Cpbest=1./(Nstat-1)*(Nstat*np.power(R,2)-1)

            
            # save for later
            Ctrue[m,n] = Cptrue[imid]
            Cbest[m,n] = Cpbest[imid]
            Cpprci = Cpprc[imid,:]
            Cpprci.sort()
            Cpred[m,:,n] = Cpprci[iprc]
            
            # adjust for noise
            Cpprc = Cpprc / (1-ns)

            # what percentile Cptrue falls in
            #Cptrue = cohmap(coh=pcoh,Nstat=Nstat)
            Cptrue=Cptrue.reshape([Cptrue.size,1])
            bgr=Cptrue>=Cpprc
            # fraction of the values that are smaller than the true value
            bgr=np.sum(bgr,axis=1)/float(Ncheck)
            # is Cptrue in each of these percentiles?
            bgr=bgr.reshape([bgr.size,1])<prc.reshape([1,prc.size])
            Nlarge[:,:,n] = Nlarge[:,:,n]+bgr

            bgr=Cptrue>=Cpprc+0.05
            # fraction of the values that are smaller than the true value
            bgr=np.sum(bgr,axis=1)/float(Ncheck)
            # is Cptrue in each of these percentiles?
            bgr=bgr.reshape([bgr.size,1])<prc.reshape([1,prc.size])
            Nlarges[:,:,n] = Nlarges[:,:,n]+bgr

            bgr=Cptrue>=Cpprc-0.05
            # fraction of the values that are smaller than the true value
            bgr=np.sum(bgr,axis=1)/float(Ncheck)
            # is Cptrue in each of these percentiles?
            bgr=bgr.reshape([bgr.size,1])<prc.reshape([1,prc.size])
            Nlargeb[:,:,n] = Nlargeb[:,:,n]+bgr
            
            # compute phase walkout, Cp
            xc=np.divide(xc,np.abs(xc))
            R=np.abs(np.mean(xc,axis=1))
            Cp=1./(Nstat-1)*(Nstat*np.power(R,2)-1)

            # save if desired
            if n==isave:
                Cpprcs = Cpprc
                Cptrues = Cptrue

    # normalize to percentage time
    Nlarge = Nlarge / float(Ntry)
    Nlarges = Nlarges / float(Ntry)
    Nlargeb = Nlargeb / float(Ntry)


    plt.close()
    f = plt.figure(figsize=(5.,12.5))
    gs,p=gridspec.GridSpec(3,1,width_ratios=[1]),[]
    gs.update(wspace=0.1,hspace=0.18)
    gs.update(left=0.16,right=0.95)
    gs.update(bottom=0.06,top=0.98)
    for gsi in gs:
        p.append(plt.subplot(gsi))
    p=np.array(p)
    pm=np.array(p).reshape([3,1])

    cols = graphical.colors(len(coh))
    p[2].plot([0,1],[0,1],color='gray')
    h,lbls=[],[]
    for k in range(0,len(coh)):
        hh,=p[2].plot(prc,Nlarge[imid,:,k],marker='o',linestyle='none',
                   color=cols[k])
        h.append(hh)
        hh,=p[2].plot(prc,Nlargeb[imid,:,k],marker='x',linestyle='--',
                      color=cols[k])
        hh,=p[2].plot(prc,Nlarges[imid,:,k],marker='x',linestyle='--',
                      color=cols[k])
        lbls.append("{0:0.2g}".format(coh[k]))
    p[2].set_ylabel('fraction of true values\nbelow percentile')
    p[2].set_xlabel('uncertainty percentile')
    p[2].text(0.2,0.9,'uncertainty\nboundaries\noverestimated',
              horizontalalignment='center',fontsize='small',verticalalignment='top')
    p[2].text(0.8,0.02,'uncertainty\nboundaries\nunderestimated',
              horizontalalignment='center',fontsize='small')
    p[2].set_aspect('equal')
    pcp = [0.5]


    p[1].plot([-5,5],[-5,5],color='gray',linewidth=1)
    p[1].plot([-5,5],np.array([-5,5])*(1-ns),
              color='gray',linewidth=1)
    ct = np.linspace(0,1,300)
    cta = np.power(ct,2)-np.power(ct,2)*4*(1-sfrc)/Nt
    cta = np.maximum(cta,0.)
    cta = np.power(cta,0.5)
    #p[1].plot(ct,cta,color='gray',linewidth=1)
    
    xmin = [1.]

    for pc in pcp:
        ii = np.argmin(np.abs(prc-pc))
        h,lbls=[],[]
        for k in range(0,len(coh)):
            #hh,=p[1].plot(Ctrue[:,k],Cpred[:,ii,k],marker='o',linestyle='none',
            #color=cols[k])
            hh,=p[1].plot(Ctrue[:,k],Cbest[:,k],marker='o',linestyle='none',
                          color=cols[k])
            h.append(hh)
            lbls.append("{0:0.2g}".format(coh[k]))
            xmin=np.min(np.append(xmin,Ctrue[:,k]))
            xmin=np.min(np.append(xmin,Cpred[:,ii,k]))
    p[1].set_xlim([xmin-0.03,1.03])
    p[1].set_ylim(p[1].get_xlim())
    p[1].set_xlabel('true phase coherence')
    p[1].set_ylabel('estimated phase coherence')
    p[1].set_aspect('equal')

    plt.sca(p[1])
    lg=plt.legend(h,lbls,loc='lower right',fontsize='small')
    lg.set_title('coherent fraction',prop={'size':'small'})

    Cpprcs = Cpprcs[imid,:]
    Cpprcs.sort()
    pcp = np.array([0.05,0.15,0.5,0.85,0.95])
    iprc = (pcp*Cpprcs.size).astype(int)
    for ii in iprc:
        p[0].plot(Cpprcs[ii]*np.ones(2),[-1,Cpprcs.size],color='k',
                  linestyle=':')
    bns = np.std(Cpprcs)*np.array([-4.,4.])+np.mean(Cpprcs)
    bns = np.minimum(bns,1.01)
    bns = np.linspace(bns[0],bns[1],30)
    nper,bnsi,other=p[0].hist(Cpprcs,bins=bns)
    p[0].set_ylim([0,np.max(nper)*1.13])
    oset = np.std(Cpprcs)*0.07
    for pci in pcp:
        k = np.argmin(np.abs(pcp-pci))
        shf = (k % 2) * (-np.max(nper)*.05)
        lbl="{0:0.0f}".format(100.*pcp[k])+'%'
        p[0].text(Cpprcs[iprc[k]]+oset,np.max(nper)*1.07+shf,lbl,fontsize='small')
    p[0].set_xlim(general.minmax(bns))
    p[0].plot(Cptrues[imid]*np.ones(2),[-1,Cpprcs.size],color='k',
              linestyle='-',linewidth=1.5)
    p[0].set_xlabel('estimated phase coherence')
    p[0].set_ylabel('number of bootstrap values')

    graphical.cornerlabels(p,'ul','small',xscl=0.02)


    fname = 'PCtesttaper_'+"{0:0.2g}".format(sfrc)
    fname = fname.replace('.','p')
    graphical.printfigure(fname,f)


#------------FOR SUMMARY FIGURES----------------------------------------    


def ffallrng(typs=['het','bwm','circ'],vrupts=None,shfs=None,rtime=None,
             rdfrc=[0.],rdlm=[100,500]):
    """
    :param      typs: types of rupture
    :param    vrupts: rupture velocity relative to wavespeed
    :param      shfs: shifts of start location relative to the center point
    :param     rtime: rise times
    :param     rdfrc: radii fractions to calculate for
    :return    ffall: picked falloff frequencies
    :return       fc: best-fitting corner frequencies
    :return      rd1: larger radii
    :return      rd2: smaller radii
    :return      typs: types of rupture
    :return    vrupts: rupture velocity relative to wavespeed
    :return      shfs: shifts of start location relative to 
                          the center point
    :return     rtime: rise times
    """

    if vrupts is None:
        vrupts=np.exp(np.linspace(np.log(0.07),np.log(1.1),8))
    if shfs is None:
        shfs=np.array([0.,1.])
    if rtime is None:
        rtime=np.exp(np.linspace(np.log(0.05),np.log(2),6))

    ffall=np.ndarray([len(typs),len(vrupts),len(shfs),len(rtime),len(rdfrc)],
                     dtype=float)
    fc=np.ndarray([len(typs),len(vrupts),len(shfs),len(rtime),len(rdfrc)],
                  dtype=float)
    rd1=np.ndarray([len(typs),len(vrupts),len(shfs),len(rtime),len(rdfrc)],
                   dtype=float)
    rd2=np.ndarray([len(typs),len(vrupts),len(shfs),len(rtime),len(rdfrc)],
                   dtype=float)

    # for mapping
    tlbls={'bwm':'boatwright-m','hetb':'boatwright-m',
           'het':'circrise','circ':'circrise','hetnoshf':'circrise'}
    slipdists={'bwm':None,'hetb':'elliptical-fractal',
               'het':'elliptical-fractal-shiftmean','circ':'elliptical',
               'hetnoshf':'elliptical-fractal'}

    # number of stations
    Ns=31
    shstat = 180./float(Ns)/2.

    rdlm = np.log(rdlm)
    vprop=6.e3
                                
    # duration of the Green's functions
    tgf = 3.

    for kt in range(0,len(typs)):
        tlbl=tlbls.get(typs[kt])
        slipd=slipdists.get(typs[kt])
        for kv in range(0,len(vrupts)):
            vrupti=vrupts[kv]*vprop
            for kf in range(0,len(rdfrc)):
                for ks in range(0,len(shfs)):
                    xyst1=np.array([shfs[ks],0.])
                    xyst2=np.array([0.,0.])
                    for kr in range(0,len(rtime)):
                        if typs[kt]!='bwm' or (ks==0 and kr==0):

                            # limit the radius
                            rmax=np.log(tgf/2.*vrupti/np.minimum(2.,rtime[kr]))
                            if rmax<=rdlm[0]:
                                rdlmi=[rmax-np.log(3),rmax]
                            else:
                                rdlmi=[rdlm[0],np.minimum(rdlm[1],rmax)]
                                
                            # pick radii
                            rd1i=np.random.rand(1)*np.diff(rdlm)+rdlmi[0]
                            rd2i=np.random.rand(1)*(rd1i-rdlm[0])+rdlmi[0]
                            rd1i,rd2i=np.exp(rd1i[0]),np.exp(rd2i[0])*0.6
                            if rdfrc[kf]>0:
                                rd2i=rd1i*rdfrc[kf]
                            rd1[kt,kv,ks,kr,kf]=rd1i
                            rd2[kt,kv,ks,kr,kf]=rd2i

                            if typs[kt] in ['bwm','circ']:
                                rndedg=[0.,0]
                            else:
                                rndedg=[0.2,5]
                                
                            # pick a set of azimuths and distances
                            stk=np.linspace(0.,360.,Ns+1)[0:-1]
                            stk=stk+(np.random.rand(Ns)-0.5)*2.*float(shstat)
                            
                            # initialize earthquakes
                            seq1=syntheq.seq(a=rd1i,ruptev=tlbl,slipdist=slipd,
                                             xyst=xyst1*rd1i,
                                             vrupt=vrupti,vprop=vprop,
                                             rndedg=rndedg)
                            seq2=syntheq.seq(a=rd2i,ruptev=tlbl,slipdist=slipd,
                                             xyst=xyst2*rd2i,
                                             vrupt=vrupti,vprop=vprop,
                                             rndedg=rndedg)
                            seq1.rtime=seq1.a/seq1.vrupt*rtime[kr]
                            seq2.rtime=seq2.a/seq2.vrupt*rtime[kr]

                            # make the slip distributions
                            seq1.makeslipdist()
                            seq2.makeslipdist()

                            # timing for everything
                            dtim=np.minimum(rd1i,rd2i)/vprop/30
                            dtim=np.maximum(dtim,1.e-4)

                            # and some fake Green's functions---same for all
                            seq1.initobs(strike=stk,takeang=90.,dtim=dtim)
                            seq2.initobs(strike=stk,takeang=90.,dtim=dtim,
                                         justloc=True)
                            seq1.initfakegf(dtim=dtim,tdec=tgf,tlen=tgf*5)
                            seq2.gf = seq1.gf.copy()
                            seq2.gftim = seq1.gftim.copy()
                            

                            # to be analysed
                            tdur = seq1.a/seq1.vrupt*\
                                   np.maximum(2,seq1.rtime*seq1.vrupt/seq1.a)
                            if tdur < tgf/2.:

                                # calculate apparent source time functions
                                seq1.calcappstf(dtim=dtim)
                                seq2.calcappstf(dtim=dtim)
                                
                                # waveforms
                                st1 = seq1.obswaveforms()
                                st2 = seq2.obswaveforms()
                                
                                # get cross-correlation
                                trange=[-.1,np.max([1.5,rd1i/seq1.vrupt,
                                                    seq1.rtime])*2]
                                trange=-.1+np.array([0.,tgf*1.5])

                                # highpass filter
                                hpf=2./np.diff(trange)[0]
                                (st1+st2).filter('highpass',freq=hpf)
                                
                                try:
                                    xc=phscoh.calcxc(st1,st2,trange,mk1='t0',mk2='t0',
                                                     nsint=None,fmax=300.,dfres=1.)    
                                    xc.calcmvout()
                                except:
                                    import code
                                    code.interact(local=locals())

                                # preferred falloff frequency
                                xc.pickffreq(cpcutoff=0.5)
                                ffall[kt,kv,ks,kr,kf]=xc.ffbest

                                # corner frequency
                                fci=earthquakefc(seq1)
                                fc[kt,kv,ks,kr,kf]=fci
                            else:
                                # garbage for this calculation
                                ffall[kt,kv,ks,kr,kf]=float('nan')
                                fc[kt,kv,ks,kr,kf]=float('nan')
                        else:
                            # just copy
                            ffall[kt,kv,ks,kr,kf]=ffall[kt,kv,0,0,kf]
                            fc[kt,kv,ks,kr,kf]=fc[kt,kv,0,0,kf]
                            rd1[kt,kv,ks,kr,kf]=rd1[kt,kv,0,0,kf]
                            rd2[kt,kv,ks,kr,kf]=rd2[kt,kv,0,0,kf]


    return ffall,fc,rd1,rd2,typs,vrupts,shfs,rtime


def ffallwrad():

    vrupts=0.8/3**0.5*np.ones(2)
    rdfrc=np.array([0.05,0.2,0.5,0.7,0.8,0.9])
    rdfrc1=np.linspace(np.log(0.05),np.log(0.6),25)
    rdfrc1=np.exp(rdfrc1)[0:-1]
    rdfrc2=np.linspace(np.log(0.6),np.log(0.95),15)
    rdfrc=np.append(rdfrc1,np.exp(rdfrc2))
    #rdfrc=np.array([0.05,0.2,0.5,0.7,0.8,0.9])
    typs=['het','hetnoshf','circ','bwm']
    ffall,fc,rd1,rd2,typs,vrupts,shfs,rtime=\
        ffallrng(typs=typs,rdfrc=rdfrc,
                 vrupts=vrupts,shfs=[0],rtime=[0.2],
                 rdlm=[30,800])

    return ffall,fc,rd1,rd2,typs,vrupts,shfs,rtime,rdfrc


def ffallrdplot(ffall,rd1,rd2,typs,vrupts,shfs,rtime,rdfrc):
    """
    :param     ffall: picked falloff frequencies
    :param       rd1: larger radii
    :param       rd2: smaller radii
    :param      typs: types of rupture
    :param    vrupts: rupture velocity relative to wavespeed
    :param      shfs: shifts of start location relative to the center point
    :param     rtime: rise times
    :param     rdfrc: radii fraction
    """

    fs,fs2='x-large','large'

    Nt,Ns = len(typs),len(shfs)

    if Nt==4:
        Nx,Ny=2,2
    else:
        Nx,Ny=1,Nt
    plt.close()
    f = plt.figure(figsize=(5.*Ny,5.5*Nx))
    gs,p=gridspec.GridSpec(Nx,Ny),[]
    gs.update(wspace=0.04,hspace=0.03)
    gs.update(left=0.07,right=0.9)
    gs.update(bottom=0.05,top=0.99)
    for gsi in gs:
        p.append(plt.subplot(gsi))
    p=np.array(p)
    pm=np.array(p).reshape([Nx,Ny])
    for ph in p.flatten():
        ph.set_aspect('equal')
    
    Np=rdfrc.size
    No=int(rd1.size/Np/Nt)
    cols=graphical.colors(Np)
    cmap=plt.get_cmap('Reds')

    cls=np.log(rdfrc)

    pcs = []
    scls = []
    rmax=900
    for kt in range(0,Nt):
        plt.sca(p[kt])
        r1,r2=rd1[kt,:,:,:,:],rd2[kt,:,:,:,:]
        f1=ffall[kt,:,:,:,:].reshape([No,Np])
        r1,r2=r1.reshape([No,Np])*2,r2.reshape([No,Np])*2
        h=[]
        for kf in range(0,Np):
            iok=r1[:,kf]<rmax
            clsi=cls[kf]*np.ones(r1.shape[0],dtype=float)
            hh=plt.scatter(r1[iok,kf],f1[iok,kf],c=clsi[iok],
                           vmin=cls[0],vmax=0.,s=50)
        #                    cmap=cmap,s=60)
        # plt.set_cmap('Reds')

        ps=p[kt].get_position()
        wd,ht=ps.width*.45,ps.height*.25
        psi=[ps.x1-wd,ps.y1-ht,wd,ht]
        pci=plt.axes(psi)

        i1=np.logical_and(r2<0.8*r1,r1<rmax)
        bins=np.linspace(np.log(0.4),np.log(5),25)
        rt = np.multiply(f1,r1)/6e3
        print('MEDIAN, STD:')
        print(np.exp(np.median(np.log(rt[i1]))))
        scls.append(np.exp(np.median(np.log(rt[i1]))))
        mdn=np.median(rt[i1])
        print(np.exp(np.std(np.log(rt[i1]))))
        i1=np.logical_and(r2<0.6*r1,r1<rmax)
        N,bns=np.histogram(np.log(rt[i1]),bins=bins)
        x1,y1=graphical.baroutlinevals(np.exp(bns),N,wzeros=True)
        ply = Polygon(np.vstack([x1,y1]).transpose())
        ply.set_edgecolor('none')
        ply.set_color('blue')
        ply.set_alpha(0.5)
        ply.set_zorder(10)
        hb=pci.add_patch(ply)
        i1=np.logical_and(r2>0.6*r1,r2<0.8*r1)
        i1=np.logical_and(i1,r1<rmax)
        N2,bns=np.histogram(np.log(rt[i1]),bins=bins)
        x2,y2=graphical.baroutlinevals(np.exp(bns),N2,wzeros=False,yi=N)
        ply = Polygon(np.vstack([x2,y2]).transpose())
        ply.set_edgecolor('none')
        ply.set_color('green')
        ply.set_alpha(0.5)
        ply.set_zorder(10)
        hb2=pci.add_patch(ply)
        i1=np.logical_and(r2>0.8*r1,r1<rmax)
        N3,bns=np.histogram(np.log(rt[i1]),bins=bins)
        x3,y3=graphical.baroutlinevals(np.exp(bns),N3,wzeros=False,yi=N+N2)
        ply = Polygon(np.vstack([x3,y3]).transpose())
        ply.set_edgecolor('none')
        ply.set_color('gold')
        ply.set_alpha(0.5)
        ply.set_zorder(10)
        hb3=pci.add_patch(ply)
        pci.set_xlim(general.minmax(np.exp(bns)))
        pci.set_ylim([0,np.max(y3)*1.05])
        pci.set_xscale('log')
        pci.set_xlabel('$f_f D/V_p$',fontsize=fs2)
        pci.set_yticks([0,np.max(N)])
        tks = np.append(np.arange(0.1,1,.1),np.arange(1,10))
        tks = tks[np.logical_and(tks>=np.exp(bns[0]),tks<=np.exp(bns[-1]))]
        tkls = np.array(['{:g}'.format(tk) for tk in tks])
        tkls[tks<1] = ''
        pci.set_xticks(tks)
        pci.set_xticklabels(tkls)
        #pci.set_xticks([])
        plt.setp(pci.get_xminorticklabels(), visible=False)
        pcs.append(pci)
        pci.plot(mdn*np.ones(2),pci.get_ylim(),
                 color='dimgray',alpha=0.7)


    xvl = p[-1].get_position().x1+0.007
    psc=[xvl,0.25,0.013,0.5]
    cbs = f.add_axes(psc)
    tks=np.log([0.1,1])
    tks=np.log(np.append([0.05],np.arange(0.1,1.01,0.1)))
    tkls=['{:g}'.format(tk) for tk in np.exp(tks)]
    cb = f.colorbar(hh,cax=cbs,orientation='vertical',
                    ticklocation='right',ticks=tks)
    cbs.tick_params(axis='y',labelsize=fs2)
    cb.set_label('$d_2/d_1$',fontsize=fs2)
    cb.set_ticklabels(tkls)
    #cbs.set_ylim(np.log([0.05,1]))
    xlm = general.minmax(cbs.get_xlim(),3)
    xlm = [-.2,1.2]
    yvl = (np.log(0.8)-cb.vmin)/(cb.vmax-cb.vmin)
    cbs.plot(xlm,np.ones(2)*yvl,color='k',linewidth=3,clip_on=False)
    yvl = (np.log(0.6)-cb.vmin)/(cb.vmax-cb.vmin)
    cbs.plot(xlm,np.ones(2)*yvl,color='k',linewidth=3,clip_on=False)
    #cbs.axhspan(xmin=-.5,xmax=1.5,ymin=0.67,ymax=0.73)
            
    scl=6.e3
    rlm=general.minmax(rd1.flatten())*2.
    rlm[1]=np.minimum(rlm[1],rmax)
    rlm=np.exp(general.minmax(np.log(rlm),1.1))
    #rlm2=np.divide(scl,general.minmax(ffall.flatten()))
    #rlm=np.exp(general.minmax(np.log(np.append(rlm,rlm2)),1.1))
    flm=general.minmax(ffall.flatten()[rd1.flatten()*2<rmax])
    flm=np.exp(general.minmax(np.log(flm),1.1))
    flm=[flm[0],215]

    xvl,yvl=rlm[0]*2,flm[0]*2.
    yvl=np.exp(general.minmax(np.log(flm),0.96)[0])
    xvl=np.exp(general.minmax(np.log(rlm),0.85)[0])
    dcts = {'het':'heterogeneous slip',
            'bwm':'Boatwright-Madariaga',
            'circ':'elliptical slip',
            'hetnoshf':'heterogeneous slip,\ntruncated zero-mean distribution'}
    for k in range(0,len(typs)):
        p[k].text(xvl,yvl,dcts.get(typs[k]),
                  horizontalalignment='left',
                  verticalalignment='bottom',
                  fontsize=fs2)

    for k in range(0,len(p.flatten())):
        ph = p.flatten()[k]
        ph.set_xscale('log')
        ph.set_yscale('log')
        #ph.set_aspect('equal')
        ph.set_xlim(rlm)
        ph.set_ylim(flm)
        ph.set_ylim(np.divide(scl*scls[k],Cplm).sort())
        ph.set_xlabel('larger event diameter $d_1$ (m)',fontsize=fs)
        ph.plot(rlm,1.*np.divide(scl,rlm),linestyle='--',color='k',
                zorder=1)
        ph.xaxis.set_tick_params(labelsize=fs2)
        ph.yaxis.set_tick_params(labelsize=fs2)

        

    for ph in pm[:,0]:
        ph.set_ylabel('$f_f$ (Hz)',fontsize=fs)

    graphical.delticklabels(pm)
    pc = np.vstack([p,pcs]).T.flatten()
    graphical.cornerlabels(pc,'ll',fontsize=fs2,xscl=0.02)
    fname='PCffallwrad'
    if not np.isinf(rmax):
        fname=fname+'_{:0.0f}'.format(rmax)
    graphical.printfigure(fname,f)
    

def ffallplot(ffall,rd1,rd2,typs,vrupts,shfs,rtime):
    """
    :param     ffall: picked falloff frequencies
    :param       rd1: larger radii
    :param       rd2: smaller radii
    :param      typs: types of rupture
    :param    vrupts: rupture velocity relative to wavespeed
    :param      shfs: shifts of start location relative to the center point
    :param     rtime: rise times
    """


    # for labelling
    lbls={'bwm':'Boatwright-Madariaga model',
          'hetb':'Boatwright-Madariaga, with hetereogeneity',
          'circ':'constant rupture speed and rise time, elliptical slip',
          'het':'constant rupture speed and rise time, heterogeneous slip',
          'bwm-fast':'Boatwright-Madariaga, with faster propagation',
          'hetb-fast':'Boatwright-Madariaga, with hetereogeneity and faster propagation',
          'circ-fast':'constant propagation and rise time, with faster propagation',
          'het-fast':'constant propagation and rise time, with heterogeneity and faster propagation',
          'bwm-slow':'Boatwright-Madariaga, with slower propagation',
          'hetb-slow':'Boatwright-Madariaga, with hetereogeneity and slower propagation',
          'circ-slow':'constant propagation and rise time, with slower propagation',
          'het-slow':'constant propagation and rise time, with heterogeneity and slower propagation',
          'bwm-shift':'Boatwright-Madariaga, with directivity',
          'hetb-shift':'Boatwright-Madariaga, with hetereogeneity and directivity',
          'circ-shift':'constant propagation and rise time, with directivity',
          'het-shift':'constant propagation and rise time, with heterogeneity and directivity'}
    
    fs,fs2='x-large','large'

    Nt,Ns = len(typs),len(shfs)
    rtime = np.atleast_1d(rtime)

    hrt=np.log(1.2/0.1)/np.log(2/0.2)
    
    plt.close()
    f = plt.figure(figsize=(9,4.*Nt))
    gs,p=gridspec.GridSpec(Nt,2,width_ratios=[hrt,1]),[]
    gs.update(wspace=0.1,hspace=0.03)
    gs.update(left=0.06,right=0.99)
    gs.update(bottom=0.07,top=0.93)
    for gsi in gs:
        p.append(plt.subplot(gsi))
    p=np.array(p).reshape([Nt,2])


    # normalize by radius
    ffalls = np.multiply(ffall,rd1)/(0.5*6.e3)
    ffalls = np.ma.masked_array(ffalls,mask=np.isnan(ffalls))
    ffalls = np.ma.median(ffalls,axis=4)

    # ffall=ffall[:,:,:,:,0]
    # rd1,rd2=rd1[:,:,:,:,0],rd2[:,:,:,:,0]


    # pick rise times to plot
    irt = np.array([rtime.size/2])
    irt=np.round(np.linspace(0,rtime.size-1,3))
    irt=general.closest(rtime,[0.2,0.45,2])
    irt=np.unique(irt.astype(int))
    lnst = ['-','--','-.',':']

    # pick rupture velocities to plot
    irv = np.array([vrupts.size/2])
    irv=np.round(np.linspace(0,vrupts.size-1,3))
    irv=general.closest(vrupts,[0.1,0.8/3**0.5,1])
    irv=np.unique(irv.astype(int))

    cols = graphical.colors(Ns)

    # plot against rupture velocity
    h=[]
    for kt in range(0,Nt):
        for ks in range(0,Ns):
            for kr in range(0,len(irt)):
                hh,=p[kt,0].plot(vrupts,ffalls[kt,:,ks,irt[kr]],
                                 linestyle=lnst[kr],color=cols[ks])
                h.append(hh)
    h=np.array(h).reshape([Nt,Ns,len(irt)])



    # plot against rise time
    h2=[]
    for kt in range(0,Nt):
        for ks in range(0,Ns):
            for kv in range(0,len(irv)):
                hh,=p[kt,1].plot(rtime,ffalls[kt,irv[kv],ks,:],
                                 linestyle=lnst[kv],color=cols[ks])
                if typs[kt]=='circ' and shfs[ks]!=0:
                    hh.set_visible('off')
                h2.append(hh)

                
    h2=np.array(h2).reshape([Nt,Ns,len(irv)])



    ht=[]
    for kt in range(0,Nt):
        if typs[kt] in ['bwm']:
            plt.sca(p[kt,0])
        else:
            plt.sca(p[kt,1])
        ps1=p[kt,0].get_position()
        ps2=p[kt,1].get_position()
        x=(ps1.x1+ps2.x0)/2.
        y=ps1.y0+ps1.height*0.92
        hh=plt.text(x,y,lbls.get(typs[kt],''),
                    transform=plt.gcf().transFigure,
                    horizontalalignment='center',
                    verticalalignment='center',
                    bbox={'facecolor':'white','alpha':1},
                    fontsize=fs)
        ht.append(ht)

    shlb = ['{:g}'.format(shf) for shf in shfs]
    lg = p[0,1].legend(h2[0,:,0],shlb,loc='lower right',fontsize=fs2)
    lg.set_title('start point / radius')
    plt.setp(lg.get_title(),fontsize=fs2)
    lg.set_zorder(30)

    ps=p[0,1].get_position()
    ax = p[0,1].add_artist(lg)
    ax.set_bbox_to_anchor((ps.x0,ps.y0),transform=plt.gcf().transFigure)
                
    ylm=general.minmax(ffalls[:,:,:,irt])
    ylm=general.minmax(np.append(ylm,general.minmax(ffalls[:,irv,:,:])))
    ylm=np.exp(general.minmax(np.log(ylm),1.1))
    ylm=np.array([0.3,2.5])
    
    if len(irt)>1:
        rlbl = ['{:g}'.format(np.round(rtime[kr],2)) for kr in irt]
        lg1=p[0,0].legend(h[0,0,:],rlbl,loc='upper left',
                          fontsize=fs2)
        lg1.set_title('$t_r$ / $(R/V_r)$')

    if len(irv)>1:
        rlbl = ['{:g}'.format(np.round(vrupts[kr],2)) for kr in irv]
        lg2=p[0,1].legend(h2[0,0,:],rlbl,loc='upper right',
                          fontsize=fs2)
        lg2.set_title('$V_r$ / $V_p$')

    ytk=np.append(np.arange(0.3,1,0.1),np.arange(1,3.1))
    ytk=ytk[np.logical_and(ytk>=ylm[0],ytk<=ylm[1])]
    ytkl=['{:g}'.format(ytki) for ytki in ytk]
    for k in range(0,len(ytk)):
        if not ytk[k] in [0.3,0.5,1.,2]:
            ytkl[k]=''

    for ph in p.flatten():
        ph.set_xscale('log')
        ph.set_ylim(ylm)
        ph.set_yscale('log')
        ph.set_yticks(ytk)
        ph.minorticks_off()
        #ph.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ph.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ph.set_yticklabels(ytkl)
        ph.xaxis.set_tick_params(labelsize=fs2)
        ph.yaxis.set_tick_params(labelsize=fs2)



    for ph in p[:,0]:
        ph.set_xlim(general.minmax(vrupts))
        ph.set_xlim([0.1,np.max(vrupts)])
        ph.set_ylabel('$f_f$ / $(V_p/D)$',fontsize=fs)
        ph.set_xlabel('$V_r$ / $V_p$',fontsize=fs)
        # ph.plot(general.minmax(vrupts),1.*(general.minmax(vrupts)/0.4),
        #         color='gray')
        ph.plot(general.minmax(vrupts),[1.,1],color='gray',linestyle=':',zorder=1)
        rct=ph.axvspan(0.3,1,0,1)
        rct.set_color('gray')
        rct.set_alpha(0.3)


    for ph in p[:,1]:
        ph.set_xlim(general.minmax(rtime))
        ph.set_xlim([0.2,general.minmax(rtime)[1]])
        ph.set_xlabel('$t_r$ / $(R/V_r)$',fontsize=fs)

    for ph in p.flatten():
        ph.set_aspect('equal')
    

    graphical.cornerlabels(p[0,:],'ll',fontsize=fs2,xscl=0.03,yscl=0.03)
    graphical.cornerlabels(p[1,0:1],'ll',fontsize=fs2,xscl=0.03,lskip=2,yscl=0.03)
    graphical.cornerlabels(p[2,:],'ll',fontsize=fs2,xscl=0.03,lskip=3,yscl=0.03)

    ity, = np.where([typ in ['bwm'] for typ in typs])
    for ii in ity:
        p[ii,1].set_visible('off')
        p[ii,1].remove()
    h2i=h[ity,:,1:].flatten()
    for hh in h2i:
        hh.remove()


    hhi = h2[np.array(typs)=='circ',np.array(shfs)!=0,:]
    hhj = h[np.array(typs)=='circ',np.array(shfs)!=0,:]
    for hh in np.append(hhi.flatten(),hhj.flatten()):
        hh.remove()

    graphical.delticklabels(p)    
    
    graphical.printfigure('PCffallplot',f,pngback=False)

#-------------TEST TAPERING UNCERTAINTY------------------------

def testtap(nsa=0.05,tmax=2.,smlr=0.7):

    # to a noise amplitude
    nsamp = (nsa/(1-nsa))**0.5

    fsb,fs,fs2='x-large','large','medium'
    rd1=300.
    rd2=rd1*smlr
    slipd='elliptical-fractal'

    # pick a set of azimuths and distances
    Ns=10
    shstat = 180./float(Ns)/2.
    stk=np.linspace(0.,180.,Ns)
    stk=stk+(np.random.rand(Ns)-0.5)*2.*float(shstat)

    # initialize earthquakes
    seq1=syntheq.seq(a=rd1,ruptev='circrise',slipdist=slipd,rndedg=[0.2,5])
    seq2=syntheq.seq(a=rd2,ruptev='circrise',slipdist=slipd,rndedg=[0.2,5])

    # timing for everything
    dtim=np.minimum(rd1,rd2)/seq1.vrupt
    dtim=0.005

    # and some fake Green's functions---same for all
    seq1.initobs(strike=stk,takeang=90.,dtim=dtim,justloc=False)
    seq2.initobs(strike=stk,takeang=90.,dtim=dtim,justloc=True)
    seq2.gf = seq1.gf.copy()
    seq2.gftim = seq1.gftim.copy()

    # make the slip distributions
    seq1.makeslipdist()
    seq2.makeslipdist()

    # set the GF to a spike
    ix=np.argmin(np.abs(seq1.gftim))
    seq1.gf=np.zeros(seq1.gf.shape,dtype=float)
    seq2.gf=np.zeros(seq2.gf.shape,dtype=float)
    seq1.gf[ix,:]=1.
    seq2.gf[ix,:]=1.

    # calculate apparent source time functions
    seq1.calcappstf(dtim=dtim)
    seq2.calcappstf(dtim=dtim)
    
    # waveforms
    st1 = seq1.obswaveforms()
    st2 = seq2.obswaveforms()
        
    # coherence
    #trange=[-.1,tmax]
    trange=[-.1,0.9]
    dfres=2.
    xcs=phscoh.calcxc(st1,st2,trange,mk1='t0',mk2='t0',
                      nsint=None,fmax=300.,dfres=2)    
    xcs.calcmvout()

    # time range
    trange=[-.1,tmax]
    hpf = 2./np.diff(trange)[0]

    # save the coherences
    Cps=[]

    for k in range(0,200):
        # new Green's functions
        seq1.initfakegf(dtim=dtim,tdec=2.,tlen=20.)
        seq2.gf = seq1.gf.copy()
        seq2.gftim = seq1.gftim.copy()

        # calculate apparent source time functions
        seq1.calcappstf(dtim=dtim)
        seq2.calcappstf(dtim=dtim)
        
        # waveforms
        st1 = seq1.obswaveforms()
        st2 = seq2.obswaveforms()

        st1=syntheq.addnoise(st1,trange=trange,nrat=nsamp,
                             pdec='same',flma=[hpf,hpf*5])
        st2=syntheq.addnoise(st2,trange=trange,nrat=nsamp,
                             pdec='same',flma=[hpf,hpf*5])

        
        # filter
        (st1+st2).filter('highpass',freq=hpf)
        
        # coherence
        xc=phscoh.calcxc(st1,st2,trange,mk1='t0',mk2='t0',
                         nsint=None,fmax=300.,dfres=dfres)    
        xc.calcmvout()
        
        # save the values
        Cps.append(xc.Cp)

    # make a grid
    prc=np.array([0.05,0.95])
    Cps = np.array(Cps)
    Cps.sort(axis=0)
    iprc=(prc*Cps.shape[0]).astype(int)

    # bootstrap percentages
    xc.calcrprc(rprc=prc)

    plt.close()
    f = plt.figure(figsize=(12,8))
    gs,p=gridspec.GridSpec(2,1,height_ratios=[0.25,1]),[]
    gs.update(wspace=0.1,hspace=0.13)
    gs.update(left=0.08,right=0.54)
    gs.update(bottom=0.07,top=0.98)
    pc=plt.subplot(gs[1])
    ps=plt.subplot(gs[0])

    fpk=np.array([np.maximum(1.,np.ceil(hpf*1.2)),7.,15.,40])
    Nf=len(fpk)
    gs,p=gridspec.GridSpec(Nf,1),[]
    gs.update(wspace=0.1,hspace=0.03)
    gs.update(left=0.61,right=0.99)
    gs.update(bottom=0.07,top=0.98)
    for gsi in gs:
        p.append(plt.subplot(gsi))
    p=np.array(p).reshape([Nf,1])

    x=np.append(xc.freq,np.flipud(xc.freq))
    y=np.append(xc.Cplim[:,0],np.flipud(xc.Cplim[:,1]))
    ply = Polygon(np.vstack([x,y]).transpose())
    ply.set_edgecolor('none')
    ply.set_color('blue')
    ply.set_alpha(0.5)
    hb=pc.add_patch(ply)

    x=np.append(xc.freq,np.flipud(xc.freq))
    y=np.append(Cps[iprc[0],:],
                np.flipud(Cps[iprc[1],:]))
    ply = Polygon(np.vstack([x,y]).transpose())
    ply.set_edgecolor('none')
    ply.set_color('red')
    ply.set_alpha(0.5)
    hr=pc.add_patch(ply)
    
    ht,=pc.plot(xcs.freq,xcs.Cp,color='k',
                linewidth=2)
    pc.set_xscale('log')
    pc.set_xlim([hpf,100])
    pc.set_ylim([-.54,1.04])

    bns=np.linspace(-0.5,1.02,100)
    ifreq=general.closest(xc.freq,fpk)
    ifreqs=general.closest(xcs.freq,fpk)
    for kf in range(0,Nf):
        cp=xc.Rrng[ifreq[kf],:]
        cp = 1./(Ns-1)*(np.power(cp,2)*Ns-1.)
        Nb,trash=np.histogram(cp,bins=bns)
        Nb=Nb.astype(float)/np.sum(Nb)/np.diff(bns)[0]
        Nbm=np.max(Nb)
        x,y=graphical.baroutlinevals(bns,Nb)
        
        ply = Polygon(np.vstack([x,y]).transpose())
        ply.set_edgecolor('none')
        ply.set_color('blue')
        ply.set_alpha(0.5)
        p[kf,0].add_patch(ply)

        cp=Cps[:,ifreq[kf]]
        Nb,trash=np.histogram(cp,bins=bns)
        Nb=Nb.astype(float)/np.sum(Nb)/np.diff(bns)[0]
        Nbms=np.max(Nb)
        x,y=graphical.baroutlinevals(bns,Nb)
        
        ply = Polygon(np.vstack([x,y]).transpose())
        ply.set_edgecolor('none')
        ply.set_color('red')
        ply.set_alpha(0.5)
        p[kf,0].add_patch(ply)

        p[kf,0].set_xlim(general.minmax(bns))
        ymx=np.maximum(Nbm,Nbms)*1.05
        p[kf,0].set_ylim([0.,ymx])
        ytks=ymx*0.9
        ytk=general.roundsigfigs(ymx*0.9,1)
        while ytk>ymx*0.9:
            ytks=ytks*0.9
            ytk=general.roundsigfigs(ytks,1)
        p[kf,0].set_yticks([0,ytk])
        
        p[kf,0].plot(xcs.Cp[ifreqs[kf]]*np.ones(2),
                     p[kf,0].get_ylim(),color='black',
                     linestyle='-',zorder=1,
                     linewidth=2)
        p[kf,0].text(-.4,ymx*0.9,'{:g}'.format(fpk[kf])+' Hz',
                       horizontalalignment='left',
                       verticalalignment='center',
                       fontsize=fs2)

        pc.plot(fpk[kf]*np.ones(2),[-3,3],color='gray',
                linestyle='-.',zorder=1)

    tm=st1[0].times()-st1[0].stats.t0
    ps.plot(tm,st1[0].data/np.max(np.abs(st1[0].data)),color='green')
    tm=st2[0].times()-st2[0].stats.t0
    ps.plot(tm,st2[0].data/np.max(np.abs(st2[0].data)),color='yellow')
    ps.set_xlim([-.5,4])
    ps.set_xlabel('time (s)',fontsize=fs2)
    ps.set_ylim([-1.1,1.1])
    ps.set_yticks([])
    ps.set_xticks(np.arange(-1.,5))
    x=[trange[0],trange[0],trange[1],trange[1],trange[0]]
    y=[-5,5,5,-5,-5]
    ply = Polygon(np.vstack([x,y]).transpose())
    ply.set_edgecolor('none')
    ply.set_color('lightgray')
    ply.set_alpha(1)
    ps.add_patch(ply)


    pc.plot(pc.get_xlim(),[0,0],color='gray',
            linestyle=':',zorder=1)
    pc.set_ylabel('$C_p$',fontsize=fsb)
    p[-1,0].set_xlabel('$C_p$',fontsize=fsb)

    lbls=['true: from ASTFs',
          'seismogram realizations',
          'bootstrapped tapers']
    lg=pc.legend([ht,hr,hb],lbls,fontsize=fs,
                 loc='lower left')

    p[Nf/2,0].set_ylabel('probability density',fontsize=fs)
    for ph in p.flatten():
        ph.xaxis.set_tick_params(labelsize=fs2)
        ph.yaxis.set_tick_params(labelsize=fs2)
    ps.yaxis.set_tick_params(labelsize=fs2)
    ps.xaxis.set_tick_params(labelsize=fs2)
    pc.yaxis.set_tick_params(labelsize=fs2)
    pc.xaxis.set_tick_params(labelsize=fs2)
    pc.set_xlabel('frequency (Hz)',fontsize=fs)
    
    graphical.cornerlabels([ps],'ul',fontsize=fs2,
                           xscl=0.02,yscl=0.04)
    graphical.cornerlabels([pc],'ul',fontsize=fs2,lskip=1,
                           xscl=0.02,yscl=0.05)
    graphical.cornerlabels(p.flatten(),'ul',fontsize=fs2,lskip=2,
                           xscl=0.02,yscl=0.04)
    graphical.delticklabels(p)

    tlbl='{:g}'.format(trange[0])+'_'+'{:g}'.format(trange[1])
    tlbl=tlbl+'_ns'+'{:g}'.format(nsa)
    tlbl=tlbl+'_rd2'+'{:g}'.format(smlr)
    tlbl=tlbl.replace('.','p')
    graphical.printfigure('PCtesttap_'+tlbl,f)



def predcoh():

    plt.close()
    f = plt.figure(figsize=(12,8))
    gs,p=gridspec.GridSpec(1,1),[]
    gs.update(wspace=0.1,hspace=0.13)
    gs.update(left=0.08,right=0.54)
    gs.update(bottom=0.07,top=0.98)
    p=plt.subplot(gs[0])

    phi = np.linspace(-180,180,1000)
    dphi = np.linspace(-180,180,1000)

    phir = phi*np.pi/180
    dphir = dphi*np.pi/180
    
    phirm,dphirm=np.meshgrid(phir,dphir)

    vl = np.multiply(1+0.5*np.exp(1j*phirm),
                     1+0.5*np.exp(-1j*(phirm+dphirm)))
    vl = np.divide(np.real(vl),np.abs(vl))

    h=p.pcolormesh(phi,dphi,vl,vmin=-1,vmax=1)

    cbs = f.add_axes([0.9,0.2,0.05,0.6])
    cb = f.colorbar(h,cax=cbs,orientation='vertical',
                    ticklocation='right')
    cbs.tick_params(axis='y',labelsize=9)
    cb.set_label('$C_p$',fontsize=9)


def plotaztest(pr0=None,pr1=None,pr2=None,pr3=None,ttl=''):

    # collect the values
    pr0,pr1,pr2,pr3=collectpairs(pr0,pr1,pr2,pr3)
    prs=[pr0,pr1,pr2,pr3]

    fs='large'
    plt.close()
    f = plt.figure(figsize=(8,10))
    gs,p=gridspec.GridSpec(2,1),[]
    gs.update(wspace=0.1,hspace=0.17)
    gs.update(left=0.15,right=0.95)
    gs.update(bottom=0.1,top=0.88)
    p=plt.subplot(gs[0])
    p2=plt.subplot(gs[1])

    cols=graphical.colors(4)
    mks=['o','s','^','d']
    vprop=8040
    h=[]
    xlm=np.hstack([2*pr[0] for pr in prs])
    xlm=np.exp(general.minmax(np.log(xlm),1.1))
    ylm=np.hstack([np.multiply(pr[1],pr[0]*2/vprop) for pr in prs])
    ylm=np.exp(general.minmax(np.log(ylm),1.1))

    bns=np.linspace(0.8,1.8,50)
    
    for k in range(0,4):
        rad,ffreq=prs[k][0],prs[k][1]
        iok=np.logical_and(rad>100,rad<2000.)
        rad,ffreq=rad[iok],ffreq[iok]

        rt=np.multiply(ffreq,rad*2/vprop)
        mds=[]
        for m in range(0,100):
            ii=np.random.choice(rt.size,rt.size,replace=True)
            mds.append(np.median(rt[ii]))
        mds.sort()
        mds=np.array(mds)
        mdfrc=np.interp(np.array([0.025,0.5,0.975])*mds.size,
                        np.arange(0.,mds.size),mds)
        N,bn=np.histogram(mds,bins=bns)
        x,y=graphical.baroutlinevals(bn,N,wzeros=False)
        p2.plot(x,y,color=cols[k])
        print(mdfrc)
        
        hh,=p.plot(rad*2,np.multiply(ffreq,rad*2/vprop),
                   marker=mks[k],linestyle='none',color=cols[k])
        h.append(hh)
        p.plot(xlm,np.ones(2)*np.median(np.multiply(ffreq,rad*2/vprop)),
               color=cols[k],linestyle='--')

    p.set_xscale('log')
    p.set_yscale('log')
    p.set_xlim(xlm)
    p.set_title(ttl)
    p.set_xlabel('diameter (m)',fontsize=fs)
    p.set_ylabel('$f_f / (V_p / D)$',fontsize=fs)
    p.set_ylim(ylm)
    p.xaxis.set_tick_params(labelsize=fs)
    p.yaxis.set_tick_params(labelsize=fs)
    p2.set_xlim(general.minmax(bns))
    p2.set_xlabel('$f_f / (V_p / D)$',fontsize=fs)
    
    
    lg=p.legend(h,['0','1','2','3'])
    
    graphical.printfigure('PCjoshaztest',f)
    
    return

def plotjoshwrad(pr):

    fs=16
    plt.close()
    f = plt.figure(figsize=(5.5,12))
    gs,p=gridspec.GridSpec(2,1,height_ratios=[2,1]),[]
    gs.update(wspace=0.1,hspace=0.22)
    gs.update(left=0.15,right=0.97)
    gs.update(bottom=0.1,top=0.97)
    p=plt.subplot(gs[0])
    p2=plt.subplot(gs[1])

    rad,ffreq=pr[0],pr[1]
    rad,rad2=rad[:,0],rad[:,1]

    iok = np.logical_and(rad>300.,rad<2500)
    rad,ffreq,rad2=rad[iok],ffreq[iok],rad2[iok]

    diam=2.*rad
    vprop=8040.
    rt=np.multiply(ffreq,rad*2/vprop)
    h=[]

    bns=np.linspace(0.2,6.,30)
    bns=np.exp(np.linspace(np.log(0.2),np.log(8.),25))
    
    mds=[]
    for m in range(0,100):
        ii=np.random.choice(rt.size,rt.size,replace=True)
        mds.append(np.median(rt[ii]))
    mds.sort()
    mds=np.array(mds)
    mdfrc=np.interp(np.array([0.025,0.5,0.975])*mds.size,
                    np.arange(0.,mds.size),mds)
    N,bn=np.histogram(rt,bins=bns,normed=False)
    x,y=graphical.baroutlinevals(bn,N,wzeros=False)
    p2.plot(x,y,color='black',zorder=2,linewidth=2)

    yst=np.max(y)*np.array([-0.05,1.2])
    p2.plot(np.ones(2)*1.2,yst,zorder=1,color='dimgray',
            linewidth=3)
    p2.set_ylim(yst)
    print(mdfrc)


    cl=np.log10(np.divide(rad2,rad))
    clm=np.log10(np.array([0.1,1.01]))
    hh=p.scatter(rad*2,ffreq, #np.multiply(ffreq,rad*2/vprop),
                 marker='o',c=cl,zorder=2,vmin=clm[0],vmax=clm[1])
    h.append(hh)

    print(np.median(np.multiply(ffreq,rad*2/vprop)))

    xlm=np.exp(general.minmax(np.log(diam),1.3))
    xlm=np.array([600,5100])
    ylm=np.exp(general.minmax(np.log(ffreq),1.1))
    ylm=np.array([1,40])

    yvl=1.2*np.divide(vprop,xlm)
    p.plot(xlm,yvl,color='dimgrey',zorder=1,linestyle='-',linewidth=1.5)


    p.set_ylim(ylm)
    p.set_xlim(xlm)
    p.set_xscale('log')
    p.set_yscale('log')
    p.set_xlabel('larger earthquake diameter $D$ (km)',fontsize=fs)
    p.set_ylabel('$f_f (Hz)$',fontsize=fs)
    xtk=np.array([600,700,800,900,1000,2000,3000,4000,5000,6000])
    xtkl=['','','','','1','2','','4','','']
    minorLocator = matplotlib.ticker.AutoMinorLocator(0)
    p.set_xticks(xtk)
    p.set_xlim(xlm)
    p.xaxis.set_minor_locator(minorLocator)
    #p.minorticks_off('x')
    p.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())    
    p.set_xticklabels(xtkl)
    p.set_aspect('equal')

    xtk=np.unique(np.hstack([np.arange(0.1,1,0.1),np.arange(1.,10,1)]))
    xtk=xtk[np.logical_and(xtk>=np.min(bns),xtk<=np.max(bns))]
    xlb=np.array([0.5,1,2,4])
    xtkl=['']*len(xtk)
    for k in range(0,len(xtk)):
        if xtk[k] in xlb:
            xtkl[k] = '{:g}'.format(xtk[k])
    print(xtkl)


    p.xaxis.set_tick_params(labelsize=fs)
    p.yaxis.set_tick_params(labelsize=fs)
    p2.xaxis.set_tick_params(labelsize=fs)
    p2.yaxis.set_tick_params(labelsize=fs)
    p2.set_xscale('log')
    p2.set_xticks(xtk)
    p2.set_xlim(general.minmax(bns))
    p2.xaxis.set_minor_locator(minorLocator)
    #p.minorticks_off('x')
    p2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())    
    p2.set_xticklabels(xtkl)
    p2.set_xlabel('$f_f / (V_p / D)$',fontsize=fs)
    p2.set_ylabel('number of simulated pairs',fontsize=fs)

    ps=p.get_position()
    p.set_position([ps.x0*0.3,ps.y0,ps.width,ps.height])
    ps=p.get_position()
    cax=plt.axes([ps.x1-0.1,ps.y0+ps.height*0.1,0.03,ps.height*0.8])
    tks = np.log10(np.array([0.1,0.2,0.5,1.]))

    tkl = ['{:0.1g}'.format(10**tk) for tk in tks]
    cb = f.colorbar(h[0],cax=cax,orientation='vertical',
                    ticklocation='right',ticks=tks)
    cax.tick_params(axis='y',labelsize=fs-5)
    cb.set_ticks(tks)
    cb.set_clim([np.log10(0.1),np.log10(1.01)])
    cb.set_ticklabels(tkl)
    cb.set_label('smaller earthquake diameter / larger earthquake diameter',
                 fontsize=fs-4)

    xtk=np.arange(0.5,2,0.25)
    xtkl=['']*len(xtk)
    print(xtk)
    for k in range(0,len(xtk)):
        if xtk[k] % 0.5 ==0:
            xtkl[k] = '{:g}'.format(xtk[k])
    #p2.set_xticks(xtk)
    #p2.set_xticklabels(xtkl)
    graphical.cornerlabels([p],loc='ll',fontsize=fs-2,xscl=0.03)    
    graphical.cornerlabels([p2],loc='ul',fontsize=fs-2,xscl=0.02,
                           yscl=0.04,lskip=1)    
    
    graphical.printfigure('PCffreqwdiam',f)
    
    return



def writeazpairs(pr0=None,pr1=None,pr2=None,pr3=None):

    # collect the values
    pr0,pr1,pr2,pr3=collectpairs(pr0,pr1,pr2,pr3)
    prs=[pr0,pr1,pr2,pr3]

    fname=os.path.join('STUDENTS','Josh','azimuthfrequencies')
    fl=open(fname,'w')
    fl.write('For each station group, the first line gives the diameters of the synthetic events in meters\n')
    fl.write('The second line gives f_f * the diameter / V_p\n')
    fl.write('\n')
    vprop=8040
    
    for k in range(0,4):
        rad,ffreq=prs[k][0],prs[k][1]
        iok=np.logical_and(rad<100,rad<1000.)
        rad,ffreq=rad[iok],ffreq[iok]

        ffreq=np.multiply(ffreq,rad*2/vprop)

        fl.write('Station group '+str(k)+'\n')
        dwrt=','.join([str(radi*2) for radi in rad])
        fwrt=','.join([str(ff) for ff in ffreq])
        fl.write(dwrt+'\n')
        fl.write(fwrt+'\n')
        
    fl.close()

def collectpairs(pr0=None,pr1=None,pr2=None,pr3=None):


    if pr2 is None:
        rad,ffreq,seqs,xc,ffreq2=joshaztest(ix=2)
        pr2=[rad,ffreq,ffreq2]

    if pr0 is None:
        rad,ffreq,seqs,xc,ffreq2=joshaztest(ix=0)
        pr0=[rad,ffreq,ffreq2]

    if pr1 is None:
        rad,ffreq,seqs,xc,ffreq2=joshaztest(ix=1)
        pr1=[rad,ffreq,ffreq2]

    if pr3 is None:
        rad,ffreq,seqs,xc,ffreq2=joshaztest(ix=3)
        pr3=[rad,ffreq,ffreq2]

    return pr0,pr1,pr2,pr3


def converttakeoffs(tkang,Vpu=5.8,Vpl=8.04):
    """
    modify the takeoff angles to mimic what would 
    happen for PmP
    :param      tkang: actual takeoff angles in degrees
    :param        Vpu: Vp in upper layer    
    :param        Vpl: Vp in lower layer    
    :return    tkangn: new takeoff angles in degrees
    :return       Vpa: an average propagation velocity
    """

    # the change in travel time for a horizontal offset x is x/Vpl
    # the change in travel time for a vertical offset y is y/(Vpu*cos(tkang))

    # we'll make it so that 
    # the change in travel time for a horizontal offset x is x/(Vpa*sin(tkangn))
    # the change in travel time for a vertical offset y is y/(Vpa*cos(tkangn))

    tkang=np.pi/180*tkang
    
    # first find the new takeoff angle
    tkangn = np.arctan(1/np.cos(tkang)*Vpu/Vpl)

    # and the new average velocity
    Vpa = Vpu*np.cos(tkangn)/np.cos(tkang)

    # to degrees
    tkangn=tkangn*180/np.pi
    
    return tkangn,Vpa


def joshaztest(evtnum=1,ix=3):
    
    # get the azimuths
    dct = joshazimuth.getprops()

    # pick one of the events
    evtnum=str(int(evtnum))
    stk = dct['evt'+evtnum+'azimuths']
    tkg = dct['evt'+evtnum+'takeoffs']
    Ns = len(stk[ix])
    stks = np.vstack([stk[ix],tkg[ix]]).T

    arrival = 'S'
    jstinfo = np.load('stinfo'+arrival+'wave.npy')            
    stk = np.array(jstinfo[:,2],dtype='float32').astype(float)
    stk = stk % 360
    Ns = len(stk)
    tkg = np.array(jstinfo[:,3],dtype='float32').astype(float)
    stks = np.vstack([stk,tkg]).T

    # pick one of the station sets

    #stks[:,0] = np.random.rand(len(stk[ix]))*180.


    isrand=True
    if isrand:
        shstat = 360./float(Ns)/2.
        stks[:,0]=np.linspace(0.,360.,Ns+1)[0:-1]
        stks[:,0]=stks[:,0]+(np.random.rand(Ns)-0.5)*2.*float(shstat)
        stks[:,0]=np.random.rand(Ns)*360.
        stks[:,1] = 90.

    # choose a new takeoff angle---one for the whole set
    usetake=False
    if usetake:
        tkangn,Vpa=converttakeoffs(np.median(tkg[ix]))
        stks[:,1]=tkangn
        vrupt = 0.8/3**0.5*(5.8/8.04)
    else:
        Vpa = 8.040
        vrupt = 0.8/3**0.5
        vrupt = 0.8
    print(stks)

    # time range
    tgf=4.
    trange=[-.1,tgf*1.5]
    trange=[-.1,5.]
    hpf = 2./np.diff(trange)[0]
    dfres = 2./np.diff(trange)[0]

    ffreq=[]
    ffreq2=[]
    rad=300*np.exp(np.random.rand(10)*np.log(20))
    print(rad)
    rad2=np.array([],dtype=float)
    ctr=0
    for radi in rad:
        ctr=ctr+1
        print('Event {:d}: {:f}'.format(ctr,radi))

        if isrand:
            stks[:,0]=np.random.rand(Ns)*360.
            stks[:,0]=np.linspace(0.,360.,Ns+1)[0:-1]
            stks[:,0]=stks[:,0]+(np.random.rand(Ns)-0.5)*2.*float(shstat)


        # time spacing
        dtim=radi*.4/(Vpa*1000)/30.
        dtim=np.maximum(dtim,1.e-4)
        
        secrad=0.1*np.exp(np.random.rand(1)[0]*np.log(0.95/0.1))
        print(secrad*radi)
        rad2=np.append(rad2,secrad*radi)
        # compute synthetics
        seqs,seqs2=appstfcalc(justsetup=True,N=Ns,eqtyp='het',rad=[radi],
                              secrad=secrad,shstat=0.,vrupts=vrupt,stk=stks,
                              vprop=Vpa*1000)
        # compute GF
        setupseqs(seqs+seqs2,tgf=tgf,dtim=dtim)

        print(seqs[0].ststrike)
        print(seqs[0].sttakeang)

        # waveforms
        st1 = seqs[0].obswaveforms()
        st2 = seqs2[0].obswaveforms()

        # filter
        (st1+st2).filter('highpass',freq=hpf)
        
        # coherence
        xc=phscoh.calcxc(st1,st2,trange,mk1='t0',mk2='t0',
                         nsint=None,fmax=300.,dfres=dfres)    

        #xc.tix=np.arange(0,xc.Ntap*3/4).astype(int)
        xc.calcmvout()
        xc.cpcutoff = 0.5
        xc.pickffreq()

        ffreq.append(xc.ffbest)
        ffreq2.append(xc.ffspl)

    ffreq=np.array(ffreq)
    ffreq2=np.array(ffreq2)
    rad=np.vstack([rad,rad2]).T

    return rad,ffreq,seqs,xc,ffreq2


def calcsourcespec(seq):
    """
    :param    seq: a synthetic earthquake
    :return  spec: spectra for each station
    :return  freq: frequencies
    :return aspec: station-averaged spectra
    """

    # time spacing and maximum frequency
    dtim=np.diff(seq.tstf())[0]
    fmax=np.minimum(100./seq.calcsdur(),0.4/dtim)

    # time windows
    cummom = seq.amom
    Ns = cummom.shape[1]
    tmom = cummom[-1,:]
    cummom = np.divide(cummom,tmom.reshape([1,Ns]))
    cummom = np.mean(cummom,axis=1)

    # find percentiles
    prc=0.999
    prc = np.array([-1.,1.])*0.5*prc+0.5
    tlm = np.interp(prc,cummom,seq.tmom)
    tlm = general.minmax(tlm,1.8)
    tlm[0] = np.maximum(tlm[0],seq.tmom[0])
    tlm[1] = np.minimum(tlm[1],seq.tmom[-1])

    # window to consider
    ix=np.logical_and(seq.tstf()>=tlm[0],seq.tstf()<=tlm[1])
    N=np.sum(ix)

    # decide on the tapers' concentration
    dfres=np.maximum(0.5/np.diff(tlm)[0],0.25/seq.calcsdur())
    NW = dfres / (1./np.diff(tlm)[0]) * 2

    # compute tapers
    [U,V] = spectrum.mtm.dpss(N,NW)

    # just select some?
    ii = V>=0.99
    U,V = U[:,ii],V[ii]

    # to initialize spectrum
    Nf=N*2
    freq=np.fft.rfftfreq(Nf,d=dtim)
    ifreq=np.logical_and(freq>0,freq<=fmax)
    freq=freq[ifreq]

    spec=np.ndarray([len(freq),Ns],dtype=float)
    
    # compute a spectrum for each station
    astf=seq.astf()[ix,:]
    for k in range(0,Ns):
        speci=astf[:,k:k+1]/tmom[k]
        speci=np.fft.rfft(np.multiply(U,speci),n=Nf,axis=0)
        speci=np.mean(np.power(np.abs(speci[ifreq,:]),2),axis=1)
        spec[:,k]=speci

    aspec=np.exp(np.mean(np.log(spec),axis=1))

    cspec=seq.cstf()[ix]/seq.cmom[-1]
    cspec=cspec.reshape([cspec.size,1])
    cspec=np.fft.rfft(np.multiply(U,cspec),n=Nf,axis=0)
    cspec=np.mean(np.power(np.abs(cspec[ifreq,:]),2),axis=1)

    # normalize
    ii=np.zeros(freq.size,dtype=bool)
    flm=0.5/seq.calcsdur()
    while np.sum(ii)<np.minimum(4,len(freq)):
        ii=freq<=flm
        nml=np.median(aspec[ii])
        nml=np.median(cspec[ii])
        flm=flm*1.5

    spec=spec/nml
    aspec=aspec/nml
    cspec=cspec/nml

    return spec,freq,cspec,aspec


def pickfc(spec,freq,fclm,flm=None,n=2):
    """
    :param   spec: spectra
    :param   freq: frequencies
    :param   fclm: limits on corner frequencies to try
    :param    flm: limits on frequencies to check
    :param      n: high-frequency falloff
    """

    # only some frequencies?
    if flm is not None:
        ix=np.logical_and(freq>=flm[0],freq<=flm[1])
        freq,spec=freq[ix],spec[ix]

    # corner frequencies to try
    fctry=np.exp(np.linspace(np.log(fclm[0]),np.log(fclm[1]),100))

    # keep track of misfits
    msft=np.ndarray(fctry.size,dtype=float)

    # log spectra
    lspec=np.log(spec)
    
    for k in range(0,len(fctry)):
        # predicted values
        pspec,trash=predspec(freq=freq,fc=fctry[k],n=n)

        # difference
        df=np.abs(lspec-np.log(pspec))

        #weighted by 1/frequency because of spacing
        df=np.divide(df,freq)

        # save
        msft[k]=np.sum(df)

    # minimum
    imin=np.argmin(msft)
    fc=fctry[imin]
        
    return fc,msft,fctry
        


def predspec(freq=None,fc=10,n=2,p=1):
    """
    :param     freq: frequencies
    :param       fc: corner frequency
    :param        n: high-frequency decay rate
    :param        p: an extra scaling factor
    :return   pspec: predicted POWER spectra
    :return    freq: frequencies
    """

    # create frequencies if not given
    if freq is None:
        freq=np.exp(np.linspace(np.log(fc/30),np.log(fc*30),3000))

    # predicted spectra
    pspec=np.power(1+np.power(freq/fc,p*n),-2/p)

    return pspec,freq


def earthquakefc(seq,pl=True):
    """
    :param       seq: a synthetic earthquake
    :param        pl: plot the resulting spectra and fit
    :return       fc: best-fitting corner frequency
    """

    n=2
    
    # compute spectra
    spec,freq,cspec,aspec=calcsourcespec(seq)

    # limits for fitting
    flm=np.array([0.1,20])/seq.calcsdur()

    # limits for fc
    fclm=np.array([0.5,10])/seq.calcsdur()
    
    # corner frequencies
    fc,msft,fctry=pickfc(aspec,freq,fclm=fclm,flm=flm,n=n)

    if pl:
        p = plt.axes()
        p.loglog(freq,spec)

        p.loglog(freq,aspec,color='k')

        for fci in fc*np.array([.3,1,3]):
            pspec,freqi=predspec(freq,fci,n=n)
            p.plot(freqi,pspec,color='gray',linestyle='--',linewidth=2)
    
    return fc


def simpstart(typs='bwm',Ns=11,vrupt=0.8*3**0.5):
    
    if typs in ['bwm','circ']:
        rndedg=[0.,0]
    else:
        rndedg=[0.2,5]
                                
    # pick a set of azimuths and distances
    shstat = 180./float(Ns)/2.
    stk=np.linspace(0.,360.,Ns+1)[0:-1]
    stk=stk+(np.random.rand(Ns)-0.5)*2.*float(shstat)

    tlbls={'bwm':'boatwright-m','hetb':'boatwright-m',
           'het':'circrise','circ':'circrise','hetnoshf':'circrise'}
    slipdists={'bwm':None,'hetb':'elliptical-fractal',
               'het':'elliptical-unsmoothed-fractal-shiftmean','circ':'elliptical',
               'hetnoshf':'elliptical-unsmoothed-fractal'}
    tlbl=tlbls.get(typs)
    slipd=slipdists.get(typs)

    
    # initialize earthquakes
    vprop=6000
    vrupt=vprop*vrupt
    rad=300.
    seq=syntheq.seq(a=rad,ruptev=tlbl,slipdist=slipd,
                     xyst=[0,0],vrupt=vrupt,vprop=vprop,
                     rndedg=rndedg)
    seq.rtime=seq.a/seq.vrupt*0.4

    seq.slipdist=slipd
    seq.makeslipdist()

    seq.initobs(takeang=90,strike=stk)

    seq.calcappstf()


    return seq

