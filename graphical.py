import matplotlib.pyplot as plt
import general
import numpy as np
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.backends.backend_pdf import PdfPages
import os
from scipy.interpolate import interp1d
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors

def atleast10(ylm):
    """
    :param   ylm: an initial set of limits
    :return ylma: limits adjusted to include at least two round values of 10
    """

    ylma = np.array(ylm).copy()
    ylml = np.log10(ylm)
    
    # within one factor of 10, set to the limits
    lm = [np.floor(ylml[0]),np.ceil(ylml[1])]
    if lm[1]-lm[0] == 1.:
        ylml = np.array(lm)
        ylma = np.power(10.,ylml)
    else:
        # are interior values the same?
        lm = [np.ceil(ylml[0]),np.floor(ylml[1])]
        if lm[1]-lm[0]<0.5:
            if ylml[0]-np.floor(ylml[0]) < np.ceil(ylml[1])-ylml[1] and ylml[0]>np.floor(ylml[0]):
                ylml[0] = lm[0]-1.
            else:
                ylml[1] = lm[1]+1.
            ylma = np.power(10.,ylml)

    return ylma


def findedge(x,y,xlm,ylm,xlog=False,ylog=False):
    """
    :param     x: x-values being plotted
    :param     y: y-values being plotted
    :param   xlm: x-limits
    :param   ylm: y-limits
    :param  xlog: is x-axis on log scale?
    :param  ylog: is y-axis on log scale?
    """

    if xlog:
        x = np.log(x)
        xlm = np.log(xlm)

    if ylog:
        y = np.log(y)
        ylm = np.log(ylm)

    # be sure to go from left to right
    ix = np.argsort(x)
    x,y=x[ix],y[ix]

    # unless interpolating from y-values
    iy = np.argsort(y)
    xs,ys=x[iy],y[iy]
    
    # left edge first
    if x[0]<=xlm[0]:
        # if values are given past the left edge
        yi = np.interp(xlm[0],x,y)
        if yi<ylm[0]:
            # if the extrapolation would be below the plot
            yi = ylm[0]
            xi = np.interp(ylm[0],ys,xs)
        elif yi>ylm[1]:
            # if the extrapolation would be above the plot
            yi = ylm[1]
            xi = np.interp(ylm[1],ys,xs)
        else:
            # if it's in range
            xi = xlm[0]
    else:
        # if there's nothing that small, find the leftmost y-value on the plot
        if y[0]>np.mean(ylm):
            yi = np.maximum(y[0],ylm[1])
        else:
            yi = np.minimum(y[0],ylm[0])
        xi = np.interp(yi,ys,xs)

    # save left values
    xyl = np.array([xi,yi])

    # and right edge
    if x[1]>=xlm[1]:
        # if values are given past the left edge
        yi = np.interp(xlm[1],x,y)
        if yi<ylm[0]:
            # if the extrapolation would be below the plot
            xi = np.interp(ylm[0],ys,xs)
            yi = ylm[0]
        elif yi>ylm[1]:
            # if the extrapolation would be above the plot
            xi = np.interp(ylm[1],ys,xs)
            yi = ylm[1]
        else:
            # if it's in range
            xi = xlm[1]
    else:
        # if there's nothing that small, find the leftmost y-value on the plot
        if y[-1]>np.mean(ylm):
            yi = np.maximum(y[-1],ylm[1])
        else:
            yi = np.minimum(y[-1],ylm[0])
        xi = np.interp(yi,ys,xs)

    # save left values
    xyr = np.array([xi,yi])

    if ylog:
        xyr[1]=np.exp(xyr[1])
        xyl[1]=np.exp(xyl[1])

    if xlog:
        xyr[0]=np.exp(xyr[0])
        xyl[0]=np.exp(xyl[0])

    return xyl,xyr

def minmax(x,bfr=1.):
    """
    :param     x:  values
    :param   bfr:  buffer
    :return   lm:  limits
    """

    # limits
    lm=np.array([np.min(x),np.max(x)])
    df=(1-bfr)*np.diff(lm)[0]/2.*np.array([-1.,1])
    lm=lm+df

    return lm


def printfigure(fname,f,ftype='pdf',pngback=False,aax=[],tight=True):
    """
    :param     fname:   file name
    :param         f:   figure handle
    :param     ftype:   figure type (default: 'pdf')
    :param   pngback:   export as png, then use convert to retrieve the original 
                           format---for high resolution images
    """
    
    if pngback:
        fname2=os.path.join(os.environ['FIGURES'],fname+'.'+ftype)
        ftype = 'png'

    fname=os.path.join(os.environ['FIGURES'],fname+'.'+ftype)
    if ftype=='pdf':
        # pp=PdfPages(fname)
        # pp.savefig(f)
        # pp.close()
        if tight:
            f.savefig(fname,format='pdf',bbox_inches='tight',
                      bbox_extra_artists=aax)
        else:
            f.savefig(fname,format='pdf')
    elif ftype=='png':
        f.savefig(fname,dpi=600)
    else:
        f.savefig(fname)

    # convert from the png to desired file type
    if pngback:
        import subprocess
        tocall=['convert',fname,fname2]
        subprocess.call(tocall)
        os.remove(fname)
        

def baroutlinevals(x,y,wzeros=False,yi=None):
    """
    :param      x:  x-values: center or edges of a histogram
    :param      y:  number of values in each bin
    :param wzeros:  include zero-values at the end to 
                       create a closed polygon (default: False)
    :param     yi:  the bottom of the histogram
                       y is added to this (default: 0)
    :return    xl:  x-locations of the edges
    :return    yl:  y-locations of the edges
    """

    x=np.atleast_1d(x)
    y=np.atleast_1d(y)
    if yi is not None:
        yi=np.atleast_1d(yi)
        yi=np.repeat(yi,int(y.size/yi.size))

    # number of bins
    N = y.size
    
    if x.size==N:
        # for bin centers
        dx = np.median(np.diff(x))
        xc = (x[0:-1]+x[1:])/2.
        xc=np.append(np.array(x[0]-dx/2.),xc)
        x=np.append(xc,np.array(x[-1]+dx/2.))
    elif x.size==N+1:
        # for bin edges
        pass
    else:
        print('Wrong number of bins')

    # which indices
    ix=np.repeat(np.arange(0,N+1),2)
    ix=ix[1:-1]

    iy=np.repeat(np.arange(0,N),2)

    # extract the limits
    xl = x[ix]
    yl = y[iy]

    if yi is not None:
        xadd = x[np.flipud(ix)]
        yl = yl + yi[iy]
        yadd = yi[np.flipud(iy)]

    # add points at zero
    if wzeros:
        xl=np.hstack([xl[0:1],xl,xl[-1:],xl[0:1]])
        yl=np.hstack([np.zeros(1),yl,np.zeros(2)])

    if yi is not None:
        xl = np.append(xl,xadd)
        yl = np.append(yl,yadd)


    return xl,yl


def linestyles(N):
    """
    :param           N: number of data types to plot
    :return        mks: a list of markers
    """

    if N==1:
        mks=['-']
    elif N==2:
        mks=['-','--']
    elif N==3:
        mks=['-','--',':']
    elif N==4:
        mks=['-','--',':','-.']
    else:
        mks=['-','--',':','-.']
        mks=mks*int(np.ceil(N/len(mks)))
        mks=mks[0:N]

    return mks


def markers(N):
    """
    :param           N: number of data types to plot
    :return        mks: a list of markers
    """

    if N==1:
        mks=['o']
    elif N==2:
        mks=['o','^']
    elif N==3:
        mks=['o','s','^']
    elif N==4:
        mks=['o','s','d','^']
    elif N==5:
        mks=['o','s','d','<','^']
    elif N==6:
        mks=['o','s','d','<','^','X']
    elif N==7:
        mks=['o','s','d','>','<','^','X']
    elif N==8:
        mks=['o','s','d','v','>','<','^','X']
    elif N==9:
        mks=['o','s','d','v','>','<','^','X','*']
    elif N==10:
        mks=['o','s','d','v','>','<','^','X','*','H']
    elif N==11:
        mks=['o','s','d','v','>','<','^','X','*','H','P']
    else:
        mks=['o','s','d','v','>','<','^','X','*','H','P']
        mks=mks*int(np.ceil(N/len(mks)))
        mks=mks[0:N]

    return mks


def colors(N,lgt=False):
    """
    :param      N:  number of colors
    :param    lgt:  lighter colors (default: False)
    :return  cols:  colors
    """
    
    if not lgt:
        red = 'firebrick'
        blue =  'darkblue'
        yellow = 'darkgoldenrod'
        green = 'darkgreen'
        pink = 'deeppink'
        violet = 'darkviolet'
        orange = 'orange'
    else:
        red = 'darksalmon'
        blue =  'lightsteelblue'
        yellow = 'palegoldenrod'
        green = 'lightgreen'
        pink = 'lightpink'
        violet = 'thistle'
        orange = 'navajowhite'

    if N==0:
        cols = []
    elif N==1:
        cols = [red]
    elif N==2:
        cols = [red,blue]
    elif N==3:
        cols = [red,yellow,blue]
    elif N==4:
        cols = [red,yellow,green,blue]
    elif N==5:
        cols = [red,orange,yellow,green,blue]
    elif N==6:
        cols = [red,orange,yellow,green,blue,violet]
    elif N==7:
        cols = [pink,red,orange,yellow,green,blue,violet]
    elif N>7:
        # original map
        colsi = colors(7,lgt=lgt)
        for k in range(0,len(colsi)):
            colsi[k]=matplotlib.colors.colorConverter.to_rgb(colsi[k])
        colsi = np.array(colsi)
        xi = np.linspace(0.,1.,colsi.shape[0])

        # to interpolate between
        colsj = np.ndarray([N,3],dtype=float)
        xj = np.linspace(0.,1.,N)

        for k in range(0,3):
            f1=interp1d(xi,colsi[:,k],
                        bounds_error=False)
            colsj[:,k]=f1(xj)
        
        # collect for output
        cols = [tuple(colsj[k,:]) for k in range(0,N)]

        
    return cols
    

def boxpolygon(xlm,ylm):
    """
    :param     xlm:  x-limits
    :param     ylm:  y-limits
    """

    # to indices
    xlm = np.atleast_1d(xlm).astype(float)
    ylm = np.atleast_1d(ylm).astype(float)

    # vertices
    vt = np.ndarray([5,2],dtype=float)
    vt[:,0] = xlm[np.array([0,1,1,0,0])]
    vt[:,1] = ylm[np.array([0,0,1,1,0])]

    # create polygon
    ply = Polygon(vt)

    return ply


def delticklabels(pm,axs='both'):
    """
    :param   pm: grid of plots to delete labels for
                   first dimension is x
    :param  axs: labels to delete 'x','y', or 'both' (default)
    """

    szx = pm.shape[1]
    szy = pm.shape[0]

    if axs is 'both':
        delticklabels(pm,'x')
        delticklabels(pm,'y')
        return
    elif axs is 'y':
        dlt = np.ones(pm.shape,dtype=bool)
        dlt[:,0]=False
        # transpose to look like x
        dlt,pm=dlt.transpose(),pm.transpose()
        dlt,pm=np.flipud(dlt),np.flipud(pm)
    elif axs is 'x':
        dlt = np.ones(pm.shape,dtype=bool)
        dlt[-1,:]=False

    # check that the panel not to change exists
    for kx in range(0,pm.shape[1]):
        ky,=np.where(~dlt[:,kx])
        while ky.any():
            ky=ky[0]
            try:
                vs=pm[ky,kx].get_visible()
                if vs=='off':
                    dlt[ky-1:ky,kx]=False
                    ky,=np.where(~dlt[:,kx])
                else:
                    ky=np.array([])
            except:
                dlt[ky-1:ky,kx]=False
                ky,=np.where(~dlt[:,kx])

    for kx in range(0,pm.shape[1]):
        for ky in range(0,pm.shape[0]):
            if pm[ky,kx].get_visible()=='off':
                dlt[ky,kx]=False
    
    if axs is 'y':
        for ph in pm[dlt]:
            ph.set_yticklabels([])
            ph.set_ylabel('')
    elif axs is 'x':
        for ph in pm[dlt]:
            ph.set_xticklabels([])
            ph.set_xlabel('')

    return


def cornerlabels(p,loc='ll',fontsize='medium',xscl=None,yscl=None,scl=None,lskip=0):
    """
    :param        p:  list of plots
    :param      loc:  location (default: 'll'--lower left)
    :param fontsize:  font size (default: 'medium')
    :param     xscl:  distance from x-edge
    :param     yscl:  distance from y-edge (default: xscl)
    :param      scl:  distance from edge
    :param    lskip:  how many letters of the alphabet to skip
    :return       h:  handles to text boxes
    """
    
    # set grid of values
    lbl='abcdefghijklmnopqrstuvwxyz'

    if scl is not None:
        xscl = scl

    if xscl is None:
        if loc[0]=='o':
            xscl = 0.95
        else:
            xscl = 0.05

    if yscl is None:
        yscl = xscl

    if loc is 'll':
        x,y=xscl,yscl
        hl = 'left'
        vl = 'bottom'
    elif loc is 'ul':
        x,y=xscl,1.-yscl
        hl = 'left'
        vl = 'top'
    elif loc is 'ur':
        x,y=1.-xscl,1.-yscl
        hl = 'right'
        vl = 'top'
    elif loc is 'lr':
        x,y=1.-xscl,yscl
        hl = 'right'
        vl = 'bottom'
    if loc is 'oll':
        x,y=-xscl,-yscl
        hl = 'left'
        vl = 'bottom'
    elif loc is 'oul':
        x,y=-xscl,1.+yscl
        hl = 'left'
        vl = 'top'
    elif loc is 'our':
        x,y=1.+xscl,1.+yscl
        hl = 'right'
        vl = 'top'
    elif loc is 'olr':
        x,y=1.+xscl,-yscl
        hl = 'right'
        vl = 'bottom'

    h=[]
    for k in range(0,len(p)):
        # correct label
        kk = int((k+lskip) % 26)
        jj = int(((k+lskip) / 26) + 1)
        ll = lbl[kk]*jj

        # plot
        hh=p[k].text(x,y,ll,transform=p[k].transAxes,
                     fontsize=fontsize,alpha=1.,backgroundcolor='w',
                     horizontalalignment=hl,verticalalignment=vl,
                     bbox=dict(edgecolor='black',facecolor='w'),
                     zorder=50)
        h.append(hh)
    return h



def timegroup(tlen,nsig=2,wdash=True):
    """
    :param   tlen:   time length in second
    :param   nsig:   number of significant figures
    :param  wdash:   add a dash, as for 10-minute
    :return   tnm:   number of values 
    :return   unt:   which unit, eg. seconds
    :return  sunt:   number of seconds per unit  
    :return  tlbl:   full label
    """

    tlen = np.atleast_1d(tlen)

    # which value to use
    lmins = np.array([1.,60.,3600.,86400.])
    ix = np.searchsorted(lmins,tlen,'left')
    ix = np.maximum(ix-1,0)

    # divide by the relevant scaling
    vl = np.divide(tlen,lmins[ix])
    sunt = lmins[ix]

    # round
    vl = general.roundsigfigs(vl,nsig)
    vl = np.atleast_1d(vl)
    
    # labels
    lbls = np.array(['seconds','minutes','hours','days'])
    lbls = lbls[ix]
    lbls = np.atleast_1d(lbls)

    tlbl = []
    tnm = []
    unt = []
    for k in range(0,len(vl)):
        vli = vl[k]
        lbli = lbls[k]

        if vli % 1==0:
            vli=int(vli)

        if vli==1 or wdash:
            lbli=lbli[0:-1]

        tnm.append(vli)
        unt.append(lbli)

        if wdash:
            lbli='-'+lbli
        else:
            lbli=' '+lbli
        tlbl.append(str(vli)+lbli)

    
    tnm=np.array(tnm)
    unt=np.array(unt)
    tlbl=np.array(tlbl)

    return tnm,unt,sunt,tlbl



def timelabel(tlen,nsig=2,wdash=True):
    """
    :param   tlen:   time length in second
    :param   nsig:   number of significant figures
    :param  wdash:   add a dash, as for 10-minute
    :return  tlbl:   label for this time
    """

    tlen = np.atleast_1d(tlen)

    # which value to use
    lmins = np.array([1.,60.,3600.,86400.])
    ix = np.searchsorted(lmins,tlen,'left')
    ix = np.maximum(ix-1,0)

    # divide by the relevant scaling
    vl = np.divide(tlen,lmins[ix])

    # round
    vl = np.atleast_1d(vl)
    vl = general.roundsigfigs(vl,nsig)

    
    # labels
    lbls = np.array(['seconds','minutes','hours','days'])
    lbls = lbls[ix]
    lbls = np.atleast_1d(lbls)

    tlbl = []
    for k in range(0,len(vl)):
        vli = vl[k]
        lbli = lbls[k]

        if vli % 1==0:
            vli=int(vli)

        if vli==1 or wdash:
            lbli=lbli[0:-1]

        if wdash:
            lbli='-'+lbli
        else:
            lbli=' '+lbli
        
        tlbl.append(str(vli)+lbli)

    return tlbl


def addscalebreak(ycnt,p=None,prc=0.03):
    """
    :param   ycnt: center of break
    :param      p: plot to use
    :param    prc: percentage of plot
    """

    if p is None:
        p = plt.gca()

    xlm=p.get_xlim()
    if p.get_xscale()=='log':
        xlm=np.log(xlm)

    xvl=np.linspace(xlm[0],xlm[1],1000)
    yvl = xvl / (np.diff(xlm)[0] / 10)
    yvl = yvl % 2
    yvl[yvl>1]=yvl[yvl>1]-2.
    yvl=np.abs(yvl)-0.5

    ylm=p.get_ylim()
    if p.get_yscale()=='log':
        ylm=np.log(ylm)
        ycnt=np.log(ycnt)
    ylm=np.diff(ylm)[0]*prc

    # shift
    yvl=yvl*ylm+ycnt

    # both sides
    xvl=np.append(xvl,np.flipud(xvl))
    yvl=np.append(yvl+ylm*0.3,np.flipud(yvl)-ylm*0.3)
    xvl=np.append(xvl,xvl[0])
    yvl=np.append(yvl,yvl[0])

    if p.get_yscale()=='log':
        yvl=np.exp(yvl)

    ply = Polygon(np.vstack([xvl,yvl]).transpose())
    ply.set_edgecolor('gray')
    ply.set_facecolor('w')
    ply.set_fill(True)
    ply.set_zorder(100)
    p.add_patch(ply)

    #p.plot(xvl,yvl)


# define an object that will be used by the legend
def multilegend(colors=['g','r','b']):
#    hh=MulticolorPatch(colors=colors)

    hh=MulticolorCircles(face_colors=colors)

    hmap={MulticolorPatch:MulticolorPatchHandler(),
          MulticolorCircles:MulticolorHandler}

    return hh,hmap

class MulticolorPatch(object):
    def __init__(self, colors):
        self.colors = colors
        
# define a handler for the MulticolorPatch object
class MulticolorPatchHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        width, height = handlebox.width, handlebox.height
        patches = []
        for i, c in enumerate(orig_handle.colors):
            patches.append(plt.Rectangle([width/len(orig_handle.colors) * i - handlebox.xdescent, 
                                          -handlebox.ydescent],
                                         width / len(orig_handle.colors),
                                         height, 
                                         facecolor=c, 
                                         edgecolor='none'))

            patch = PatchCollection(patches,match_original=True)

            handlebox.add_artist(patch)
            return patch



class MulticolorCircles:
    """
    For different shapes, override the ``get_patch`` method, and add the new
    class to the handler map, e.g. via

    ax_r.legend(ax_r_handles, ax_r_labels, handlelength=CONF.LEGEND_ICON_SIZE,
            borderpad=1.2, labelspacing=1.2,
            handler_map={MulticolorCircles: MulticolorHandler})
    """

    def __init__(self, face_colors, edge_colors=None, face_alpha=1,
                 radius_factor=1):
        """
        """
        assert 0 <= face_alpha <= 1, f"Invalid face_alpha: {face_alpha}"
        assert radius_factor > 0, "radius_factor must be positive"
        self.rad_factor = radius_factor
        self.fc = [mcolors.colorConverter.to_rgba(fc, alpha=face_alpha)
                   for fc in face_colors]
        self.ec = edge_colors
        if edge_colors is None:
            self.ec = ["none" for _ in self.fc]
        self.N = len(self.fc)

    def get_patch(self, width, height, idx, fc, ec):
        """
        """
        w_chunk = width / self.N
        radius = min(w_chunk / 2, height) * self.rad_factor
        xy = (w_chunk * idx + radius, radius)
        patch = plt.Circle(xy, radius, facecolor=fc, edgecolor=ec)
        return patch

    def __call__(self, width, height):
        """
        """
        patches = []
        for i, (fc, ec) in enumerate(zip(self.fc, self.ec)):
            patch = self.get_patch(width, height, i, fc, ec)
            patches.append(patch)
        result = PatchCollection(patches, match_original=True)
        #
        return result

class MulticolorHandler:
    """
    """
    @staticmethod
    def legend_artist(legend, orig_handle, fontsize, handlebox):
        """
        """
        width, height = handlebox.width, handlebox.height
        patch = orig_handle(width, height)
        handlebox.add_artist(patch)
        return patch


def adjust_text(h,mindst=0.04,shfx=0.005,shfy=0.):
    """
    :param          h: handles to the texts
    :param    minxdst: minimum distance, as fraction of the plot
    :param       shfx: how much to step in x direction
    :param       shfy: how much to step in y direction
    """

    # check they're all on the same plot
    pref=h[0].axes
    isame=np.array([hh.axes == pref for hh in h])
    if np.sum(isame)<len(h):
        h2=[h[k] for k in np.where(~isame)[0]]
        adjust_text(h2,mindst=mindst,shfx=shfx,shfy=shfy)
        h=[h[k] for k in np.where(isame)[0]]

        
    xlm=np.array(pref.get_xlim())
    if pref.get_xscale()=='log':
        xlm=np.log(xlm)

    ylm=np.array(pref.get_ylim())
    if pref.get_yscale()=='log':
        ylm=np.log(ylm)

    # get values
    xvl=np.array([hh._x for hh in h])
    if pref.get_xscale()=='log':
        xvl=np.log(xvl)
    yvl=np.array([hh._y for hh in h])
    if pref.get_yscale()=='log':
        yvl=np.log(yvl)

    xvl=(xvl-xlm[0])/np.diff(xlm)[0]
    yvl=(yvl-ylm[0])/np.diff(ylm)[0]

    i1=np.arange(0,xvl.size)
    i1,i2=np.meshgrid(i1,i1)
    ii=i2>i1
    i1,i2=i1[ii],i2[ii]

    # shift distances if they're too small
    dmin2=mindst**2
    dst2=np.power(xvl[i2]-xvl[i1],2)+np.power(yvl[i2]-yvl[i1],2)
    iclose=np.where(dst2<dmin2)[0]
    ctr=0
    while iclose.size and ctr<1000:
        ix=np.random.choice(iclose,1)[0]
        j1,j2=i1[ix],i2[ix]
        xvl[j1],yvl[j1]=xvl[j1]-shfx,yvl[j1]-shfy
        xvl[j2],yvl[j2]=xvl[j2]+shfx,yvl[j2]+shfy

        dst2=np.power(xvl[i2]-xvl[i1],2)+np.power(yvl[i2]-yvl[i1],2)
        iclose=np.where(dst2<dmin2)[0]

        ctr=ctr+1

    # back to data units
    xvl=xvl*np.diff(xlm)[0]+xlm[0]
    yvl=yvl*np.diff(ylm)[0]+ylm[0]
    if pref.get_xscale()=='log':
        xvl=np.exp(xvl)
    if pref.get_yscale()=='log':
        yvl=np.exp(yvl)

    # reset values
    for k in range(0,len(h)):
        h[k].set_x(xvl[k])
        h[k].set_y(yvl[k])

    
