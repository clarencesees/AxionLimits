#================================PlotFuncs.py==================================#
# Created by Ciaran O'Hare 2020

# Description:
# This file has many functions which are used throughout the project, but are
# all focused around the bullshit that goes into making the plots

#==============================================================================#

from numpy import *
from numpy.random import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from matplotlib import colors
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
from scipy.stats import norm
import matplotlib.patheffects as pe
from scipy.constants import m_e

pltdir = 'plots/'
pltdir_png = pltdir+'plots_png/'

#==============================================================================#
def col_alpha(col,alpha=0.1):
    rgb = colors.colorConverter.to_rgb(col)
    bg_rgb = [1,1,1]
    return [alpha * c1 + (1 - alpha) * c2
            for (c1, c2) in zip(rgb, bg_rgb)]
#==============================================================================#


def PlotBound(ax,filename,edgecolor='k',facecolor='crimson',alpha=1,lw=1.5,y2=1e10,zorder=0.1,
              linestyle='-',skip=1,FillBetween=True,edgealpha=1,rescale_m=False,
              scale_x=1,scale_y=1,start_x=0,end_x=nan,MinorEdgeScale=1.5,AddMinorEdges=False):
    dat = loadtxt(filename)
    if end_x/end_x==1:
        dat = dat[start_x:end_x,:]
    else:
        dat = dat[start_x:,:]
    dat[:,0] *= scale_x
    dat[:,1] *= scale_y
    if rescale_m:
        dat[:,1] = dat[:,1]/dat[:,0]
    if FillBetween:
        ax.fill_between(dat[0::skip,0],dat[0::skip,1],y2=y2,color=facecolor,alpha=alpha,zorder=zorder,lw=0)
    else:        
        ax.fill(dat[0::skip,0],dat[0::skip,1],color=facecolor,alpha=alpha,zorder=zorder,lw=0)
    ax.plot(dat[0::skip,0],dat[0::skip,1],color=edgecolor,zorder=zorder,lw=lw,linestyle=linestyle,alpha=edgealpha)
    if skip>1:
        ax.plot([dat[-2,0],dat[-1,0]],[dat[-2,1],dat[-1,1]],color=edgecolor,zorder=zorder,lw=lw,linestyle=linestyle,alpha=edgealpha)
    if AddMinorEdges:
        ax.plot([dat[-1,0],dat[-1,0]],[dat[-1,1],MinorEdgeScale*dat[-1,1]],color=edgecolor,zorder=zorder,lw=lw,linestyle=linestyle,alpha=edgealpha)
        ax.plot([dat[0,0],dat[0,0]],[dat[0,1],MinorEdgeScale*dat[0,1]],color=edgecolor,zorder=zorder,lw=lw,linestyle=linestyle,alpha=edgealpha)
    return

def line_background(lw,col):
    return [pe.Stroke(linewidth=lw, foreground=col), pe.Normal()]



def FilledLimit(ax,dat,text_label='',col='ForestGreen',edgecolor='k',zorder=1,linestyle='-',\
                    lw=2,y2=1e0,edgealpha=0.6,text_on=False,text_pos=[0,0],\
                    ha='left',va='top',clip_on=True,fs=15,text_col='k',rotation=0,facealpha=1,path_effects=None,textalpha=1):
    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor=None,facecolor=col,alpha=facealpha,zorder=zorder)
    plt.plot(dat[:,0],dat[:,1],linestyle=linestyle,color=edgecolor,alpha=edgealpha,zorder=zorder,lw=lw)
    if text_on:
        plt.text(text_pos[0],text_pos[1],text_label,fontsize=fs,color=text_col,\
            ha=ha,va=va,clip_on=clip_on,rotation=rotation,rotation_mode='anchor',path_effects=path_effects,alpha=textalpha)
    return

def UnfilledLimit(ax,dat,text_label='',col='ForestGreen',edgecolor='k',zorder=1,\
                    lw=2,y2=1e0,edgealpha=0.6,text_on=False,text_pos=[0,0],\
                    ha='left',va='top',clip_on=True,fs=15,text_col='k',rotation=0,facealpha=1,\
                     linestyle='--'):
    plt.plot(dat[:,0],dat[:,1],linestyle=linestyle,color=edgecolor,alpha=edgealpha,zorder=zorder,lw=lw)
    if text_on:
        plt.text(text_pos[0],text_pos[1],text_label,fontsize=fs,color=text_col,\
            ha=ha,va=va,clip_on=clip_on,rotation=rotation,rotation_mode='anchor')
    return

# Black hole superradiance constraints on the axion mass
# can be used for any coupling
def BlackHoleSpins(ax,C,label_position,whichfile='Mehta',fs=20,col='k',alpha=0.4,\
                   PlotLine=True,rotation=90,linecolor='k',facecolor='k',text_col='k',text_on=True,zorder=0.1):
    y2 = ax.get_ylim()[-1]

    # arxiv: 2009.07206
    # BH = loadtxt("limit_data/BlackHoleSpins.txt")
    # if PlotLine:
    #     plt.plot(BH[:,0],BH[:,1],color=col,lw=3,alpha=min(alpha*2,1),zorder=0)
    # plt.fill_between(BH[:,0],BH[:,1],y2=0,edgecolor=None,facecolor=col,zorder=0,alpha=alpha)
    # if text_on:
    #     plt.text(label_position[0],label_position[1],r'{\bf Black hole spins}',fontsize=fs,color=text_col,\
    #          rotation=rotation,ha='center',rotation_mode='anchor')

    # arxiv: 2011.11646
    dat = loadtxt('limit_data/fa/BlackHoleSpins_'+whichfile+'.txt')
    dat[:,1] = dat[:,1]*C
    plt.fill_between(dat[:,0],dat[:,1],y2=0,lw=3,alpha=alpha,color=facecolor,zorder=zorder)
    if PlotLine:
        plt.plot(dat[:,0],dat[:,1],'-',lw=3,alpha=0.7,color=linecolor,zorder=zorder)
    if text_on:
        plt.text(label_position[0],label_position[1],r'{\bf Black hole spins}',fontsize=fs,color=text_col,\
            rotation=rotation,ha='center',rotation_mode='anchor')

    return

def UpperFrequencyAxis(ax,N_Hz=1,tickdir='out',xtick_rotation=0,labelsize=25,xlabel=r"$\nu_a$ [Hz]",lfs=40,tick_pad=8,tfs=25,xlabel_pad=10):
    m_min,m_max = ax.get_xlim()
    ax2 = ax.twiny()
    ax2.set_xlabel(xlabel,fontsize=lfs,labelpad=xlabel_pad)
    ax2.set_xscale('log')
    plt.xticks(rotation=xtick_rotation)
    ax2.tick_params(labelsize=tfs)
    ax2.tick_params(which='major',direction=tickdir,width=2.5,length=13,pad=tick_pad)
    ax2.tick_params(which='minor',direction=tickdir,width=1,length=10)
    locmaj = mpl.ticker.LogLocator(base=10.0, subs=(1.0, ), numticks=50)
    locmin = mpl.ticker.LogLocator(base=10.0, subs=arange(2, 10)*.1,numticks=100)
    ax2.xaxis.set_major_locator(locmaj)
    ax2.xaxis.set_minor_locator(locmin)
    ax2.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax2.set_xlim([m_min*241.8*1e12/N_Hz,m_max*241.8*1e12/N_Hz])
    plt.sca(ax)

def UpperFrequencyAxis_Simple(ax,tickdir='out',xtick_rotation=0,labelsize=25,xlabel=None,lfs=40,tick_pad=8,tfs=25,xlabel_pad=10):
    m_min,m_max = ax.get_xlim()
    ax2 = ax.twiny()
    ax2.set_xscale('log')
    ax2.set_xlabel(xlabel,fontsize=lfs,labelpad=xlabel_pad)
    ax2.tick_params(labelsize=tfs)
    ax2.tick_params(which='major',direction=tickdir,width=2.5,length=13,pad=tick_pad)
    ax2.tick_params(which='minor',direction=tickdir,width=1,length=10)
    locmaj = mpl.ticker.LogLocator(base=10.0, subs=(1.0, ), numticks=50)
    locmin = mpl.ticker.LogLocator(base=10.0, subs=arange(2, 10)*.1,numticks=100)
    ax2.xaxis.set_major_locator(locmaj)
    ax2.xaxis.set_minor_locator(locmin)
    ax2.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax2.set_xticks(10.0**arange(-9,18))
    ax2.set_xticklabels(['nHz','','',r'\textmu Hz','','','mHz','','','Hz','','','kHz','','','MHz','','','GHz','','','THz','','','PHz','','']);
    ax2.set_xlim([m_min*241.8*1e12,m_max*241.8*1e12])
    plt.sca(ax)
    return

def AlternativeCouplingAxis(ax,scale=1,tickdir='out',labelsize=25,ylabel=r"$g_\gamma$ [GeV$^{-1}$]",lfs=40,tick_pad=8,tfs=25,ylabel_pad=60):
    g_min,g_max = ax.get_ylim()
    ax3 = ax.twinx()
    ax3.set_ylim([g_min*scale,g_max*scale])
    ax3.set_ylabel(ylabel,fontsize=lfs,labelpad=ylabel_pad,rotation=-90)
    ax3.set_yscale('log')
    ax3.tick_params(labelsize=tfs)
    ax3.tick_params(which='major',direction=tickdir,width=2.5,length=13,pad=tick_pad)
    ax3.tick_params(which='minor',direction=tickdir,width=1,length=10)
    locmaj = mpl.ticker.LogLocator(base=10.0, subs=(1.0, ), numticks=50)
    locmin = mpl.ticker.LogLocator(base=10.0, subs=arange(2, 10)*.1,numticks=100)
    ax3.yaxis.set_major_locator(locmaj)
    ax3.yaxis.set_minor_locator(locmin)
    ax3.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    plt.sca(ax)

def FigSetup(xlab=r'$m_a$ [eV]',ylab='',\
                 g_min = 1.0e-19,g_max = 1.0e-6,\
                 m_min = 1.0e-12,m_max = 1.0e7,\
                 lw=2.5,lfs=45,tfs=25,tickdir='out',figsize=(16.5,11),\
                 Grid=False,Shape='Rectangular',\
                 mathpazo=False,TopAndRightTicks=False,majorticklength=13,minorticklength=10,\
                xtick_rotation=20.0,tick_pad=8,x_labelpad=10,y_labelpad=10,\
             FrequencyAxis=False,N_Hz=1,upper_xlabel=r"$\nu_a$ [Hz]",**freq_kwargs):

    plt.rcParams['axes.linewidth'] = lw
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=tfs)

    if mathpazo:
            plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
            })

    if Shape=='Wide':
        fig = plt.figure(figsize=(16.5,5))
    elif Shape=='Rectangular':
        fig = plt.figure(figsize=(16.5,11))
    elif Shape=='Square':
        fig = plt.figure(figsize=(14.2,14))
    elif Shape=='Custom':
        fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(111)

    ax.set_xlabel(xlab,fontsize=lfs,labelpad=x_labelpad)
    ax.set_ylabel(ylab,fontsize=lfs,labelpad=y_labelpad)

    ax.tick_params(which='major',direction=tickdir,width=2.5,length=majorticklength,right=TopAndRightTicks,top=TopAndRightTicks,pad=tick_pad)
    ax.tick_params(which='minor',direction=tickdir,width=1,length=minorticklength,right=TopAndRightTicks,top=TopAndRightTicks)

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim([m_min,m_max])
    ax.set_ylim([g_min,g_max])

    locmaj = mpl.ticker.LogLocator(base=10.0, subs=(1.0, ), numticks=50)
    locmin = mpl.ticker.LogLocator(base=10.0, subs=arange(2, 10)*.1,numticks=100)
    ax.xaxis.set_major_locator(locmaj)
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    locmaj = mpl.ticker.LogLocator(base=10.0, subs=(1.0, ), numticks=100)
    locmin = mpl.ticker.LogLocator(base=10.0, subs=arange(2, 10)*.1,numticks=100)
    ax.yaxis.set_major_locator(locmaj)
    ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    plt.xticks(rotation=xtick_rotation)

    if Grid:
        ax.grid(zorder=0)

    if FrequencyAxis:
        UpperFrequencyAxis(ax,N_Hz=N_Hz,tickdir='out',\
                           xtick_rotation=xtick_rotation,\
                           xlabel=upper_xlabel,\
                           lfs=lfs/1.3,tfs=tfs,tick_pad=tick_pad-2,**freq_kwargs)

    return fig,ax


#==============================================================================#
class AxionElectron():
    def QCDAxion(ax,text_on=True,C_logwidth=10,KSVZ_on=False,DFSZ_on=True,Hadronic_on=True,fs=20,DFSZ_col='gold',KSVZ_col='#857c20',Hadronic_col='goldenrod',DFSZ_label_mass=5e-9,KSVZ_label_mass=5e-9,Hadronic_label_mass=5e-8):
        ## QCD Axion band:
        g_min,g_max = ax.get_ylim()
        m_min,m_max = ax.get_xlim()

        # Mass-coupling relation
        def g_x(C_ae,m_a):
            return 8.943e-11*C_ae*m_a
        DFSZ_u = 1.0/3.0
        DFSZ_l = 2.0e-5
        KSVZ = 2e-4
        Had_u = 5e-3
        Had_l = 1.5e-4

        # QCD Axion models
        n = 200
        m = logspace(log10(m_min),log10(m_max),n)
        rot = 45.0
        trans_angle = plt.gca().transData.transform_angles(array((rot,)),array([[0, 0]]))[0]
        if DFSZ_on:
            col = DFSZ_col
            plt.fill_between(m,g_x(DFSZ_l,m)/1e-3,y2=g_x(DFSZ_u,m)/1e-3,facecolor=col,zorder=0,alpha=0.2)
            plt.plot(m,g_x(DFSZ_l,m)/1e-3,'k-',lw=3,zorder=0)
            plt.plot(m,g_x(DFSZ_u,m)/1e-3,'k-',lw=3,zorder=0)
            plt.plot(m,g_x(DFSZ_l,m)/1e-3,'-',lw=2,zorder=0,color=col)
            plt.plot(m,g_x(DFSZ_u,m)/1e-3,'-',lw=2,zorder=0,color=col)
            if text_on:
                plt.text(DFSZ_label_mass,g_x(DFSZ_u,DFSZ_label_mass)/1.5/1e-3,r'{\bf DFSZ}',fontsize=fs,rotation=trans_angle,ha='left',va='top',rotation_mode='anchor',clip_on=True,color=DFSZ_col,path_effects=line_background(1,'k'))
        if KSVZ_on:
            col = KSVZ_col
            plt.plot(m,g_x(KSVZ,m)/1e-3,'-',lw=2,zorder=0.02,color=col)
            if text_on:
                plt.text(KSVZ_label_mass,g_x(KSVZ,KSVZ_label_mass)*2.1/1e-3,r'{\bf KSVZ}',fontsize=fs*0.7,rotation=trans_angle,color=col,ha='left',va='top',rotation_mode='anchor',clip_on=True)
        if Hadronic_on:
            col = Hadronic_col
            plt.fill_between(m,g_x(Had_l,m)/1e-3,y2=g_x(Had_u,m)/1e-3,facecolor=col,zorder=0.01,alpha=0.2)
            plt.plot(m,g_x(Had_l,m)/1e-3,'k-',lw=3,zorder=0.01)
            plt.plot(m,g_x(Had_u,m)/1e-3,'k-',lw=3,zorder=0.01)
            plt.plot(m,g_x(Had_l,m)/1e-3,'-',lw=2,zorder=0.01,color=col)
            plt.plot(m,g_x(Had_u,m)/1e-3,'-',lw=2,zorder=0.01,color=col)
            if text_on:
                plt.text(Hadronic_label_mass,g_x(Had_u,Hadronic_label_mass)/1.5/1e-3,r'{\bf KSVZ-like}',fontsize=fs-5,rotation=trans_angle,ha='left',va='top',rotation_mode='anchor',clip_on=True,color=Hadronic_col,path_effects=line_background(1,'k'))

        return

    def XENON1T(ax,col='darkred',fs=14,text_on=False,zorder=0.51,lw=1.5,text_shift=[1,1*1e3],**kwargs):
        # XENON1T S2 analysis arXiv:[1907.11485]
        dat = loadtxt("limit_data/AxionElectron/XENON1T_DM_S2.txt")
        plt.fill_between(dat[:,0],dat[:,1]/1e-3,y2=1e3,edgecolor=None,facecolor=col,zorder=zorder)
        plt.plot(dat[:,0],dat[:,1]/1e-3,'k-',alpha=1,zorder=zorder,lw=lw)

        # XENON1T S1+S2 analysis arXiv:[2006.09721]
        dat = loadtxt("limit_data/AxionElectron/XENON1T_DM_S1S2.txt")
        plt.fill_between(dat[:,0],dat[:,1]/1e-3,y2=1e3,edgecolor=None,facecolor=col,zorder=zorder)
        plt.plot(dat[:,0],dat[:,1]/1e-3,'k-',alpha=1,zorder=zorder,lw=lw)

        # XENON1T Single electron analysis arXiv:[2112.12116]
        dat = loadtxt("limit_data/AxionElectron/XENON1T_DM_SE.txt")
        plt.fill_between(dat[:,0],dat[:,1]/1e-3,y2=1e3,edgecolor=None,facecolor=col,zorder=zorder)
        plt.plot(dat[:,0],dat[:,1]/1e-3,'k-',alpha=1,zorder=zorder,lw=lw)

        if text_on:
            plt.text(text_shift[0]*3.2e2,text_shift[1]*4e-14,r'{\bf XENON1T}',fontsize=fs,color=col,ha='center',va='top',clip_on=True,**kwargs)
            #plt.text(text_shift[0]*1.2e2,text_shift[1]*2.5e-14,r'(DM)',fontsize=fs,color=col,ha='center',va='top',clip_on=True,**kwargs)
        return

    def XENONnT(ax,col='darkred',fs=17,text_on=True,zorder=0.51,lw=1.5,text_shift=[1,1*1e3],**kwargs):
        # XENONnT ALP DM
        dat = loadtxt("limit_data/AxionElectron/XENONnT.txt")
        plt.fill_between(dat[:,0],dat[:,1]/1e-3,y2=1e3,edgecolor=None,facecolor=col,zorder=zorder)
        plt.plot(dat[:,0],dat[:,1]/1e-3,'k-',alpha=1,zorder=zorder,lw=lw)

        if text_on:
            plt.text(text_shift[0]*0.5e3,text_shift[1]*0.8e-14,r'{\bf XENON}',fontsize=fs,color=col,ha='center',va='top',clip_on=True)

    def XENONnT_Solar(ax,col='#0e6e37',fs=19,text_on=True,zorder=0.52,lw=2,text_shift=[1,1*1e3],**kwargs):
        # Solar axions
        dat = loadtxt("limit_data/AxionElectron/XENONnT_Solar.txt")
        plt.fill_between(dat[:,0],dat[:,1]/1e-3,y2=1e3,edgecolor=None,facecolor=col,zorder=zorder)
        plt.plot(dat[:,0],dat[:,1]/1e-3,'k-',alpha=1,zorder=zorder,lw=lw)
        if text_on:
            plt.text(text_shift[0]*1.2e-8,text_shift[1]*2.5e-12,r'{\bf XENONnT (Solar axions)}',fontsize=fs,color='w',ha='center',clip_on=True,path_effects=line_background(1,'k'),**kwargs)
        return

    def SolarBasin(ax,col='#7d203c',fs=10,text_on=True,lw=1.5,text_shift=[0.8,1*1e3],zorder=0.6,**kwargs):
        # Solar axion basin arXiv:[2006.12431]
        dat = loadtxt("limit_data/AxionElectron/XENON1T_S2_SolarAxionBasin.txt")
        plt.fill_between(dat[:,0],dat[:,1]/1e-3,y2=1e3,edgecolor=None,facecolor=col,zorder=zorder)
        plt.plot(dat[:,0],dat[:,1]/1e-3,'k-',alpha=1,zorder=zorder,lw=lw)
        if text_on:
            plt.text(text_shift[0]*3e3,text_shift[1]*2e-11,r'\begin{center}{\bf XENON1T} \linebreak (Solar basin)\end{center}',fontsize=fs,color='w',ha='center',va='top',clip_on=True,path_effects=line_background(1,'k'),**kwargs)
        return

    def LUX(ax,col='indianred',fs=14,text_on=True,lw=1.5,text_pos=[0.2e-8,7e-12*1e3],zorder=0.52,**kwargs):
        # LUX arXiv:[1704.02297]
        dat = loadtxt("limit_data/AxionElectron/LUX.txt")
        plt.fill_between(dat[:,0],dat[:,1]/1e-3,y2=1e3,edgecolor=None,facecolor=col,zorder=zorder)
        plt.plot(dat[:,0],dat[:,1]/1e-3,'k-',alpha=1,zorder=zorder,lw=lw)

        if text_on:
            plt.text(text_pos[0],text_pos[1],r'{\bf LUX (Solar axions)}',fontsize=fs,color='w',ha='left',va='top',clip_on=True,path_effects=line_background(1,'k'),**kwargs)
        return

    def PandaX(ax,col='firebrick',fs=10,text_on=True,lw=1.5,text_pos=[1.2e3,4.5e-13*1e3],zorder=0.53,rotation=20,**kwargs):
        dat = loadtxt("limit_data/AxionElectron/PandaX.txt")
        plt.fill_between(dat[:,0],dat[:,1]/1e-3,y2=1e3,edgecolor=None,facecolor=col,zorder=zorder)
        plt.plot(dat[:,0],dat[:,1]/1e-3,'k-',alpha=1,zorder=zorder,lw=lw)

        if text_on:
            plt.text(text_pos[0],text_pos[1],r'{\bf PandaX}',fontsize=fs-2,color='w',ha='left',va='top',rotation=rotation,clip_on=True,path_effects=line_background(1,'k'),**kwargs)
        return

    def GERDA(ax,col='#d13617',fs=10,text_on=True,text_pos=[0.5e5,1.5e-11*1e3],zorder=0.52,lw=1.5,text_col='w',rotation=45,**kwargs):
        dat = loadtxt("limit_data/AxionElectron/GERDA.txt")
        plt.fill_between(dat[:,0],dat[:,1]/1e-3,y2=1e3,edgecolor=None,facecolor=col,zorder=zorder)
        plt.plot(dat[:,0],dat[:,1]/1e-3,'k-',alpha=1,zorder=zorder,lw=lw)

        if text_on:
            plt.text(text_pos[0],text_pos[1],r'{\bf GERDA}',fontsize=fs,color=text_col,ha='left',va='top',clip_on=True,path_effects=line_background(1,'k'),rotation=rotation,**kwargs)
        return

    def EDELWEISS(ax,col='#8f2a1f',projection=False,fs=10,text_col='w',text_on=True,text_pos=[1.25e4,1.2e-12*1e3],zorder=0.57,lw=1.5,rotation=55,**kwargs):
        # EDELWEISS arXiv:[1808.02340]
        dat = loadtxt("limit_data/AxionElectron/EDELWEISS.txt")
        plt.fill_between(dat[:,0],dat[:,1]/1e-3,y2=1e3,edgecolor=None,facecolor=col,zorder=zorder)
        plt.plot(dat[:,0],dat[:,1]/1e-3,'k-',alpha=1,zorder=zorder,lw=lw)

        if projection:
            dat = loadtxt("limit_data/AxionElectron/Projections/EDELWEISS.txt")
            plt.plot(dat[:,0],dat[:,1]/1e-3,'--',color=col,zorder=zorder,lw=lw)
        if text_on:
            plt.text(text_pos[0],text_pos[1],r'{\bf EDELWEISS',fontsize=fs,rotation=rotation,color=text_col,path_effects=line_background(1,'k'),clip_on=True)

        return

    def SuperCDMS(ax,col='#800f24',fs=12,text_on=True,text_pos=[3.0e4,8.0e-10*1e3],text_col='w',zorder=0.58,rotation=60,lw=1.5,**kwargs):
        # SuperCDMS arXiv:[1911.11905]
        dat = loadtxt("limit_data/AxionElectron/SuperCDMS.txt")
        plt.fill_between(dat[:,0],dat[:,1]/1e-3,y2=1e3,edgecolor=None,facecolor=col,zorder=zorder)
        plt.plot(dat[:,0],dat[:,1]/1e-3,'-',color='k',alpha=1,zorder=zorder,lw=lw)
        if text_on:
            plt.text(text_pos[0],text_pos[1],r'{\bf SuperCDMS}',fontsize=fs-1,color=text_col,ha='left',va='top',alpha=1.0,rotation=rotation,clip_on=True,path_effects=line_background(1,'k'),**kwargs)
        return

    def DarkSide(ax,col='#921f24',fs=11,text_on=True,text_pos=[4.3e1,1.9e-12*1e3],text_col='w',zorder=0.55,rotation=-50,lw=1.5,**kwargs):
        dat = loadtxt("limit_data/AxionElectron/DarkSide.txt")
        plt.fill_between(dat[:,0],dat[:,1]/1e-3,y2=1e3,edgecolor=None,facecolor=col,zorder=zorder)
        plt.plot(dat[:,0],dat[:,1]/1e-3,'-',color='k',alpha=1,zorder=zorder,lw=lw)
        if text_on:
            plt.text(text_pos[0],text_pos[1],r'{\bf DarkSide}',fontsize=fs-1,color=text_col,ha='left',va='top',alpha=1.0,rotation=rotation,clip_on=True,path_effects=line_background(1,'k'),**kwargs)
        return

    def DARWIN(ax,col='brown',fs=14,text_on=True,text_pos=[0.3e3,2e-14*1e3],zorder=0.1,lw=3,**kwargs):
        # DARWIN arXiv:[1606.07001]
        dat = loadtxt("limit_data/AxionElectron/Projections/DARWIN.txt")
        plt.plot(dat[:,0],dat[:,1]/1e-3,'--',color=col,alpha=1.0,zorder=zorder,lw=lw)
        if text_on:
            plt.text(text_pos[0],text_pos[1],r'{\bf DARWIN}',fontsize=fs,color=col,ha='left',va='top',clip_on=True,**kwargs)
        return

    def LZ(ax,col='crimson',fs=14,text_on=True,text_pos=[2.3e3,0.8e-14*1e3],lw=3,zorder=0.1,**kwargs):
        # DARWIN arXiv:[2102.11740]
        dat = loadtxt("limit_data/AxionElectron/Projections/LZ.txt")
        plt.plot(dat[:,0],dat[:,1]/1e-3,'--',color=col,alpha=1.0,zorder=zorder,lw=lw)
        if text_on:
            plt.text(text_pos[0],text_pos[1],r'{\bf LZ}',fontsize=fs,color=col,ha='left',va='top',clip_on=True,**kwargs)
        return

    def QUAX(ax,col='orangered',fs=15,text_on=True,text_pos=[46e-6,5.1e-10*1e3],lw=1,zorder=10.0,text_rot=-90,path_effects=line_background(1,'k'),**kwargs):
        # QUAX https://inspirehep.net/literature/1777123
        dat = loadtxt("limit_data/AxionElectron/QUAX.txt")
        plt.fill_between(dat[:,0],dat[:,1]/1e-3,y2=1e3,color=col,alpha=0.4,zorder=zorder)
        plt.plot(dat[:,0],dat[:,1]/1e-3,'-',color=col,alpha=1.0,zorder=zorder,lw=lw,path_effects=line_background(lw+2,'k'))
        if text_on:
            plt.text(text_pos[0],text_pos[1],r'{\bf QUAX}',fontsize=fs,color=col,rotation=text_rot,ha='left',va='top',clip_on=True,path_effects=path_effects,**kwargs)
        return
    
    def UWA(ax,col='pink',fs=15,text_on=True,text_pos=[12e-6,0.9e-6*1e3],lw=1,zorder=10.0,text_rot=90,path_effects=line_background(1,'k'),**kwargs):
        dat = loadtxt("limit_data/AxionElectron/UWA.txt")
        plt.fill_between(dat[:,0],dat[:,1]/1e-3,y2=1e3,color=col,alpha=0.4,zorder=zorder)
        plt.plot(dat[:,0],dat[:,1]/1e-3,'-',color=col,alpha=1.0,zorder=zorder,lw=lw,path_effects=line_background(lw+2,'k'))
        if text_on:
            plt.text(text_pos[0],text_pos[1],r'{\bf UWA}',fontsize=fs,color=col,rotation=text_rot,ha='left',va='top',clip_on=True,path_effects=path_effects,**kwargs)
        return

    def MagnonQND(ax,col='#942b3e',fs=15,text_on=True,text_pos=[10e-6,0.7e-4*1e3],lw=1,zorder=10.0,text_rot=90,path_effects=line_background(1,'k'),**kwargs):
        dat = loadtxt("limit_data/AxionElectron/Magnons.txt")
        plt.fill_between(dat[:,0],dat[:,1]/1e-3,y2=1e3,color=col,alpha=0.4,zorder=zorder)
        plt.plot(dat[:,0],dat[:,1]/1e-3,'-',color=col,alpha=1.0,zorder=zorder,lw=lw,path_effects=line_background(lw+2,'k'))
        if text_on:
            plt.text(text_pos[0],text_pos[1],r'{\bf QND}',fontsize=fs,color=col,rotation=text_rot,ha='left',va='top',clip_on=True,path_effects=path_effects,**kwargs)
        return


    def RedGiants(ax,col=[0.0, 0.66, 0.42],text_pos=[0.8e-8,2e-13*1e3],text_on=True,zorder=0.5,fs=19,lw=2,**kwargs):
        # Red Giants arXiv:[2007.03694]
        dat = loadtxt("limit_data/AxionElectron/RedGiants_HighMass.txt")
        plt.fill_between(dat[:,0],dat[:,1]/1e-3,y2=1e3,edgecolor=None,facecolor=col,zorder=zorder)
        plt.plot(dat[:,0],dat[:,1]/1e-3,color='k',alpha=1,zorder=zorder,lw=lw)
        if text_on: plt.text(text_pos[0],text_pos[1],r'{\bf Red giants (}$\omega${\bf Cen)}',fontsize=fs,color='w',clip_on=True,path_effects=line_background(1,'k'),ha='center',**kwargs)
        return dat

    def Xrays(ax,col='green',text_shift=[1,1*1e3],text_on=True,zorder=0.5,fs=17,rotation=-73,alpha=0.3,**kwargs):
        dat = loadtxt("limit_data/AxionElectron/Xray_1loop.txt")
        plt.fill_between(dat[:,0],dat[:,1]/1e-3,y2=1e3,edgecolor=None,facecolor=col,zorder=zorder,alpha=alpha)
        plt.plot(dat[:,0],dat[:,1]/1e-3,':',color='k',zorder=zorder,lw=2,alpha=1)

        if text_on:
            plt.text(2.5e3*text_shift[0],3.17e-20*text_shift[1],r'{\bf X-rays} (EM-anomaly free)',fontsize=fs,color='k',clip_on=True,rotation=rotation,**kwargs)
            #plt.text(1.32e4*text_shift[0],1.2e-15*text_shift[1],r'(EM anomaly-free ALP)',fontsize=fs*0.85,color='w',clip_on=True,rotation=rotation,**kwargs)

            return

    def SolarNu(ax,col='seagreen',text_pos=[0.8e-8,3.8e-11*1e3],text_on=True,zorder=0.7,fs=19,lw=2,**kwargs):
        # Solar neutrinos arXiv:[0807.2926]
        dat = loadtxt("limit_data/AxionElectron/SolarNu.txt")
        plt.fill_between(dat[:,0],dat[:,1]/1e-3,y2=1e3,edgecolor=None,facecolor=col,zorder=zorder)
        plt.plot(dat[:,0],dat[:,1]/1e-3,color='k',alpha=1,zorder=zorder,lw=lw)
        if text_on: plt.text(text_pos[0],text_pos[1],r'{\bf Solar} $\nu$',fontsize=fs,color='w',clip_on=True,path_effects=line_background(1,'k'),**kwargs,ha='center')
        return

    def WhiteDwarfHint(ax,col='k',text_pos=[1e-7,1e-13*1e3],facealpha=0.3,zorder=1.0,text_on=True,fs=20,**kwargs):
        # White dwarf hint arXiv:[1708.02111]
        dat = loadtxt("limit_data/AxionElectron/WDhint.txt")
        plt.fill_between(dat[:,0],dat[:,1]/1e-3,color=col,edgecolor=None,lw=0.001,zorder=zorder,alpha=facealpha)
        if text_on: plt.text(text_pos[0],text_pos[1],r'{\bf White dwarf hint}',fontsize=fs,clip_on=True,**kwargs)
        return

    def StellarBounds(ax,fs=30,Hint=True,text_on=True):
        AxionElectron.RedGiants(ax,text_on=text_on)
        AxionElectron.SolarNu(ax,text_on=text_on)
        if Hint:
            AxionElectron.WhiteDwarfHint(ax,text_on=text_on)
        return

    def IrreducibleFreezeIn(ax,text_label=r'{\bf Freeze-in}',text_pos=[6.5e5,6.2e-16*1e3],col='#376631',
                        edgecolor='k',text_col='w',fs=17,zorder=0.009,text_on=True,lw=1,facealpha=1,rotation=-73,edgealpha=1):
        dat = loadtxt("limit_data/AxionElectron/IrreducibleFreezeIn.txt")
        FilledLimit(ax,dat,text_label,text_pos=text_pos,col=col,text_col=text_col,
                    rotation=rotation,edgecolor=edgecolor,fs=fs,
                    zorder=zorder,text_on=text_on,lw=lw,ha='right',facealpha=facealpha,edgealpha=edgealpha,path_effects=line_background(1,'k'))
        return
    
    def Comagnetometers(ax,col=[0.75, 0.2, 0.2],fs=19,text_on=True,zorder=2,lw=1.5,text_shift=[1,1*1e3],Projection=False,**kwargs):
        dat = loadtxt("limit_data/AxionElectron/OldComagnetometers.txt")
        plt.fill_between(dat[:,0],dat[:,1]/1e-3,y2=1e3,edgecolor=None,facecolor=col,zorder=zorder)
        plt.plot(dat[:,0],dat[:,1]/1e-3,'k-',alpha=1,zorder=zorder,lw=lw)

        if text_on:
            plt.text(text_shift[0]*0.15e-14,text_shift[1]*1.5e-5,r'\begin{center}{\bf Old \linebreak comagnetometers} \linebreak (K-He)\end{center}',fontsize=fs,color='w',ha='center',va='top',clip_on=True,path_effects=line_background(1,'k'))
        
        if Projection:
            dat = loadtxt("limit_data/AxionElectron/Projections/FutureComagnetometers.txt")
            plt.fill_between(dat[:,0],dat[:,1]/1e-3,y2=1e3,edgecolor=None,alpha=0.01,facecolor='darkred',zorder=zorder-0.01,lw=0)
            plt.plot(dat[:,0],dat[:,1]/1e-3,'--',color='darkred',alpha=1,zorder=zorder-0.01,lw=lw)
            if text_on:
                plt.text(0.3e-18,1.8e-12,r'{\bf Future comagnetometers}',fontsize=14,color='darkred',ha='center',va='top',clip_on=True)

        return
    
    def ElectronStorageRing(ax,col='darkred',fs=14,text_on=True,zorder=2,lw=1.5,text_shift=[1,1*1e3],Projection=False,**kwargs):
        dat = loadtxt("limit_data/AxionElectron/Projections/ElectronStorageRing.txt")
        plt.fill_between(dat[:,0],dat[:,1]/1e-3,y2=1e3,edgecolor=None,alpha=0.05,facecolor=col,zorder=zorder-0.01,lw=0)
        plt.plot(dat[:,0],dat[:,1]/1e-3,'--',color=col,alpha=1,zorder=zorder-0.01,lw=lw)
        if text_on:
            plt.text(0.3e-18,3.15e-13,r'{\bf Electron Storage Ring}',fontsize=fs,color=col,ha='center',va='top',clip_on=True)

        return
    
    def FermionicAxionInterferometer(ax,col='#870032',fs=13,text_on=True,zorder=10,lw=1.5,text_shift=[1,1*1e3],Projection=False,**kwargs):
        dat = loadtxt("limit_data/AxionElectron/FermionicAxionInterferometer.txt")
        plt.fill_between(dat[:,0],dat[:,1]/1e-3,y2=1e3,edgecolor=None,facecolor=col,zorder=zorder)
        plt.plot(dat[:,0],dat[:,1]/1e-3,'k-',alpha=1,zorder=zorder,lw=lw)

        if text_on:
            plt.text(text_shift[0]*2.0e-8,text_shift[1]*6e-5,r'\begin{center} {\bf Fermionic axion \linebreak interferometer} \end{center}',fontsize=fs,color=col,ha='center',va='top',clip_on=True,path_effects=line_background(0.5,'k'))
        return

    def TorsionPendulumDM(ax,col='#a83248',fs=19,text_on=True,zorder=1.9,lw=1.5,text_shift=[1,1*1e3],Projection=False,**kwargs):
        dat = loadtxt("limit_data/AxionElectron/TorsionPendulum-DM.txt")
        plt.fill_between(dat[:,0],dat[:,1]/1e-3,y2=1e3,edgecolor=None,facecolor=col,zorder=zorder)
        plt.plot(dat[:,0],dat[:,1]/1e-3,'k-',alpha=1,zorder=zorder,lw=lw)

        if text_on:
            plt.text(text_shift[0]*0.1e-19,text_shift[1]*2e-8,r'\begin{center} {\bf Torsion \linebreak pendulum} \end{center}',fontsize=fs,color='w',ha='center',va='top',clip_on=True,path_effects=line_background(1,'k'))
        
        if Projection:
            dat = loadtxt("limit_data/AxionElectron/Projections/TorsionPendulum-DM.txt")
            plt.fill_between(dat[:,0],dat[:,1]/1e-3,y2=1e3,edgecolor=None,alpha=0.05,facecolor='darkred',zorder=-10,lw=0)
            plt.plot(dat[:,0],dat[:,1]/1e-3,'--',color='darkred',alpha=1,zorder=-10,lw=lw)
            if text_on:
                plt.text(9e-16,0.6e-14,r'\begin{center}{\bf Torsion \linebreak pendulum}\end{center}',fontsize=14,rotation=-15,color='darkred',ha='center',va='top',clip_on=True)

        return
    
    def TorsionPendulumSpin(ax,col=[0.2,0.2,0.2],fs=19,text_on=True,zorder=1.9,lw=1.5,text_shift=[1,1*1e3],**kwargs):
        dat = loadtxt("limit_data/AxionElectron/TorsionPendulum-Spin.txt")
        plt.fill_between(dat[:,0],dat[:,1]/1e-3,y2=1e3,edgecolor=None,facecolor=col,zorder=zorder)
        plt.plot(dat[:,0],dat[:,1]/1e-3,'k-',alpha=1,zorder=zorder,lw=lw)

        if text_on:
            plt.text(text_shift[0]*0.065e-7,text_shift[1]*1.6e-7,r'\begin{center} {\bf Torsion pendulum} \linebreak (dipole-dipole force)\end{center}',fontsize=fs,color='w',ha='center',va='top',clip_on=True,path_effects=line_background(1,'k'))
        return
    
    def Electron_gminus2(ax,col='gray',fs=19,text_on=True,zorder=1.9,lw=1.5,text_shift=[1,1*1e3],**kwargs):
        dat = loadtxt("limit_data/AxionElectron/Electron_g-2.txt")
        plt.fill_between(dat[:,0],dat[:,1]/1e-3,y2=1e3,edgecolor=None,facecolor=col,zorder=zorder)
        plt.plot(dat[:,0],dat[:,1]/1e-3,'k-',alpha=1,zorder=zorder,lw=lw)

        if text_on:
            plt.text(text_shift[0]*1e0,text_shift[1]*3.5e-5,r'{\bf Electron $g-2$}',fontsize=fs,color='w',ha='center',va='top',clip_on=True,path_effects=line_background(1,'k'))
        return
    
    def AxionWindMultilayer(ax,col='crimson',fs=13,text_on=True,zorder=-1,lw=1.5,text_shift=[1,1*1e3],rotation=50,SinglePhoton=True,**kwargs):
        dat = loadtxt("limit_data/AxionElectron/Projections/AxionWindMultilayer.txt")
        plt.fill_between(dat[:,0],dat[:,1]/1e-3,y2=1e3,edgecolor=None,alpha=0.05,facecolor=col,zorder=zorder-0.01,lw=0)
        plt.plot(dat[:,0],dat[:,1]/1e-3,'--',color=col,alpha=1,zorder=zorder-0.01,lw=lw)
        if text_on:
            plt.text(0.5e-5,1.15e-13,r'\begin{center}{\bf  Axion wind \linebreak multilayer}\end{center}',rotation=rotation,fontsize=fs,color=col,ha='center',va='top',clip_on=True)

        if SinglePhoton:
            dat = loadtxt("limit_data/AxionElectron/Projections/AxionWindMultilayer_SinglePhoton.txt")
            plt.plot(dat[:,0],dat[:,1]/1e-3,'--',color=col,alpha=1,zorder=zorder-0.01,lw=lw)
            if text_on:
                plt.text(0.7e-5,1.2e-15,r'\begin{center}{\bf  Axion wind multilayer \linebreak (single photon)}\end{center}',rotation=rotation,fontsize=fs*0.9,color=col,ha='center',va='top',clip_on=True)

        return
    

    def Semiconductors(ax,col='#3d1d01',fs=12,text_on=True,text_pos=[0.7e0,6.7e-9*1e3],lw=2,rotation=-88,zorder=1,**kwargs):
        dat = loadtxt("limit_data/AxionElectron/Projections/Semiconductors.txt")
        plt.plot(dat[:,0],dat[:,1]/1e-3,'--',color=col,alpha=1.0,zorder=zorder,lw=lw)
        if text_on:
            plt.text(text_pos[0],text_pos[1],r'{\bf Semiconductors}',fontsize=fs,color=col,ha='left',va='top',rotation=rotation,clip_on=True,**kwargs)
        return
    
    def Superconductors(ax,col='#3d1d01',fs=12,text_on=True,text_pos=[1.1e-3,9e-9*1e3],lw=2,rotation=-75,zorder=1,**kwargs):
        dat = loadtxt("limit_data/AxionElectron/Projections/Superconductors.txt")
        plt.plot(dat[:,0],dat[:,1]/1e-3,'-.',color=col,alpha=1.0,zorder=zorder,lw=lw)
        if text_on:
            plt.text(text_pos[0],text_pos[1],r'{\bf Superconductors}',fontsize=fs,color=col,ha='left',va='top',rotation=rotation,clip_on=True,**kwargs)
        return
    
        
    def SpinOrbitCoupling(ax,col='#3d1d01',fs=12,text_on=True,text_pos=[1.8e-2,9e-9*1e3],lw=2,rotation=-86,zorder=1,**kwargs):
        dat = loadtxt("limit_data/AxionElectron/Projections/SpinOrbitCoupling.txt")
        plt.plot(dat[:,0],dat[:,1]/1e-3,':',color=col,alpha=1.0,zorder=zorder,lw=lw)
        if text_on:
            plt.text(text_pos[0],text_pos[1],r'\begin{center}{\bf Spin-orbit}\end{center}',fontsize=fs,color=col,ha='left',va='top',rotation=rotation,clip_on=True,**kwargs)
        return

    def NVCenters(ax,col='red',fs=14,text_on=True,text_shift=[1,1*1e3],lw=2,zorder=-0.5,rotation=0,**kwargs):
        # NV center dc magnetometery
        dat = loadtxt("limit_data/AxionElectron/Projections/NVCenters.txt")
        plt.fill_between(dat[:,0],dat[:,1]/1e-3,y2=1e3,color=col,alpha=0.2,zorder=zorder)
        plt.plot(dat[:,0],dat[:,1]/1e-3,'--',color=col,alpha=0.7,zorder=zorder,lw=lw)
        if text_on:
            plt.text(text_shift[0]*2.4e-12,text_shift[1]*3e-14,r'{\bf NVCenters}',rotation=rotation,alpha=0.7,fontsize=fs-1,color=col,ha='center',va='top',clip_on=True,**kwargs)
        return
    
    def YIG(ax,col='#850735',fs=13,text_on=True,text_shift=[1,1*1e3],lw=2,zorder=-0.5,rotation=-90,**kwargs):
        # NV center dc magnetometery
        dat = loadtxt("limit_data/AxionElectron/Projections/YIG.txt")
        plt.fill_between(dat[:,0],dat[:,1]/1e-3,y2=1e3,color=col,alpha=0.2,zorder=zorder)
        plt.plot(dat[:,0],dat[:,1]/1e-3,'--',color=col,alpha=0.7,zorder=zorder,lw=lw)
        if text_on:
            plt.text(text_shift[0]*5e-4,text_shift[1]*1.25e-13,r'{\bf YIG}',rotation=rotation,alpha=0.7,fontsize=fs-1,color=col,ha='center',va='top',clip_on=True,**kwargs)
        return

    def UndergroundDetectors(ax,projection=False,fs=20,text_on=True):
        AxionElectron.XENONnT_Solar(ax,fs=fs+10,text_on=text_on)
        AxionElectron.PandaX(ax,fs=fs,text_on=text_on)
        AxionElectron.XENON1T(ax,fs=fs-2,text_on=text_on)
        AxionElectron.XENONnT(ax,fs=fs-2,text_on=text_on)
        AxionElectron.SolarBasin(ax,fs=fs-2,text_on=text_on)
        AxionElectron.SuperCDMS(ax,fs=fs,text_on=text_on)
        AxionElectron.EDELWEISS(ax,fs=fs-5,projection=projection,text_on=text_on)
        AxionElectron.DarkSide(ax,fs,text_on=text_on)
        if projection:
            AxionElectron.DARWIN(ax,fs=fs,text_on=text_on)
            AxionElectron.LZ(ax,fs=fs,text_on=text_on)
            AxionElectron.Semiconductors(ax,fs=fs-5,text_on=text_on)
        return

    def Haloscopes(ax,projection=False,fs=20,text_on=True):
        if projection:
            AxionElectron.Magnon(ax,fs=fs,text_on=text_on)
            AxionElectron.MagnonScan(ax,fs=fs,text_on=text_on)
        return
#==============================================================================#


#==============================================================================#
def MySaveFig(fig,pltname,pngsave=True):
    fig.savefig(pltdir+pltname+'.pdf',bbox_inches='tight')
    if pngsave:
        fig.set_facecolor('w') # <- not sure what matplotlib fucked up in the new version but it seems impossible to set png files to be not transparent now
        fig.savefig(pltdir_png+pltname+'.png',bbox_inches='tight',transparent=False)

def cbar(mappable,extend='neither',minorticklength=8,majorticklength=10,\
            minortickwidth=2,majortickwidth=2.5,pad=0.2,side="right",orientation="vertical"):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(side, size="5%", pad=pad)
    cbar = fig.colorbar(mappable, cax=cax,extend=extend,orientation=orientation)
    cbar.ax.tick_params(which='minor',length=minorticklength,width=minortickwidth)
    cbar.ax.tick_params(which='major',length=majorticklength,width=majortickwidth)
    cbar.solids.set_edgecolor("face")

    return cbar

def MySquarePlot(xlab='',ylab='',\
                 lw=2.5,lfs=45,tfs=25,size_x=13,size_y=12,Grid=False):
    plt.rcParams['axes.linewidth'] = lw
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=tfs)
    plt.rcParams.update({"text.usetex": True,"font.family": "serif","font.serif": ["Palatino"],})
    fig = plt.figure(figsize=(size_x,size_y))
    ax = fig.add_subplot(111)

    ax.set_xlabel(xlab,fontsize=lfs)
    ax.set_ylabel(ylab,fontsize=lfs)

    ax.tick_params(which='major',direction='in',width=2,length=13,right=True,top=True,pad=7)
    ax.tick_params(which='minor',direction='in',width=1,length=10,right=True,top=True)
    if Grid:
        ax.grid()
    return fig,ax

def MyDoublePlot(xlab1='',ylab1='',xlab2='',ylab2='',\
                 wspace=0.25,lw=2.5,lfs=45,tfs=25,size_x=20,size_y=11,Grid=False):
    plt.rcParams['axes.linewidth'] = lw
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=tfs)
    plt.rcParams.update({"text.usetex": True,"font.family": "serif","font.serif": ["Palatino"],})
    fig, axarr = plt.subplots(1, 2,figsize=(size_x,size_y))
    gs = gridspec.GridSpec(1, 2)
    gs.update(wspace=wspace)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax1.tick_params(which='major',direction='in',width=2,length=13,right=True,top=True,pad=7)
    ax1.tick_params(which='minor',direction='in',width=1,length=10,right=True,top=True)
    ax2.tick_params(which='major',direction='in',width=2,length=13,right=True,top=True,pad=7)
    ax2.tick_params(which='minor',direction='in',width=1,length=10,right=True,top=True)

    ax1.set_xlabel(xlab1,fontsize=lfs)
    ax1.set_ylabel(ylab1,fontsize=lfs)

    ax2.set_xlabel(xlab2,fontsize=lfs)
    ax2.set_ylabel(ylab2,fontsize=lfs)

    if Grid:
        ax1.grid()
        ax2.grid()
    return fig,ax1,ax2


def MyDoublePlot_Vertical(xlab1='',ylab1='',xlab2='',ylab2='',\
                     hspace=0.05,lw=2.5,lfs=45,tfs=30,size_x=15,size_y=14,Grid=False,height_ratios=[2.5,1]):
    plt.rcParams['axes.linewidth'] = lw
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=tfs)
    plt.rcParams.update({"text.usetex": True,"font.family": "serif","font.serif": ["Palatino"],})


    fig, axarr = plt.subplots(2,1,figsize=(size_x,size_y))
    gs = gridspec.GridSpec(2, 1,height_ratios=height_ratios)
    gs.update(hspace=hspace)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    ax1.tick_params(which='major',direction='in',width=2,length=13,right=False,top=True,pad=10)
    ax1.tick_params(which='minor',direction='in',width=1,length=10,right=False,top=True)

    ax2.tick_params(which='major',direction='in',width=2,length=13,right=False,top=True,pad=10)
    ax2.tick_params(which='minor',direction='in',width=1,length=10,right=False,top=True)

    ax1.set_xlabel(xlab1,fontsize=lfs)
    ax1.set_ylabel(ylab1,fontsize=lfs)

    ax2.set_xlabel(xlab2,fontsize=lfs)
    ax2.set_ylabel(ylab2,fontsize=lfs)


    if Grid:
        ax1.grid()
        ax2.grid()
    return fig,ax1,ax2



def MyTriplePlot(xlab1='',ylab1='',xlab2='',ylab2='',xlab3='',ylab3='',\
                 wspace=0.25,lw=2.5,lfs=45,tfs=25,size_x=20,size_y=7,Grid=False):
    plt.rcParams['axes.linewidth'] = lw
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=tfs)
    plt.rcParams.update({"text.usetex": True,"font.family": "serif","font.serif": ["Palatino"],})
    fig, axarr = plt.subplots(1, 3,figsize=(size_x,size_y))
    gs = gridspec.GridSpec(1, 3)
    gs.update(wspace=wspace)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])

    ax1.tick_params(which='major',direction='in',width=2,length=13,right=True,top=True,pad=7)
    ax1.tick_params(which='minor',direction='in',width=1,length=10,right=True,top=True)

    ax2.tick_params(which='major',direction='in',width=2,length=13,right=True,top=True,pad=7)
    ax2.tick_params(which='minor',direction='in',width=1,length=10,right=True,top=True)

    ax3.tick_params(which='major',direction='in',width=2,length=13,right=True,top=True,pad=7)
    ax3.tick_params(which='minor',direction='in',width=1,length=10,right=True,top=True)

    ax1.set_xlabel(xlab1,fontsize=lfs)
    ax1.set_ylabel(ylab1,fontsize=lfs)

    ax2.set_xlabel(xlab2,fontsize=lfs)
    ax2.set_ylabel(ylab2,fontsize=lfs)

    ax3.set_xlabel(xlab3,fontsize=lfs)
    ax3.set_ylabel(ylab3,fontsize=lfs)

    if Grid:
        ax1.grid()
        ax2.grid()
        ax3.grid()
    return fig,ax1,ax2,ax3
#==============================================================================#

#==============================================================================#
def reverse_colourmap(cmap, name = 'my_cmap_r'):
    reverse = []
    k = []

    for key in cmap._segmentdata:
        k.append(key)
        channel = cmap._segmentdata[key]
        data = []

        for t in channel:
            data.append((1-t[0],t[2],t[1]))
        reverse.append(sorted(data))

    LinearL = dict(zip(k,reverse))
    my_cmap_r = mpl.colors.LinearSegmentedColormap(name, LinearL)
    return my_cmap_r
#==============================================================================#




from matplotlib import patches
from matplotlib import text as mtext
import numpy as np
import math

class CurvedText(mtext.Text):
    """
    A text object that follows an arbitrary curve.
    """
    def __init__(self, x, y, text, axes, **kwargs):
        super(CurvedText, self).__init__(x[0],y[0],' ', **kwargs)

        axes.add_artist(self)

        ##saving the curve:
        self.__x = x
        self.__y = y
        self.__zorder = self.get_zorder()

        ##creating the text objects
        self.__Characters = []
        for c in text:
            if c == ' ':
                ##make this an invisible 'a':
                t = mtext.Text(0,0,'a')
                t.set_alpha(0.0)
            else:
                t = mtext.Text(0,0,c, **kwargs)

            #resetting unnecessary arguments
            t.set_ha('center')
            t.set_rotation(0)
            t.set_zorder(self.__zorder +1)

            self.__Characters.append((c,t))
            axes.add_artist(t)


    ##overloading some member functions, to assure correct functionality
    ##on update
    def set_zorder(self, zorder):
        super(CurvedText, self).set_zorder(zorder)
        self.__zorder = self.get_zorder()
        for c,t in self.__Characters:
            t.set_zorder(self.__zorder+1)

    def draw(self, renderer, *args, **kwargs):
        """
        Overload of the Text.draw() function. Do not do
        do any drawing, but update the positions and rotation
        angles of self.__Characters.
        """
        self.update_positions(renderer)

    def update_positions(self,renderer):
        """
        Update positions and rotations of the individual text elements.
        """

        #preparations

        ##determining the aspect ratio:
        ##from https://stackoverflow.com/a/42014041/2454357

        ##data limits
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        ## Axis size on figure
        figW, figH = self.axes.get_figure().get_size_inches()
        ## Ratio of display units
        _, _, w, h = self.axes.get_position().bounds
        ##final aspect ratio
        aspect = ((figW * w)/(figH * h))*(ylim[1]-ylim[0])/(xlim[1]-xlim[0])

        #points of the curve in figure coordinates:
        x_fig,y_fig = (
            np.array(l) for l in zip(*self.axes.transData.transform([
            (i,j) for i,j in zip(self.__x,self.__y)
            ]))
        )

        #point distances in figure coordinates
        x_fig_dist = (x_fig[1:]-x_fig[:-1])
        y_fig_dist = (y_fig[1:]-y_fig[:-1])
        r_fig_dist = np.sqrt(x_fig_dist**2+y_fig_dist**2)

        #arc length in figure coordinates
        l_fig = np.insert(np.cumsum(r_fig_dist),0,0)

        #angles in figure coordinates
        rads = np.arctan2((y_fig[1:] - y_fig[:-1]),(x_fig[1:] - x_fig[:-1]))
        degs = np.rad2deg(rads)


        rel_pos = 10
        for c,t in self.__Characters:
            #finding the width of c:
            t.set_rotation(0)
            t.set_va('center')
            bbox1  = t.get_window_extent(renderer=renderer)
            w = bbox1.width
            h = bbox1.height

            #ignore all letters that don't fit:
            if rel_pos+w/2 > l_fig[-1]:
                t.set_alpha(0.0)
                rel_pos += w
                continue

            elif c != ' ':
                t.set_alpha(1.0)

            #finding the two data points between which the horizontal
            #center point of the character will be situated
            #left and right indices:
            il = np.where(rel_pos+w/2 >= l_fig)[0][-1]
            ir = np.where(rel_pos+w/2 <= l_fig)[0][0]

            #if we exactly hit a data point:
            if ir == il:
                ir += 1

            #how much of the letter width was needed to find il:
            used = l_fig[il]-rel_pos
            rel_pos = l_fig[il]

            #relative distance between il and ir where the center
            #of the character will be
            fraction = (w/2-used)/r_fig_dist[il]

            ##setting the character position in data coordinates:
            ##interpolate between the two points:
            x = self.__x[il]+fraction*(self.__x[ir]-self.__x[il])
            y = self.__y[il]+fraction*(self.__y[ir]-self.__y[il])

            #getting the offset when setting correct vertical alignment
            #in data coordinates
            t.set_va(self.get_va())
            bbox2  = t.get_window_extent(renderer=renderer)

            bbox1d = self.axes.transData.inverted().transform(bbox1)
            bbox2d = self.axes.transData.inverted().transform(bbox2)
            dr = np.array(bbox2d[0]-bbox1d[0])

            #the rotation/stretch matrix
            rad = rads[il]
            rot_mat = np.array([
                [math.cos(rad), math.sin(rad)*aspect],
                [-math.sin(rad)/aspect, math.cos(rad)]
            ])

            ##computing the offset vector of the rotated character
            drp = np.dot(dr,rot_mat)

            #setting final position and rotation:
            t.set_position(np.array([x,y])+drp)
            t.set_rotation(degs[il])

            t.set_va('center')
            t.set_ha('center')

            #updating rel_pos to right edge of character
            rel_pos += w-used
