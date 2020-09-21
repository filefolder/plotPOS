#!/usr/bin/python3

#plot timeseries from gnss POS files
# robert.pickle@anu.edu.au

#BETA 0.1 


import math, glob, datetime, os, sys, time, argparse

if len(sys.argv) < 2:
	print("USE: plotPOS.py <file or directory containing .pos files> <flags> (-h for help)"); exit()


import scipy.ndimage.filters
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
#for the filter
from scipy.signal import lfilter, butter, lfilter_zi, filtfilt
from scipy.ndimage.filters import gaussian_filter #for outlier removal
from scipy import sin, cos, tan, arctan, arctan2, arccos, pi
from scipy.interpolate import UnivariateSpline #for filter


### set up commandline arguments
parser = argparse.ArgumentParser(description='VERSION 0.1 // plot POS files and fix offsets and other things')

#first element
parser.add_argument('targetdir_or_file', action="store", help="filename or directory containing POS files",type=str)
parser.add_argument('-plotdir', 	 action="store", dest='plotdir',				help='where output files are put', default='./plots')
parser.add_argument('-eqfiles', 	 action="store", dest='renamefiles',			help='rename files (full path, comma separated)')
parser.add_argument('-numproc', 	 action="store", dest='numproc',type=int,		help='number processes',default=4)
parser.add_argument('-days_per_data',action="store", dest='days_per_data',type=int,	help='days combined per datapoint (default = 1 day/file)',default=1)

parser.add_argument('-save', 			action="store_true",dest='save',			help='save plots to plotdir?',default=True) #why is this needed exactly
parser.add_argument('-fix_offsets', 	action="store_true",dest='fix_offsets',		help='fix offsets listed in eqfiles? (FALSE)',default=False)
parser.add_argument('-dont_overwrite', 	action="store_true",dest='dont_overwrite',	help='dont overwrite existing plots (FALSE)',default=False)
parser.add_argument('-savenpz', 		action="store_true",dest='savenpz',			help='save data as numpy binary? (FALSE)',default=False)
parser.add_argument('-interactive', 	action="store_true",dest='interactive',		help='enter interactive plot mode? (FALSE)',default=False)
parser.add_argument('-vert_residual', 	action="store_true",dest='vert_residual',	help='take residual of vertical data? (FALSE)', default=False)
parser.add_argument('-write_verts', 	action="store_true",dest='write_verts',		help='NOT IN USE // write vertical velocity data to file? (FALSE)',default=False) #eventually ALL velocity data..
parser.add_argument('-write_offsets', 	action="store_true",dest='write_offsets',	help='write calculated offsets to an eqfile? (FALSE)',default=False)
parser.add_argument('-plot_spectral', 	action="store_true",dest='plot_spectral',	help='IN PROGRESS // also plot spectral power density information? (FALSE)',default=False)

argresults = parser.parse_args()

"""
######################################################### done with functions, program begins 
outdir="/home/bob/gnss/nu_plotsfigs/"
renamedir = "/home/bob/gnss/"
renamefiles = ['custom.eq','nz.eq','itrf14.less.eq','banish.eq'] #GLOBK FORMAT // rename MDO1     MDO1_XPS 2009 05 01 00 00 2100 02 13 00 00 

#### set parameters TODO set flags!

numproc = 20
days_per_data = 7 #TODO: should calculate this from data itself

########## output program settings
#SAVING and STORING these plots
save=True
interactive=False #plot for interactive viewing
overwrite=True #overwrite? setting this to False will skip pre-existing for speed
savenpz=False #save all the data in a npz (e.g. a single plot) to a binary, useful to work with later or interactively

#take residual of vertical data?
vert_residual = False
#write verticals to a .vel structured file?
write_verts = False
#are we interested in a "fixed" version with offsets corrected?
do_fixoffset=True
#want to assess spectral stuff? for hopefully coming up with alternate way to calc random walk /flicker et al. possibly also white 
#if so, probably will want to fix offsets (above)
plot_spectral=False

#write out an eq file of offsets? only works with do_fixoffset=True (for now)
write_offsets=True
"""
#turn the args into vars
fileordir =		argresults.targetdir_or_file
plotdir = 		str(argresults.plotdir).rstrip('/')
renamefiles = 	argresults.renamefiles
if renamefiles: renamefiles = list(renamefiles.split(','))
if not renamefiles: renamefiles = []
numproc = 		argresults.numproc
days_per_data = argresults.days_per_data
save = 			argresults.save
dont_overwrite= argresults.dont_overwrite
savenpz = 		argresults.savenpz
interactive = 	argresults.interactive
vert_residual = argresults.vert_residual
write_verts =	argresults.write_verts
write_offsets = argresults.write_offsets
fix_offsets = 	argresults.fix_offsets
plot_spectral = argresults.plot_spectral


#################################PARAMETERS YOU PROBABLY WONT NEED TO CHANGE
days_per_year=365.24 #if you want to be specific... 

#spectral constants
samplerate = days_per_year/days_per_data
seglen = 512
window_type='hann'

#minimums
mincontinuouslen = 20 #if there are less than this amount of data points then assumption is this is campaign mode data (no renames, no filters)
minlen = 2 #sites must have this much data to plot

#definition of baseline 3 month filter
filterlen = days_per_data/days_per_year*4 

#plotting format stuff (TODO add more)
dot_color = 'royalblue'
errorbar_color = 'grey'
ytickfontsize = 10

#default filenames / erase them or nah
outoffsetfile="plotpos.offsets.eq" #the only way this works is if each process quickly writes their offsets all at the same time
if write_offsets: os.system("rm plotpos.offsets.eq") #can be smarter but afraid of rm-ing stuff 

#output file for vertical velocities in GMT format
vert_vel_file = "verticals.vel"
###################################

#do a little prep / warning

#tmp location
if sys.platform == 'linux': tmpdir = "/dev/shm/" #with trailing slash please
else: tmpdir = '/tmp/'


if (fix_offsets or write_offsets) and not renamefiles: 
	print("no eqfiles given! not fixing offsets") 
	fix_offsets = False; write_offsets = False

if write_verts and not fix_offsets: print("warning, vertical velocities calculated without fixing offset!")

#ok we're going to assume a single variable at first, which we must determine is a directory or a single file
if os.path.isfile(fileordir): #it's just the one file
	fyles = [fileordir]

if os.path.isdir(fileordir): #searching this directory
	if interactive: print("interactive mode must be done with one file only"); exit()
	if savenpz: print("can only save npz data for one site at a time! not saving.. "); savenpz = False
	if not os.path.isdir(plotdir): os.system("mkdir %s" % plotdir) #make sure output dir exists
	if fileordir[-1] != "/": fileordir+="/"
	fyles = glob.glob(fileordir+"*.pos")
	fyles.sort()


#print(hello)
#######################define functions here

def getfromPOS(fyle):
	f = open(fyle,'r')
	yyyymmdd=[];dates=[];mjds=[];sn=[]; se=[];su=[]; lats=[]; lons=[]; heights=[];dlats=[]; dlons=[];dheights=[]#;x=[];y=[];z=[];sx=[];sy=[];sz=[]
	for line in f:
		if line[0:15]=='4-character ID:': name=line[16:20]; continue
		if line[0:15]=='Station name  :': altname=str(line[16:33]).strip(); continue		
		if "NEU Reference position" in line:
			reflat =     float(line.split()[4])
			reflon =     float(line.split()[5])
			refheight =  float(line.split()[6])
			continue
		if line[0]!=' ': continue
		line = line.split()
		yyyymmdd.append( int(line[0] ) )
		mjds.append(     float(line[2] ) )
		lats.append(     float(line[12]) )
		lons.append(     float(line[13]) )
		heights.append(  float(line[14]) )
		dlats.append(    float(line[15]) )
		dlons.append(    float(line[16]) )
		dheights.append( float(line[17]) )
		sn.append(       float(line[18]) )
		se.append(       float(line[19]) )
		su.append(       float(line[20]) )
#		x.append(        float(line[3]) )
#		y.append(        float(line[4]) )
#		z.append(        float(line[5]) )
#		sx.append(       float(line[6]) )
#		sy.append(       float(line[7]) )
#		sz.append(       float(line[8]) )
	f.close()
	return name,altname,yyyymmdd,reflat,reflon,refheight,mjds,lats,lons,heights,dlats,dlons,dheights,sn,se,su

def mjd_to_decyear(x):
	#mjd 0 is nov 17 1858, or 1858.88
	return [round(i/days_per_year + 1858.88,4) for i in x]

def date_to_doy(date, format='%Y%m%d'): #gives year, doy
    date = datetime.datetime.strptime(date,format)
    new_year_day = datetime.datetime(year=date.year, month=1, day=1)
    return date.year, (date - new_year_day).days + 1

def toYearFraction(date):
    def sinceEpoch(date): # returns seconds since epoch
        return time.mktime(date.timetuple())
    s = sinceEpoch
    year = date.year
    startOfThisYear = datetime(year=year, month=1, day=1)
    startOfNextYear = datetime(year=year+1, month=1, day=1)
    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = np.round(yearElapsed/yearDuration,4)
    return date.year + fraction

def reject_outliers(data,m):
	data = np.array(data)
	if len(data) == 1: return data	
	d = np.abs(data - np.median(data))
	mdev = np.median(d)
	if mdev == 0 : return data
	s = d/mdev if mdev else 0.
	return data[s<m]

def vect_rejected_outliers(x,data,m): #returns shortened vectors with bad data gone
	data = np.array(data); x = np.array(x)
	if len(data) == 1: return data	
	d = np.abs(data - np.median(data))
	mdev = np.median(d)
	if mdev == 0 : return data
	s = d/mdev if mdev else 0.
	tfarray = s<m #true/false
	return x[tfarray],data[tfarray] #this doesn't work. 


def reject_noisy(t,y,errors,maxerror=.030): #remove data that has very large errorbars
	#prob need to convert to np
	errors = np.array(errors); t = np.array(t); y = np.array(y)
	#TODO: calc std dev, not some idiotic limit
	boolvect = errors<maxerror
	return t[boolvect], y[boolvect],errors[boolvect]
	

#TODO: need to make this more of a rolling STD calculator
def reject_gaussian_outliers(t,y,errors,sig=3,dist=20): #remove data that is off by 20 mm from 3std filter
	errors = np.array(errors); t = np.array(t); y = np.array(y)
	f = gaussian_filter(y,sigma=sig, mode='mirror',truncate=3,order=0)
	boolvect = abs(y-f) < dist
	return t[boolvect], y[boolvect],errors[boolvect]




def spherical_distance(lat1, long1, lat2, long2):
    phi1 = 0.5*pi - lat1
    phi2 = 0.5*pi - lat2
    r = 0.5*(6378137 + 6356752) # mean radius in meters
    t = sin(phi1)*sin(phi2)*cos(long1-long2) + cos(phi1)*cos(phi2)
    return r * arccos(t)

def ellipsoidal_distance(lat1, long1, lat2, long2):
	#courtesy of https://www.johndcook.com/blog/2018/11/24/spheroid-distance/
    a = 6378137.0 # equatorial radius in meters 
    f = 1/298.257223563 # ellipsoid flattening 
    b = (1 - f)*a 
    tolerance = 1e-11 # to stop iteration

    phi1, phi2 = lat1, lat2
    U1 = arctan((1-f)*tan(phi1))
    U2 = arctan((1-f)*tan(phi2))
    L1, L2 = long1, long2
    L = L2 - L1

    lambda_old = L + 0

    while True:   
        t = (cos(U2)*sin(lambda_old))**2
        t += (cos(U1)*sin(U2) - sin(U1)*cos(U2)*cos(lambda_old))**2
        sin_sigma = t**0.5
        cos_sigma = sin(U1)*sin(U2) + cos(U1)*cos(U2)*cos(lambda_old)
        sigma = arctan2(sin_sigma, cos_sigma) 
    
        sin_alpha = cos(U1)*cos(U2)*sin(lambda_old) / sin_sigma
        cos_sq_alpha = 1 - sin_alpha**2
        cos_2sigma_m = cos_sigma - 2*sin(U1)*sin(U2)/cos_sq_alpha
        C = f*cos_sq_alpha*(4 + f*(4-3*cos_sq_alpha))/16
    
        t = sigma + C*sin_sigma*(cos_2sigma_m + C*cos_sigma*(-1 + 2*cos_2sigma_m**2))
        lambda_new = L + (1 - C)*f*sin_alpha*t
        if abs(lambda_new - lambda_old) <= tolerance:
            break
        else:
            lambda_old = lambda_new

    u2 = cos_sq_alpha*((a**2 - b**2)/b**2)
    A = 1 + (u2/16384)*(4096 + u2*(-768+u2*(320 - 175*u2)))
    B = (u2/1024)*(256 + u2*(-128 + u2*(74 - 47*u2)))
    t = cos_2sigma_m + 0.25*B*(cos_sigma*(-1 + 2*cos_2sigma_m**2))
    t -= (B/6)*cos_2sigma_m*(-3 + 4*sin_sigma**2)*(-3 + 4*cos_2sigma_m**2)
    delta_sigma = B * sin_sigma * t
    s = b*A*(sigma - delta_sigma)
    return s



def getmultivectstats(data): #each vector must be same length
	meanvect=[];medianvect=[];modevect=[];stdvect=[];varvect=[]
	for j in range(len(data[0])): #go through each each vector element
		temp = [];
		for i in range(len(data)): #go through each vector
			temp.append(data[i][j])
		temp = reject_outliers(temp,4)
		meanvect.append(np.mean(temp))
		medianvect.append(np.median(temp))
		modevect.append(np.median(temp))
		stdvect.append(np.std(temp))
		varvect.append(np.var(temp))
	return meanvect,medianvect,modevect,stdvect,varvect


def get_dt(x): #tests if a time series is regularly spaced or not, and also returns the space
	x = np.array(x)
	y = x[1:]
	x = x[0:-1]
	gaps = y-x
	avggap = np.mean(y-x)
	if avggap % 1 == 0: return True,int(avggap)
	else:return False,avggap



#this version handles irregularly spaced data WITH error weighting. NEEDS WORK 
def quickfilter(t,x,e,dt=7,freq=7/days_per_year,filttype='lowpass'):
	if len(x) <= 3: return np.array([np.mean(x) for barf in x])
	freq = max(0.5/len(t),freq) #makes sure we don't filter longer than possible
	#get our spline
	spl = UnivariateSpline(t, x, w=[sigma**-1 for sigma in e], s=2/freq ,k=2, ext=3, check_finite=False) ##what is k (order) ?? strangely 1 seems to work best?
	#^^^^ smoothing factor s is tricky, but logically should be proportional to filter length. maxing out at half the wavelength seems to work ok 

	#now convert the original time vector t to one where the missing gaps are filled
	#t = np.arange(t[0],t[-1]+dt/days_per_year,step=dt/days_per_year) #sometimes returns an incorrect-length array
	t = np.linspace(t[0],t[-1],num=len(t)) #better? 

	x = spl(t) #converts obpect to an array, at each time t (now gap-free) 
	# Create an order 3?4? lowpass butterworth filter.
	b, a = butter(4,freq,btype=filttype) #N * size of filt (e.g. 7 days) per year
	# Apply the filter to xn.  Use lfilter_zi to choose the initial condition of the filter.
	zi = lfilter_zi(b, a)
	z, _ = lfilter(b, a, x, zi=zi*x[0])
	# Apply the filter again, to have a result filtered at an order the same as filtfilt.
	z2, _ = lfilter(b, a, z, zi=zi*z[0])
	# Use filtfilt to apply the filter.
	return filtfilt(b, a, x, method='gust') #gustaffson's method
	#return filtfilt(b, a, x, method='pad') #padding (mirrored) 


#can use gaussian_filter (imported at top). the sigma is effectively the filter length... number of data points?
#hmm = gaussian_filter(x0,sigma=days_per_year/days_per_data,mode='mirror',truncate=3) #gaussian year filter, removing data std > 3
#use a series of these filters to calculate (weighted) offsets
# ***doesn't work that great. hard to know what value of truncate is appropriate 
def calc_offset_gauss(x,unit=days_per_data):
	lefts = []; rights = []
	for days in np.arange(10,350,unit): #taking a lot of averages
		left = gaussian_filter(x,sigma=days/unit,mode='mirror',order=0,truncate=.1)[0]
		lefts.append( left)
		right = gaussian_filter(x,sigma=days/unit,mode='mirror',order=0,truncate=.1)[-1]
		rights.append( right )
	return np.mean(lefts), np.mean(rights)


#need a function to calculate residual timeseries
def getresidual(time,val,sig):
	p = np.polyfit(time,val,deg=1,w=[1/i for i in sig]) #for gaussian uncert use 1/sig
	fit = np.polyval(p,time)
	return val-fit


#ugly but works
def decyear2doy(x):
	decx = x%1
	doy = int(np.round(decx*days_per_year,0)+1)
	outstr = "%s-%s" % (int(x),doy)
	return outstr

def moveXtoback(sub_eqdata):
	for m in range(len(sub_eqdata)):
		for eq in sub_eqdata: 
			if eq[1][5:6] == "X": sub_eqdata.append(eq); sub_eqdata.remove(eq)
	


#TODO set minimum time spacing to "days_per_data" length
#TODO for campaign: derive and plot linear fits (different from original file?)

def plotcampaign(times,XXX,sigXXX,axX): #campaign or shortened data, typically  without renames ***TODO eventually need to just integrate all of these
	ymid = np.mean(XXX); ystd = max(1.0,np.std(XXX)); ymin = min(-3,ymid-4*ystd); ymax = max(3,ymid+4*ystd)
	#can set the potential y-limits now
	if axX == axU: ymin = ymid-2*ystd; ymax = ymid+2*ystd
	#if 4*ystd < max(sigXXX):  #rescale if sigma off the charts. careful of this, if a single campaign datapoint has a mondo error it causes huge problems 
	#	ymin = ymid -2*max(sigXXX)
	#	ymax = ymid +2*max(sigXXX) 
	axX.set_ylim((ymin,ymax)) #ploting +/- 4 std from the mean
	axX.axhline(y=0,linewidth=2,color='k') #plot the zero axis for (sparse) campaign sites
	#TODO need to change colour for days which are XPS/XCL
	#if sub_eqdata[i][1][5:8] == "XPS":
	#	axX.errorbar(times0,x0,yerr=s0,ecolor='hotpink',marker='o',markersize=2,linestyle='',zorder=0)
	#else: axX.errorbar(times0,x0,yerr=s0,ecolor='grey',marker='o',markersize=3,linestyle='',zorder=0)
	axX.errorbar(times,XXX,yerr=sigXXX,color=dot_color,ecolor=errorbar_color,marker='o',markersize=3,linestyle='')
	axX.grid(color='grey',linewidth=1,linestyle=':',alpha=0.5,axis='y',which='both')
	axX.grid(color='grey',linewidth=1,linestyle=':',alpha=0.5,axis='x',which='major')
	axX.tick_params(axis='y',labelsize=ytickfontsize)
	axX.yaxis.set_minor_locator(MultipleLocator(1.0))
	axX.xaxis.set_minor_locator(MultipleLocator(3./12))
	return

def plotcontinuous(times,sub_eqdata,XXX,sigXXX,axX,plot=True):
	#can set the potential y-limits now
	ymid = np.mean(XXX); ystd = np.std(XXX); ymin = ymid-4*ystd; ymax = ymid+4*ystd #TODO make this more sane
	#TODO: possibly re-config error if bigger than the yscale?
	if axX == axU: ymin = ymid-3*ystd; ymax = ymid+3*ystd
	if plot: axX.set_ylim((ymin,ymax)) #ploting +/- 4 std from the mean
	#deal with instances of no offsets, or only 1 GPS listing
	if len(sub_eqdata) == 0 : #easy
		if plot:
			axX.errorbar(times,XXX,yerr=sigXXX,color=dot_color,ecolor=errorbar_color,marker='o',markersize=3,linestyle='')
			axX.plot(times,quickfilter(times,XXX,sigXXX,dt=days_per_data,freq=days_per_data/days_per_year*4),linewidth=3,color='k') #1 year=365.24
			axX.plot(times,quickfilter(times,XXX,sigXXX,dt=days_per_data,freq=days_per_data/days_per_year),linewidth=2,color='r') #plot 1 year filter
	pre_pos = []; post_pos =[]; offsets=[]
	for i in range(len(sub_eqdata)):

		#skip all XCL
		if sub_eqdata[i][1][5:8] == "XCL": continue
		 
		starttime = sub_eqdata[i][2]
		endtime = sub_eqdata[i][3]

		#need to slice this with indexes
		times0 = []; x0 = []; s0 = []
		for j in range(len(times)):
			if times[j] >= starttime and times[j] < endtime: 
				times0.append(times[j])
				x0.append(XXX[j])
				s0.append(sigXXX[j])

		if len(times0) == 0: #no data in this rename.
			offsets.append(0)
			continue
		
		#attempt to discourage bad data (std or size of error) from influence. needs work but managed to remove the super crazy bad data
		times0,x0,s0 = reject_gaussian_outliers(times0,x0,s0,sig=3,dist=20)

		#have to do this right away to establish plot in which to build on
		if plot: 
			if sub_eqdata[i][1][5:8] == "XPS":
				axX.errorbar(times0,x0,yerr=s0,color=dot_color,ecolor='hotpink',marker='o',markersize=3,linestyle='',zorder=0)
			else: axX.errorbar(times0,x0,yerr=s0,color=dot_color,ecolor=errorbar_color,marker='o',markersize=3,linestyle='',zorder=0)


		#set up filter lengths
		filt_3mo = days_per_data/days_per_year*4 #3 months
		filt_6mo = days_per_data/days_per_year*2 #6
		filt_9mo = days_per_data/days_per_year*4/3 #9
		filt_yr  = days_per_data/days_per_year #year


		#create low pass estimates to overlay on the data
		f0 = quickfilter(times0,x0,s0,dt=days_per_data,freq=filt_3mo) 
		f1 = quickfilter(times0,x0,s0,dt=days_per_data,freq=filt_6mo) 
		f2 = quickfilter(times0,x0,s0,dt=days_per_data,freq=filt_9mo)
		f3 = quickfilter(times0,x0,s0,dt=days_per_data,freq=filt_yr)


		offset = 0.0
		if sub_eqdata[i][1][5:6].upper() != "X": 
			if not pre_pos: #doesn't exist yet, so we're only defining the preoffset for the NEXT segment e.g. [-1]
				try: pre_pos = (3*f0[-1] + 6*f1[-1] + 9*f2[-1] + 12*f3[-1])/30 #weighted. works OK until something more clever pops into my head
				except Exception as ex: print("problem with preoffset, site %s rename %s" % (site,sub_eqdata[i][1])); print(ex)
			else:
				try: post_pos = (3*f0[0] + 6*f1[0] + 9*f2[0] + 12*f3[0])/30 #weighted
				except Exception as ex: print("problem with postoffset, site %s rename %s" % (site,sub_eqdata[i][1])); print(ex)
				#if either of these pre/post offsets are unable to be calculated, append an offset of zero
				if pre_pos and post_pos:
					offset = post_pos - pre_pos
				else:
					offset = 0.1 #set to a specific value for testing. should never happen

				offsets.append(offset)

				pre_pos =  (3*f0[-1] + 6*f1[-1] + 9*f2[-1] + 12*f3[-1])/30 #setup the next segment..


			if plot:
				#plot the label/offset amount. only plot labels for N, plot offset for all
				axX.axvline(x=sub_eqdata[i][2],color='purple',linewidth=2,linestyle='--')
				if round(offset,1) != 0.0:
					if axX == axN:
						axX.text(sub_eqdata[i][2]+.002,ymin+.78*(ymax-ymin),"%s %.1f mm" % (sub_eqdata[i][1][5:8],offset),fontsize=11,color='purple',rotation=270)
					else:
						axX.text(sub_eqdata[i][2]+.002,ymin+.85*(ymax-ymin),"%.1f mm" % offset,fontsize=11,color='purple',rotation=270)
				else:
					if axX == axN:
						axX.text(sub_eqdata[i][2]+.002,ymin+.92*(ymax-ymin),"%s" % sub_eqdata[i][1][5:8],fontsize=11,color='purple',rotation=270)

		#print("offsets = ",offsets)
		if plot and sub_eqdata[i][1][5:8] != "XCL" and sub_eqdata[i][1][5:8] != "XPS":
			#if we're plotting the filters, need to get a time array that is the same size (same method used in the quickfilter function)
			#t_nogaps = np.arange(times0[0],times0[-1]+np.round(days_per_data/days_per_year,4),step=np.round(days_per_data/days_per_year,4))
			#some issue here at times, where length of t_nogaps is one off from length of filter f0/f1 et al. fix a stupid way
			#if len(t_nogaps) == len(f0)-1: t_nogaps = np.append(t_nogaps, t_nogaps[-1] + np.round(days_per_data/days_per_year,4))
			#elif len(t_nogaps) == len(f0)+1: t_nogaps = t_nogaps[:-1]

			#for weird rename borders the length of time array can be off slightly. just force t_nogaps to be the same length as the filters
			t_nogaps = np.linspace(times0[0],times0[-1],num=len(f0))

			axX.plot(t_nogaps,f0,linewidth=1,color='k') #plot 3 month filter
			#axX.plot(t_nogaps,f1,linewidth=2,color='g') #plot 1/2 year filter
			axX.plot(t_nogaps,f3,linewidth=2,color='r') #plot 1 year filter

	if plot:
		for eq in sub_eqdata: #plot the XPS/XCL vertical rename breaks which have no offset data
			xc = eq[2]
			if eq[1][5:8].upper() == "XPS":
				axX.text(xc+.002,ymin+.92*(ymax-ymin),"%s" % sub_eqdata[i][1][5:8],fontsize=9,color='hotpink',rotation=270)
				axX.axvline(x=xc,color='hotpink',linewidth=2,linestyle=':')
				axX.axvline(x=eq[3],color='hotpink',linewidth=2,linestyle=':')
			if eq[1][5:8].upper() == "XCL":
				axX.text(xc+.002,ymin+.92*(ymax-ymin),"%s" % sub_eqdata[i][1][5:8],fontsize=9,color='red',rotation=270)
				axX.axvline(x=xc,color='red',linewidth=2,linestyle=':')
				axX.axvline(x=eq[3],color='red',linewidth=2,linestyle=':')
	
		#calculate and print the vertical velocity now (TODO: all velocities?)
		if axX == axU:
			#solves for slope (mm/yr), so essentially Uvel. sigma is the sqrt of the covariance
			[Uvel,Uint],cov_mat = np.polyfit(times,XXX,w = [_i**-1 for _i in sigXXX],deg=1,full=False,cov='unscaled')
			slope_err = cov_mat[0,0]**.5
			axX.text(.77,.04,"%.1f+/-%.1f mm/yr" % (Uvel,slope_err), transform=axX.transAxes,fontsize=18,color='red')
			#write the vert dat to a vel file 7.46527702579 46.8770977358 0 0 0 0 0 0 0 0.190957205564 0 0 ZIMM_GPS
			#print("%.5f %.5f 0 0 0 0 0 0 0 0 %.4f 0 0 %s" % (reflat,reflon,site))		
			#vertfyle = open("vertical.vel",'a')
			

		axX.grid(color='grey',linewidth=1,linestyle=':',alpha=0.5,axis='x',which='major')
		axX.tick_params(axis='y',labelsize=ytickfontsize)
		#TODO: need some logic here about the grid size
		if (ymax-ymin) <= 40 :
			axX.grid(color='grey',linewidth=1,linestyle=':',alpha=0.5,axis='y',which='both')
			axX.yaxis.set_major_locator(MultipleLocator(1.0))
		if (ymax-ymin) > 40 and (ymax-ymin) <= 100 : #turn off grid if wider than .4 meters
			axX.grid(color='grey',linewidth=1,linestyle='-.',alpha=0.5,axis='y',which='both')
			axX.yaxis.set_major_locator(MultipleLocator(10.0))
			axX.yaxis.set_minor_locator(MultipleLocator(5.0))
		if ymax - ymin > 100: #turn off grid if wider than 1 meters
			axX.grid(color='grey',linewidth=1,linestyle='--',alpha=0.5,axis='y',which='both')
			axX.yaxis.set_major_locator(MultipleLocator(20.0))
			axX.yaxis.set_minor_locator(MultipleLocator(10.0))
		
		axX.xaxis.set_minor_locator(MultipleLocator(3./12)) #quarterly
		axX.xaxis.set_major_locator(MultipleLocator(1.)) #yearly

	return offsets


############################################ start code here



#load all the POS data we're going to process
data = []
for fyle in fyles:
	site,altname,yyyymmdd,reflat,reflon,refheight,mjds,lats,lons,heights,dlats,dlons,dheights,sn,se,su = getfromPOS(fyle)
	print(fyle+"   site: "+site+ " loaded")
	if len(lats) != len(sn): print("problem with loading data for site %s!!!!!!" % site)
	data.append( [site,[reflat,reflon,refheight],mjd_to_decyear(mjds),lats,lons,heights,dlats,dlons,dheights,sn,se,su,altname] )


#sensible to sort these alphabetically
data.sort(key=lambda x: x[0])

#if we're only doing a single site, change the thread count
if len(data) == 1: numproc = 1


#dumb lil helper function to convert line elements to integer, if possible. not using 
def turn2ints(line):
	for i in range(len(line)):
		try:
			line[i] = int(line[i])
		except:
			continue


#load all renames
eqdata = []; orig_eq_line = []
for fyle in renamefiles:
	f = open(fyle,'r')
	for line in f:
		if line[0] != ' ' or len(line) == 0 : continue
		line = line.split()
		if len(line) == 0 : continue #sometimes a line is just one blank space on accident
		if len(line) < 13: print("line %s in eqfile %s is malformed" % (line,fyle))
		#turn2ints(line)
		#turn to ints (if possible)
		for i in range(len(line)):
			try: line[i] = int(line[i])
			except: continue		
		if line[0].lower() == "rename":
			orig_eq_line.append(line) #we're going to need this later, to write out offsets
			site8char = str(line[1]).upper()
			rename = line[2].upper()
			start = datetime.datetime(year=line[3], month=line[4], day=line[5], minute=line[6], second=line[7])
			start_decyear = start.year + float(start.strftime('%j'))/days_per_year
			end = datetime.datetime(year=line[8], month=line[9], day=line[10], minute=line[11], second=line[12])
			end_decyear = end.year + float(end.strftime('%j'))/days_per_year
			#make sure start and end times are reasonable, also round them to 4 decimals
			start_decyear = round(max(1995., start_decyear),4) #TODO: 1995 is hard encoded as beginning of any analysis, may have to be more flexible
			end_decyear = round( min( float(datetime.datetime.now().strftime('%Y')) + float(datetime.datetime.now().strftime('%j'))/days_per_year, end_decyear)  , 4) 			
			if end_decyear-start_decyear > days_per_data/days_per_year : #avoid any tiny renames too small to do anything with *TODO a better way to deal with this
				eqdata.append([site8char,rename,start_decyear,end_decyear])
	f.close()


#now that we have allllll the data loaded, we're going through each of our files/sites


def do_job(sub_data): #sub_data is data[i]
	global fig,axN,axE,axU #not 100% sure why this is needed but, it is

	site = sub_data[0]
	altname = sub_data[12] #the longer, descriptive version
	savename = plotdir+"/%s.png" % site
	if fix_offsets and len(sub_data[6]) >= mincontinuouslen : savename = plotdir+"/%s_fixed.png" % site
	savename = savename.replace('//','/') #make sure we haven't screwed up the file path

	if os.path.exists(savename) and dont_overwrite : 
		print("file %s exists and not overwriting, skipping" % savename )
		return

	#global times, each (and only) datapoint has a value
	times = sub_data[2]


	#### set up the EQ file array... a bit complex
	#select a subset pertaining to current site we're processing 
	sub_eqdata = [ele for ele in eqdata if ele[0]==site]
	
	#sort by start time, but then also move the XPS and XCL to the rear
	sub_eqdata.sort(key=lambda x: x[3])
	#need to sort so the X's are at the end
	moveXtoback(sub_eqdata)

	#if there are XPS/XCL with ranges that are greater than the data, set equal to extremes
	for ele in sub_eqdata:
		if ele[2] < min(sub_data[2]): ele[2] = min(sub_data[2])
		if ele[3] > max(sub_data[2]): ele[3] = max(sub_data[2])

	#make sure we've got at least one *PS rename in there besides any XPS
	if 'PS' not in '\t'.join([sub_eqdata[i][1] for i in range(len(sub_eqdata))]).replace('XPS',''): #a mouthful but works
		sub_eqdata.append([site,site+"_GPS", min(sub_data[2]), max(sub_data[2])])


	#sort by start time, but then also move the XPS and XCL to the rear AGAIN
	sub_eqdata.sort(key=lambda x: x[3])
	#need to sort so the X's are at the end
	moveXtoback(sub_eqdata)

	#OK, what if we've got 2PS (or 3PS or..) but no GPS preceeding it? 
	if min(sub_data[2]) < sub_eqdata[0][2]:
		sub_eqdata.append([site,site+"_GPS", min(sub_data[2]), sub_eqdata[0][2]]  )

	#sort by start time, but then also move the XPS and XCL to the rear AGAIN
	sub_eqdata.sort(key=lambda x: x[3])
	#need to sort so the X's are at the end
	moveXtoback(sub_eqdata)

	#problem where a rename exists entirely before (or after) our data span (e.g. ALGO!!). find and remove... 
	for m in range(len(sub_eqdata)):
		for ele in sub_eqdata:
			if ele[3] <= min(sub_data[2]) or ele[2] >= max(sub_data[2]): sub_eqdata.remove(ele)


	#TODO need a sanity check if we have a lot of overlapping XPS/XCL. YAR3 as it currently is. doesn't seem to be that big a deal

	#get residuals of NEU. 2 parts
	#first, get the lon/lat/up residuals relative to their (a?) reference
	reflat = sub_data[1][0]; reflon = sub_data[1][1]; refheight = sub_data[1][2]

	#second, get residual
	N_r = getresidual(times,sub_data[6],sub_data[9]) 
	E_r = getresidual(times,sub_data[7],sub_data[10]) #converts to np array
	#U_r = getresidual(times,sub_data[8],sub_data[11]) #normalised?
	U_r = np.array(sub_data[8]) #raw (preferred. TODO add a setting to switch vert_residual

	sN = np.array(sub_data[9]); sE = np.array(sub_data[10]); sU = np.array(sub_data[11]) #sigmas

	#convert all these to mm here and now
	N_r = 1000*N_r; E_r = 1000*E_r; U_r = 1000*U_r; sN=1000*sN; sE=1000*sE; sU=1000*sU

	################################################################all data loaded now plot
	print("trying to plot %s.... " % site,end="")

	fig,(axN,axE,axU) = plt.subplots(nrows=3,ncols=1,sharex=True,figsize=(12,14)) #TODO: changing this size affects the WRMS/NRMS text placement. need ratios not offsets

	if len(N_r) < mincontinuouslen: 
		plotcampaign(times,N_r,sN,axN)
		plotcampaign(times,E_r,sE,axE)
		plotcampaign(times,U_r,sU,axU)
	else:
		if fix_offsets and sub_eqdata : 
			try:
				noffsets = plotcontinuous(times,sub_eqdata,N_r,sN,axN,plot=False) #these are in mm
			except Exception as ex:
				print("problem first plotcont in site %s" %site )
				print(ex)
				exit()
			eoffsets = plotcontinuous(times,sub_eqdata,E_r,sE,axE,plot=False)
			uoffsets = plotcontinuous(times,sub_eqdata,U_r,sU,axU,plot=False)
			offsettimes = []
			for i in range(len(sub_eqdata)-1): 
				if sub_eqdata[i][2] < sub_eqdata[i+1][2] and sub_eqdata[i][1][5:6] != "X" and sub_eqdata[i+1][1][5:6] != "X" : 
					offsettimes.append(sub_eqdata[i+1])

			if len(noffsets) != len(offsettimes): 
				print("problem with offset times for site %s / has %d offsets but %d offsettimes" % (site,len(noffsets),len(offsettimes)))
				print(noffsets); print(offsettimes)
				return
			#now to actually fix (Residual Rixed = _rf)
			N_rf = N_r; E_rf = E_r; U_rf = U_r
			for m in range(len(offsettimes)):
				for n in range(len(times)):
					if times[n] >= offsettimes[m][2]:
						N_rf[n] -= noffsets[m]; E_rf[n] -= eoffsets[m]; U_rf[n] -= uoffsets[m]
			#now plot as normal. except now we have already removed the first offset rename... so need to catch it in the plotcont function 
			try:			
				trash = plotcontinuous(times,sub_eqdata,N_rf,sN,axN)
			except Exception as ex:
				print("problem 2nd plotcont in site %s" %site )
				print(ex)
				exit()
			trash = plotcontinuous(times,sub_eqdata,E_rf,sE,axE)
			trash = plotcontinuous(times,sub_eqdata,U_rf,sU,axU)
			#save to an array, which we are then going to use to make a velocity output file
			#savenpz.append([[site,reflat,reflon],U_rf,sU])
		else : #not fixing the offsets
			noffsets = plotcontinuous(times,sub_eqdata,N_r,sN,axN) #these are in mm
			eoffsets = plotcontinuous(times,sub_eqdata,E_r,sE,axE)
			uoffsets = plotcontinuous(times,sub_eqdata,U_r,sU,axU)
						


	#axN.tick_params(axis='y',width=0.001)

	axN.set_ylabel("N (mm)",fontsize = 16)
	axE.set_ylabel("E (mm)",fontsize = 16)
	axU.set_ylabel("U (mm)",fontsize = 16)


	#set the y-range
	#TODO: set this to integer mm for cryin out loud
	#TODO (esp for campaign) make sure scale range is greater than the sigmas!!


	#plt.yticks(np.round(np.arange(ymin-.001,ymax+.001,.001),3) )
	#axN.ylim((ymid-.010,ymid+.010))

	###calculate white noise from the 3 month filter #TODO here need to move text via ratio of figure, not precise amount
	if len(N_r) >= mincontinuouslen: 
		f0 = quickfilter(times,N_r,sN,dt=days_per_data,freq=filterlen,filttype='highpass') #3 months
		f0 = reject_outliers(f0,3); 
		axN.text(0.075, 0.03, "WRMS = %.1f mm" % np.std(f0),fontsize=12,color='darkblue',horizontalalignment='center',verticalalignment='center',transform = axN.transAxes)
		f0 = quickfilter(times,E_r,sE,dt=days_per_data,freq=filterlen,filttype='highpass') #3 months
		f0 = reject_outliers(f0,3); 
		axE.text(0.075, 0.03, "WRMS = %.1f mm" % np.std(f0),fontsize=12,color='darkblue',horizontalalignment='center',verticalalignment='center',transform = axE.transAxes) 
		f0 = quickfilter(times,U_r,sU,dt=days_per_data,freq=filterlen,filttype='highpass') #3 months
		f0 = reject_outliers(f0,3); 
		axU.text(0.075, 0.03, "WRMS = %.1f mm" % np.std(f0),fontsize=12,color='darkblue',horizontalalignment='center',verticalalignment='center',transform = axU.transAxes) 


	
	#TODO change title if offsets have been "fixed" ? right now it says in file name and is pretty obvious.. 
	axN.set_title("%s %s (%.4f,%.4f)" % (site,altname,reflat,reflon),fontsize=20)
	plt.tight_layout()
	fig.subplots_adjust(hspace=0)

	if savenpz and len(fyles)==1 : #makes no sense to save if we're doing a bunch of files
		if fix_offsets: 
			np.savez(site+"_FIXED",times=times,sN=sN,sE=sE,sU=sU,N_r=N_rf,E_r=E_rf,U_r=U_rf)
		else:
			np.savez(site,         times=times,sN=sN,sE=sE,sU=sU,N_r=N_r,E_r=E_r,U_r=U_r)
		

	if save:
		plt.savefig(savename,bbox_inches='tight')
		print(savename+" saved")


	if write_offsets: #write out the calculate offsets into the same eq file format?
		#establish/overwrite a new file
		of = open(outoffsetfile,'a') #TODO: need a sane way to deal with this. deleting the file first?
		sub_sub_eqdata = []
		for i in range(len(sub_eqdata)-1): 
			if sub_eqdata[i][2] < sub_eqdata[i+1][2] and sub_eqdata[i+1][1][5:6] != "X" : # and sub_eqdata[i+1][1][5:6] != "X" : 
				sub_sub_eqdata.append(sub_eqdata[i+1])
		for i in range(len(sub_sub_eqdata)): #now write out the lines
			#now find original eq line...
			ogeq = [j for j in orig_eq_line if j[2]==sub_sub_eqdata[i][1] ][0] #last [0] is because double nested. #TODO potential problem here if there's a double listing!! check length first?
			#now format this output string..
			eq_str = " %s %s %s  %4d %02d %02d %02d %02d  2099 %02d %02d %02d %02d %7.2f %7.2f %7.2f NEU " % (ogeq[0],ogeq[1],ogeq[2],ogeq[3],ogeq[4],ogeq[5],ogeq[6],ogeq[7],ogeq[9],ogeq[10],ogeq[11],ogeq[12], -noffsets[i],-eoffsets[i],-uoffsets[i]) #offsets in mm, negative to reverse
			#add any of the comments
			for ele in ogeq[13:]: eq_str += ' '+str(ele)
			#write to a new file?
			of.write(eq_str+"\n")
		of.close() #a ton of open/closes...TODO is there a better way?

	if interactive:
		#format the date axis for interactive use
		axU.fmt_xdata=decyear2doy #"any function that takes a scalar and returns a string"
		axN.fmt_xdata=decyear2doy;axE.fmt_xdata=decyear2doy #also do the other ones i guess
		plt.show()


	plt.close()

	""" TODO/NOT WORKING
	#calculate U velocity and write to vel format
	if write_verts: 
		[Uvel,Uint],cov_mat = np.polyfit(times,U_rf,w = [1./_i for _i in sU],deg=1,full=False,cov='unscaled')
		slope_err = cov_mat[0,0]**.5
		v = open(vert_vel_file,'a')
		#with open(vert_vel_file,'a') as v:
		v.write(" %.5f %.5f 0 %.4f 0 0 0 %.4f 0 0 0 0 0 %s_GPS\n" % (reflon,reflat,Uvel,slope_err,site))
		v.close()
	"""

	if plot_spectral:
		#probably have to convert from mm realm back to m to get units right. DENSITY = V**2/Hz, SPECTRUM V**2
		freq, psd_N = scipy.signal.welch(N_r,fs=samplerate,nfft=seglen,nperseg=seglen,scaling='density',window=window_type)
		freq, psd_E = scipy.signal.welch(E_r,fs=samplerate,nfft=seglen,nperseg=seglen,scaling='density',window=window_type)
		freq, psd_U = scipy.signal.welch(U_r,fs=samplerate,nfft=seglen,nperseg=seglen,scaling='density',window=window_type)
		#mean_u,median_u,mode_u,std_u,var_u = getmultivectstats(allpsds_height)  this is if we're averaging a ton of sites
		#mean_n,median_n,mode_n,std_n,var_n = getmultivectstats(allpsds_lat)
		#mean_e,median_e,mode_e,std_e,var_e = getmultivectstats(allpsds_lon)
		#TODO: need to rename this plot / don't use plt. / need to learn matplotlib better!!
		plt.loglog(freq,psd_N,label="N")
		plt.loglog(freq,psd_E,label="E")
		plt.loglog(freq,psd_U,label="U")
		plt.xlim(.08,12)
		plt.ylim(.001,101)
		xcoords = [1./10,1./9, 1./8, 1./7, 1./6, 1/5., 1/4., 1/3.,.5,1,2,4,12]
		#Tticks = [round(1./i,1) for i in xcoords]
		Tticks = [10,9,8,7,6,5,4,3,2,1,0.5,0.2,0.1]		
		plt.xticks(xcoords, Tticks)
		plt.xlabel("1/f (Years)"); plt.ylabel("PSD (mm^2/year)")
		plt.legend()
		plt.show()
		#TODO save spectra option?
		plt.close()
		"""
		#TODO: at this point need to fit a function to this Y = WN + RW*f^2 (then later incorporate f^1, etc) *use envelope function
		from scipy import optimize
		def wnfit(x,a): return a
		def rwfit(x,b): return b*x*2
		#need to do two versions. freq less than 3 months (linear..already done above?), freq more than 3
		tfarray = freq>1./(3./12) #an array of true/false where longer period (for RW) are "false" and short period high freq for white noise
		#whitenoise
		wnparams, wnparams_covariance = optimize.curve_fit(wnfit,freq[tfarray],psd_N[tfarray],p0=[.0015]) #this works.. technically. but we need to assess the structure of the PSD
		plt.plot(freq[tfarray],psd_N[tfarray]); plt.show() #cool but no labels or anything TODO TODO 

		rwparams, rwparams_covariance = optimize.curve_fit(rwfit,freq[~tfarray],psd_N[~tfarray],p0=[wnparams[0],5e-7])
		#TODO stilla  lot of problems here. white noise is not working out
		"""



#here's our actual process
print ("starting up %d threads..." % numproc)
if __name__=="__main__":
	with mp.Pool(processes=numproc) as pool:
		pool.map_async(do_job,data)
		pool.close()
		pool.join()

