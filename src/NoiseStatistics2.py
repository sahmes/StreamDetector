import os, re, numpy as np
from AuxiliaryFunctions import ReadConfig
from scipy.special import erf, erfinv
import scipy.optimize, scipy.interpolate

Points = 64
Conf = ReadConfig()

DataDir = Conf['datadir']
OutputFileName = os.path.join(DataDir, 'noise-statistics.dat')

def LogNormalFit_CDF(x,y):
    fitfunc = lambda p, x: 0.5 + 0.5*erf((np.log(x)-p[0])/(p[1]*np.sqrt(2)))
    errfunc = lambda p, x, y: (y - fitfunc(p, x))**2
    Y1 = erfinv(2*y[0]-1)
    Y2 = erfinv(2*y[-1]-1)
    X1 = np.log(x[0])
    X2 = np.log(x[-1])
    mu = (Y1*X2-Y2*X1)/(Y1-Y2)
    sigma = np.sqrt(0.5)*(X1-X2)/(Y1-Y2)
    guess = [mu, sigma]
    out = scipy.optimize.leastsq(errfunc, guess, args=(x, y), full_output=0)
    if 1 <= out[1] <= 4:
        return out[0]
    else:
        return asarray[np.nan, np.nan]

N_bg = []
for FileName in os.listdir(DataDir):
    Match = re.findall(r'^noisesamples-(\d\d\d\d).dat', FileName)
    if len(Match)!=1: continue
    N_bg.append(int(Match[0]))
N_bg = np.asarray(sorted(N_bg), dtype=int)

Data = np.empty((Points, len(N_bg)), dtype=np.double)

File = open(OutputFileName, 'w')
for i in range(len(N_bg)):
    FileName = os.path.join(DataDir, 'noisesamples-%04d.dat'%N_bg[i])
    NoiseSamples = np.loadtxt(FileName, usecols=[1])
    NoiseSamples.sort()
    Fraction_data = np.arange(len(NoiseSamples),dtype=np.double)/(len(NoiseSamples))
    index_min = len(NoiseSamples)/2
    index_max = np.where(Fraction_data>=erf(2.5/np.sqrt(2)))[0][0]-1
    delta_min = NoiseSamples[index_min] # the median
    delta_max = NoiseSamples[index_max] # 2.5 sigma
    indices = np.linspace(index_min, index_max, Points, dtype=int)
    index_min = np.where(Fraction_data>=erf(2.0/np.sqrt(2)))[0][0]
    mu, sigma = LogNormalFit_CDF(NoiseSamples[index_min:index_max], Fraction_data[index_min:index_max])
    Extrapolator = lambda x: 0.5 + 0.5*erf((log(x)-mu)/(sigma*sqrt(2)))
    Interpolator = scipy.interpolate.interp1d(NoiseSamples[indices], Fraction_data[indices], assume_sorted=True, bounds_error=False, fill_value=(0.5, np.nan))
    SignalProbability = lambda x: Interpolator(x) if x<=delta_max else Extrapolator(x)
    File.write('# N = %d [interpolation data]\n' % N_bg[i])
    for j in range(Points):
        File.write('%22.16e%23.16e\n' % (NoiseSamples[indices[j]], 1-Fraction_data[indices[j]]))
    File.write('# END\n\n')
    File.write('# N = %d [extrapolation data]\n' % N_bg[i])
    File.write('%22.16e%23.16e\n' % (mu, sigma))
    File.write('# END\n\n')
File.close()
