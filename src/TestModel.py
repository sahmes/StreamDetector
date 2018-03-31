from AuxiliaryFunctions import *
import sys, os, numpy as np

# Command line arguments: N_samples
if len(sys.argv)==2:
    N_samples = int(sys.argv[1])
else:
    print 'Usage: %s N_bg N_samples' % (sys.argv[0])
    raise SystemExit
SaveEvery = 32

Conf = ReadConfig()
N_bg_min, N_bg_max, DataDir = Conf['N_bg_min'], Conf['N_bg_max'], Conf['datadir']

FileName = 'test-0.dat'
i = 1
while FileName in os.listdir(DataDir):
    FileName = 'test-%d.dat' % (i)
    i += 1
File = open(os.path.join(DataDir, FileName), 'w')

print 'Output file name: %s' % (FileName)
print 'Flushing every %d iterations.' % (SaveEvery)
sys.stdout.flush()

ParameterEstimatorInstance = ParameterEstimator()

i = 0
while i < N_samples:
    # Test sample generation
    while True:
        alpha = np.deg2rad(np.random.rand()*360.-180.)
        b = np.random.rand()**2*0.5*np.sqrt(2)
        w = np.random.rand()*np.sqrt(2)
        if GetArea(alpha, b, w) > 0.5: continue
        if CheckParameters(alpha, b, w): break
    N_bg = int(10**(np.log10(N_bg_min)+np.random.rand()*(np.log10(N_bg_max)-np.log10(N_bg_min))))
    A = np.random.rand()
    x_bg, y_bg, x_fg, y_fg = GenerateCoordinates(N_bg, A, alpha, b, w)
    x, y = np.append(x_bg, x_fg), np.append(y_bg, y_fg)

    Result = ParameterEstimatorInstance(x,y)
    if Result is None: continue
    i += 1
    p, N_tot, N_bg_pred, N_fg_pred, A_pred, alpha_pred, s_err, b_pred, b_err, w_pred, w_err = Result


    N_fg = len(x_fg)
    File.write('%22.16e%5d%23.16e%5d%23.16e%23.16e%23.16e%24.16e%24.16e%23.16e%23.16e%23.16e%23.16e%23.16e%23.16e%23.16e\n' % (p, N_bg, N_bg_pred, N_fg, N_fg_pred, A, A_pred, alpha, alpha_pred, s_err, b, b_pred, b_err, w, w_pred, w_err))
    if (i%SaveEvery==SaveEvery-1): File.flush()

File.close()
