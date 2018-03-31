import keras
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



class ParameterEstimator():
    def __init__(self):


=================================================================

Noise = NoiseStatistics()
BackgroundControler = BackgroundControl()

model = keras.models.load_model(os.path.join(DataDir, 'model.h5'))

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
    x0, y0 = x.copy(), y.copy()

    H = CalculateCorrelation(x, y)

    N_tot = len(x)
    if N_tot > 8000: continue # need to read this maximum value from somewhere
    delta = BackgroundControler.delta(H, N_tot)
    p = 1-Noise(delta, N_tot)[0]
    if p > 0.1: continue
    i += 1

    # Preparation for prediction
    N_tot = len(x) # We already did it, but this is a separate part of the program
    Norm = N_tot*(N_tot-1)/2
    H_bg = BackgroundControler.Background*Norm
    H_all = np.array(FourPossibleMatrices(H))
    H_bg = H_bg.reshape(1,64,64).repeat(4,axis=0)
    H_bg = H_bg[:,BackgroundControler.Mask]
    H_all = H_all[:,BackgroundControler.Mask]
    xi = np.asarray((H_all-H_bg)/H_bg, dtype=np.float32)

    s_pred, b_pred, w_pred = model.predict(xi) # Remember those are the NORMALIZED b & w.
    s_pred = s_pred[:,0]
    s_pred[1] = np.sqrt(1-s_pred[1]**2)
    s_pred[3] = np.sqrt(1-s_pred[3]**2)
    b_pred = b_pred[:,0]*np.sqrt(0.5)
    w_pred = w_pred[:,0]*np.sqrt(0.5)

    s_err = np.std(s_pred)
    b_err = np.std(b_pred)
    w_err = np.std(w_pred)
    s_pred = np.mean(s_pred)
    b_pred = np.mean(b_pred)
    w_pred = np.mean(w_pred)

    alpha_pred, N_bg_pred, N_fg_pred, A_pred = MatchFilter(x, y, s_pred, b_pred, w_pred)

    N_fg = len(x_fg)
    File.write('%22.16e%5d%23.16e%5d%23.16e%23.16e%23.16e%24.16e%24.16e%23.16e%23.16e%23.16e%23.16e%23.16e%23.16e%23.16e\n' % (p, N_bg, N_bg_pred, N_fg, N_fg_pred, A, A_pred, alpha, alpha_pred, s_err, b, b_pred, b_err, w, w_pred, w_err))
    if (i%SaveEvery==SaveEvery-1): File.flush()

File.close()
