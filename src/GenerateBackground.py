import os, sys, numpy as np
from AuxiliaryFunctions import ReadConfig, GenerateBackground, CalculateCorrelation

def MakeBackgroundSymmetric(Input):
    Output = 0.5*(Input[:,:32] + Input[:,32:])
    Output = 0.5*(Output + np.fliplr(Output))
    Output = Output.repeat(2,0).reshape((64,64))
    return Output

# Command line arguments: N_bg, N_samples
if len(sys.argv)==3:
    N_bg      = int(sys.argv[1])
    N_samples = int(sys.argv[2])
else:
    print 'Usage: %s N_bg N_samples' % (sys.argv[0])
    raise SystemExit
SaveEvery = 32

Conf = ReadConfig()
DataDir = Conf['datadir']

FileName = 'bg-%04d-0.dat' % N_bg
i = 1
while FileName in os.listdir(DataDir):
    FileName = 'bg-%04d-%d.dat' % (N_bg, i)
    i += 1
open(os.path.join(DataDir, FileName), 'w').close() # Just create the file, so other instances of the application won't use it.

print 'Calculating the background for N_bg=%d.' % (N_bg)
print 'Averaging up to %d samples.' % (N_samples)
print 'Output file name: %s' % (FileName)
print 'Saving every %d iterations.' % (SaveEvery)
sys.stdout.flush()

Counts  = np.zeros((64,64), dtype=int) # accumulate the counts
Squares = np.zeros((64,64), dtype=int) # accumulate the squares

for i in range(N_samples):
    sys.stdout.flush()
    x_bg, y_bg = GenerateBackground(N_bg)
    H = CalculateCorrelation(x_bg, y_bg)
    Counts  += H
    Squares += H**2
    if (i%SaveEvery==SaveEvery-1) or (i==N_samples-1):
        File = open(os.path.join(DataDir, FileName), 'w')
        mu = Counts/np.double(i+1)
        sigma = np.sqrt(Squares/np.double(i+1) - mu**2)
        mu = MakeBackgroundSymmetric(mu)
        sigma = MakeBackgroundSymmetric(sigma)
        mu = mu.flatten()
        sigma = sigma.flatten()
        File.write('# N = %d\n' % (i+1))
        for j in range(len(mu)):
            File.write('%22.16e%23.16e\n' % (mu[j], sigma[j]))
        File.close()
