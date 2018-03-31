import os, sys, numpy as np
from AuxiliaryFunctions import ReadConfig, GenerateBackground, CalculateCorrelation, BackgroundControl

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

FileName = 'noisesamples-%04d-0.dat' % N_bg
i = 1
while FileName in os.listdir(DataDir):
    FileName = 'noisesamples-%04d-%d.dat' % (N_bg, i)
    i += 1
FullFileName = os.path.join(DataDir, FileName)
open(FullFileName, 'w').close() # Just create the file, so other instances of the application won't use it.

print 'Calculating the background delts for N_bg=%d.' % (N_bg)
print 'Generating up to %d samples.' % (N_samples)
print 'Output file name: %s' % (FullFileName)
print 'Flushing every %d iterations.' % (SaveEvery)
sys.stdout.flush()

BackgroundControler = BackgroundControl()

ZeroPadInt = '%%0%dd' % (int(np.log10(N_samples))+1)

File = open(FullFileName, 'w')
for i in range(N_samples):
    x_bg, y_bg = GenerateBackground(N_bg)
    H = CalculateCorrelation(x_bg, y_bg)
    delta = BackgroundControler.delta(H)
    File.write((ZeroPadInt+'%23.16e\n')  % (i, delta))
    if (i%SaveEvery==SaveEvery-1):
        File.flush()
File.close()