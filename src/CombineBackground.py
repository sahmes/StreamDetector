import os, re, numpy as np
from AuxiliaryFunctions import ReadConfig

Conf = ReadConfig()
DataDir = Conf['datadir']

FileList = {}
for FileName in os.listdir(DataDir):
    Match = re.findall(r'^bg-(\d\d\d\d)-(\d+).dat', FileName)
    if len(Match)!=1: continue
    N_bg = int(Match[0][0])
    if not (N_bg in FileList.keys()): FileList[N_bg] = []
    FileList[N_bg].append(FileName)

for N_bg in FileList:
    print 'Combining %d files with N_bg=%d.' % (len(FileList[N_bg]), N_bg)

    N = 0
    Counts  = np.zeros(64*64)
    Squares = np.zeros(64*64)
    for FileName in FileList[N_bg]:
        FullFileName = os.path.join(DataDir, FileName)
        with open(FullFileName, 'r') as File: Line = File.readline()
        N_ = int(Line[6:])
        N += N_
        mu_, sig_ = np.loadtxt(FullFileName, skiprows=1, unpack=True)
        Counts += mu_*N_
        Squares += (sig_**2 + mu_**2)*N_

    # Save
    File = open(os.path.join(DataDir, 'bg-%04d.dat'%N_bg), 'w')
    File.write('# N = %d\n' % N)
    mu = Counts/np.double(N)
    sigma = np.sqrt(Squares/np.double(N) - mu**2)
    for j in range(len(mu)):
        File.write('%22.16e%23.16e\n' % (mu[j], sigma[j]))
    File.close()
