import os, re, numpy as np
from AuxiliaryFunctions import ReadConfig

Conf = ReadConfig()
DataDir = Conf['datadir']

FileList = {}
for FileName in os.listdir(DataDir):
    Match = re.findall(r'^noisesamples-(\d\d\d\d)-(\d+).dat', FileName)
    if len(Match)!=1: continue
    N_bg = int(Match[0][0])
    if not (N_bg in FileList.keys()): FileList[N_bg] = []
    FileList[N_bg].append(FileName)

for N_bg in FileList:
    print 'Combining %d files with N_bg=%d.' % (len(FileList[N_bg]), N_bg)

    delta = np.array([], dtype=np.double)
    for FileName in FileList[N_bg]:
        delta_ = np.loadtxt(os.path.join(DataDir, FileName), usecols=[1])
        delta  = np.append(delta, delta_)

    # Save
    ZeroPadInt = '%%0%dd' % (int(np.log10(len(delta)))+1)
    File = open(os.path.join(DataDir, 'noisesamples-%04d.dat'%N_bg), 'w')
    for i in range(len(delta)): File.write((ZeroPadInt+'%23.16e\n')  % (i, delta[i]))
    File.close()
