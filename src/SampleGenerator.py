import numpy as np
import h5py, sys
from AuxiliaryFunctions import * # We really need almost everything

# Command line arguments: NumberOfSamples
if len(sys.argv) == 2:
    NumberOfSamples  = sys.argv[1]
else:
    raise ValueError
NumberOfSamples = int(NumberOfSamples)

Conf = ReadConfig()
N_bg_min, N_bg_max, DataDir = Conf['N_bg_min'], Conf['N_bg_max'], Conf['datadir']

FileName = 'streams-0.h5'
i = 1
while FileName in os.listdir(DataDir):
    FileName = 'streams-%d.h5' % i
    i += 1
FullFileName = os.path.join(DataDir, FileName)
h5py.File(FullFileName, 'w').close()  # Just create the file, so other instances of the application won't use it.

Noise = NoiseStatistics()
Background = BackgroundControl()
Buffer = BufferClass('H', 'N_bg', 'N_fg', 'A', 'alpha', 'b', 'w', 'p')

BufferSize = 32
i = 0
while i < NumberOfSamples:
    while True:
        alpha = np.deg2rad(np.random.rand()*360.-180.)
        b = np.random.rand()**2*0.5*np.sqrt(2)
        w = np.random.rand()*np.sqrt(2)
        if GetArea(alpha, b, w) > 0.5: continue
        if CheckParameters(alpha, b, w): break
    N_bg = int(10**(np.log10(N_bg_min)+np.random.rand()*(np.log10(N_bg_max)-np.log10(N_bg_min))))
    A = np.random.rand()
    x_bg, y_bg, x_fg, y_fg = GenerateCoordinates(N_bg, A, alpha, b, w)
    N_fg = len(x_fg)
    N_tot = N_bg + N_fg
    if N_tot > Noise.N_arr[-1]: continue
    x, y = np.append(x_bg, x_fg), np.append(y_bg, y_fg)
    H = CalculateCorrelation(x, y)
    delta = Background.delta(H, N_tot)
    p = 1-Noise(delta, N_tot)[0]
    if p > 0.1: continue
    #print 'Sample %06d with %d total stars; p=%.2e' % (i, N_tot, p)
    i += 1
    sys.stdout.flush()
    Buffer.Append(H=H, N_bg=N_bg, N_fg=N_fg, A=A, alpha=alpha, b=b, w=w, p=p)

    if (Buffer.Size==BufferSize) or (i==NumberOfSamples):
        File = h5py.File(FullFileName, 'a')
        IndicesInFile = np.array([int(Key) for Key in File.keys()])
        if len(IndicesInFile)==0: FileIndex=0
        else: FileIndex = IndicesInFile.max()+1
        for j in range(Buffer.Size):
            Group = File.create_group('%06d' % (FileIndex+j))
            Group['H'] = np.asarray(Buffer.Data['H'][j], dtype=int)
            Group['N_bg'] = Buffer.Data['N_bg'][j]
            Group['N_fg'] = Buffer.Data['N_fg'][j]
            Group['A'] = np.float32(Buffer.Data['A'][j])
            Group['alpha'] = np.float32(Buffer.Data['alpha'][j])
            Group['b'] = np.float32(Buffer.Data['b'][j])
            Group['w'] = np.float32(Buffer.Data['w'][j])
            Group['p'] = np.float32(Buffer.Data['p'][j])
        File.close()
        Buffer.Reset()
