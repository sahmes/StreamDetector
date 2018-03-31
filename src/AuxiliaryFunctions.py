import os, re, numpy as np, h5py, scipy.interpolate, copy
from scipy.special import erf

def ReadConfig():
    FileName = 'streams.conf' # Hardcoded
    Conf = {}
    with open(FileName) as File:
        for Line in File:
            Line = Line.strip()
            if Line.startswith('#') or Line=='': continue
            if Line.find('=')==-1: # Defined but has no value
                Conf[Line] = None
                continue
            Key, Value = Line.split('=', 1)
            try:
                Value = int(Value)
            except:
                try:
                    Value = np.double(Value)
                except:
                    Value = Value.strip()
            Conf[Key.strip()] = Value
    return Conf

def ReadData(FileName):
    if type(FileName)==str:
        FileName = [FileName]
    elif not isinstance(FileName, list):
        raise ValueError('Cannot understand the FileName argument, it should be a string or a list of strings.')
    Keys  = ['H', 'N_bg', 'N_fg', 'A', 'alpha', 'b', 'w', 'p']
    Types = [None, int, int, np.float32, np.float32, np.float32, np.float32, np.float32] # None means I'll initialize it manually
    Data = {}
    for FileName_ in FileName:
        File = h5py.File(FileName_, 'r')
        DatumShape = File['000000']['H'][...].shape
        N = len(File.keys()) # Number of samples in this case, nothing to do with N_bg and N_fg, which are parameters
        Data_ = {}
        for Key, Type in zip(Keys, Types):
            if not (Type is None): Data_[Key] = np.empty(N, dtype=Type)
        Data_['H'] = np.empty((N, DatumShape[0], DatumShape[1]), dtype=int)
        for i in range(N):
            i_str = '%06d'%i
            for Key in Data_.keys(): Data_[Key][i] = File[i_str][Key][...]
        File.close()
        if len(FileName)==1: return Data_
        if Data=={}: Data=Data_.copy()
        else:
            for Key in Keys:
                Data[Key] = np.append(Data[Key], Data_[Key], axis=0)
    return Data

def FilterData(Data, Key, MinVal=None, MaxVal=None):
    Mask1, Mask2 = True, True
    if not (MinVal is None): Mask1 = Data[Key] > MinVal
    if not (MaxVal is None): Mask2 = Data[Key] < MaxVal
    Mask = Mask1 & Mask2
    for Key_ in Data:
        Data[Key_] = Data[Key_][Mask]
    return Data

def SplitData(Data, Ratio=0.5):
    DataA = {}
    DataB = {}
    N = len(Data['N_bg']) # Number of samples
    Idx = int(N*Ratio)
    for Key in Data.keys():
        DataA[Key] = Data[Key][:Idx]
        DataB[Key] = Data[Key][Idx:]
    return DataA, DataB

class BackgroundControl:
    def __init__(self, Mode='interpolate'):
        Conf = ReadConfig()
        DataDir = Conf['datadir']
        N_bg = []
        for FileName in os.listdir(DataDir):
            Match = re.findall(r'^bg-(\d\d\d\d).dat', FileName)
            if len(Match)!=1: continue
            N_bg.append(int(Match[0]))
        N_bg = np.asarray(sorted(N_bg), dtype=int)

        Weight = np.empty(len(N_bg), dtype=np.double)
        mu     = []
        sigma  = []
        for i in range(len(N_bg)):
            FileName = os.path.join(DataDir, 'bg-%04d.dat'%N_bg[i])
            with open(FileName, 'r') as File: Line = File.readline()
            Weight[i] = int(Line[6:]) * N_bg[i]*(N_bg[i]-1)/2
            mu_, sigma_ = np.loadtxt(FileName, skiprows=1, unpack=True)
            mu_    = mu_.reshape((64,64))
            sigma_ = sigma_.reshape((64,64))
            mu.append(mu_)
            sigma.append(sigma_)

        self.Background = np.zeros((64,64), dtype=np.double)
        Weight = Weight/Weight.max()
        for i in range(len(N_bg)):
            self.Background += Weight[i]*mu[i]/np.sum(mu[i])
        self.Background = self.Background/np.sum(Weight)
        self.Mask = self.Background > 0

        if Mode.lower()=='extrapolate':
            Slope = np.empty((64,64))
            Intercept = np.empty((64,64))
            for i in range(64):
                for j in range(64):
                    if self.Background[i,j]==0:
                        Slope[i,j], Intercept[i,j] = np.nan, np.nan
                        continue
                    s = [sigma[k][i,j] for k in range(len(N_bg))]
                    Slope[i,j], Intercept[i,j] = np.polyfit(np.log(N_bg), np.log(s), 1)

            self.Noise = lambda N: exp(Intercept)*N**Slope
        elif Mode.lower()=='interpolate':
            y_values = np.array(sigma)
            y_values[y_values==0] = 1.e-6
            y_values = np.log(y_values)
            Interpolant = scipy.interpolate.interp1d(np.log(N_bg), y_values, axis=0)
            def Noise(N_bg):
                sigma = np.exp(Interpolant(np.log(N_bg)))
                sigma[~self.Mask] = np.nan
                return sigma
            self.Noise = Noise
        else:
            raise ValueError
    def delta(self, H, N=None):
        if N is None: N = 0.5*(1 + np.sqrt(1+8*np.sum(H)))
        H_bg  = (self.Background*(N*(N-1)/2))
        sigma = self.Noise(N)
        Delta = (H-H_bg)/sigma
        Delta = Delta[self.Mask]
        rms = np.sqrt(np.mean(Delta**2))
        return rms
    def SubtractBackground(self, H, N=None):
        if N is None: N = 0.5*(1 + np.sqrt(1+8*np.sum(H)))
        H_bg  = (self.Background*(N*(N-1)/2))
        H_bg[~self.Mask] = np.nan
        Result = (H - H_bg)/H_bg
        return Result

class FancyInterpolator:
    def __init__(self, Interpolator, LeftExtrapolator, RightExtrapolator):
        self.Interpolator      = copy.copy(Interpolator)
        self.LeftExtrapolator  = copy.copy(LeftExtrapolator)
        self.RightExtrapolator = copy.copy(RightExtrapolator)
        self.MinVal = Interpolator.x[0]
        self.MaxVal = Interpolator.x[-1]
    def __call__(self, x):
        x = np.atleast_1d(x)
        LeftMask   = (x < self.MinVal)
        MiddleMask = (self.MinVal<=x)&(x<=self.MaxVal)
        RightMask  = (x > self.MaxVal)
        Result = np.empty_like(x)
        Result[MiddleMask] = self.Interpolator(x[MiddleMask])
        if callable(self.LeftExtrapolator): Result[LeftMask] = self.LeftExtrapolator(x[LeftMask])
        else: Result[LeftMask] = self.LeftExtrapolator
        if callable(self.RightExtrapolator): Result[RightMask] = self.RightExtrapolator(x[RightMask])
        else: Result[RightMask] = self.RightExtrapolator
        return Result

def ReadNoiseData():
    Conf = ReadConfig()
    DataDir = Conf['datadir']
    FileName = os.path.join(DataDir, 'noise-statistics.dat')
    Modes = ['interpolation', 'extrapolation']
    Data = {} # Dictionary of dictionaries
    for Mode in Modes: Data[Mode] = {} # Initialize the inner dictionaries.
    File = open(FileName, 'r')
    for l, Line in enumerate(File):
        if Line.strip()=='': continue
        Match = re.findall(r'^\s*#\s*N\s*=\s*(\d*)\s*\[(\w+)', Line)
        if len(Match) > 0:
            if len(Match[0])!=2:
                raise ValueError('Problem interpreting line %d in %s.' % (l, FileName))
            N_bg = int(Match[0][0])
            Mode = Match[0][1]
            if Mode in Modes:
                Data[Mode][N_bg] = []
            else:
                raise ValueError('Problem interpreting line %d in %s.' % (l, FileName))
            continue
        Match = re.findall(r'^\s*#\s*END', Line)
        if len(Match)==1:
            N_bg, Mode = None, None
            continue
        Data[Mode][N_bg].append([np.double(Item) for Item in Line.split()])
    File.close()
    for Mode in Data:
        for N_bg in Data[Mode]:
            Data[Mode][N_bg] = np.array(Data[Mode][N_bg])
    return Data

def CreateInterpolants(Data):
    N_bg_arr = sorted(Data['extrapolation'].keys())
    SignalProbability_arr = []
    class LogNormal:
        def __init__(self, mu, sigma): self.mu, self.sigma = mu, sigma
        def __call__(self, x): return 0.5 + 0.5*erf((np.log(x)-self.mu)/(self.sigma*np.sqrt(2)))
    for i, N_bg in enumerate(N_bg_arr):
        Extrapolator = LogNormal(Data['extrapolation'][N_bg][0][0], Data['extrapolation'][N_bg][0][1])
        Interpolator = scipy.interpolate.interp1d(Data['interpolation'][N_bg][:,0], 1-Data['interpolation'][N_bg][:,1], kind='cubic', copy=True, assume_sorted=True, bounds_error=False, fill_value=(np.nan, np.nan))
        SignalProbability_arr.append(FancyInterpolator(Interpolator, 0.5, Extrapolator))
    return N_bg_arr, SignalProbability_arr

class NoiseStatistics:
    def __init__(self):
        Data = ReadNoiseData()
        self.N_arr, self.SignalProbability_arr = CreateInterpolants(Data)
    def __call__(self, x, N):
        if not ((self.N_arr[0]<=N)&(N<=self.N_arr[-1])):
            raise ValueError('Value of N not in range')
        i = np.searchsorted(self.N_arr, N)
        W = np.double(self.N_arr[i]-self.N_arr[i-1])
        w1 = (N-self.N_arr[i-1])/W
        w2 = 1-w1
        if N==self.N_arr[i]: return self.SignalProbability_arr[i](x)
        return self.SignalProbability_arr[i-1](x)*w2 + self.SignalProbability_arr[i](x)*w1

def GenerateBackground(N_bg):
    x_bg = np.random.rand(N_bg)-0.5
    y_bg = np.random.rand(N_bg)-0.5
    return x_bg, y_bg

class BufferClass:
    def __init__(self, *Keys):
        self.Data = {}
        self.Keys = list(Keys)
        self.Reset()
    def Reset(self):
        self.Size = 0
        for Key in self.Keys:
            self.Data[Key] = []
    def Append(self, **Kargs):
        for Key in Kargs:
            if not (Key in self.Keys): raise ValueError
            self.Data[Key].append(Kargs[Key])
        self.Size += 1

def CheckParameters(alpha, b, w):
    X = np.array([-0.5,-0.5,0.5,0.5])
    Y = np.array([0.5,-0.5,0.5,-0.5])
    return np.abs(np.sum(np.sign(-np.sin(alpha)*X + np.cos(alpha)*Y - (b+w/2)))) != 4.0

def GenerateForeground(N_fg, alpha, b, w, BatchSize=1024):
    x_fg, y_fg = np.array([]), np.array([])
    while len(x_fg) < N_fg:
        x_tmp = np.random.rand(BatchSize)-0.5
        y_tmp = np.random.rand(BatchSize)-0.5
        C = (-np.sin(alpha)*x_tmp + np.cos(alpha)*y_tmp - (b-0.5*w) > 0) & (-np.sin(alpha)*x_tmp + np.cos(alpha)*y_tmp - (b+0.5*w) < 0)
        x_fg = np.append(x_fg, x_tmp[C])
        y_fg = np.append(y_fg, y_tmp[C])
    x_fg = x_fg[:N_fg]
    y_fg = y_fg[:N_fg]
    return x_fg, y_fg

def GetArea(alpha, b, w, GridSize=128):
    x = np.linspace(-0.5, 0.5, GridSize)
    X, Y = np.meshgrid(x,x)
    C = (-np.sin(alpha)*X + np.cos(alpha)*Y - (b-0.5*w) > 0) & (-np.sin(alpha)*X + np.cos(alpha)*Y - (b+0.5*w) < 0)
    return np.sum(C)/np.double(GridSize**2)

def GenerateCoordinates(N_bg, A, alpha, b, w):
    x_bg, y_bg = GenerateBackground(N_bg)
    Area = GetArea(alpha, b, w)
    N_fg = int(N_bg*Area*A)
    if N_fg > 0:
        x_fg, y_fg = GenerateForeground(N_fg, alpha, b, w)
        return x_bg, y_bg, x_fg, y_fg
    return x_bg, y_bg, np.array([]), np.array([])

def CalculateCorrelation(x, y, distance_bins=64, angle_bins=64):
    N = len(x)
    dx = np.empty(N*(N-1)/2, dtype=np.float32)
    dy = np.empty(N*(N-1)/2, dtype=np.float32)
    for i in range(N):
        j = i*N-i*(i+1)/2
        dx[j:(j+N-i-1)] = x[i] - x[(i+1):]
        dy[j:(j+N-i-1)] = y[i] - y[(i+1):]
    Distances = np.sqrt(dx**2 + dy**2)
    Angles = np.arctan2(dy, dx)
    Angles[Angles<0] = Angles[Angles<0] + np.pi
    H = np.histogram2d(Distances, Angles, bins=[np.linspace(0,np.sqrt(2),distance_bins+1),np.linspace(0,np.pi,angle_bins+1)])[0]
    H = np.asarray(H, dtype=int)
    return H

def FourPossibleMatrices(H):
    if H.shape!=(64,64): raise ValueError
    H_rot = np.empty_like(H)
    H_rot[:,:32] = H[:,32:]
    H_rot[:,32:] = H[:,:32]
    H_tilde = np.fliplr(H)
    H_tilde_rot = np.fliplr(H_rot)
    return H, H_rot, H_tilde, H_tilde_rot

def MatchFilter(x, y, s_pred, b_pred, w_pred):
    alpha_pred_arr = [np.arcsin(s_pred),
                    np.pi-np.arcsin(s_pred),
                    -np.arcsin(s_pred),
                    np.arcsin(s_pred)-np.pi]
    N_stream_arr = np.empty(4, dtype=int)
    for i, alpha_pred in enumerate(alpha_pred_arr):
        C = (-np.sin(alpha_pred)*x + np.cos(alpha_pred)*y - (b_pred-0.5*w_pred) > 0) & (-np.sin(alpha_pred)*x + np.cos(alpha_pred)*y - (b_pred+0.5*w_pred) < 0)
        N_stream_arr[i] = np.sum(C)
    i = np.argmax(N_stream_arr)
    alpha_pred    = alpha_pred_arr[i]
    N_stream_pred = N_stream_arr[i]
    Area = GetArea(alpha_pred, b_pred, w_pred)
    if (Area <= 0) or (Area >= 1):
        return 0, 0, 0, 0
    N_tot = len(x)
    N_bg_pred = (N_tot-N_stream_pred)/(1-Area)
    N_fg_pred = N_stream_pred - N_bg_pred*Area
    if N_fg_pred < 0: N_fg_pred=0
    A_pred = N_fg_pred/(N_bg_pred*Area)
    return alpha_pred, N_bg_pred, N_fg_pred, A_pred

class ParameterEstimator:
    def __init__(self):
        import keras
        self.Noise = NoiseStatistics()
        self.Background = BackgroundControl()
        Conf = ReadConfig()
        DataDir = Conf['datadir']
        self.model = keras.models.load_model(os.path.join(DataDir, 'model.h5'))
    def __call__(self, x, y):
        N_tot = len(x)
        if N_tot > self.Noise.N_arr[-1]: return None
        H = CalculateCorrelation(x, y)

        delta = self.Background.delta(H, N_tot)
        p = 1-self.Noise(delta, N_tot)[0]
        if p > 0.1: return None

        # Preparation for prediction
        N_tot = len(x) # We already did it, but this is a separate part of the program
        Norm = N_tot*(N_tot-1)/2
        H_bg = self.Background.Background*Norm
        H_all = np.array(FourPossibleMatrices(H))
        H_bg = H_bg.reshape(1,64,64).repeat(4,axis=0)
        H_bg = H_bg[:,self.Background.Mask]
        H_all = H_all[:,self.Background.Mask]
        xi = np.asarray((H_all-H_bg)/H_bg, dtype=np.float32)

        s_pred, b_pred, w_pred = self.model.predict(xi) # Remember those are the NORMALIZED b & w.
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

        return p, N_tot, N_bg_pred, N_fg_pred, A_pred, alpha_pred, s_err, b_pred, b_err, w_pred, w_err
