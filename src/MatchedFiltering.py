#first, filtering thingy
from pylab import *
from AuxiliaryFunctions import GenerateCoordinates, CalculateCorrelation, BackgroundControl, GetArea
from NoiseControl import NoiseControl
NoiseControler = NoiseControl()
BackgroundControler = BackgroundControl()

N_bg = 2000
A = 1.
alpha = rand()*2*pi-pi
b = 0.125
w = 0.25

x_bg, y_bg, x_fg, y_fg = GenerateCoordinates(N_bg, A, alpha, b, w)
x, y = np.append(x_bg, x_fg), np.append(y_bg, y_fg)
H = CalculateCorrelation(x, y)
N_fg = len(x_fg)
N_tot = N_bg + N_fg
delta = BackgroundControler.SNR(H, N_tot)
p = 1-NoiseControler(delta, N_tot)

alpha_tag = alpha % pi
s_pred = sin(alpha_tag)
b_pred = b
w_pred = w


def MatchFilter(x, y, s_pred, b_pred, w_pred):
    alpha_pred_arr = [arcsin(s_pred),
                    pi-arcsin(s_pred),
                    -arcsin(s_pred),
                    arcsin(s_pred)-pi]
    N_stream_arr = empty(4, dtype=int)
    for i, alpha_pred in enumerate(alpha_pred_arr):
        C = (-np.sin(alpha_pred)*x + np.cos(alpha_pred)*y - (b-0.5*w_pred) > 0) & (-np.sin(alpha_pred)*x + np.cos(alpha_pred)*y - (b_pred+0.5*w_pred) < 0)
        N_stream_arr[i] = sum(C)
    i = argmax(N_stream_arr)
    alpha_pred    = alpha_pred_arr[i]
    N_stream_pred = N_stream_arr[i]
    Area = GetArea(alpha_pred, b_pred, w_pred)
    N_tot = len(x)
    N_bg_pred = (N_tot-N_stream_pred)/(1-Area)
    N_fg_pred = N_stream_pred - N_bg_pred*Area
    A_pred = N_fg_pred/(N_bg_pred*Area)
    return N_bg_pred, N_fg_pred, A_pred

print MatchFilter(x, y, s_pred, b_pred, w_pred)

#x = np.linspace(-0.5, 0.5, GridSize)
#X, Y = np.meshgrid(x,x)
#C = (-np.sin(alpha)*X + np.cos(alpha)*Y - (b-0.5*w) > 0) & (-np.sin(alpha)*X + np.cos(alpha)*Y - (b+0.5*w) < 0)




#return np.sum(C)/np.double(GridSize**2)



#plot(x, y, '.')
#gca().set_aspect('equal')
#show()