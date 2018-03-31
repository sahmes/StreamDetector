import os, re, numpy as np
from AuxiliaryFunctions import ReadConfig, ReadData, BackgroundControl
from keras.layers import Input, Dense
from keras.models import Model

Conf = ReadConfig()
DataDir = Conf['datadir']

if 'model.h5' in os.listdir(DataDir):
    print 'There is already a model.h5 file in the data directory. Remove or rename it and try again.'
    raise SystemExit

FileList = []
for FileName in os.listdir(DataDir):
    Match = re.findall(r'^streams-(\d+).h5', FileName)
    if len(Match)!=1: continue
    FileList.append(os.path.join(DataDir, FileName))

Data = ReadData(FileList)

Mode = 'NoTest'
Epochs = 10

alpha_ = Data['alpha']
alpha_[alpha_<0] = np.pi+alpha_[alpha_<0]
Data['s'] = np.sin(alpha_)
Data['b'] = Data['b']/np.sqrt(0.5)
Data['w'] = Data['w']/np.sqrt(0.5)

Background = BackgroundControl()
N_tot = Data['N_bg'] + Data['N_fg']
Norm = N_tot*(N_tot-1)/2
Samples = len(Norm)

BG = Background.Background
H_bg = BG.copy()
H_bg = H_bg.reshape((1, H_bg.shape[0], H_bg.shape[1])).repeat(Samples,axis=0)
for i in range(Samples): H_bg[i,...]*=Norm[i]
H_bg = H_bg[:,Background.Mask]

xi = np.asarray((Data['H'][:,Background.Mask]-H_bg)/H_bg, dtype=np.float32)
Data['xi'] = xi

del alpha_, H_bg, Data['H'] # Note needed

if Mode=='NoTest':
    Data_train = Data
else:
    Data_train, Data_test = SplitData(Data, Ratio=0.75) # We can pass the "validation_split" argument to fit, but this makes it easier to plot the results.

inputs = Input(shape=(3324,), name='xi')
x = Dense(1662, activation='relu')(inputs)
x = Dense(162, activation='relu')(x)
outputs = [Dense(1, name='s', activation='sigmoid')(x),
           Dense(1, name='b', activation='sigmoid')(x),
           Dense(1, name='w', activation='sigmoid')(x)]
model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='mae')
if Mode=='NoTest':
    History = model.fit(Data_train, Data_train, epochs=Epochs)
else:
    History = model.fit(Data_train, Data_train, validation_data=(Data_test, Data_test), epochs=Epochs) # It's a good idea to keep the history
model.save(os.path.join(DataDir, 'model.h5'))
