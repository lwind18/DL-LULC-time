"""
Created on Wed Aug 31 16:49:06 2022
@author: Dong.Luo
feature [0,1]
Four classes: copland(0), grassland(1), savanna(2), forest(3)
v40: dense=32, epoch=50, lr=0.0001, batchsize=16, units=128. try to get a best model *
"""
# using CNN and LSTM to classify time series MODIS image (09-16-2021)
from tifffile import imread, imwrite
import rasterio
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import *
import matplotlib.pyplot as plt
# from keras.utils import np_utils 
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, BatchNormalization
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
##****************************************************************************************************
## check GPU device
import tensorflow as tf 
gpus = tf.config.list_physical_devices('GPU')
# print (gpus)

for gpu in gpus:
    print (gpu)
    tf.config.experimental.set_memory_growth(gpu, True)
    # tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*30)])

if gpus==[]:
    print ('There is no GPU on the machine')

if tf.test.gpu_device_name(): 
    # print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    print ('Default GPU Device has just been printed by invoking "if tf.test.gpu_device_name()"\n\n' )
##*******************************************************************************************************    
# Slice one image to many patches    https://zhuanlan.zhihu.com/p/39361808
# define a function to use tifffile package to read RS images in each year with  5 channels shape (height, width, depth)
def img_slice (img):
    dt = imread(img)
    shp = dt.shape
    dt0 = dt.reshape(-1, shp[0], shp[1], shp[2])
    dttf = tf.convert_to_tensor(dt0)
    dtpth = tf.image.extract_patches(images=dttf, sizes=[1, 15, 15, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')
    overpth = tf.reshape(dtpth,[-1,15,15,shp[2]])
    return overpth

# perpare feature data (SITE A) with the order: 2015, 2016, 2017, 2018, 2019, 2020
dta0 = imread("./train/MCD43A4_201509BOAAOI.SA.tif")
print(dta0.shape)    
imga07 = dta0.reshape(-1, 225, 225, 5)
imgsta07 = tf.convert_to_tensor (imga07)
pacha07 = tf.image.extract_patches(images=imgsta07, sizes=[1, 15, 15, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID') # SAME   VALID
dta07 = tf.reshape(pacha07,[-1,15,15,5])
#dta07 = img_slice('/content/drive/My Drive/Q1_newmodel/MCD43A4_201509BOASA.tif')
ptha070 = dta07.numpy()
dta08 = img_slice('./train/MCD43A4_201609BOAAOI.SA.tif')
ptha080 = dta08.numpy()
dta09 = img_slice('./train/MCD43A4_201709BOAAOI.SA.tif')
ptha090 = dta09.numpy()
dta10 = img_slice('./train/MCD43A4_201809BOAAOI.SA.tif')
ptha100 = dta10.numpy()
dta11 = img_slice('./train/MCD43A4_201909BOAAOI.SA.tif')
ptha110 = dta11.numpy()
dta12 = img_slice('./train/MCD43A4_202009BOAAOI.SA.tif')
ptha120 = dta12.numpy()
# create sequence dataset
seqdta = []
for k in zip(ptha070, ptha080,ptha090,ptha100,ptha110,ptha120):
    seqdta.append(k)
print(np.array(seqdta).shape) # (n_sample, time_step, height, width, depth)

# perpare feature data (SITE B) with the order: 2015, 2016, 2017, 2018, 2019, 2020
dtb0 = imread("./train/MCD43A4_201509BOAAOI.SB.tif")
print(dtb0.shape)    
imgb07 = dtb0.reshape(-1, 225, 225, 5)
imgstb07 = tf.convert_to_tensor (imgb07)
pachb07 = tf.image.extract_patches(images=imgstb07, sizes=[1, 15, 15, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID') # SAME   VALID
dtb07 = tf.reshape(pachb07,[-1,15,15,5])
pthb070 = dtb07.numpy()
dtb08 = img_slice('./train/MCD43A4_201609BOAAOI.SB.tif')
pthb080 = dtb08.numpy()
dtb09 = img_slice('./train/MCD43A4_201709BOAAOI.SB.tif')
pthb090 = dtb09.numpy()
dtb10 = img_slice('./train/MCD43A4_201809BOAAOI.SB.tif')
pthb100 = dtb10.numpy()
dtb11 = img_slice('./train/MCD43A4_201909BOAAOI.SB.tif')
pthb110 = dtb11.numpy()
dtb12 = img_slice('./train/MCD43A4_202009BOAAOI.SB.tif')
pthb120 = dtb12.numpy()
# create sequence dataset
seqdtb = []
for i in zip(pthb070, pthb080,pthb090,pthb100,pthb110,pthb120):
    seqdtb.append(i)
print(np.array(seqdtb).shape) # (n_sample, time_step, height, width, depth)

# perpare feature data (SITE C) with the order: 2015, 2016, 2017, 2018, 2019, 2020
dtc0 = imread("./train/MCD43A4_201509BOAAOI.SC.tif")
print(dtc0.shape)    
imgc07 = dtc0.reshape(-1, 225, 225, 5)
imgstc07 = tf.convert_to_tensor (imgc07)
pachc07 = tf.image.extract_patches(images=imgstc07, sizes=[1, 15, 15, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID') # SAME   VALID
dtc07 = tf.reshape(pachc07,[-1,15,15,5])
pthc070 = dtc07.numpy()
dtc08 = img_slice('./train/MCD43A4_201609BOAAOI.SC.tif')
pthc080 = dtc08.numpy()
dtc09 = img_slice('./train/MCD43A4_201709BOAAOI.SC.tif')
pthc090 = dtc09.numpy()
dtc10 = img_slice('./train/MCD43A4_201809BOAAOI.SC.tif')
pthc100 = dtc10.numpy()
dtc11 = img_slice('./train/MCD43A4_201909BOAAOI.SC.tif')
pthc110 = dtc11.numpy()
dtc12 = img_slice('./train/MCD43A4_202009BOAAOI.SC.tif')
pthc120 = dtc12.numpy()
# create sequence dataset
seqdtc = []
for i in zip(pthc070, pthc080,pthc090,pthc100,pthc110,pthc120):
    seqdtc.append(i)
print(np.array(seqdtc).shape) # (n_sample, time_step, height, width, depth)

# create label (SITE A) in 2020
dtla12 = imread('./train/MCD12Q1.IBGP_rc_a.label2020.tif')   # trainlab478A4202012luc3
print(dtla12.shape)
imga2 = dtla12.reshape(-1, 225, 225, 1)
imgsta2 = tf.convert_to_tensor (imga2)
slclaba12 = tf.image.extract_patches(images=imgsta2, sizes=[1, 15, 15, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')
labptha12 = tf.reshape(slclaba12,[-1,15,15,1])
print(labptha12.shape)
centlaba12 = []
for i in labptha12:
    # difference with odd and even. Even: [n/2-1:n/2+1, n/2-1:n/2+1]
    # https://stackoverflow.com/questions/36318806/fastest-way-to-access-middle-four-elements-in-numpy-array
    centera = i[int((i.shape[0]-1)/2), int((i.shape[1]-1)/2)]
    centlaba12.append(int(centera))
print(np.array(centlaba12).shape)

# create label (SITE B) in 2020
dtlb12 = imread('./train/MCD12Q1.IBGP_rc_b.label2020.tif')   # trainlab478A4202012luc3
print(dtlb12.shape)
imgb2 = dtlb12.reshape(-1, 225, 225, 1)
imgstb2 = tf.convert_to_tensor (imgb2)
slclabb12 = tf.image.extract_patches(images=imgstb2, sizes=[1, 15, 15, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')
labpthb12 = tf.reshape(slclabb12,[-1,15,15,1])
print(labpthb12.shape)
centlabb12 = []
for i in labpthb12:
    # difference with odd and even. Even: [n/2-1:n/2+1, n/2-1:n/2+1]
    # https://stackoverflow.com/questions/36318806/fastest-way-to-access-middle-four-elements-in-numpy-array
    centerb = i[int((i.shape[0]-1)/2), int((i.shape[1]-1)/2)]
    centlabb12.append(int(centerb))
print(np.array(centlabb12).shape)

# create label (SITE C) in 2020
dtlc12 = imread('./train/MCD12Q1.IBGP_rc_c.label2020.tif')   # trainlab478A4202012luc3
print(dtlc12.shape)
imgc2 = dtlc12.reshape(-1, 225, 225, 1)
imgstc2 = tf.convert_to_tensor (imgc2)
slclabc12 = tf.image.extract_patches(images=imgstc2, sizes=[1, 15, 15, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')
labpthc12 = tf.reshape(slclabc12,[-1,15,15,1])
print(labpthc12.shape)
centlabc12 = []
for i in labpthc12:
    # difference with odd and even. Even: [n/2-1:n/2+1, n/2-1:n/2+1]
    # https://stackoverflow.com/questions/36318806/fastest-way-to-access-middle-four-elements-in-numpy-array
    centerc = i[int((i.shape[0]-1)/2), int((i.shape[1]-1)/2)]
    centlabc12.append(int(centerc))
print(np.array(centlabc12).shape)
##*********************************************************UP: PRORESS FEATURE AND LABEL FROM SIATE A AND B**************************************************************************

# create model input dataset for SITE A
lucdta = [np.array(seqdta), np.array(centlaba12)]

# create model input dataset for SITE B
lucdtb = [np.array(seqdtb), np.array(centlabb12)]

# create model input dataset for SITE C
lucdtc = [np.array(seqdtc), np.array(centlabc12)]
#################################################################
# This step is to seperate different types of patches in site A
balulca0 = []
balulca1 = []
balulca2 = []
balulca3 = []
balulca4 = []

for m, n in zip(lucdta[0], lucdta[1]):   # m and n are np.array
    if n == 0:
        crop = m
        balulca0.append(crop)
    elif n == 1:
        gras = m
        balulca1.append(gras)
    elif n == 2:
        sava = m
        balulca2.append(sava)
    elif n == 3:
        fore = m
        balulca3.append(fore)
    else:
        back = m
        balulca4.append(back)
print('cropland:', np.array(balulca0).shape)   
print('grassland', np.array(balulca1).shape)   
print('savanna', np.array(balulca2).shape)
print('forest', np.array(balulca3).shape)

# 0 is cropland
balabela0 = [l0 for l0 in centlaba12 if l0 == 0]
trainsa0 = [balulca0, np.array(balabela0)]

# 1 is grassland
balabela1 = [l1 for l1 in centlaba12 if l1 == 1]
trainsa1 = [balulca1, np.array(balabela1)]

# 2 is savanna
balabela2 = [l2 for l2 in centlaba12 if l2 == 2]
trainsa2 = [balulca2, np.array(balabela2)]

#3 is forest
balabela3 = [l3 for l3 in centlaba12 if l3 == 3]
trainsa3 = [balulca3, np.array(balabela3)]
#####################################################
# This step is to seperate different types of patches in site B
balulcb0 = []
balulcb1 = []
balulcb2 = []
balulcb3 = []
balulcb4 = []
for p, q in zip(lucdtb[0], lucdtb[1]):   # m and n are np.array
    if q == 0:
        crop = p
        balulcb0.append(crop)
    elif q == 1:
        gras = p
        balulcb1.append(gras)
    elif q == 2:
        sava = p
        balulcb2.append(sava)
    elif q == 3:
        fore = p
        balulcb3.append(fore)
    else:
        back = p
        balulcb4.append(back)
print('cropland', np.array(balulcb0).shape)   
print('grassland', np.array(balulcb1).shape)   
print('savanna', np.array(balulcb2).shape)
print('forest', np.array(balulcb3).shape)

# 0 is cropland
balabelb0 = [l0 for l0 in centlabb12 if l0 == 0]
trainsb0 = [balulcb0, np.array(balabelb0)]

# 1 is grassland
balabelb1 = [l1 for l1 in centlabb12 if l1 == 1]
trainsb1 = [balulcb1, np.array(balabelb1)]

# 2 is savanna
balabelb2 = [l2 for l2 in centlabb12 if l2 == 2]
trainsb2 = [balulcb2, np.array(balabelb2)]

#3 is forest
balabelb3 = [l3 for l3 in centlabb12 if l3 == 3]
trainsb3 = [balulcb3, np.array(balabelb3)]

#####################################################
# This step is to seperate different types of patches in site C
balulcc0 = []
balulcc1 = []
balulcc2 = []
balulcc3 = []
balulcc4 = []
for p, q in zip(lucdtc[0], lucdtc[1]):   # m and n are np.array
    if q == 0:
        crop = p
        balulcc0.append(crop)
    elif q == 1:
        gras = p
        balulcc1.append(gras)
    elif q == 2:
        sava = p
        balulcc2.append(sava)
    elif q == 3:
        fore = p
        balulcc3.append(fore)
    else:
        back = p
        balulcc4.append(back)
print('cropland', np.array(balulcc0).shape)   
print('grassland', np.array(balulcc1).shape)   
print('savanna', np.array(balulcc2).shape)
print('forest', np.array(balulcc3).shape)

# 0 is cropland
balabelc0 = [l0 for l0 in centlabc12 if l0 == 0]
trainsc0 = [balulcc0, np.array(balabelc0)]

# 1 is grassland
balabelc1 = [l1 for l1 in centlabc12 if l1 == 1]
trainsc1 = [balulcc1, np.array(balabelc1)]

# 2 is savanna
balabelc2 = [l2 for l2 in centlabc12 if l2 == 2]
trainsc2 = [balulcc2, np.array(balabelc2)]

#3 is forest
balabelc3 = [l3 for l3 in centlabc12 if l3 == 3]
trainsc3 = [balulcc3, np.array(balabelc3)]
#########################################################################################################
# working on each land use and land cover in both sites to generate training patches
# crop is 0
fea0 = np.concatenate((np.array(trainsa0[0]), np.array(trainsb0[0]), np.array(trainsc0[0])), axis=0)
lab0 = np.concatenate((np.array(trainsa0[1]), np.array(trainsb0[1]), np.array(trainsc0[1])), axis=0)
X0, y0 = [fea0, lab0]
trainx0, testx0, trainy0, testy0 = train_test_split(X0,  y0, test_size=0.2)

# grass is 1
fea1 = np.concatenate((np.array(trainsa1[0]), np.array(trainsb1[0]), np.array(trainsc1[0])), axis=0)
lab1 = np.concatenate((np.array(trainsa1[1]), np.array(trainsb1[1]), np.array(trainsc1[1])), axis=0)
X1, y1 = [fea1, lab1]
trainx1, testx1, trainy1, testy1 = train_test_split(X1,  y1, test_size=0.2)

# savanna is 2
fea2 = np.concatenate((np.array(trainsa2[0]), np.array(trainsb2[0]), np.array(trainsc2[0])), axis=0)
lab2 = np.concatenate((np.array(trainsa2[1]), np.array(trainsb2[1]), np.array(trainsc2[1])), axis=0)
X2, y2 = [fea2, lab2]
trainx2, testx2, trainy2, testy2 = train_test_split(X2,  y2, test_size=0.2)

# forest is 3
fea3 = np.concatenate((np.array(trainsa3[0]), np.array(trainsb3[0]), np.array(trainsc3[0])), axis=0)
lab3 = np.concatenate((np.array(trainsa3[1]), np.array(trainsb3[1]), np.array(trainsc3[1])), axis=0)
X3, y3 = [fea3, lab3]
trainx3, testx3, trainy3, testy3 = train_test_split(X3,  y3, test_size=0.2)

# combine train and test
trainx = np.concatenate((trainx0, trainx1, trainx2, trainx3), axis=0) 
testx = np.concatenate((testx0, testx1, testx2, testx3), axis=0) 
trainy = np.concatenate((trainy0, trainy1, trainy2, trainy3), axis=0)
testy = np.concatenate((testy0, testy1, testy2, testy3), axis=0) 

print(trainx.shape)
print(testx.shape)

# convert numpy array to tensor
trainX = tf.convert_to_tensor(trainx)
print(trainX.shape)
testX = tf.convert_to_tensor(testx)
# convert class vectors to binary class matrices
trainy = tf.keras.utils.to_categorical(trainy, 4) 
testy = tf.keras.utils.to_categorical(testy, 4)
###################################################################################################
####################################################################################################
# define the model. First CNN, but using timedistributed to wrap it. Then LSTM
model = Sequential()
model.add(TimeDistributed(Conv2D(16, (3,3), padding='same', activation= 'relu'), input_shape=(6,15,15,5))) # shape(time_step, H, W, D)
#model.add(TimeDistributed(BatchNormalization()))

model.add(TimeDistributed(Conv2D(32, (3,3), activation='relu')))

model.add(TimeDistributed(Conv2D(64, (3,3), padding='same', activation='relu')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
#model.add(TimeDistributed(BatchNormalization()))

#model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu')))
#model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))

model.add(TimeDistributed(Conv2D(128, (3,3), activation= 'relu')))
#model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))


model.add(TimeDistributed(Flatten()))
#model.add(Dropout(0.5))

#model.add(LSTM(units=256, return_sequences=True))
model.add(LSTM(units=128))    # I can have couple LSTMs, but the last one shoule be the return_sequences=Flase in this case
#model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))
print(model.summary())

opt = optimizers.Adam(learning_rate=0.0001)   # or 0.0001, but 0.00001 is better    RMSprop   Adam   SGD
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])    

mdfit = model.fit(trainX, trainy,
                batch_size=16,
                epochs=50, verbose=2, shuffle=True,
                validation_data=(testX, testy))

score = model.evaluate(testX, testy, verbose=2)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


model.save('./Q1cnnlstmmodel.modis.v40.h5')
#######################################################################################################################
#######################################################################################################################
# perpare feature data (SITE C) with the order: 2015, 2016, 2017, 2018, 2019, 2020
dtc0 = imread("./test/MCD43A4_201509BOAAOI.SE.tif")
print(dtc0.shape)    
imgc07 = dtc0.reshape(-1, 225, 225, 5)
imgstc07 = tf.convert_to_tensor (imgc07)
pachc07 = tf.image.extract_patches(images=imgstc07, sizes=[1, 15, 15, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID') # SAME   VALID
dtc07 = tf.reshape(pachc07,[-1,15,15,5])
pthc070 = dtc07.numpy()
dtc08 = img_slice('./test/MCD43A4_201609BOAAOI.SE.tif')
pthc080 = dtc08.numpy()
dtc09 = img_slice('./test/MCD43A4_201709BOAAOI.SE.tif')
pthc090 = dtc09.numpy()
dtc10 = img_slice('./test/MCD43A4_201809BOAAOI.SE.tif')
pthc100 = dtc10.numpy()
dtc11 = img_slice('./test/MCD43A4_201909BOAAOI.SE.tif')
pthc110 = dtc11.numpy()
dtc12 = img_slice('./test/MCD43A4_202009BOAAOI.SE.tif')
pthc120 = dtc12.numpy()
# create sequence dataset
seqdtc = []
for k in zip(pthc070, pthc080,pthc090,pthc100,pthc110,pthc120):
    seqdtc.append(k)

rdy_seqdtc = np.array(seqdtc)
print(rdy_seqdtc.shape) # (n_sample, time_step, height, width, depth)


## LSTM have the same prediction function with CNN 
pred = model.predict(rdy_seqdtc)
pred0 = np.argmax(pred, axis=1)

predimg = pred0.reshape(211,211)
# plt.imshow(predimg, cmap='spring')
# plt.show()
#################################################################################################
# evalaute results
with rasterio.open("./test/MCD12Q1.IBGP_rc_e.label2020.tif") as dst:
  dto = dst.read()
  geoto = dst.profile
dto_2d = np.squeeze(dto, axis=0)
# this part is working to get the shape (211, 211)
dt_2d_left = np.delete(dto_2d, np.s_[0:7], axis=0)
dt_2d_right = np.delete(dt_2d_left, np.s_[-7:], axis=0)
dt_2d_up = np.delete(dt_2d_right, np.s_[0:7], axis=1)
dt_2d_down = np.delete(dt_2d_up, np.s_[-7:], axis=1)
print(dt_2d_down.shape)
####################
msk = dt_2d_down!=4
predimg[np.logical_not(msk)] =4

gtf = dt_2d_down.flatten()
cnnf = predimg.flatten()
# CNN result and groud truth
print(classification_report(gtf, cnnf))
print(confusion_matrix(gtf, cnnf))
print('cohen_kappa_score:', cohen_kappa_score(gtf, cnnf))
print('accuracy_score', accuracy_score(gtf, cnnf))

#####################################################################################################
## the way to save raster with profile
meta = geoto.copy()
meta['driver'] = 'GTiff'
meta['width'] = 211
meta['height'] = 211
meta['nodata'] = 5
out_file = './result/MCD43A4202009SE.CNNLSTM.v40.tif'
with rasterio.open(out_file, 'w', **meta) as dst:
  dst.write(predimg, 1)

with rasterio.open('./result/MCD43A4202009SE.GT.tif', 'w', **meta) as dst:
  dst.write(dt_2d_down, 1)