"""
Created on Wed Aug 31 16:01:34 2022
@author: Dong.Luo
features is[0,1]
label: 0= cropland, 1= grassland, 2=savanne, 3=forest, 4=background
find record from previous code files
see previous code. from v23, use new label data (combine modis and mapbimos)
v25: epoch=50, batchsize=16, lr=0.0001, dropout=0.5, dense =50, kerel=(3,3). try to get a best model
"""
## This is the CNN model
## four classes: cropland (0), grassland (1), savanna (2), forest (3)
## currently, it looks like site A training data has some issues (10-21-2021)
from tifffile import imread, imwrite
import rasterio
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import *
from keras.utils import np_utils 
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras import optimizers

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
##**********************************************************SITE A*****************************************************************************
# Using tifffile to read site A tif file with 5 channels (skikit-image to resize it)
dta = imread("./train/MCD43A4_202009BOAAOI.SA.tif")
print(dta.shape)    #(420, 420, 5)
imga = dta.reshape(-1, 225, 225, 5)
imgsta = tf.convert_to_tensor (imga)
pacha = tf.image.extract_patches(images=imgsta, sizes=[1, 15, 15, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID') # SAME   VALID
pacha = tf.reshape(pacha,[-1,15,15,5])
print(pacha.shape)

# Using tifffile to read site A label tif file with 1 channel
dtla = imread("./train/MCD12Q1.IBGP_rc_a.label2020.tif")    
imgla = dtla.reshape(-1, 225, 225, 1)
imgstla = tf.convert_to_tensor (imgla)
labpacha = tf.image.extract_patches(images=imgstla, sizes=[1, 15, 15, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')
labpacha = tf.reshape(labpacha,[-1,15,15,1])
print(labpacha.shape)
# create the label of each patch
centlaba = []
for i in labpacha:     # i is a array (15, 15, 1)
    # difference with odd and even. Even: [n/2-1:n/2+1, n/2-1:n/2+1]
    # https://stackoverflow.com/questions/36318806/fastest-way-to-access-middle-four-elements-in-numpy-array
    centera = i[int((i.shape[0]-1)/2), int((i.shape[1]-1)/2)]
    centlaba.append(int(centera))    # in pycharm, i need to remove int and change 0 to 0.0; 1 to 1.0

# Create input data. What I did is add ".numpy" convert tensor to numpy to match label
lucdta = [pacha.numpy(), np.array(centlaba)]  # list data

# This step is to seperate each land use and land cover type. 
# crop =0, grass=1, savan=2, forest=3, water=4, others=5
ballulca0 = []
ballulca1 = []
ballulca2 = []
ballulca3 = []
ballulca4 = []
for m, n in zip(lucdta[0], lucdta[1]):   # m and n are np.array
    if n == 0:
        crop = m
        ballulca0.append(crop)
    elif n == 1:
        gras = m
        ballulca1.append(gras)
    elif n == 2:
        sava = m
        ballulca2.append(sava)
    elif n == 3:
        fore = m
        ballulca3.append(fore)
    else:
        back = m
        ballulca4.append(back)
print('cropland', np.array(ballulca0).shape)   # (n_sample, 15, 15, 5)
print('grassland', np.array(ballulca1).shape)   # (n_sample, 15, 15, 5)
print('savanna', np.array(ballulca2).shape)
print('forest', np.array(ballulca3).shape)

# 0 is cropland
balabela0 = [l0 for l0 in centlaba if l0 == 0]
trainsa0 = [ballulca0, np.array(balabela0)]
# print(np.array(trainsa0[0]).shape)

# 1 is grassland
balabela1 = [l1 for l1 in centlaba if l1 == 1]
trainsa1 = [ballulca1, np.array(balabela1)]

# 2 is savanna
balabela2 = [l2 for l2 in centlaba if l2 == 2]
trainsa2 = [ballulca2, np.array(balabela2)]

# 3 is forest
balabela3 = [l3 for l3 in centlaba if l3 == 3]
trainsa3 = [ballulca3, np.array(balabela3)]

##****************************************UP: SITE A     DOWN: SITE B******************************************************************
# Using tifffile to read site B tif file with 7 channels (skikit-image to resize it)
dtb = imread("./train/MCD43A4_202009BOAAOI.SB.tif")        
print(dtb.shape)    #(300, 300, 5)
imgb = dtb.reshape(-1, 225, 225, 5)
imgstb = tf.convert_to_tensor (imgb)
pachb = tf.image.extract_patches(images=imgstb, sizes=[1, 15, 15, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID') # SAME   VALID
pachb = tf.reshape(pachb,[-1,15,15,5])
print(pachb.shape)

# Using tifffile to read label tif file with 1 channel
dtlb = imread("./train/MCD12Q1.IBGP_rc_b.label2020.tif")   
imglb = dtlb.reshape(-1, 225, 225, 1)
imgstlb = tf.convert_to_tensor (imglb)
labpachb = tf.image.extract_patches(images=imgstlb, sizes=[1, 15, 15, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')
labpachb = tf.reshape(labpachb,[-1,15,15,1])
print(labpachb.shape)
# create the label of each patch
centlabb = []
for i in labpachb:     # i is a array (15, 15, 1)
    # difference with odd and even. Even: [n/2-1:n/2+1, n/2-1:n/2+1]
    # https://stackoverflow.com/questions/36318806/fastest-way-to-access-middle-four-elements-in-numpy-array
    centerb = i[int((i.shape[0]-1)/2), int((i.shape[1]-1)/2)]
    centlabb.append(int(centerb))    # in pycharm, i need to remove int and change 0 to 0.0; 1 to 1.0

# Create input data. What I did is add ".numpy" convert tensor to numpy to match label
lucdtb = [pachb.numpy(), np.array(centlabb)]  # list data

# This step is to balance the unburn and burn areas
ballulcb0 = []
ballulcb1 = []
ballulcb2 = []
ballulcb3 = []
ballulcb4 = []
for m, n in zip(lucdtb[0], lucdtb[1]):   # m and n are np.array
    if n == 0:
        crop = m
        ballulcb0.append(crop)
    elif n == 1:
        gras = m
        ballulcb1.append(gras)
    elif n == 2:
        sava = m
        ballulcb2.append(sava)
    elif n == 3:
        fore = m
        ballulcb3.append(fore)
    else:
        back = m
        ballulcb4.append(back)
print('cropland', np.array(ballulcb0).shape)   # (n_sample, 15, 15, 7)
print('grassland', np.array(ballulcb1).shape)   # (n_sample, 15, 15, 7)
print('savanna', np.array(ballulcb2).shape)
print('forest', np.array(ballulcb3).shape)

balabelb0 = [l0 for l0 in centlabb if l0 == 0]
trainsb0 = [ballulcb0, np.array(balabelb0)]
# print(np.array(trainsb0[0]).shape)

balabelb1 = [l1 for l1 in centlabb if l1 == 1]
trainsb1 = [ballulcb1, np.array(balabelb1)]

balabelb2 = [l2 for l2 in centlabb if l2 == 2]
trainsb2 = [ballulcb2, np.array(balabelb2)]

balabelb3 = [l3 for l3 in centlabb if l3 == 3]
trainsb3 = [ballulcb3, np.array(balabelb3)]

##****************************************UP: SITE B     DOWN: SITE C******************************************************************
# Using tifffile to read site B tif file with 7 channels (skikit-image to resize it)
dtc = imread("./train/MCD43A4_202009BOAAOI.SC.tif")        
print(dtc.shape)    #(300, 300, 5)
imgc = dtc.reshape(-1, 225, 225, 5)
imgstc = tf.convert_to_tensor (imgc)
pachc = tf.image.extract_patches(images=imgstc, sizes=[1, 15, 15, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID') # SAME   VALID
pachc = tf.reshape(pachc,[-1,15,15,5])
print(pachc.shape)

# Using tifffile to read label tif file with 1 channel
dtlc = imread("./train/MCD12Q1.IBGP_rc_c.label2020.tif")   
imglc = dtlc.reshape(-1, 225, 225, 1)
imgstlc = tf.convert_to_tensor (imglc)
labpachc = tf.image.extract_patches(images=imgstlc, sizes=[1, 15, 15, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')
labpachc = tf.reshape(labpachc,[-1,15,15,1])
print(labpachc.shape)
# create the label of each patch
centlabc = []
for i in labpachc:     # i is a array (15, 15, 1)
    # difference with odd and even. Even: [n/2-1:n/2+1, n/2-1:n/2+1]
    # https://stackoverflow.com/questions/36318806/fastest-way-to-access-middle-four-elements-in-numpy-array
    centerc = i[int((i.shape[0]-1)/2), int((i.shape[1]-1)/2)]
    centlabc.append(int(centerc))    # in pycharm, i need to remove int and change 0 to 0.0; 1 to 1.0

# Create input data. What I did is add ".numpy" convert tensor to numpy to match label
lucdtc = [pachc.numpy(), np.array(centlabc)]  # list data

# This step is to balance the unburn and burn areas
ballulcc0 = []
ballulcc1 = []
ballulcc2 = []
ballulcc3 = []
ballulcc4 = []
for m, n in zip(lucdtc[0], lucdtc[1]):   # m and n are np.array
    if n == 0:
        crop = m
        ballulcc0.append(crop)
    elif n == 1:
        gras = m
        ballulcc1.append(gras)
    elif n == 2:
        sava = m
        ballulcc2.append(sava)
    elif n == 3:
        fore = m
        ballulcc3.append(fore)
    else:
        back = m
        ballulcc4.append(back)
print('cropland', np.array(ballulcc0).shape)   # (n_sample, 15, 15, 7)
print('grassland', np.array(ballulcc1).shape)   # (n_sample, 15, 15, 7)
print('savanna', np.array(ballulcc2).shape)
print('forest', np.array(ballulcc3).shape)

balabelc0 = [l0 for l0 in centlabc if l0 == 0]
trainsc0 = [ballulcc0, np.array(balabelc0)]
# print(np.array(trainsb0[0]).shape)

balabelc1 = [l1 for l1 in centlabc if l1 == 1]
trainsc1 = [ballulcc1, np.array(balabelc1)]

balabelc2 = [l2 for l2 in centlabc if l2 == 2]
trainsc2 = [ballulcc2, np.array(balabelc2)]

balabelc3 = [l3 for l3 in centlabc if l3 == 3]
trainsc3 = [ballulcc3, np.array(balabelc3)]
##*****************************************************END OF DATA PROCESS******************************************************************************
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
# convert class vectors to binary class matrices
trainy = tf.keras.utils.to_categorical(trainy, 4)
testy = tf.keras.utils.to_categorical(testy, 4)
###################################################################################################
###################################################################################################
# CNN model using keras
model = Sequential()
model.add(Conv2D(16, (3, 3), padding='same', activation= 'relu', input_shape=(15,15,5))) # 
#model.add(BatchNormalization())

model.add(Conv2D(32, (3, 3), activation='relu'))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
#model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
#model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))
model.summary()
                                                                                               
opt = optimizers.Adam(learning_rate=0.0001)   #SGD   RMSprop  Adam    0.0001 is good 
model.compile(loss='CategoricalCrossentropy', optimizer=opt, metrics=['acc'])

mdfit = model.fit(trainx, trainy,
                  batch_size=16,
                  epochs=50, verbose=2, shuffle=True,
                  validation_data=(testx, testy))
               
# check the key of accuracy such as 'acc','loss'
history_dict = mdfit.history
print(history_dict.keys())

loss, acc = model.evaluate(testx, testy, verbose=2)
print(loss)
print(acc)

# save the cnn model
model.save('./Q1cnnmodel.modis.v25.h5')
################################################################################################
################################################################################################
##*********************************************************************************************************
# Using tifffile to read site A tif file with 5 channels (skikit-image to resize it)
dto = imread("./test/MCD43A4_202009BOAAOI.SE.tif")     
print(dto.shape)    
imgto = dto.reshape(-1, 225, 225, 5)
imgsto = tf.convert_to_tensor (imgto)
pachto = tf.image.extract_patches(images=imgsto, sizes=[1, 15, 15, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID') # SAME   VALID
pachto = tf.reshape(pachto,[-1,15,15,5])
print(pachto.shape)

# load trained model
#model = tf.keras.models.load_model('./Q1cnnmodel.modis.v2.h5')

## LSTM have the same prediction function with CNN 
pred = model.predict(pachto)
pred0 = np.argmax(pred, axis=1)

predimg = pred0.reshape(211,211)
# plt.imshow(predimg, cmap='winter')
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
out_file = './result/MCD43A4202009SE.CNN.v25.tif'
with rasterio.open(out_file, 'w', **meta) as dst:
  dst.write(predimg, 1)
  
with rasterio.open('./result/MCD43A4202009SE.GT.tif', 'w', **meta) as dst:
  dst.write(dt_2d_down, 1)