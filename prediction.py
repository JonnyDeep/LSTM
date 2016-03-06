# -*- coding: utf-8 -*-


import numpy as np
np.random.seed(1337)
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from six.moves import cPickle
import os
import string


pklFilePath="/home/hlq/Python/mypkl/"
fileList=os.listdir(pklFilePath)
#ACC loss
resultFile="/home/hlq/Python/lstm/result2.txt"  
#预测结果
predict="/home/hlq/Python/lstm/predict.txt"
FileEntity=open(resultFile,'w+')
predictEntity=open(predict,'w+')
if(os.path.exists(pklFilePath)==False):
    os.makedirs(pklFilePath)




#读取文件数据
def readPklFile(filePath):
    f=open(pklFilePath+filePath,'rb')
    x,y=cPickle.load(f)
    
    '''
    print(len(x))
    print("y len:"+"%i"%(len(y)))
    print(type(y))
    print(len(y))
    print(y)
    '''
    length=len(y)
    y=y.reshape(length,1,1)
    return x,y
    

#数据归一化
def Nomalization(data):
    #三维数组
    for i in range(len(data)):
        if data[i][0][0]<0.5:
            data[i][0][0]=0
        else:
            data[i][0][0]=1
    return data

#计算ACC
def CalculateAcc(OriginalData,PredictData):
    m=len(OriginalData)
    tp=0
    tn=0
    for i in range(m):
        if OriginalData[i][0][0]==PredictData[i][0][0]:
            if PredictData[i][0][0]==1:
                tp=tp+1
            else:
                tn=tn+1
    acc=float(tp+tn)/m
    return acc
print("Building model")
model=Sequential()
#input_length:蛋白质的残基数
#input_dim:残基的参数，前20
layer=LSTM(output_dim=20,activation='tanh',return_sequences=True,input_dim=20)
#input_x=layer.input
#print input_x.ndim
#print layer.input
model.add(layer)
model.add(Dropout(0.3))
layer1=LSTM(output_dim=20,activation='tanh',return_sequences=True,input_dim=20)
model.add(layer1)
model.add(Dropout(0.3))
#print layer1.input.shape
layer2=LSTM(output_dim=1,activation='tanh',return_sequences=True,input_dim=20)
model.add(layer2)
model.add(Activation('sigmoid'))
#print layer2.input.shape


model.compile(loss='binary_crossentropy',optimizer='adam',class_mode="binary")



print("training model")

'''
x_train,y_train=readPklFile("1a0a.pkl")
print x_train
print y_train
print type(x_train)
print type(y_train)
print len(x_train)
if(len(x_train)==0):
    print "NULL"
if x_train==[]:
    print 1
else:
    print 2

model.fit(x_train,y_train,nb_epoch=40,batch_size=32,verbose=1,show_accuracy=True)
x_train,y_train=readPklFile("1a0i.pkl")
classes=model.predict(x_train,batch_size=32,verbose=1)
y_test=Nomalization(classes)
loss,acc=model.test_on_batch(x_train,y_test,accuracy=True)
ACC=CalculateAcc(y_train,y_test)
print 'myACC'+str(ACC)
print 'loss'+str(loss)+'\tacc'+str(acc)   
FileEntity.writelines('1a0i'+'\n'+"acc: "+"%i"%acc+"\n")
FileEntity.close() 
'''


i=0
for file in fileList:
    print file
    print i
    x_train,y_train=readPklFile(file)
    #print x_train
    if x_train.ndim<3:
        continue
    if(len(x_train)==0):
        continue
    if i<1800:
        model.fit(x_train,y_train,nb_epoch=40,batch_size=32,verbose=1,show_accuracy=True)
    else:
        classes=model.predict(x_train,batch_size=32,verbose=1)
        y_test=Nomalization(classes)
        predictEntity.writelines(file+'\n'+"original:\n"+str(y_train)+"\npredict:\n"+str(y_test)+'\n')
        loss,acc=model.test_on_batch(x_train,y_train,accuracy=True)
        
        #print loss,acc
        #acc= CalculateAcc(y_train,y_test)
        FileEntity.writelines(file+'\n'+'loss: '+str(loss)+"\tacc: "+str(acc)+"\n")        
        
        #model.evaluate(x_train,y_train,batch_size=32,verbose=1,show_accuracy=True)
    i=i+1
FileEntity.close()
predictEntity.close()
print "finish"
