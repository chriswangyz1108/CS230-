from keras import backend as K
from keras.models import load_model
import cv2
import numpy as np
import os
from keras.applications.resnet50 import preprocess_input
K.set_image_dim_ordering("th")

model = load_model('model.h5')
model.compile(loss='categorical_crossentropy', optimizer='Adadelta', metrics = ['accuracy'])
while(1):
    folder = input('Type the folder (ex Test Set/Nicole Kidman/): ')
    for i in os.listdir(str(folder)):
	    
        try:
            filepath = i 
            img = cv2.imread(str(folder) + str(filepath))
            #x = preprocess_input(img)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,(200,200))
            #x = preprocess_input(img)
            x = img.transpose(2,0,1)
            x = np.expand_dims(x,axis=0)
            x = preprocess_input(x.astype('float32'))
            pred = model.predict(x).argmax(-1)[0]
            if pred==0:
                print(str(i) + ' is: Barack Obama')
            if pred==1:
                print(str(i) + ' is: David Beckham')
            if pred==2:
                print(str(i) + ' is: Emma Watson')
            if pred==3:
                print(str(i) + ' is: Michael Jordan')
            if pred==4:
                print(str(i) + ' is: Michelle Obama')
            if pred==5:
                print(str(i) + ' is: Nicole Kidman')
        except Exception as e:
            print('Error!!!! ',e)