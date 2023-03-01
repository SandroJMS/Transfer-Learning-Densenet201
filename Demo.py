# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from PIL import Image   
import cv2
import os
import absl.logging
import random

absl.logging.set_verbosity(absl.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_im_cv2(path, img_rows, img_cols):
    img = cv2.imread(path)
    resized = (cv2.resize(img, (img_cols, img_rows)))
    return resized

folders=[
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
    '10']

classList=[
    'Drive Safe',
    'Adjust Radio',
    'Reach Behind',
    'Talk Passenger',
    'Hair & Makeup',
    'Drink',
    'Text Right',
    'Text Left',
    'Talk Right',
    'Talk Left',
     ]

print('Loading Model')
pathModel = '.\models\model.h5'
print('Model Loaded')

model = tf.keras.models.load_model(pathModel)

path1 = '.\demo_dataset\{}'
path2 = '.\demo_dataset\{}\{}'

while True:
    for i in classList: print(classList.index(i),'-',i)
    print('input "exit" to quit:')
    ImgClass = input('choose class (0 - 9): ')
    print('\n')
    
    if ImgClass == 'exit':
        break
    
    pathFolder = path1.format(folders[int(ImgClass)])

    ImgName = random.choices(os.listdir(pathFolder),k=1)

    pathImg = path2.format(folders[int(ImgClass)],ImgName[0])
    print(pathImg)
        
    img = get_im_cv2(pathImg,224,224)
        
    y_pred = model.predict(np.resize(img,[1,224,224,3]))
        
    y1=np.argmax(y_pred, axis=1)


    print('Prediction: ',classList[y1[0]],'\n\n')

    img = Image.open(pathImg)
    img.show()