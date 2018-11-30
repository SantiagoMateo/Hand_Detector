import os
import re
import glob
import numpy as np
import os.path as path
from scipy import ndimage, misc
from keras.models import model_from_json
import matplotlib.pyplot as plt
import cv2
import imutils
from collections import deque
from imutils.video import VideoStream
import autopy

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'



def draw(x_test, model):
    labels = ['Mano abierta', 'Puño', 'Indice', 'Nada']

    x_test = x_test / 255.0
    x_test = [x_test, x_test]
    x_test = np.asarray(x_test)

    p = model.predict(x_test)
    indices = np.argmax(p, 1)

    return labels[indices[0]]



def load_m():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    return loaded_model

def Filtro (Img):
    ColorLow = (17,77,126)#(25, 70, 50)
    ColorHigh = (45,255,255)#(37, 236, 255)
    hsv = cv2.cvtColor(Img, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv, ColorLow, ColorHigh)
    kernel = np.ones((3,3),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, None, iterations=1)
    return mask

def main():
    bufferSize = 64
    pts = deque(maxlen=bufferSize)
    vs = VideoStream(src=0).start()

    model = load_m()
    while True:     
        Img = vs.read()                
        Img = cv2.flip(Img, 1 )
        filas,columnas = Img.shape[:2]
        
        MaskBW = Filtro (Img)
        Mask= cv2.bitwise_and(Img,Img,mask=MaskBW)
        
        NewMask = misc.imresize(Mask, (120, 160))
        Title = draw(NewMask, model)

        cnts = cv2.findContours(MaskBW.copy(), cv2.RETR_EXTERNAL,	cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        center = None

        IsClick = 0
        if len(cnts) > 0 :
            if Title == 'Mano abierta' :
                IsClick = 0
                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)

                Px = 1919
                Py = 1079

                center = (int(x), int(y))
                autopy.mouse.smooth_move((x*Px)//columnas, (y*Py)//filas)
                
                if radius > 50:
                    cv2.circle(Img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            elif Title == 'Puño':
                if IsClick == 0:  
                    autopy.mouse.click(button = autopy.mouse.Button.LEFT)                
                    IsClick = 1
                
            
        print('Predicted: ' + Title)  
        cv2.imshow('Imagen', Img)
    
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    
    vs.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
