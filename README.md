# Hand_Detector
Hand Detect that move cursor, python project

University Santo Tomas Faculty of Engineering Electronic

Miguel Elkin Jimenez Avila Santiago Mateo Guerrero Romero Jhonatan Alexander Navas Ballesteros

Description of project
This project is a hand detect that move the cursor according to position of hand, also detect fist and the program do click.

We used a classifier to the detection hand and fist, this classifier is a CNN (Convolution Neuronal Network).

We used a color filter to the localitation hand and fist, this filter take the yellow color.

Next we going to show the flow diagram:

diagrama

We use Tensorflow to create CNN, this net have 4 layers. Information for CNN:

red

Information for trainning:
curva

The filter is a band-pass filter to the yellow color, because we used a yellow glove to does easy localization process.

Images samples
Detect open hand and location in the screen, get position of the hand and set position of the cursor.

manoabierta

Detect the fist and does click. https://user-images.githubusercontent.com/45437733/49186418-95af2d00-f332-11e8-833a-4d42d40f982f.png

puno

Steps for run program:
In your terminal, you go to folder "DetectHand".
Run this line python test2.py
Put the yellow glove
Try to move the cursor and then try to do click
Dependencies:
cv2
tensorflow
keras
numpy
scipy
os
re
glob
os.path
matplotlib
imutils
collections
autopy
