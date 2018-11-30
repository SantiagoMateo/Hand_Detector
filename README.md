# Hand_Detector

Hand Detect that move cursor, python project 

University Santo Tomas
Faculty of Engineering Electronic

Miguel Elkin Jimenez Avila 
Santiago Mateo Guerrero Romero
Jhonatan Alexander Navas Ballesteros


# Description of project
This project is a hand detect that move the cursor according to position of hand, also detect fist and the program do click.

We used a classifier to the detection hand and fist, this classifier is a CNN (Convolution Neuronal Network).

We used a color filter to the localitation hand and fist, this filter take the yellow color. 

Next we going to show the flow diagram:

![diagrama](https://user-images.githubusercontent.com/45437733/49232259-9d69e280-f3c1-11e8-9efb-2bff439234ce.jpg)


We use Tensorflow to create CNN, this net have 4 layers. Information for CNN:

![red](https://user-images.githubusercontent.com/45437733/49232392-d4d88f00-f3c1-11e8-94d2-81fd0156701c.JPG)

Information for trainning:  
![curva](https://user-images.githubusercontent.com/45437733/49232442-efab0380-f3c1-11e8-9a8f-4d84310c0aec.JPG)

The filter is a band-pass filter to the yellow color, because we used a yellow glove to does easy localization process.

# Images samples
Detect open hand and location in the screen, get position of the hand and set position of the cursor.

![manoabierta](https://user-images.githubusercontent.com/45437733/49232518-1832fd80-f3c2-11e8-87f9-eff49211df53.png)

Detect the fist and does click.
https://user-images.githubusercontent.com/45437733/49186418-95af2d00-f332-11e8-833a-4d42d40f982f.png

![puno](https://user-images.githubusercontent.com/45437733/49232538-25e88300-f3c2-11e8-9dc9-8696e493ce48.png)


# Steps for run program:

1. In your terminal, you go to folder "DetectHand".
2. Run this line python test2.py
3. Put the yellow glove 
4. Try to move the cursor and then try to do click 

# Dependencies:

- cv2
- tensorflow
- keras
- numpy
- scipy 
- os
- re
- glob
- os.path
- matplotlib
- imutils
- collections
- autopy
