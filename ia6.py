import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import cv2
import numpy as np
from PIL import Image
import keras
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
import pandas as pd 

model = load_model('path_to_my_model.h5')
(X_entrenamiento, Y_entrenamiento), (X_pruebas, Y_pruebas) = mnist.load_data()

image = cv2.imread('ia6.jpg')
image = cv2.resize(image, (800, 800), interpolation=cv2.INTER_AREA)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.blur(gray, (3, 3))
canny = cv2.Canny(gray, 100, 150)
cnts, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # OpenCV 4
count = 0
Diccionario = {}

for c in cnts:
    area = cv2.contourArea(c)
    epslion = 0.1 * cv2.arcLength(c, True)
    aprox = cv2.approxPolyDP(c, epslion, True)
    M = cv2.moments(c)
    este = cv2.drawContours(image, [c], 0, (0, 255, 0), 2)

    if len(aprox) == 4 and area > 50:

        x, y, w, h = cv2.boundingRect(c)
        placa = gray[y:y + h, x:x + w]

        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            cv2.circle(image, (cx, cy), 3, (0, 0, 255), -1)
            cv2.putText(image, "center", (cx - 20, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        guardar = cv2.imwrite('placa_{}.jpg'.format(count), placa)
        imagen1 = cv2.imread('placa_{}.jpg'.format(count))  # placa 8
        imagen1 = cv2.resize(imagen1, (28, 28), interpolation=cv2.INTER_AREA)
        imagen1 = cv2.cvtColor(imagen1, cv2.COLOR_BGR2GRAY)

        for i in range(imagen1.shape[0]):
            for j in range(imagen1.shape[1]):
                if (imagen1[i][j] < 100):
                    imagen1[i][j] = 0
                else:
                    imagen1[i][j] = 255

        for i in range(3):
            for j in range(imagen1.shape[1]):
                imagen1[i][j] = 0
        for i in range(3):
            for j in range(imagen1.shape[1]):
                imagen1[27 - i][j] = 0
        for i in range(imagen1.shape[1]):
            for j in range(3):
                imagen1[i][j] = 0
        for i in range(imagen1.shape[1]):
            for j in range(3):
                imagen1[i][27 - j] = 0

        #cv2.imshow('imagen', imagen1)
        cv2.waitKey(0)
        
        imagen1 = np.expand_dims(imagen1, axis=0)
        imagen1 = np.expand_dims(imagen1, axis=3)
        imagen1 = imagen1.astype('float32') / 255
        probabilidades = np.multiply(model.predict(imagen1),100)
        esun = np.argmax(probabilidades)
        Diccionario[('placa_{}'.format(count))] = esun, np.array([0.0025*(cx-400), -0.0025*(cy-400)]) #probabilidades'
        count = count + 1
print(Diccionario)

a = list(Diccionario)
print('\n',Diccionario[str(a[0])][0])
posiciones = []
ca = 0
posiciones.append([1.1894, -1.11614])
plt.plot(1.1894,-1.11614,marker = "o", color="green")
for i in range(len(a)):
    #print('entrando')
    if Diccionario[str(a[i])][0] == 2:
     #   print('entro')
        ca = ca+1
        posiciones.append(Diccionario[str(a[i])][1])
        plt.plot(float(posiciones[ca][0]),float(posiciones[ca][1]),marker = "o", color="red")
        plt.plot([posiciones[ca-1][0], posiciones[ca][0]],[posiciones[ca-1][1] , posiciones[ca][1]],color="green")
        

plt.xlim(-1.3,1.3)
plt.ylim(-1.3,1.3)
print(posiciones)
pd.DataFrame(posiciones).to_csv('Posiciones_movil.csv')
plt.show()
