import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import os
def Imagen(path):
    Imagen_list = []
    for archivo in os.listdir(path):
        if archivo.endswith('.dcm'):
            Imagen_list.append(pydicom.dcmread(os.path.join(path, archivo)))
            
    Imagen_list.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    for Imagen_list_ in Imagen_list:
        plt.imshow(Imagen_list_.pixel_array)
        plt.show()

    for Imagen_list_ in Imagen_list:
        print(Imagen_list_.filename)

    return Imagen_list
while True:
    menu=input("""elija una opcion 
               1. conteo celular
               2. visualizacion de imagenes
               3. salir
               """)
    if menu=="1":
        ruta = r'C:\Users\dario\Segundo-Punto-1\Imagen celula.jpg'
        imagen = cv2.imread(ruta)
        img = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        azul_min = np.array([100, 50, 50])   
        azul_max = np.array([140, 255, 255])
        mask = cv2.inRange(imagen_hsv, azul_min, azul_max)
        kernel_cierre = np.ones((30,30), np.uint8)
        mask_cerrada = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_cierre)
        regiones_azules = cv2.bitwise_and(img, img, mask=mask_cerrada)
        gray = cv2.cvtColor(regiones_azules, cv2.COLOR_BGR2GRAY)
        binarizada = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        kernele = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)) 
        final = cv2.erode(binarizada, kernele, iterations=1)
        imagen_bn_invertida = cv2.bitwise_not(final)
        contornos, _ = cv2.findContours(imagen_bn_invertida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_contornos = len(contornos)
        print(f"el numero de celulas es {num_contornos}")
        plt.imshow(imagen_bn_invertida, cmap='gray')
        plt.axis('off')
        plt.show()
    elif menu=="2":
        path = r'C:\Users\dario\Segundo-Punto-1\archivosDCM'
        Imagen(path)
    else:
        break
