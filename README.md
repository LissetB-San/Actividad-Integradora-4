# Actividad-Integradora-4
## Integrantes del equipo
Juan Pablo Ramos Sanabria, 
Diego Alberto Alvarez Rodríguez, 
César Francisco Barraza Aguilar, 
César Buenfil Vázquez y 
Lisset Botello Santiago.

## Planteamiento del problema 
En esta actividad se explora la técnica de detección de objetos utilizando la red neuronal You Only Look Once (YOLO), la cual fue entrenada para reconocer un total de 80 objetos. La base de datos que se utilizó para entrenar la red neuronal fue la de COCO: https://cocodataset.org/#home

El objetivo es diseñar un detector de objetos personalizado, al reentrenar la red YOLO para identificar un objeto diferente a los ya establecidos en la base de datos COCO.

## Desarrollo del código
De acuerdo con lo estipulado anteriormente se decidió realizar el entrenamiento para la detección de **tacos**, siendo éste nuestro objeto personalizado.

Los pasos para realizar el entrenamieento fueron con base al tutorial de https://www.youtube.com/playlist?list=PLKHYJbyeQ1a3tMm-Wm6YLRzfW1UmwdUIN. Al ver que el entrenamiento tenía una duración de 10 horas, decidimos interrumpirlo en 4 horas solamente.

## Codigo
Como ya lo mencionamos, en esta ocasión utilizaremos utilizaremos Darknet y Yolov4 para generar predicciones de una categoría nueva a nuestra elección. Para comenzar primero clonaremos Darknet del repositorio de AlexeyAB's utilizando el siguiente código.

´´´
!git clone https://github.com/AlexeyAB/darknet
´´´

Para poder utilizar las herramientas con más facilidad y rapidez habilitaremos el uso de un GPU y OpenCV con el siguiente comando.

%cd darknet
!sed -i 's/OPENCV=0/OPENCV=1/' Makefile
!sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile
!sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile

Verificamos de igual manera a CUDA

!/usr/local/cuda/bin/nvcc --version

Por último, aplicaremos el siguiente comando para construir a darknet y poder utilizar su carpeta de ejecución y así poder entrenar y correr detector de objetos.

!make

Como ya lo mencionamos, la herramienta Yolov4 ya ha sido entrenada con una base de datos de 80 clases. El siguiente paso será el de descargar los weights de la herramienta utilizando el siguiente comando con su dirección. 

!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights

Generamos las funciones imShow(), Upload() y Download() para poder visualizar nuestros resultados y subir o descargar archivos al Cloud VM. 

def imShow(path):
  import cv2
  import matplotlib.pyplot as plt
  %matplotlib inline

  image = cv2.imread(path)
  height, width = image.shape[:2]
  resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)

  fig = plt.gcf()
  fig.set_size_inches(18, 10)
  plt.axis("off")
  plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
  plt.show()

def upload():
  from google.colab import files
  uploaded = files.upload() 
  for name, data in uploaded.items():
    with open(name, 'wb') as f:
      f.write(data)
      print ('saved file', name)
  
def download(path):
  from google.colab import files
  files.download(path)

Como la idea es entrenar a Yolov4 para detectar una categoría que no contiene, es necesario generar una base de datos con los que podamos trabajar. Google Open Images es una buena opción para recolectar los datos necesarios. Implementando la herramienta OIDv4 podremos realizar esto en cuestión de minutos. Los siguientes comandos son utilizados para descargar la herramienta y verificar los requerimientos.

git clone https://github.com/EscVM/OIDv4_ToolKit.git
pip3 install -r requirements.txt

Ahora es momento de generar la base de datos. Con el siguiente comando introduciremos la descripción del objeto que queremos aprender a identificar y la herramienta recabará los datos de la página Google Open Images.

python main.py downloader --classes 'Taco' --type_csv train --limit 1200

Repetiremos el mismo procedimiento para generar una base de datos de validación. 

python main.py downloader --classes 'Taco' --type_csv validation --limit 240

Una vez teniendo nuestra base de datos, es necesario cambiar el su formato para poder utilizarlo con Yolov4. Primero debemos abrir el archivo classes.txt y editarlo para que contenga la clase que queremos entrenar. En esta ocasión sería Taco. Después corremos el siguiente comando para cambiar el formato de la base de datos. 

python convert_annotations.py

El siguiente paso será el de borrar los documentos con los formatos antiguos. Para eso ejecutamos el siguiente comando. Es preciso detallar que la dirección puede ser diferente para cada caso. 

rm -r OID/Dataset/train/'Vehicle registration plate'/Label/
rm -r OID/Dataset/validation/'Vehicle registration plate'/Label/

Ahora debemos de cargar los recursos al Cloud VM. Para esto primero es recomendable cambiar el nombre de la base de datos de entrenamiento a “obj” y generar un zip dentro de nuestro drive y repetir lo mismo con la base de datos de validación y llamarla “test”. Utilizando los siguientes comandos encontraremos los archivos, los copiaremos y los abriremos la nube. Este metodo sera mas rapido. 

!ls /mydrive/yolov4
!cp /mydrive/yolov4/obj.zip ../
!cp /mydrive/yolov4/test.zip ../
!unzip ../obj.zip -d data/
!unzip ../test.zip -d data/

Lo siguiente será configurar los documentos necesarios para correr exitosamente Yolov4. Primero comenzaremos con el archivo de configuración. Con el siguiente comando guardaremos este archivo en drive para poder editarlo fácilmente. 

!cp cfg/yolov4-custom.cfg /mydrive/yolov4/yolov4-obj.cfg

Modificaremos el archivo de configuración a los siguientes parámetros.

subdivisions=16
width=416
height=416
max_batches = 6000
steps=4800,5400
filters=32
size=3

Continuamos con generar dos archivos de texto llamados obj.names y obj.data. Para el obj.names hacemos lo mismo que hicimos anteriormente para el documento classes.txt. Para el archivo obj.data ponemos la siguiente informacion.

classes = 1
train = data/train.txt
valid = data/test.txt
names = data/obj.names
backup = /mydrive/yolov4/backup

Con los siguientes comandos subimos los dos archivos previamente generados.

!cp /mydrive/yolov4/obj.names ./data
!cp /mydrive/yolov4/obj.data  ./data

En el siguiente paso descargaremos dos archivos de la direccion https://github.com/theAIGuysCode/YOLOv4-Cloud-Tutorial/tree/master/yolov4 y los subimos al Cloud VM desde nuestro drive con los siguientes comandos.

!cp /mydrive/yolov4/generate_train.py ./
!cp /mydrive/yolov4/generate_test.py ./

Y corremos los archivos anteriores con los siguientes comandos. 

!python generate_train.py
!python generate_test.py

Un atajo que tomaremos para agilizar el proceso de entrenamiento y obtener resultados más precisos, será el de utilizar los weights de Yolov4 previamente entrenados para una red neuronal convolucional. Con el siguiente comando podremos obtener esta documentación. 

!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137

Por fin podemos comenzar a entrenar nuestro modelo. Con el siguiente comando empezaremos este proceso. Toma en cuenta que esto puede tomar mucho tiempo. 

!./darknet detector train <path to obj.data> <path to custom config> yolov4.conv.137 -dont_show -map
 
Una vez finalizado el entrenamiento hemos completado con el procedimiento. Volveremos a configurar nuestro archivo para evaluar con los siguientes comandos.
 
%cd cfg
!sed -i 's/batch=64/batch=1/' yolov4-obj.cfg
!sed -i 's/subdivisions=16/subdivisions=1/' yolov4-obj.cfg
%cd ..

Por último correremos nuestro modelo y guardaremos una foto de este en nuestro drive con los siguientes comandos. 

!./darknet detector test data/obj.data cfg/yolov4-obj.cfg /mydrive/yolov4/backup/yolov4-obj_best.weights /mydrive/yolov4/tacotest.jpg -thresh 0.3
imShow('predictions.jpg')

# Resultado
Reconocimiento de tacos como resultado del entrenamiento de YOLO.
 <p align="center">
  <img src="https://github.com/LissetB-San/Actividad-Integradora-4/blob/master/resultado.jpg">
</p>
