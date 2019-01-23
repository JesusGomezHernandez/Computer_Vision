# Detección de objetos en tiempo real.

# Importamos las bibliotecas
import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio

# Definimos las función que hará las detecciones
def detect(frame, net, transform): # Esta función toma como argumentos, un frame, una red neuronal SSD, y una transformación que se aplica a las imágenes. La función devuelve el frame con la caja o cajas del detector.

    height, width = frame.shape[:2] # Tomamos las dimensiones del frame. Aplica el  metodo shape al frame para sacar los dos primeros valores del vector frame (height, weight, canales de color:puede ser 1 para b/n y 3 para color).

    frame_t = transform(frame)[0] # Se aplica la transformación que se había recibido como argumento al frame.

    x = torch.from_numpy(frame_t).permute(2, 0, 1) # Convertimos el frame en un tensor de torch.

    x = Variable(x.unsqueeze(0)) # Se le aplica el metodo 'Variable' para convertir cada frame en una variable de torch, y añadimos una dimensión ficticia que corresponde con el batch. Se mete como primera dimensión: 0, en la NN. 

    y = net(x) # Alimentamos la red neuronal SSD con la imagen y obtenemos la salida 'y'.

    detections = y.data # Aplicamos el método 'data' para crear el tensor de la detecciones contenidas en la salida 'y'. 
    # detections = [batch, number of classes, number of occurence, (score, x0, Y0, x1, y1)]
    # detections = [batch (de la salida), número de clases (clases de objetos detectados:perro,maceta), number of occurencias (de cada clase:si hay 3 macetas/frame), tupla de 5 elems:(score, x0, y0, x1, y1)]. -> Esta tupla: Por cada ocurrencia en cada clase, en el bacth, tendremos el score y las coordendas vertice sup izq e inf der. Si el score > 0.6 se muestra la ocurrencia y si < 0.6 no, esto se consigue con el bucle siguiente.

    scale = torch.Tensor([width, height, width, height]) # Creamos un un tensor (objeto) de dimensiones [anchura, altura, anchura, altura]. Es un 'truco' para poder normalizar los rectángulos.
  
    for i in range(detections.size(1)): # Por cada clase (el segundo argumento de 'detections')

        j = 0 # Inicializamos el bucle con la variable j, que corresponde con el número de ocurrencias dentro de cada clase.

        while detections[0, i, j, 0] >= 0.6: # Tomamos las ocurrencias j dentro de la clase i, que tienen un score > 0.6. 

            pt = (detections[0, i, j, 1:] * scale).numpy() # Sacamos las coordenadas de los puntos superior izquierdo e inferior derecho del rectángulo del detector que cumple la condicion anterior >0.6. Con 'scale' se "normalizan las coordenadas al tamaño del frame. Al final lo tenemos que convertir a un array de numpy porque cv2 no trabaja con los tensores de torch, y tiene que insertar los cuadros que ha predicho la red neuronal.

            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2) # Dibujamos el rectángulo alrededor del objeto detectado.

            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA) # Incluímos la etiqueta de la calse arriba a la derecha del rectángulo. El i-1 es para ponerle la etiqueta de la clase i correspondiente (porque empieza por 0).
            j += 1
    return frame # La función devuelve el frame original con el rectángulo del detector y la etiqueta del objeto detectado.

# Creating the SSD neural network
net = build_ssd('test') # Creamos un objeto, que es nuestra red neuronal SSD.

net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage))  # Tomamos los pesos de otra red neuronal que ha sido previamente entrenada (ssd300_mAP_77.43_v2.pth). torch.load, carga los pesos  del modelo preentrenado en un tensor y load_state los carga  en la NN (net).

# Creating the transformation
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0)) # Creamos un objeto de la clase 'BaseTransform', esta clase lleva a cabo las transformaciones necesarias para que la imagen pueda ser la entrada de la NN. El primer agurmento es lared.size, el segundo un tupla de 3 que normaliza los colores a la escala a la que la red ha sido entrenada, estos valores vienen dados por la red.

# Doing some Object Detection on a video
reader = imageio.get_reader('lolo.mp4') # Abrimos el vídeo

fps = reader.get_meta_data()['fps'] # Sacamos el número de fps (frames por segundo)

writer = imageio.get_writer('lolo_output.mp4', fps = fps) # Creamos un video como salida con esta misma secuencia de frames.

for i, frame in enumerate(reader): # Iteramos por los frames del video de salida.

    frame = detect(frame, net.eval(), transform) # Le aplicamos la función detect (definida anteriormente) para detectar los objetos en el frame.

    writer.append_data(frame) # Vamos añadiendo los frames al archivo de salida.
    print(i) # Sacamos el número de frames procesados
writer.close() # Cerramos el proceso de creación del vídeo de salida.
