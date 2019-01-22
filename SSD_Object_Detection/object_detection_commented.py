# Object Detection

# Importing the libraries
import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio

# Defining a function that will do the detections
def detect(frame, net, transform): # We define a detect function that will take as inputs, a frame, a ssd neural network, and a transformation to be applied on the images, and that will return the frame with the detector rectangle.
    height, width = frame.shape[:2] # We get the height and the width of the frame. Aplica el  metodo shape al frame para sacar los dos primeros valores del vector frame (height, weight, canales de color:puede ser 1 para p/n y 3 para color)
    frame_t = transform(frame)[0] # We apply the transformation to our frame.
    x = torch.from_numpy(frame_t).permute(2, 0, 1) # We convert the frame into a torch tensor.
    x = Variable(x.unsqueeze(0)) # We add a fake dimension corresponding to the batch. Hay que meterlo así en la NN. Se mete como primera dimensión : 0. Se le aplica el metodo variable para convertirlo en una variable: cada frame. LA convetimos en una variable de torch
    y = net(x) # We feed the neural network ssd with the image and we get the output y.
    detections = y.data # We create the detections tensor contained in the output y. 
    # detencttions = Contiene : [batch (pero de la salida), número de clases (clses de objetos detectados), number of occurences (de cada clase:si hay 3 perros/frame), tupla de 5 elems:(score, x0, y0, x1, y1)]. -> Esta tupla: Por cada ocurrencia en cada clase, en el bacth, tendremos el score y las coordendas vertice sup izq e inf der. Si el score > 0.6 se muestra la ocurrencia y si < 0.6 no, esto se consigue con el bucle siguiente.
    scale = torch.Tensor([width, height, width, height]) # We create a tensor object of dimensions [width, height, width, height]. Tiene que ser asi, es un trick para poder normalizar los rectángulos
    for i in range(detections.size(1)): # For every class: (el segundo argumento de detentions, número de clases)
        j = 0 # We initialize the loop variable j that will correspond to the occurrences of the class.
        while detections[0, i, j, 0] >= 0.6: # We take into account all the occurrences j of the class i that have a matching score larger than 0.6. El último 0 es el elemento de la tupla de 4 anterior. Este buble while hacer referencia a cada objeto j, dentro de cada clase i
            pt = (detections[0, i, j, 1:] * scale).numpy() # We get the coordinates of the points at the upper left and the lower right of the detector rectangle que cumple la condicion anterior >0.6. Con el scale se "normalizan las coordenadas al tamamño del frame. AL final lo tenemos que convertir a numpy array porque cv2 no trabaja con ellos, y tiene qu einsertar los cuadros que ha predicho la red neuronal.
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2) # We draw a rectangle around the detected object.
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA) # We put the label of the class right above the rectangle. El i-1 es para ponerle la etiqueta de la clase i correspondiente (xq empieza por 0)
            j += 1 # We increment j to get to the next occurrence.
    return frame # We return the original frame with the detector rectangle and the label around the detected object.

# Creating the SSD neural network
net = build_ssd('test') # We create an object that is our neural network ssd.
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage)) # We get the weights of the neural network from another one that is pretrained (ssd300_mAP_77.43_v2.pth).
# torch.load, carga los weights  del modelo preentrenado en un tensor y load_state los carga  en la NN (net) 
# Creating the transformation
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0)) # We create an object of the BaseTransform class, a class that will do the required transformations so that the image can be the input of the neural network.
# El primer agurmento es lared.size, el segudo un tupla de 3 que normaliza los colores a la escala a la que la red ha sido entrenada, son valores que vienen dados.
# Doing some Object Detection on a video
reader = imageio.get_reader('lolo.mp4') # We open the video.
fps = reader.get_meta_data()['fps'] # We get the fps frequence (frames per second).
writer = imageio.get_writer('output.mp4', fps = fps) # We create an output video with this same fps frequence.
for i, frame in enumerate(reader): # We iterate on the frames of the output video:
    frame = detect(frame, net.eval(), transform) # We call our detect function (defined above) to detect the object on the frame.
    writer.append_data(frame) # We add the next frame in the output video.
    print(i) # We print the number of the processed frame.
writer.close() # We close the process that handles the creation of the output video.