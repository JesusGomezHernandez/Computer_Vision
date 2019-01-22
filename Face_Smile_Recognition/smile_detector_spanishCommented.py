# Homework Solution

# Importamos la biblioteca
import cv2

# Cargamos los cascades del clasificador. Creamos un objeto para cada clasificación
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Definimos la función que hara las detecciones
def detect(gray, frame): # Creamos una función que toma como entrada una imagen en b/n de la imagen original (frame), y nos devuelve la misma imagen con los rectángulos del detector.
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # Toma 3 argumentos: imagen b/n, escala de reducción de la imagen y número mínimo de vecinos. Aplicamos el método detectMultiScale a face_cascade para localizar una o varias caras en la imagen. 
    for (x, y, w, h) in faces: # Para cada cara detectada, sacamos las coordenadas de los rectángulos de las caras:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # Pintamos el rectángulo alrededor de la cara
        roi_gray = gray[y:y+h, x:x+w] # Obtenemos la región de interés en la imagen b/n
        roi_color = frame[y:y+h, x:x+w] # Obtenemos la región de interés en la imagen original
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 22) # Aplicamos el método detectMultiScale a eye_cascade para localizar los ojos en la imagen. 
        for (ex, ey, ew, eh) in eyes: #Para cada ojo detectado
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2) # Pintamos un rectágulo alredeor de los ojos, pero dentro del referente de la cara. 
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22) # Repetimos con la sonrisa.
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2) 
    return frame # Devuelve la imagen con los rectángulos del detector.

# Reconocimiento a traveś de la webcam

video_capture = cv2.VideoCapture(0) # Encendemos la cámara
while True: # Creamos un bucle infinito hasta el break.
    _, frame = video_capture.read() # Nos quedamos con el último frame (_, frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Lo transformamos en b/n
    canvas = detect(gray, frame) # Aplicamos la función para tener la salida deseada.
    cv2.imshow('Video', canvas) # Mostramos las salidas
    if cv2.waitKey(1) & 0xFF == ord('q'): # Si presionamos la tecla 'q'
        break # Detenemos el bucle.
video_capture.release() # Apagamos la webcam.
cv2.destroyAllWindows() # Destruimos las ventanas de las imágenes.
