import cv2
import numpy as np
from PIL import Image
import os

amostra = 1
numeroAmostra = 5
tamanho_imagem = 220,220

id = input('Digite seu identificador: ')
l, a = 220, 220
network = cv2.dnn.readNetFromCaffe('../content\\deploy.prototxt.txt', '../content\\res10_300x300_ssd_iter_140000.caffemodel')

classificadorFace = cv2.CascadeClassifier('../content/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while amostra <= numeroAmostra:
    conectado, frame = cap.read()

    frameCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    facesDetectadas = classificadorFace.detectMultiScale(frameCinza)
    for (x, y, l, a) in facesDetectadas:
        cv2.rectangle(frame, (x, y), (x + l, y + a), (0,0,255), 2)

    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)

    if key == 27:
        imagemFace = cv2.resize(frameCinza[y:y + a, x:x + l], tamanho_imagem)
        cv2.imwrite("../images/teste/ra" + id + "." + str(amostra) + ".jpg", imagemFace)
        print("[foto capturada com sucesso]")
        print(imagemFace.shape)

        amostra += 1


cap.release()
cv2.destroyAllWindows()

face = cv2.resize(imagemFace, (60, 80))
imagem_teste = f"../images/teste/ra{id}.{amostra - 1}.jpg"
imagem = Image.open(imagem_teste).convert('L')
imagem_np = np.array(imagem, 'uint8')
print(imagem_np.shape)

imagem = cv2.cvtColor(imagem_np, cv2.COLOR_GRAY2BGR)
(h, w) = imagem.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(imagem, (100, 100)), 1.0, (100, 100), (104.0, 117.0, 123.0))
network.setInput(blob)
deteccoes = network.forward()

conf_min = 0.7
imagem_cp = imagem.copy()
roi = None
for i in range(0, deteccoes.shape[2]):
    confianca = deteccoes[0, 0, i, 2]
    if confianca > conf_min:
        bbox = deteccoes[0, 0, i, 3:7] * np.array([w, h, w, h])
        (start_x, start_y, end_x, end_y) = bbox.astype('int')
        roi = imagem_cp[start_y:end_y, start_x:end_x]
        text = "{:.2f}%".format(confianca * 100)
        cv2.putText(imagem, text, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
        cv2.rectangle(imagem, (start_x, start_y), (end_x, end_y), (0,255,0), 2)
face = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

def detecta_face(network, path_imagem, conf_min=0.7):
    imagem = Image.open(path_imagem).convert('L')
    imagem = np.array(imagem, 'uint8')
    imagem = cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR)
    (h, w) = imagem.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(imagem, (100, 100)), 1.0, (100, 100), (104.0, 117.0, 123.0))
    network.setInput(blob)
    deteccoes = network.forward()

    face = None
    for i in range(0, deteccoes.shape[2]):
        confianca = deteccoes[0, 0, i, 2]
        if confianca > conf_min:
            bbox = deteccoes[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = bbox.astype('int')
            roi = imagem[start_y:end_y, start_x:end_x]
            roi = cv2.resize(roi, (60, 80))
            cv2.rectangle(imagem, (start_x, start_y), (end_x, end_y), (0,255,0), 2)
            face = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return face, imagem

