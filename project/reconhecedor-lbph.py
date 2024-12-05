import cv2

classificadorFace = "classificadorLBPHYale.yml"

detectorFace = cv2.CascadeClassifier("../content/haarcascade_frontalface_default.xml")
reconhecedor = cv2.face.LBPHFaceRecognizer.create()
reconhecedor.read(classificadorFace)
largura, altura = 220, 220
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
camera = cv2.VideoCapture(0)

while (True):
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    facesDetectadas = detectorFace.detectMultiScale(imagemCinza,
                                                    scaleFactor=1.5,
                                                    minSize=(30,30))
    for (x, y, l, a) in facesDetectadas:
        imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (121,29,237), 2)
        id, confianca = reconhecedor.predict(imagemFace)
        nome = ""
        if id == 1371392222015:
            nome = 'Giuliana'
        elif id == 1371392222033:
            nome = 'Silvana'
        else: nome = 'Não detectado'

        print(nome)

        cv2.putText(imagem, nome, (x,y +(a+30)), font, 2, (237,133,77))
        cv2.putText(imagem, str(confianca), (x,y + (a+50)), font, 1, (237,133,77))

    cv2.imshow("Face", imagem)
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
