from tkinter import Scale
import cv2

#Carregamento do arquivo de treinamento de faces e imagens
carregaAlgoritmo = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
#Escolhendo a imagem que será analisada
for i in range(2,7):
    num = i
    num = str(num)
    imagem = 'fotos/'+num+'.jpg'
    img = cv2.imread(imagem)
    #Deixando a imagem escolhida cinza
    imgCinza = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #Dimensão da imagem
    #scaleFactor(fator de escala):Parâmetro que especifica o quanto o tamanho da imagem é reduzido em cada escala de imagem.
    #minNeighbors(minVizinhos):Parâmetro que especifica quantos vizinhos cada retângulo candidato deve ter para retê-lo.
    #minSize:Tamanho mínimo possível do objeto. Objetos menores que isso são ignorados.
    faces = carregaAlgoritmo.detectMultiScale(imgCinza,scaleFactor=1.5,minNeighbors=7,minSize=(50,50))
    print(faces,len(faces))

    #eixos -> x,y
    #altura -> a
    #largura -> l
    for(x, y, l, a) in faces:
        cv2.rectangle(img,(x,y),(x+l,y+a),(0,0,255),2)
    cv2.imshow('Faces',img) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()