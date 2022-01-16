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
    faces = carregaAlgoritmo.detectMultiScale(imgCinza)
    print(faces)

    #eixos -> x,y
    #altura -> a
    #largura -> l
    for(x, y, l, a) in faces:
        cv2.rectangle(img,(x,y),(x+l,y+a),(0,255,0),2)
    cv2.imshow('Faces',img) 
    cv2.waitKey()