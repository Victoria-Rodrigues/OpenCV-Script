import cv2

#Carregamento do arquivo de treinamento de faces e imagens
carregaAlgoritmoFace = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
carregaAlgoritmoOlho = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
#Escolhendo a imagem que será analisada
for i in range(2,11):
    num = i
    num = str(num)
    imagem = 'fotos/'+num+'.jpg'
    img = cv2.imread(imagem)
    #Deixando a imagem escolhida cinza
    imgCinza = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = carregaAlgoritmoFace.detectMultiScale(imgCinza,scaleFactor=1.015,minNeighbors=7)

    for(x, y, l, a) in faces:
        leituraImagem = cv2.rectangle(img,(x,y),(x+l,y+a),(255,0,255),2)
        localOlho = leituraImagem[y:y + a, x:x + l]#retangula da face
        localOlhoCinza = cv2.cvtColor(localOlho, cv2.COLOR_BGR2GRAY)
        detectado = carregaAlgoritmoOlho.detectMultiScale(localOlhoCinza,scaleFactor=1.1,minNeighbors=7,minSize=(30,30),maxSize=(100,100))
        for(x0, y0, l0, a0) in detectado:
            cv2.rectangle(localOlho,(x0,y0),(x0+l0,y0+a0),(0,0,255),2 )
    cv2.imshow('Detecta Face e olhos',img) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()