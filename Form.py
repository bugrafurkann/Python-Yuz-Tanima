import cv2
import os
import pytesseract
from tkinter import *
from tkinter import messagebox
import numpy as np
from PIL import Image,ImageTk

kisiIsmi=""
window = Tk()
window.title("Yüz Tanıma Sistemi")
window.geometry("300x100")
def ButtonFunc():
 cam = cv2.VideoCapture(0)
 cam.set(3, 640)
 cam.set(4, 480) 

 face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

 face_id = input('\n enter user id end press <return> ==>  ')
 kisiIsmi = input('\n ismini giriniz : ')

 messagebox.showinfo( "Yüz Kaydı Başlıyor", "Kameraya Bakın Ve Bekleyin")

 count = 0

 while(True):

     ret, img = cam.read()
     # img = cv2.flip(img, -1)
     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     faces = face_detector.detectMultiScale(gray, 1.3, 5)

     for (x,y,w,h) in faces:

         cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
         count += 1
         
         cv2.imwrite("dataset/"+kisiIsmi+'.'+ str(count) + '.' + str(face_id) + ".jpg", gray[y:y+h,x:x+w])
        #  cv2.imwrite("dataset/"+str(count)+'.' + str(face_id) + ".jpg", gray[y:y+h,x:x+w])
         cv2.imshow('image', img)
        
    
     k = cv2.waitKey(100) & 0xff
     if k == 27:
         break
     elif count >= 30:
         break
 
 messagebox.showinfo( "Yüz Kaydı Başaralı", str(count)+" tane kayıt yapıldı.")
  
 cam.release()
 cv2.destroyAllWindows()
def ButtonKaydet():
 path = 'dataset'

 recognizer = cv2.face.LBPHFaceRecognizer_create()
 detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

 def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids
 messagebox.showinfo( "Yüz Kaydı Başladı", "Dataset üzerinde kayıtlı yüzler öğreniliyor.")

 faces,ids = getImagesAndLabels(path)
 recognizer.train(faces, np.array(ids))

 recognizer.write('trainer/trainer.yml') 
 messagebox.showinfo("Başarıyla Öğrenildi.",str(format(len(np.unique(ids))))+" tane yüz öğrenildi.")
def ButtonYuzTaramasiYap():
 recognizer = cv2.face.LBPHFaceRecognizer_create()
 recognizer.read('trainer/trainer.yml')
 cascadePath = "haarcascade_frontalface_default.xml"
 faceCascade = cv2.CascadeClassifier(cascadePath)

 font = cv2.FONT_HERSHEY_SIMPLEX

 id = 0

 names = ['None0', 'Furkan', 'Bugra', 'Mert', 'Hasan', 'Unknown'] 

 cam = cv2.VideoCapture(0)
 cam.set(3, 640)
 cam.set(4, 480) 

 minW = 0.1*cam.get(3)
 minH = 0.1*cam.get(4)

 while True:

    ret, img =cam.read()
    # img = cv2.flip(img, -1)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
       
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100-confidence))
        else:
            id = "Tanınmadı"
            confidence = "  {0}%".format(round(100-confidence))
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    cv2.imshow('camera',img) 

    k = cv2.waitKey(10) & 0xff # 'ESC'
    if k == 27:
        break

 print("\n [INFO] Exiting Program and cleanup stuff")
 cam.release()
 cv2.destroyAllWindows()
B = Button(window, text ="Yüzleri Öğren", command = ButtonFunc)
B.pack()
B = Button(window, text ="Yüz Kaydet", command = ButtonKaydet)
B.pack()
B = Button(window, text ="Tarama Sistemini Çalıştır", command = ButtonYuzTaramasiYap)
B.pack()
window.mainloop()







