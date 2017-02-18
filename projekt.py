# -*- coding: utf-8 -*-

#Wir brauchen: OpenCV, Numpy (Arrays etc.) und OS (für Pfadangaben)
import cv2
import numpy as np
import os


#User Input zur Pfadangabe oder für Befehle (z.B. 'process')
command = input('Befehl oder Pfad zum Bild angeben: ')

#Prozessierung von Emotionstrainingsdaten 
#Gesichtserkennung in Trainingsdaten und Abspeicherung dieser Gesichter
#Um nur die Emotionen in den Trainingsdaten zu haben und keine Körper etc.
if command == 'process':
    #Unprozessierte Bilder liegen in:
    upath = 'bilder/unprocessed'
    #Prozessierte Bilder sollen in:
    ppath = 'bilder/train'

    #Die 4 Cascade Classifier zur Gesichtserkennung
    face_cascade1 = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
    face_cascade2 = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt.xml')
    face_cascade3 = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')
    face_cascade4 = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt_tree.xml')

    #Für jeden Ordner im unprozessierten Ordner (sollten 0,1,2,3 benannt sein)
    for folder in os.listdir(upath):
        #Speicher den Pfad zu dem Ordner (z.B. 'bilder/unprocessed/0')
        folderpath = os.path.join(upath, folder)
        #Für jede Datei in diesem Ordner
        for item in os.listdir(folderpath):
            #Initialisiere face und faceimg 
            face = []
            faceimg = []
            #Wenn die Datei ein jpg bild ist
            if '.jpg' or '.JPG' or '.jpeg' or '.JPEG' in item:
                #Lade das Bild
                unprocessedimg = cv2.imread(os.path.join(folderpath, item))
                #Wenn das Bild existiert (und nicht fehlerhaft ist)
                if unprocessedimg is not None:
                    #Übersetze das Bild in Graustufen
                    uimggray = cv2.cvtColor(unprocessedimg, cv2.COLOR_BGR2GRAY)
                    #Wenn der erste Classifier ein Gesicht gefunden hat wird face darauf gesetzt
                    #Sonst wird der nächste Classifier verwendet, usw.
                    if len(face_cascade1.detectMultiScale(uimggray, 1.1, 5)) > 0:
                        face = face_cascade1.detectMultiScale(uimggray, 1.1, 5)
                        
                    elif len(face_cascade2.detectMultiScale(uimggray, 1.1, 5)) > 0:
                        face = face_cascade2.detectMultiScale(uimggray, 1.1, 5)
                    
                    elif len(face_cascade3.detectMultiScale(uimggray, 1.1, 5)) > 0:
                        face = face_cascade3.detectMultiScale(uimggray, 1.1, 5)
                    
                    elif len(face_cascade4.detectMultiScale(uimggray, 1.1, 5)) > 0:
                        face = face_cascade4.detectMultiScale(uimggray, 1.1, 5)
                    
                    #Wenn mindestens ein Gesicht gefunden wurde, wähle diesen Ausschnitt
                    if len(face) > 0:
                        for(x, y, w, h) in face:
                            faceimg = unprocessedimg[y:y+h, x:x+w]
                        print(os.path.join(ppath, folder, item))
                        #Speichere das gesicht im train-Ordner (z.B.'bilder/train/0')
                        cv2.imwrite(os.path.join(ppath, folder, item), faceimg)
                        print('Bild gespeichert')
                    #Wenn kein Gesicht gefunden wurde, gibt es nur diese Meldung
                    else:
                        print('Kein Gesicht gefunden!')
                        
#Ist der command etwas anderes als process
else:   
    #Man kann in OpenCV3 leider kein SVM laden, 
    #deswegen muss es immer neu aufgebaut werden
    #Trainingsdaten liegen im Ordner:
    path = 'bilder/train'
    
    #eine leere Liste für Bilder für initialisiert
    images = list()
    #eine leere Liste für Ordnernamen für initialisiert
    foldername = list()
    
    #für jeden Ordner in 'bider/train' (sollten 0,1,2,3 heißen)
    for folder in os.listdir(path):
        #Speicher den Pfad zu dem Ordner (z.B. 'bilder/train/0')
        folderpath = os.path.join(path, folder)
        #Für jede Datei in diesem Ordner
        for item in os.listdir(folderpath):
            #Wenn die Datei ein jpg Bild ist
            if '.jpg' or '.JPG' or '.jpeg' or '.JPEG' in item:
                #Lade das Bild als Graustufenbild
                newimage = cv2.imread(os.path.join(folderpath, item), 0)
                #Wenn das Bild existiert (und nicht fehlerhaft ist)
                if newimage is not None:
                    #Setze die Größe des Bildes auf 100, 100, um die Bilder vergleichen zu können
                    #Alle Bilder müssen für das SVM gleich groß sein
                    #100x100 lieferte die besten Ergebnisse
                    newimageresize = cv2.resize(newimage, (100, 100))
                    #Speichere das Bild in der Liste 'images' als 1 Dimensionaler Array
                    images.append(newimageresize.ravel())
                    #Speichere den Ordnernamen in der Liste 'foldername'
                    foldername.append(int(folder))
    
    #Trainingsdaten sind die Bilder als Fließkommazahlen
    features = np.array(images, dtype=np.float32)
    
    #Labels sind die Ordnernamen als Integer (0,1,2 oder 3)
    labels = np.array(foldername, dtype=np.int)
    
    #Testdaten liegen im Ordner:
    testpath = 'bilder/test'
    #eine leere Liste für Bilder für initialisiert
    testimages = list()
    #eine leere Liste für Ordnernamen für initialisiert
    testfoldername = list()
    
    #für jeden Ordner in 'bider/test' (sollten 0,1,2,3 heißen)
    for testfolder in os.listdir(testpath):
        #Speicher den Pfad zu dem Ordner (z.B. 'bilder/test/0')
        testfolderpath = os.path.join(testpath, testfolder)
        #Für jede Datei in diesem Ordner
        for testitem in os.listdir(testfolderpath):
            #Wenn die Datei ein jpg Bild ist
            if '.jpg' or '.JPG' or '.jpeg' or '.JPEG' in testitem:
                #Lade das Bild als Graustufenbild                
                testnewimage = cv2.imread(os.path.join(testfolderpath, testitem), 0)
                #Wenn das Bild existiert (und nicht fehlerhaft ist)                
                if testnewimage is not None:
                    #Setze die Größe des Bildes auf 100, 100, um die Bilder vergleichen zu können
                    #Alle Bilder müssen für das SVM gleich groß sein
                    testnewimageresize = cv2.resize(testnewimage, (100, 100))
                    #Speichere das Bild in der Liste 'testimages' als 1 Dimensionaler Array
                    testimages.append(testnewimageresize.ravel())
                    #Speichere den Ordnernamen in der Liste 'foldername'
                    #(braucht man später für die Genauigkeit der Predictions)
                    testfoldername.append(int(testfolder))
    
    #Testdaten sind die Bilder als Fließkommazahlen    
    test = np.array(testimages, dtype=np.float32)
    
    #Erstelle die SVM, mit Typ C_SVC und linearem Kernel
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    
    #Trainiere die SVM mit den Trainingsdaten und Labels
    svm.train(features, cv2.ml.ROW_SAMPLE, labels)
    #Predict gibt mehr Dinge aus als nur die Predicteten Labels
    #Die Labels sind in Position [1] gespeichert
    #Predicte die Labels der Testdaten und wandel sie in einen 1 Dimensionalen Array um
    pre = svm.predict(test)[1].ravel()
    
    #Für jedes predictete Label in dem vorher erstellten Array
    for i in pre:
        #Wenn das Label 0 ist ist es Happiness etc. (so wie die Ordner beschriftet sind)
        if(i == 0):
            print('Happiness')
        elif(i == 1):
            print('Sadness')
        elif(i == 2):
            print('Fear') 
        elif(i == 3):
            print('Anger')        
        else:
            print('Anderes')
    
    #Überprüfung der Genauigkeit
    positives = 0
    accuracy = 0
    #num als einfache Zählervariable
    num = 0
    #Solange num kleiner ist als die Länge des Prediction-arrays
    while num < len(pre):
        #Wenn die Prediction gleich mit dem Label des Testbildes ist
        if(pre[num] == testfoldername[num]):
            #Richtige Labels werden gezählt
            positives = positives + 1
        #Zählervariable wird hochgesetzt
        num = num + 1
    
    #Genauigkeit ist die Anzahl der richtig predicteten Labels durch die Anzahl der Predictions
    accuracy = positives / len(pre)
    
    print('\nTrainingData:')
    print(len(images))
    print('\n')
    print('Accuracy:')
    print(accuracy)
    print('Positives:')
    print(positives)
    print('Gesamt:')
    print(len(pre))
    print('\n')
    
    #Die SVM wird in einer XML-Datei abgespeichert 
    #Falls man in einer späteren OpenCV-Version SVMs wieder laden kann, statt sie immer neu aufbauen zu müssen
    svm.save("data.xml")
    
    #Das (eigentliche) Bild mit dem Benutzerpfad wird eingelesen
    img = cv2.imread(command)
    
    #Bild wird in Graustufen übersetzt
    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #Variablen face und faceimg werden auf 0 gesetzt
    face = []
    faceimg = []
    
    #4 verschiedene Cascade Classifier werden reingeladen
    face_cascade1 = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
    face_cascade2 = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt.xml')
    face_cascade3 = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')
    face_cascade4 = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt_tree.xml')
    
    #Wenn der erste Classifier ein Gesicht gefunden hat wird face darauf gesetzt
    #Sonst wird der nächste Classifier verwendet, usw.
    if len(face_cascade1.detectMultiScale(imggray, 1.1, 5)) > 0:
        face = face_cascade1.detectMultiScale(imggray, 1.1, 5)

    elif len(face_cascade2.detectMultiScale(imggray, 1.1, 5)) > 0:
        face = face_cascade2.detectMultiScale(imggray, 1.1, 5)

    elif len(face_cascade3.detectMultiScale(imggray, 1.1, 5)) > 0:
        face = face_cascade3.detectMultiScale(imggray, 1.1, 5)

    elif len(face_cascade4.detectMultiScale(imggray, 1.1, 5)) > 0:
        face = face_cascade4.detectMultiScale(imggray, 1.1, 5)
    
    #Wenn mindestens ein Gesicht gefunden wurde, zeichne ein Rechteck darum
    if len(face) > 0:
        for(x,y,w,h) in face:
            cv2.rectangle(img, (x-2,y-2), (x+w+2, y+h+2), (230, 10, 100), 2)
            faceimg = img[y:y+h, x:x+w]
            faceimggray = imggray[y:y+h, x:x+w]
            
        #Zeige das Bild, Fenster lassen sich mit beliebiger Taste schließen
        cv2.imshow('IMAGE', img)
        cv2.imshow('FACEIMAGE', faceimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        #Setze die Größe des Bildes auf 100, 100, um die Bilder vergleichen zu können
        #Alle Bilder müssen für das SVM gleich groß sein
        faceimgresize = cv2.resize(faceimggray, (100, 100))
        
        #Gleiche Prozedur wie bei Trainings- und Testdaten
        #Die Daten zum trainieren, testen un predicten müssen gleich aufgebaut sein
        faceimgs = list()
        faceimgs.append(faceimgresize.ravel())
        topredict = np.array(faceimgs, dtype=np.float32)
        
        #Das Bild wird predicted und die Prediction in einen 1 Dimensionalen Array gepackt
        prediction = svm.predict(topredict)[1].ravel()
        
        print('Emotion im gewählten Bild:')
        #Für jedes predictete Label in dem vorher erstellten Array
        for predict in prediction:
            #Wenn das Label 0 ist ist es Happiness etc. (so wie die Ordner beschriftet sind)
            if(predict == 0):
                print('Happiness')
            elif(predict == 1):
                print('Sadness')
            elif(predict == 2):
                print('Fear') 
            elif(predict == 3):
                print('Anger')        
            else:
                print('Anderes')


    #Wenn kein Gesicht gefunden wurde, wird das in der Konsole geschrieben    
    else:
        print("Kein Gesicht gefunden!")