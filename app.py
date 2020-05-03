# Check For test files
from glob import glob
jpg_files = glob('test/*.jpg')
png_files = glob('test/*.png')
img_files = jpg_files + png_files

if len(img_files) !=0:
    import numpy as np
    from numpy import loadtxt
    from keras.models import load_model
    from keras.models import model_from_json
    from keras.preprocessing import image
    import cv2
    import os
    from datetime import datetime
    # to store temp files and the result
    try:  
        os.mkdir('temp')  
    except OSError as error:  
        print(error)

    #load the trained model
    json_file = open('model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model/model.h5")
    print("Loaded model from disk")
    #loaded_model.summary()

    # classifier for faces
    face_cascade = cv2.CascadeClassifier('include/haarcascade_frontalface_default.xml')
    # Analysing all pictures in test folder
    for img_file in img_files:
        img = cv2.imread(img_file)
        # Convert into grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect faces
        #this returns the 4 corners of the face
        faces_list = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces_list)!=0:
            print("Detected "+ str(len(faces_list))+" faces in " + img_file)
            faces = { id:face for id,face in enumerate(faces_list)}
            test=[]
            # Detect and extract the faces from the image
            for face_id in faces:
                x, y, w, h = faces[face_id] 
                face_img = img[y:y+h, x:x+w]
                cv2.imwrite('temp/'+str(face_id)+'.png',face_img)
                test_img = image.load_img('temp/'+str(face_id)+'.png',target_size=(48,48,1)
                                          , color_mode='grayscale')
                test_img = image.img_to_array(test_img)
                test_img = test_img/255
                test.append(test_img)
            T = np.array(test)

            print("Predicting Emotions")
            # Predict the emotion of the face
            prediction = loaded_model.predict_classes(T)

            # draw a rectangle around the face and label it
            emotions=['Anger','Contempt','Disgust','Fear','Happiness','Sadness','Surprise']
            for face_id in faces:
                x, y, w, h = faces[face_id] 
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(img,emotions[prediction[face_id]],(x,y-8),cv2.FONT_HERSHEY_SIMPLEX,max(min(img.shape[0],img.shape[1])/1250,0.6),(255,0,0),2)
                os.remove('temp/'+str(face_id)+'.png')
            print("Writing Output for " + img_file)
            filename =  img_file.replace('test/','temp/res'+datetime.now().strftime("_%m-%d-%Y_%H-%M-%S_"))
            cv2.imwrite(filename, img)
        else:
            print("No faces detected in " + img_file)

    print("Completed\nCheck the results in the temp folder")
else:
    print("No test Images found. \nPlease create a directory called test and place your pictures in it.")
