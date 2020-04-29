# this is the finall app that will load the trained model and take input from
# webcam. work on it after the model is done
# load and evaluate a saved model
import numpy as np
from numpy import loadtxt
from keras.models import load_model
from keras.models import model_from_json
from keras.preprocessing import image

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
loaded_model.summary()

test_img=[]
img = image.load_img('test/s.png', 
                         target_size=(48,48,1),
                         color_mode="grayscale")
  # convert image to an numpy array
img =  image.img_to_array(img)
# to keep the values from 0 to 1
img = img/255
test_img.append(img)
test = np.array(test_img)
prediction = loaded_model.predict_classes(test)
emotions=['anger','contempt','disgust','fear','happy','sadness','surprise']
print(emotions[prediction[0]])