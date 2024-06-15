import os
# from tensorflow import keras
from tensorflow.keras.preprocessing import image
# from keras import image
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

# Upgraded version for new keras an tf, new version tensorflow is 2.16.1

#get the list of categories :
categories = os.listdir("/Users/marek/Programowanie/Object_Detection_Web_Browswer/FineTuningMobilenetV3/Fish_Kaggle/Fish_Dataset/dataset_for_model/train")
categories.sort()
print(categories)


#load the saved model :
#old path
# modelSavedPath = "/Users/marek/Programowanie/Object_Detection_Web_Browswer/FineTuningMobilenetV3/Fish_Kaggle/Fish_Dataset/dataset_for_model/FishV3.h5"
#.keras model
# modelSavedPath = "/Users/marek/Programowanie/Object_Detection_Web_Browswer/FineTuningMobilenetV3/Fish_Kaggle/Fish_Dataset/dataset_for_model/FishV4.keras"
#new path
#modelSavedPath = "/Users/marek/Programowanie/Object_Detection_Web_Browswer/FineTuningMobilenetV3/Fish_Kaggle/FishModels/H5/FishV3.h5"
modelSavedPath = "/Users/marek/Programowanie/Object_Detection_Web_Browswer/FineTuningMobilenetV3/Fish_Kaggle/Fish_Dataset/dataset_for_model/saved_model.keras"


# Old version
# model = tf.keras.models.load_model(modelSavedPath)
# Loading model for tensorflow 2.16.1
model = tf.keras.models.load_model(
    modelSavedPath, custom_objects=None, compile=True, safe_mode=True
)

#predict the image

def classify_image(imageFile):
    x=[]

    img = Image.open(imageFile)
    img.load()
    #deprecated version
    #img = img.resize((320,320), Image.ANTIALIAS)

    img = img.resize((224,224), Image.Resampling.LANCZOS)
    x= image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    print(x.shape)
    pred = model.predict(x)
    print(pred)

    #get the highest prediction value
    categoryValue =np.argmax(pred, axis=1)
    categoryValue = categoryValue[0]
    #index 5
    print(categoryValue+1)

    result = categories[categoryValue+1]
    return result


# img_path ="/Users/marek/Programowanie/Object_Detection_Web_Browswer/FineTuningMobilenetV3/seabas-test.png"
# img_path ="/Users/marek/Programowanie/Object_Detection_Web_Browswer/FineTuningMobilenetV3/seabass-test3.png"
img_path ="/Users/marek/Programowanie/Object_Detection_Web_Browswer/FineTuningMobilenetV3/red-sea-bream-test.png"

resultText = classify_image(img_path)
print(resultText)

img = cv2.imread(img_path)
img = cv2.putText(img, resultText, (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()