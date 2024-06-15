# transfer learning MobileNet-V3 large
# This version will be saving in savedModel version. I think that it might be a preferred version 
#import libraries from keras 
from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import tensorflowjs as tfjs
import tensorflow as tf

trainPath ="/Users/marek/Programowanie/Object_Detection_Web_Browswer/FineTuningMobilenetV3/Fish_Kaggle/Fish_Dataset/dataset_for_model/train"

ValidPath ="/Users/marek/Programowanie/Object_Detection_Web_Browswer/FineTuningMobilenetV3/Fish_Kaggle/Fish_Dataset/dataset_for_model/validate"

trainGenerator = ImageDataGenerator(
    rotation_range=15 , width_shift_range = 0.1,
    height_shift_range=0.1, brightness_range=(0, 0.2)).flow_from_directory(trainPath, target_size=(224,224), batch_size=32)


ValidGenerator = ImageDataGenerator(
    rotation_range=15 , width_shift_range = 0.1,
    height_shift_range=0.1, brightness_range=(0, 0.2)).flow_from_directory(ValidPath, target_size=(224,224), batch_size=32)

#Build Model

baseModel = MobileNetV3Large( weights= "imagenet", input_shape=(224, 224, 3),include_top=False)

x = baseModel.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)


predictionLayer = Dense(9, activation='softmax')(x)

model = Model(inputs = baseModel.input, outputs = predictionLayer)

print(model.summary())

#freeze the layers of the MobileNetV3 (this is already trained we do not want to change this)

for layer in model.layers[:5]:
    layer.trainable = False

# Compile

# optimizer = Adam(learning_rate = 0.0001)
optimizer = Adam(learning_rate = 0.0001)
model.compile(loss = "categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

#train
model.fit(trainGenerator, validation_data= ValidGenerator, epochs=5)

# modelSavedPath = "/Users/marek/Programowanie/Object_Detection_Web_Browswer/FineTuningMobilenetV3/Fish_Kaggle/Fish_Dataset/dataset_for_model/FishV3.h5"
# model.save(modelSavedPath)
# modelSavedPath2 = "/Users/marek/Programowanie/Object_Detection_Web_Browswer/FineTuningMobilenetV3/Fish_Kaggle/Fish_Dataset/dataset_for_model/FishV3.keras"
modelSavedPathSavedModel = "/Users/marek/Programowanie/Object_Detection_Web_Browswer/FineTuningMobilenetV3/Fish_Kaggle/Fish_Dataset/dataset_for_model/saved_model"


# model.save(modelSavedPathSavedModel)
# save_format='tf'
model.export(modelSavedPathSavedModel)
saved_model_added_layers ="/Users/marek/Programowanie/Object_Detection_Web_Browswer/FineTuningMobilenetV3/Fish_Kaggle/Fish_Dataset/saved_model4"
# model.export()

# tfjs.converters.save_keras_model(model, saved_model_added_layers)

# def convert():
#     #load model
#     model = tf.keras.models.load_model(
#     modelSavedPathSavedModel, custom_objects=None, compile=True, safe_mode=False)
#     # modelSavedPath,custom_objects={'Conv2D': conv})

#     model.summary()


#     tfjs.converters.save_keras_model(model, saved_model_added_layers)



# convert()
