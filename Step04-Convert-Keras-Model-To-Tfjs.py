import tensorflowjs as tfjs
import tensorflow as tf
import keras

# I used command prompt, this I think not working properly?????
# //this version works in my case but not know if the model is working in the web

#      tensorflowjs_converter \
#     --input_format=tf_saved_model \
#     --output_node_names='/Users/marek/Programowanie/Object_Detection_Web_Browswer/FineTuningMobilenetV3/Fish_Kaggle/Fish_Dataset/dataset_for_model/saved_model/mobilenet'  \
#     /Users/marek/Programowanie/Object_Detection_Web_Browswer/FineTuningMobilenetV3/Fish_Kaggle/Fish_Dataset/dataset_for_model/saved_model/ \
#     /Users/marek/Programowanie/Object_Detection_Web_Browswer/FineTuningMobilenetV3/Fish_Kaggle/Fish_Dataset/dataset_for_model/saved_model/web_model

#old path
#modelSavedPath = "/Users/marek/Programowanie/Object_Detection_Web_Browswer/FineTuningMobilenetV3/Fish_Kaggle/Fish_Dataset/dataset_for_model/FishV3.h5"
#Use a .keras format
#.keras model
modelSavedPath = "/Users/marek/Programowanie/Object_Detection_Web_Browswer/FineTuningMobilenetV3/Fish_Kaggle/Fish_Dataset/dataset_for_model/saved_model"

tfjs_target_dir="/Users/marek/Programowanie/Object_Detection_Web_Browswer/FineTuningMobilenetV3/Fish_Kaggle/Fish_Dataset/dataset_for_model/tfjs-model"
tfjs_target_dir_v2="/Users/marek/Programowanie/Object_Detection_Web_Browswer/FineTuningMobilenetV3/Fish_Kaggle/Fish_Dataset/dataset_for_model/tfjs-model-v4"
saved_model_added_layers ="/Users/marek/Programowanie/Object_Detection_Web_Browswer/FineTuningMobilenetV3/Fish_Kaggle/Fish_Dataset/dataset_for_model/saved_model_from_h6"
def convert():
    #load model
    # model = keras.models.load_model(
    # modelSavedPath, custom_objects=None, compile=True, safe_mode=True)
    # modelSavedPath,custom_objects={'Conv2D': conv})
    #it is fos .pb models
    model = keras.layers.TFSMLayer(modelSavedPath, call_endpoint="serving_default")
    # model.summary()


    # tfjs.converters.convert_tf_saved_model(model, saved_model_added_layers)



convert()


