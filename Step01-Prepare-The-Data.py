#From Tutorial Data
#Python 3.9.16
#pip install tensorflow==2.10
#pip install numpy opencv-python

import os
import random
import shutil

splitsize = .85
categories = []

source_folder = "/Users/marek/Programowanie/Object_Detection_Web_Browswer/FineTuningMobilenetV3/Fish_Kaggle/Fish_Dataset/Fish_Dataset"
folders = os.listdir(source_folder)
print(folders)


for subfolder in folders:
    if os.path.isdir(source_folder +"/" + subfolder):
        categories.append(subfolder)

categories.sort()
print(categories)

# create a target folder to keep training data and validation
target_folder = "/Users/marek/Programowanie/Object_Detection_Web_Browswer/FineTuningMobilenetV3/Fish_Kaggle/Fish_Dataset/dataset_for_model"
existDataSetPath = os.path.exists(target_folder)
if existDataSetPath == False:
    os.mkdir(target_folder)

# create a function for split the data for train and validation

def split_data(SOURCE, TRAINING, VALIDATION, SPLIT_SIZE):
    files=[]

    for filename in os.listdir(SOURCE):
        file = SOURCE + filename
        print(file)
        if os.path.getsize(file) > 0 :
            files.append(filename)
        else:
            print(filename + " is 0 length, ignore it ...")
    print(len(files))

    trainingLength = int(len(files) * SPLIT_SIZE)
    shuffleSet = random.sample(files, len(files))
    trainingSet = shuffleSet[0:trainingLength]
    validSet = shuffleSet[0:trainingLength]

    #copy the train images
    for filename in trainingSet:
        thisFile = SOURCE + filename
        destination = TRAINING + filename
        shutil.copyfile(thisFile, destination)

    #copy the validation images
    for filename in validSet:
        thisFile = SOURCE + filename
        destination = VALIDATION + filename
        shutil.copyfile(thisFile, destination) 

trainPath = target_folder + "/train"
print("TRAIN PATH")
print(trainPath)
validatePath = target_folder + "/validate" 

#create the target folder
existDataSetPath = os.path.exists(trainPath)
if not(existDataSetPath):
    print("inside")
    os.mkdir(trainPath)

existDataSetPath = os.path.exists(validatePath)
if existDataSetPath ==False:
    os.mkdir(validatePath)

# run function for each of the folders

for category in categories:
    trainDestPath = trainPath + "/" + category 
    validateDestPath = validatePath + "/" + category

    if os.path.exists(trainDestPath)==False:
        os.mkdir(trainDestPath)
    if os.path.exists(validateDestPath)==False:
        os.mkdir(validateDestPath)

    sourcePath = source_folder +"/" + category + "/" +category+"/"
    trainDestPath =trainDestPath + "/"
    validateDestPath = validateDestPath + "/"

    print("Copy from: "+ sourcePath + "to : " + trainDestPath + " and " +validateDestPath)

    #For test only
    #split_data(source_folder+"/"+"Black Sea Sprat/Black Sea Sprat/","","","")
    split_data(sourcePath, trainDestPath, validateDestPath, splitsize)
    