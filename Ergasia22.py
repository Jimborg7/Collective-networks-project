import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import model_selection
from zipfile import ZipFile
import os
import gdown


def find(file_name, path):
    for root, dirs, files in os.walk(path):
        if file_name in files:
            return os.path.join(root, file_name)

output = 'COVID-19_Radiography_Dataset.zip'


def get_data():
    ###Download Data from Google Drive###
    try:
        print("Currently Downloading...")
        url = 'https://drive.google.com/uc?id=1SJKJrd1p15jZpWSij1JEYHykCBROUO78'
        gdown.download(url, output, quiet=False)

        print("Download Completed")
    except UserWarning:
        print("No other key option...")
    except:
        print("Something went wrong...\n Error during downloading.")

def unzipfile(output):
    ### Unzip Data ###
    path = "C:\\Users\\"
    directory = find(output, path)
    print("Directory for the downloaded file is -> ", directory)
    with ZipFile(output, 'r') as zip:
        zip.extractall()
        print('File has been unzipped!;)!')
    data_dir = os.path.splitext(directory)[0]
    return data_dir


def prepare_datasets(data_dir, train_pct=0.6, val_pct=0.2, test_pct=0.2, batch_size=64, img_size=(299, 299)):
    tf.keras.utils.image_dataset_from_directory
    # (x_train,x_test,y_train,y_test) =
    devel_ds = tf.data.Dataset
    train_ds = tf.data.Dataset
    val_ds = tf.data.Dataset
    test_ds = tf.data.Dataset
    classes = []

data_dir = unzipfile(output)
