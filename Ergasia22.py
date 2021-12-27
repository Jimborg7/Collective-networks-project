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

def prepare_datasets(data_dir, train_pct=0.7, val_pct=0.3, test_pct=0.2, batch_size=32, img_size=(256, 256)):
    # To run the program without GPU it is important to reduce the size of data that will be trained
    # All the data - test (here is a portion of the data)
    devel_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.8,
        color_mode='rgb',
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size)

    # Training set
    train_size = int(train_pct * (tf.data.experimental.cardinality(devel_ds).numpy()))
    # train_ds = tf.data.Dataset.range(3500)
    train_ds = devel_ds.take(train_size)

    # Validate set
    validate_size = int(val_pct * (tf.data.experimental.cardinality(devel_ds).numpy()))
    # val_ds = tf.data.Dataset.range(1000)
    val_ds = devel_ds.skip(train_size)

    # Test set
    test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=test_pct,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size)
    # List with the names of the 4 classes
    classes = devel_ds.class_names
    print(classes)
    y = np.concatenate([y for x, y in devel_ds])
    plt.hist(y, list(range(len(classes) + 1)))
    plt.show()

    return devel_ds, train_ds, val_ds, test_ds, classes    
    
    
    
data_dir = unzipfile(output)
devel_ds, train_ds, val_ds, test_ds, classes = prepare_datasets(data_dir)


