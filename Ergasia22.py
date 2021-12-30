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
    # To run the program without GPU it is important to reduce the size of data that will be trained
    # All the data - test (here is a portion of the data)

    # AUTOTUNE = tf.data.AUTOTUNE

    devel_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=train_pct + val_pct,
        # Gia olo to sunolo ginetai iso me to test_pct ara validation_split = test_pct
        color_mode='rgb',
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode = 'categorical'    )

    # Training set
    train_size = int(round((train_pct + 0.15) * (tf.data.experimental.cardinality(devel_ds).numpy())))
    train_ds = devel_ds.take(train_size)

    # Validate set
    validate_size = int(round((val_pct + 0.05) * (tf.data.experimental.cardinality(devel_ds).numpy())))
    val_tmp = devel_ds.skip(train_size)
    val_ds = val_tmp.take(validate_size)

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

    # See the results in a graph to check the number of elements
    '''y = np.concatenate([y for x, y in devel_ds])
    plt.hist(y, list(range(len(classes) + 1)))
    plt.show()'''

    # Take care of performance and blocking
    # devel_ds = devel_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    # test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return devel_ds, train_ds, val_ds, test_ds, classes
 

def cnn1(num_classes):
    # Normalizing the images from the [0,255] values of RGB to[0,1]
    model = keras.Sequential([
        layers.Rescaling(1. / 255, input_shape=(299, 299, 3)),
        layers.Conv2D(8, kernel_size=(3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(strides=2),
        layers.Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(strides=2),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def cnn2(num_classes):
    model = keras.Sequential([
        tf.keras.layers.Rescaling(1. / 255, input_shape=(299,299,3)),
        layers.Conv2D(32, kernel_size=(3,3), padding='same',activation='relu'),
        layers.Conv2D(32, kernel_size=(3,3), padding='same', activation='relu'),
        layers.MaxPooling2D(strides=4),
        layers.Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'),
        layers.Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'),
        layers.MaxPooling2D(strides=2),
        layers.Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'),
        layers.Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'),
        layers.MaxPooling2D(strides=2),
        layers.Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'),
        layers.Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'),
        layers.Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'),
        layers.MaxPooling2D(strides=2),
        layers.Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'),
        layers.MaxPooling2D(strides=2),
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def confusion_matrix(model, test_ds):
    y_test = []
    y_pred = []
    for x_1,y_1 in test_ds:
        y_pred_1 = model.predict(x_1)
        y_test.append(y_1)
        y_pred.append(y_pred_1)
    y_true = np.concatenate(y_test)
    y_p = np.concatenate(y_pred)
    y_hat = tf.argmax(y_p,axis = 1)
    cm = tf.math.confusion_matrix(y_true,y_hat)
    return cm, y_hat

# Download data
# get_data()
# data_dir = unzipfile(output)

data_dir = "C:\\Users\\jimge\\Desktop\\Ergasia_Diou_2h\\ergasia_diou\\COVID-19_Radiography_Dataset"
devel_ds, train_ds, val_ds, test_ds, classes = prepare_datasets(data_dir)
num_classes = len(classes)
model = cnn1(num_classes)

model.summary()
print(train_ds)

# Training part
batch_size = 64
epochs = 20

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.99),
    loss="categorical_crossentropy",
    metrics=['accuracy'])

model.fit(
    x = train_ds,
    validation_data=val_ds,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=callback,
    verbose='auto')

cm , yhat = confusion_matrix(model, test_ds)
print(cm)


model2 = cnn2(num_classes)
model2.summary()
model2.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.99),
    loss="categorical_crossentropy",
    metrics=['accuracy'])

model2.fit(
    x = train_ds,
    validation_data=val_ds,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=callback,
    verbose='auto')

cm2,y_hat2 = confusion_matrix(model2, test_ds)
print(cm2)



