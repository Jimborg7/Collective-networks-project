#Start by taking the data for Covid-19_Radiography_Dataset which includes over 21.000 files and 4 categories
#Code for google colab

from google.colab import drive 
drive._mount('/content/drive')

!cp '/content/drive/MyDrive/Covid-19/COVID-19_Radiography_Dataset.zip' .
!unzip -q -n COVID-19_Radiography_Dataset.zip
data_dir = '/content/COVID-19_Radiography_Dataset'
print('Done')

#Necessary Imports 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

#Prepare datasets: get images from main directory. Then split the dataset to devel_ds and test_ds. Devel_ds is the combination of tran_ds and val_ds. Because the images come at categories with labels it is needed to include the label_mode parameter.
def prepare_datasets(data_dir, train_pct=0.6, val_pct=0.2, test_pct=0.2, batch_size=64, img_size=(299, 299)):

  #AUTOTUNE = tf.data.AUTOTUNE

  devel_ds = tf.keras.utils.image_dataset_from_directory(
      data_dir,
      validation_split=train_pct + val_pct,
      # Gia olo to sunolo ginetai iso me to test_pct ara validation_split = test_pct
      color_mode='rgb',
      subset="training",
      seed=123,
      image_size=img_size,
      batch_size=batch_size)

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

  return devel_ds, train_ds, val_ds, test_ds, classes

#Convolutional network for question 3

def cnn1(num_classes,data_augmented):
  # Normalizing the images from the [0,255] values of RGB to[0,1]
  model = keras.Sequential([
      data_augmented,
      tf.keras.layers.Rescaling(1./255, input_shape=(299, 299, 3)),
      layers.Conv2D(8, kernel_size=(3, 3), padding='same', activation='relu'),
      layers.MaxPooling2D(strides=2),
      layers.Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu'),
      layers.MaxPooling2D(strides=2),
      layers.Dropout(0.2),
      layers.Flatten(),
      layers.Dense(32, activation='relu'),
      layers.Dense(num_classes, activation='softmax')
  ])
  return model


#Confusion matrix for question 3
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

#Convolutional network for question 4
def cnn2(num_classes,data_augmented):
  model = keras.Sequential([
      data_augmented,
      tf.keras.layers.Rescaling(1. /255, input_shape=(299,299,3)),
      layers.Conv2D(32, kernel_size=(3,3),padding='same',activation='relu'),
      layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'),
      layers.MaxPooling2D(strides=4),
      layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
      layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
      layers.MaxPooling2D(strides=2),
      layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
      layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
      layers.MaxPooling2D(strides=2),
      layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'),
      layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'),
      layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'),
      layers.MaxPooling2D(strides=2),
      layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'),
      layers.MaxPooling2D(strides=2),
      layers.Dropout(0.2),
      layers.Flatten(),
      layers.Dense(1024, activation='relu'),
      layers.Dense(num_classes, activation='softmax')
  ])
  return model

#Pre designed  & trained network to create model for training. Question 5
def pretrained(num_classes ):

  input = layers.Input(shape = (224,224,3))
  x = tf.keras.applications.efficientnet.EfficientNetB0(
          include_top = True,
          weights = None,
          classes = num_classes,
         )(input)
  model = tf.keras.Model(input,x)
  return model

#For OverFitting Later
def data_augm(img_size):
  data_augmentation = keras.Sequential([
      tf.keras.layers.RandomFlip("horizontal", input_shape=(img_size, img_size, 3)),
      tf.keras.layers.RandomRotation(0.1),
      tf.keras.layers.RandomZoom(0.1)
  ])
  return data_augmentation

  #Start the modulation
  
devel_ds, train_ds, val_ds, test_ds, classes = prepare_datasets(data_dir)
num_classes = len(classes)
data_augmentation = data_augm(299)
  
#First Model   
model = cnn1(num_classes,data_augmentation)
model.summary()
batch_size = 64
epochs = 20  

#Early Stopping   
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3) 
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.99),
    loss="sparse_categorical_crossentropy",
    metrics=['accuracy'])  

#Model.fit - training
history = model.fit(
    x = train_ds,
    validation_data=val_ds,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=callback,
    verbose='auto')

#Plot the results to check for overfitting
accuracy = history.history['accuracy']
loss = history.history['loss']
val_accuracy = history.history['val_accuracy']
val_loss = history.history['val_loss']

plt.plot(accuracy)
plt.plot(val_accuracy)
plt.title("Accuracy Results")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(['Train','Validation'])
plt.show()

plt.plot(loss)
plt.plot(val_loss)
plt.title("Loss Results")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(['Train','Validation'])
plt.show()

#Print Confusion matrix to check the errors
cm , yhat = confusion_matrix(model, test_ds)
print(cm)

#Second Model 
model2 = cnn2(num_classes, data_augmentation)
model2.summary()
model2.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.99),
    loss="sparse_categorical_crossentropy",
    metrics=['accuracy'])

history2 =model2.fit(
    x = train_ds,
    validation_data=val_ds,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=callback,
    verbose='auto')

#Plot the results 
accuracy = history2.history['accuracy']
loss = history2.history['loss']

val_accuracy = history2.history['val_accuracy']
val_loss = history2.history['val_loss']


plt.plot(accuracy)
plt.plot(val_accuracy)
plt.title("Accuracy Results")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(['Train','Validation'])
plt.show()

plt.plot(loss)
plt.plot(val_loss)
plt.title("Loss Results")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(['Train','Validation'])
plt.show()

#Confusion Table for model 2
cm2, y_hat2 = confusion_matrix(model2, test_ds)
print(cm2)

#Model 3 with pre trained network
model3 = pretrained(num_classes)
model3.summary()

epochs = 5
batch_size = 32
size = (224, 224)
train_ds = train_ds.map(lambda image, label: (tf.image.resize(image, size), label))
val_ds = val_ds.map(lambda image, label: (tf.image.resize(image, size), label))
test_ds = test_ds.map(lambda image, label: (tf.image.resize(image, size), label))

model3.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.99),
    loss="sparse_categorical_crossentropy",
    metrics=['accuracy'])
history3 = model3.fit(
    x=train_ds,
    validation_data=val_ds,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=callback,
    verbose='auto')

#Plot results fro third model
accuracy = history3.history['accuracy']
loss = history3.history['loss']

val_accuracy = history3.history['val_accuracy']
val_loss = history3.history['val_loss']


plt.plot(accuracy)
plt.plot(val_accuracy)
plt.title("Accuracy Results")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(['Train','Validation'])
plt.show()

plt.plot(loss)
plt.plot(val_loss)
plt.title("Loss Results")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(['Train','Validation'])
plt.show()

#Confusion Matrix 3
cm3, y_hat3 = confusion_matrix(model3, test_ds)
print(cm3, y_hat3)

