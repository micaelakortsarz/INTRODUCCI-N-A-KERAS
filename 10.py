from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

def Plot_Resultados(epocas,loss,ac,ac_test,label):
    with plt.style.context('seaborn-darkgrid'):
        plt.grid(True)

        ax1 = plt.subplot(311)
        plt.plot(epocas,loss, label=label)
        plt.ylabel(r'Loss')
        plt.xlabel(r'Epocas transcurridas')
        plt.legend()
        ax2 = plt.subplot(312)
        plt.plot(epocas,ac, label=label)
        plt.ylabel(r'Accuracy')
        plt.xlabel(r'Epocas transcurridas')
        plt.legend()
        plt.tight_layout()
        ax3 = plt.subplot(313)
        plt.plot(epocas, ac_test, label=label)
        plt.ylabel(r'Accuracy test')
        plt.xlabel(r'Epocas transcurridas')

        plt.legend()
        plt.tight_layout()

plt.figure(figsize=(20, 8), constrained_layout=True)
epocas=np.linspace(1,50,50)
epoc=50
#Obtengo y preproceso los datos
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
mean = np.mean(x_train,axis=0)
x_train = (x_train - mean) / 255.0
x_test = (x_test - mean) / 255.0
y_train=keras.utils.to_categorical(y_train, num_classes=10)
y_test=keras.utils.to_categorical(y_test, num_classes=10)
input_img = keras.Input(shape=(32, 32, 3))
x = layers.Conv2D(64, (5, 5), activation='relu', padding='same',strides=(4, 4))(input_img)
x = layers.MaxPooling2D((3, 3), padding='same',strides=(2, 2))(x)
do=keras.layers.Dropout(0.2, noise_shape=None, seed=None)(x)
x = layers.Conv2D(96, (5, 5), activation='relu', padding='same',strides=(1, 1))(do)
x = layers.MaxPooling2D((3, 3), padding='same',strides=(2, 2))(x)
do=keras.layers.Dropout(0.1, noise_shape=None, seed=None)(x)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same',strides=(1, 1))(do)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same',strides=(1, 1))(x)
x = layers.Conv2D(96, (3, 3), activation='relu', padding='same',strides=(1, 1))(x)
x = layers.MaxPooling2D((3, 3), padding='same',strides=(2, 2))(x)
do=keras.layers.Dropout(0.2, noise_shape=None, seed=None)(x)
flat=layers.Flatten()(do)
x = keras.layers.Dense(4096, activation="relu")(flat)
do=keras.layers.Dropout(0.2, noise_shape=None, seed=None)(x)
x = keras.layers.Dense(4096, activation="relu")(do)
do=keras.layers.Dropout(0.2, noise_shape=None, seed=None)(x)
output= keras.layers.Dense(10, activation="softmax")(do)
model = keras.Model(input_img, output)

opt=keras.optimizers.Adam(learning_rate=5e-4)
model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
h=model.fit(x_train, y_train,epochs=epoc,batch_size=128,shuffle=True,validation_data=(x_test, y_test))

Plot_Resultados(epocas,h.history['loss'],h.history['accuracy'],h.history['val_accuracy'],'AlexNet')

input_img = keras.Input(shape=(32, 32, 3))
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same',strides=(1, 1))(input_img)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same',strides=(1, 1))(x)

x = layers.MaxPooling2D((3, 3), padding='same',strides=(2, 2))(x)
do=keras.layers.Dropout(0.2, noise_shape=None, seed=None)(x)

x = layers.Conv2D(64, (3, 3), activation='relu', padding='same',strides=(1, 1))(do)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same',strides=(1, 1))(x)

x = layers.MaxPooling2D((3, 3), padding='same',strides=(2, 2))(x)
do=keras.layers.Dropout(0.2, noise_shape=None, seed=None)(x)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same',strides=(1, 1))(do)

x = layers.Conv2D(128, (3, 3), activation='relu', padding='same',strides=(1, 1))(x)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same',strides=(1, 1))(x)

x = layers.MaxPooling2D((3, 3), padding='same',strides=(2, 2))(x)
do=keras.layers.Dropout(0.2, noise_shape=None, seed=None)(x)

x = layers.Conv2D(256, (3, 3), activation='relu', padding='same',strides=(1, 1))(do)
x = layers.Conv2D(256, (3, 3), activation='relu', padding='same',strides=(1, 1))(x)
x = layers.Conv2D(256, (3, 3), activation='relu', padding='same',strides=(1, 1))(x)

x = layers.MaxPooling2D((3, 3), padding='same',strides=(2, 2))(x)
do=keras.layers.Dropout(0.2, noise_shape=None, seed=None)(x)
x = layers.Conv2D(256, (3, 3), activation='relu', padding='same',strides=(1, 1))(do)
x = layers.Conv2D(256, (3, 3), activation='relu', padding='same',strides=(1, 1))(x)
x = layers.MaxPooling2D((3, 3), padding='same',strides=(2, 2))(x)
do=keras.layers.Dropout(0.2, noise_shape=None, seed=None)(x)

flat=layers.Flatten()(do)
x = keras.layers.Dense(8192, activation="relu")(flat)
do=keras.layers.Dropout(0.2, noise_shape=None, seed=None)(x)

x = keras.layers.Dense(4096, activation="relu")(do)
do=keras.layers.Dropout(0.2, noise_shape=None, seed=None)(x)

output= keras.layers.Dense(10, activation="softmax")(do)
model = keras.Model(input_img, output)

opt=keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
h=model.fit(x_train, y_train,epochs=epoc,batch_size=128,shuffle=True,validation_data=(x_test, y_test))
Plot_Resultados(epocas,h.history['loss'],h.history['accuracy'],h.history['val_accuracy'],'VGG16')
plt.savefig('p10cifar10.pdf')
plt.show()
