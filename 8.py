from keras.datasets import mnist
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
(x_train, y_train), (x_test, y_test) = mnist.load_data()
epocas=np.linspace(1,50,50)
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
y_train=keras.utils.to_categorical(y_train, num_classes=10)
y_test=keras.utils.to_categorical(y_test, num_classes=10)

permutation = np.random.permutation(28*28)
x_train_perm = x_train.reshape(x_train.shape[0], -1)
x_train_perm = x_train_perm[:,permutation]
x_train_perm = x_train_perm.reshape(x_train.shape)
x_test_perm = x_test.reshape(x_test.shape[0], -1)
x_test_perm = x_test_perm[:,permutation]
x_test_perm = x_test_perm.reshape(x_test.shape)

plt.figure(figsize=(20, 8))
input_img = keras.Input(shape=(28, 28, 1))
opt=keras.optimizers.Adam(learning_rate=1e-2)
x = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((1, 1), padding='same')(x)
flat=layers.Flatten()(x)
do=keras.layers.Dropout(0.3)(flat)
x = keras.layers.Dense(100, activation="relu")(do)
dense=layers.Dense(10,activation='softmax')(x)
model = keras.Model(input_img, dense)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
h=model.fit(x_train, y_train,epochs=50,batch_size=128,validation_data=(x_test, y_test))
Plot_Resultados(epocas,h.history['loss'],h.history['accuracy'],h.history['val_accuracy'],'Convolucional')

opt=keras.optimizers.Adam(learning_rate=1e-2)
input_img = keras.Input(shape=(28, 28, 1))
flat=keras.layers.Flatten()(input_img)
do=keras.layers.Dropout(0.3)(flat)
x = keras.layers.Dense(2500, activation="relu")(do)
do=keras.layers.Dropout(0.2)(flat)
x = keras.layers.Dense(1000, activation="relu")(do)
output=keras.layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs=input_img, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer=opt , metrics=["accuracy"])
model.summary()

h=model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, batch_size=128, verbose=2)
Plot_Resultados(epocas,h.history['loss'],h.history['accuracy'],h.history['val_accuracy'],'Densas')

input_img = keras.Input(shape=(28, 28, 1))
opt=keras.optimizers.Adam(learning_rate=1e-2)
x = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((1, 1), padding='same')(x)
flat=layers.Flatten()(x)
do=keras.layers.Dropout(0.3)(flat)
x = keras.layers.Dense(100, activation="relu")(do)
dense=layers.Dense(10,activation='softmax')(x)
model = keras.Model(input_img, dense)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
h=model.fit(x_train_perm, y_train,epochs=50,batch_size=128,validation_data=(x_test_perm, y_test))
Plot_Resultados(epocas,h.history['loss'],h.history['accuracy'],h.history['val_accuracy'],'Convolucional con permutaciones')

opt=keras.optimizers.Adam(learning_rate=1e-2)
input_img = keras.Input(shape=(28, 28, 1))
flat=keras.layers.Flatten()(input_img)
do=keras.layers.Dropout(0.3)(flat)
x = keras.layers.Dense(2500, activation="relu")(do)
do=keras.layers.Dropout(0.2)(flat)
x = keras.layers.Dense(1000, activation="relu")(do)
output=keras.layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs=input_img, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer=opt , metrics=["accuracy"])
model.summary()

h=model.fit(x_train_perm, y_train, validation_data=(x_test_perm, y_test), epochs=50, batch_size=128, verbose=2)
Plot_Resultados(epocas,h.history['loss'],h.history['accuracy'],h.history['val_accuracy'],'Densas con permutaciones')



plt.savefig('p9.pdf')
plt.show()
