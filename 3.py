from keras.datasets import imdb
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
        plt.plot(epocas,[2*(ac[i]-ac_test[i])/(ac[i]+ac_test[i]) for i in range(len(ac))], label=label)
        plt.ylabel(r'Overfitting')
        plt.xlabel(r'Epocas transcurridas')
        plt.legend()
        plt.tight_layout()


plt.figure(figsize=(10, 8), constrained_layout=True)
epoc=15
epocas=np.linspace(1,epoc,epoc)

top_words = 10000

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=top_words)

max_sequence_length=500
reg=keras.regularizers.l2(1e-5)
opt=keras.optimizers.Adam(learning_rate=5e-5)

X_train = pad_sequences(x_train, maxlen=max_sequence_length, value = 0.0)
X_test = pad_sequences(x_test, maxlen=max_sequence_length, value = 0.0)
y_test = np.array(y_test).astype("float32")
y_train = np.array(y_train).astype("float32")

#Modelo con capas densas

inputs =keras.Input(shape=(max_sequence_length,))
en=keras.layers.Embedding(top_words, max_sequence_length)(inputs)
flat=keras.layers.Flatten()(en)
x = keras.layers.Dense(20, activation="relu")(flat)
x = keras.layers.Dense(10, activation="relu")(x)
output=keras.layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs, outputs=output)
model.compile(loss='binary_crossentropy', optimizer=opt , metrics=["binary_accuracy"])
model.summary()
h=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epoc, batch_size=128, verbose=2)
Plot_Resultados(epocas,h.history['loss'],h.history['binary_accuracy'],h.history['val_binary_accuracy'],'Densas')# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


#Modelo con reg L2
inputs =keras.Input(shape=(max_sequence_length,))
en=keras.layers.Embedding(top_words, max_sequence_length)(inputs)
flat=keras.layers.Flatten()(en)
x = keras.layers.Dense(20, activation="relu",kernel_regularizer = reg, bias_regularizer= reg)(flat)
x = keras.layers.Dense(10, activation="relu",kernel_regularizer = reg, bias_regularizer= reg)(x)
output=keras.layers.Dense(1, activation="sigmoid",kernel_regularizer = reg, bias_regularizer= reg)(x)
model = keras.Model(inputs=inputs, outputs=output)
model.compile(loss='binary_crossentropy', optimizer=opt , metrics=["binary_accuracy"])
model.summary()
h=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epoc, batch_size=128, verbose=2)
Plot_Resultados(epocas,h.history['loss'],h.history['binary_accuracy'],h.history['val_binary_accuracy'],'Densas con Regularización L2')
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

#Modelo con reg L2 y DropOut

inputs =keras.Input(shape=(max_sequence_length,))
en=keras.layers.Embedding(top_words, max_sequence_length)(inputs)
flat=keras.layers.Flatten()(en)
x = keras.layers.Dense(20, activation="relu",kernel_regularizer = reg, bias_regularizer= reg)(flat)
do=keras.layers.Dropout(0.4, noise_shape=None, seed=None)(x)
x = keras.layers.Dense(10, activation="relu",kernel_regularizer = reg, bias_regularizer= reg)(do)
do=keras.layers.Dropout(0.4, noise_shape=None, seed=None)(x)
output=keras.layers.Dense(1, activation="sigmoid",kernel_regularizer = reg, bias_regularizer= reg)(do)
model = keras.Model(inputs=inputs, outputs=output)
model.compile(loss='binary_crossentropy', optimizer=opt , metrics=["binary_accuracy"])
model.summary()
h=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epoc, batch_size=128, verbose=2)
Plot_Resultados(epocas,h.history['loss'],h.history['binary_accuracy'],h.history['val_binary_accuracy'],'Densas con L2 y DropOut')
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


#Modelo con DropOut y BatchN
inputs =keras.Input(shape=(max_sequence_length,))
en=keras.layers.Embedding(top_words, max_sequence_length)(inputs)
flat=keras.layers.Flatten()(en)
bn=keras.layers.BatchNormalization()(flat)
x = keras.layers.Dense(20, activation="relu")(bn)
do=keras.layers.Dropout(0.4, noise_shape=None, seed=None)(x)
bn=keras.layers.BatchNormalization()(do)
x = keras.layers.Dense(10, activation="relu")(bn)
do=keras.layers.Dropout(0.4, noise_shape=None, seed=None)(x)
bn=keras.layers.BatchNormalization()(do)
output=keras.layers.Dense(1, activation="sigmoid")(bn)
model = keras.Model(inputs=inputs, outputs=output)
model.compile(loss='binary_crossentropy', optimizer=opt , metrics=["binary_accuracy"])
model.summary()
h=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epoc, batch_size=128, verbose=2)
Plot_Resultados(epocas,h.history['loss'],h.history['binary_accuracy'],h.history['val_binary_accuracy'],'Densas DropOut y BN')
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

#Punto 4


inputs =keras.Input(shape=(max_sequence_length,))
en=keras.layers.Embedding(top_words,32)(inputs)
x = keras.layers.Conv1D(32, 2, padding='same', activation='relu')(en)
x = keras.layers.Conv1D(64, 2, padding='same', activation='relu')(x)
x = keras.layers.Conv1D(96, 2, padding='same', activation='relu')(x)
mp=keras.layers.MaxPooling1D(pool_size=2)(x)
do=keras.layers.Dropout(0.4)(mp)
flat=keras.layers.Flatten()(do)
output=keras.layers.Dense(1, activation="sigmoid")(flat)
model = keras.Model(inputs=inputs, outputs=output)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["binary_accuracy"])
model.summary()
h=model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=epoc, batch_size=128, verbose=2)
Plot_Resultados(epocas,h.history['loss'],h.history['binary_accuracy'],h.history['val_binary_accuracy'],'Convolucional 1D')
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


plt.savefig('p4.pdf')
plt.show()
