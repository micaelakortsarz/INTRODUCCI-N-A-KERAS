from tensorflow.keras.datasets import cifar10
from tensorflow import keras
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
epocas=np.linspace(1,500,500)
epoc=500
#Obtengo y preproceso los datos
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
mean = np.mean(x_train,axis=0)
x_train = (x_train - mean) / 255.0
x_test = (x_test - mean) / 255.0
x_train=np.reshape(x_train, (x_train.shape[0], np.prod(x_train.shape[1:])))
x_test=np.reshape(x_test, (x_test.shape[0], np.prod(x_test.shape[1:])))
y_train=keras.utils.to_categorical(y_train, num_classes=10)
y_test=keras.utils.to_categorical(y_test, num_classes=10)

#Creo y entreno mi modelo para los puntos 2.3 y 2.4
reg=keras.regularizers.l2(1e-5)
opt=keras.optimizers.SGD(learning_rate=5e-3)
loss_p3=keras.losses.MeanSquaredError()
loss_p4=keras.losses.CategoricalCrossentropy(from_logits=True)

inputs = keras.Input(shape=(3072,))
l = keras.layers.Dense(100, activation='sigmoid',kernel_regularizer = reg, bias_regularizer= reg)(inputs)
output= keras.layers.Dense(10, activation='linear',kernel_regularizer = reg, bias_regularizer= reg)(l)
model = keras.Model(inputs=inputs, outputs=output)
model.summary()

model.compile(optimizer=opt, loss=loss_p3, metrics=["accuracy"])
history_p3 = model.fit(x_train, y_train, epochs=epoc, batch_size=64, validation_data=(x_test, y_test), verbose=1)
Plot_Resultados(epocas,history_p3.history['loss'],history_p3.history['accuracy'],history_p3.history['val_accuracy'],'Punto 3')


reg=keras.regularizers.l2(1e-5)
opt=keras.optimizers.SGD(learning_rate=5e-3)
inputs = keras.Input(shape=(3072,))
l2 = keras.layers.Dense(100, activation='sigmoid',kernel_regularizer = reg, bias_regularizer= reg)(inputs)
output2= keras.layers.Dense(10, activation='linear',kernel_regularizer = reg, bias_regularizer= reg)(l2)
model2 = keras.Model(inputs=inputs, outputs=output2)
model2.compile(optimizer=opt, loss=loss_p4, metrics=["accuracy"])
history_p4 = model2.fit(x_train, y_train, epochs=epoc, batch_size=64, validation_data=(x_test, y_test), verbose=1)
Plot_Resultados(epocas,history_p4.history['loss'],history_p4.history['accuracy'],history_p4.history['val_accuracy'],'Punto 4')
#Creo y entreno los clasificadores lineales SM y SVM
reg=keras.regularizers.l2(1e-5)
opt=keras.optimizers.SGD(learning_rate=1e-3)
reg1=keras.regularizers.l2(1e-4)
opt1=keras.optimizers.SGD(learning_rate=1e-3)
inputs = keras.Input(shape=(3072,))
output_svm= keras.layers.Dense(10, activation='linear',kernel_regularizer = reg1, bias_regularizer= reg1)(inputs)
output_sm= keras.layers.Dense(10, activation='linear',kernel_regularizer = reg, bias_regularizer= reg)(inputs)
model_svm = keras.Model(inputs=inputs, outputs=output_svm)
model_sm = keras.Model(inputs=inputs, outputs=output_sm)

model_sm.compile(optimizer=opt, loss=loss_p4, metrics=["accuracy"])
history_sm = model_sm.fit(x_train, y_train, epochs=epoc, batch_size=64, validation_data=(x_test, y_test), verbose=1)
Plot_Resultados(epocas,history_sm.history['loss'],history_sm.history['accuracy'],history_sm.history['val_accuracy'],'SM')
model_svm.compile(optimizer=opt1, loss='hinge', metrics=["accuracy"])
history_svm=model_svm.fit(x_train, y_train, epochs=epoc, batch_size=64, validation_data=(x_test, y_test), verbose=1)
#acc = model.evaluate(x_test, y_test, verbose=1)
Plot_Resultados(epocas,history_svm.history['loss'],history_svm.history['accuracy'],history_svm.history['val_accuracy'],'SVM')

plt.savefig('p2_3.pdf')
plt.show()


from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

x_train=np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
y_train=np.array([[1],[0],[0],[1]])
x_test=x_train
y_test=y_train
opt=keras.optimizers.SGD(learning_rate=0.5)
acc=keras.metrics.BinaryAccuracy(name="binary_accuracy", threshold=0.9)
inputs = keras.Input(shape=(2,))

l = keras.layers.Dense(2, activation='tanh')(inputs)
output= keras.layers.Dense(1, activation='tanh',kernel_initializer=keras.initializers.RandomNormal(stddev=0.5))(l)
model = keras.Model(inputs=inputs, outputs=output)
model.summary()
model.compile(optimizer=opt, loss='mse', metrics=["binary_accuracy"])
history_aq1 = model.fit(x_train, y_train, epochs=100, batch_size=4, validation_data=(x_test, y_test), verbose=1)

l2 = keras.layers.Dense(1, activation='tanh',kernel_initializer=keras.initializers.RandomNormal(stddev=0.5))(inputs)
concat=keras.layers.Concatenate(axis=1)([l2, inputs])
output = keras.layers.Dense(1, activation='tanh',kernel_initializer=keras.initializers.RandomNormal(stddev=0.5))(concat)
model2 = keras.Model(inputs=inputs, outputs=output)
model.summary()
model2.compile(optimizer=opt, loss='mse', metrics=["binary_accuracy"])
history_aq2 = model2.fit(x_train, y_train, epochs=100, batch_size=4, validation_data=(x_test, y_test), verbose=1)
epocas=np.linspace(1,100,100)
plt.figure()
with plt.style.context('seaborn-darkgrid'):
      plt.grid(True)
      ax1 = plt.subplot(211)
      plt.plot(epocas,history_aq1.history['loss'], label='Arquitectura 1')
      plt.plot(epocas,history_aq2.history['loss'], label='Arquitectura 2')
      plt.ylabel(r'Loss')
      plt.xlabel(r'Epocas transcurridas')
      plt.legend()
      ax2 = plt.subplot(212)
      plt.plot(epocas,history_aq1.history['binary_accuracy'], label='Arquitectura 1')
      plt.plot(epocas,history_aq2.history['binary_accuracy'], label='Arquitectura 2')
      plt.ylabel(r'Accuracy')
      plt.xlabel(r'Epocas transcurridas')
      plt.legend()
      plt.tight_layout()
      plt.legend()
      plt.tight_layout()

plt.savefig('p2_6.pdf')
plt.show()