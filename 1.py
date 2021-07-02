from tensorflow import keras
from keras.datasets import boston_housing
import numpy as np
import matplotlib.pyplot as plt

#Obtencion de los datos
(X_train, Y_train), (X_test, Y_test) =boston_housing.load_data(path="boston_housing.npz", test_split=0.25, seed=113)
#Preprocesado
X_train=X_train.astype('float32')
X_test=X_test.astype('float32')
mean = np.mean(X_train,axis=0)
std = np.std(X_train,axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

#Crear arquitectura
inputs = keras.Input(shape=(len(X_train[0]),))
x = keras.layers.Dense(1, activation='linear')(inputs)
model = keras.Model(inputs=inputs, outputs=x)
model.summary()

#Entrenar y probar el modelo
opt = keras.optimizers.Adam(learning_rate=0.75)
model.compile(opt, 'mse')
hist=model.fit(X_train, Y_train,epochs=50, batch_size=len(X_train),verbose=0)


x=np.linspace(0,50,100)
Y_pred=model.predict(X_test,batch_size=len(X_test))

with plt.style.context('seaborn-darkgrid'):

      plt.grid(True)

      ax1 = plt.subplot(211)
      plt.scatter(Y_test,Y_pred)
      plt.plot(x,x,color='red',label='Identidad')
      plt.ylabel(r'$Y_{Predict}$')
      plt.xlabel(r'$Y_{True}$')
      plt.legend()
      ax2 = plt.subplot(212)
      plt.plot(hist.history['loss'])
      plt.ylabel(r'Función de costo')
      plt.xlabel(r'Epocas transcurridas')
      plt.legend()
      plt.tight_layout()
plt.savefig('p1.pdf')
plt.show()