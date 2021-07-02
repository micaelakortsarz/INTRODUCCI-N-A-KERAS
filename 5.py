import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

def mapeo_real(x):
    return 4*x*(1-x)

#Generación de los datos
x=np.linspace(0,1,10000)
y=mapeo_real(x)
test_x=np.linspace(0,1,250)
test_y=mapeo_real(test_x)
#Crear modelo
inputs = keras.Input(shape=(1,))
l1 = keras.layers.Dense(5, activation='tanh',
                           kernel_regularizer = keras.regularizers.l2(1e-5), bias_regularizer= keras.regularizers.l2(1e-5))(inputs)
l2=keras.layers.Concatenate(axis=1)([l1, inputs])
outputs = keras.layers.Dense(1, activation='linear',kernel_regularizer = keras.regularizers.l2(1e-5), bias_regularizer= keras.regularizers.l2(1e-5))(l2)
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()
opt = keras.optimizers.Adam(learning_rate=0.005)
model.compile(
    optimizer=opt,
    loss="mae"
)
results = model.fit(
    x, y,
    epochs=10000,
    batch_size=10000,
    validation_data=(test_x, test_y)
)
y_pred=model.predict(test_x)

plt.show()
with plt.style.context('seaborn-darkgrid'):
      plt.grid(True)
      ax1 = plt.subplot(211)
      plt.scatter(test_y,y_pred)
      plt.plot(x,x,color='red', label='Identidad')
      plt.ylabel(r'$Y_{predecido}$')
      plt.xlabel(r'$Y_{Real}$')
      plt.legend()
      plt.tight_layout()
      ax2 = plt.subplot(212)
      plt.plot(results.history['loss'], label='Entrenamiento')
      plt.plot(results.history['val_loss'],label='Validación')
      plt.ylabel(r'Función de costo')
      plt.xlabel(r'Epocas transcurridas')
      plt.legend()
      plt.tight_layout()
plt.savefig('p5.pdf')
plt.show()