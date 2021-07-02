import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

def impute_zero_field(data, field):
    nonzero_vals = data.loc[data[field] != 0, field]
    avg = np.sum(nonzero_vals) / len(nonzero_vals)
    k = len(data.loc[ data[field] == 0, field])   # num of 0-entries
    data.loc[ data[field] == 0, field ] = avg

def preprocess(X_train,zero_fields):
  for field in zero_fields:
      impute_zero_field(X_train, field)


pdata = pd.read_csv('diabetes.csv')
features = list(pdata.columns.values)
features.remove('Outcome')
X = pdata[features]
y = pdata['Outcome']
#zero_fields = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
zero_fields=[]
preprocess(X,zero_fields)
X = X.values
Y = y.values

kfold = StratifiedKFold(n_splits=5, shuffle=True)
cvscores = []
i=0
acc=[None,None,None,None,None]
acc_t=[None,None,None,None,None]
costo=[None,None,None,None,None]
for train, test in kfold.split(X, Y):
  opt=keras.optimizers.Adam(learning_rate=1e-3)
  inputs =keras.Input(shape=(8,))
  flat=keras.layers.Flatten()(inputs)
  #do=keras.layers.Dropout(0.2, noise_shape=None, seed=None)(flat)
  x = keras.layers.Dense(12, activation="relu", kernel_initializer='uniform')(flat)
  #do=keras.layers.Dropout(0.1, noise_shape=None, seed=None)(x)
  x1 = keras.layers.Dense(18, activation="relu",kernel_initializer='uniform')(x)
  output=keras.layers.Dense(1, activation="sigmoid",kernel_initializer='uniform')(x1)
  model = keras.Model(inputs=inputs, outputs=output)
  model.compile(loss='binary_crossentropy', optimizer=opt , metrics=['accuracy'])
  model.summary()
  h=model.fit(X[train], Y[train], epochs=500, batch_size=128, verbose=2,validation_data=(X[test],Y[test]))
  costo[i]=h.history['loss']
  acc[i]=h.history['accuracy']
  acc_t[i]=h.history['val_accuracy']
  i+=1
  scores = model.evaluate(X[test], Y[test], verbose=0)
  cvscores.append(scores[1] * 100)
print(np.mean(cvscores), np.std(cvscores))
acc_m=np.mean(acc,axis=0)
acc_std=np.std(acc,axis=0)
acc_t_m=np.mean(acc_t,axis=0)
acc_t_std=np.std(acc_t,axis=0)
c_m=np.mean(costo,axis=0)
c_std=np.std(costo,axis=0)
epocas=np.linspace(1,500,500)

plt.figure(figsize=(20, 10))
with plt.style.context('seaborn-darkgrid'):
  plt.grid(True)

  ax=plt.subplot(311)
  ax.plot(epocas,acc_m,label="Media")
  ax.fill_between(epocas,acc_m-acc_std,acc_m+acc_std,alpha=0.5,label="Std")
  ax.set_ylabel(r'Accuracy con train')
  ax.set_xlabel(r'Epocas transcurridas')
  plt.legend()
  ax2 = plt.subplot(312)
  ax2.plot(epocas,acc_t_m,label="Media")
  ax2.fill_between(epocas,acc_t_m-acc_t_std,acc_t_m+acc_t_std,alpha=0.5,label="Std")
  ax2.set_ylabel(r'Accuracy con test')
  ax2.set_xlabel(r'Epocas transcurridas')
  plt.legend()
  ax3 = plt.subplot(313)
  ax3.plot(epocas,c_m,label="Media")
  ax3.fill_between(epocas,c_m-c_std,c_m+c_std,alpha=0.5,label="Std")
  ax3.set_ylabel(r'Función de costo')
  ax3.set_xlabel(r'Epocas transcurridas')
  plt.legend()
plt.tight_layout()

pdata = pd.read_csv('diabetes.csv')
features = list(pdata.columns.values)
features.remove('Outcome')
X = pdata[features]
y = pdata['Outcome']

zero_fields = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
preprocess(X,zero_fields)
X = X.values
Y = y.values

kfold = StratifiedKFold(n_splits=5, shuffle=True)
cvscores = []
i=0
acc=[None,None,None,None,None]
acc_t=[None,None,None,None,None]
costo=[None,None,None,None,None]
for train, test in kfold.split(X, Y):
  opt=keras.optimizers.Adam(learning_rate=1e-3)
  inputs =keras.Input(shape=(8,))
  flat=keras.layers.Flatten()(inputs)
  #do=keras.layers.Dropout(0.2, noise_shape=None, seed=None)(flat)
  x = keras.layers.Dense(12, activation="relu", kernel_initializer='uniform')(flat)
  #do=keras.layers.Dropout(0.1, noise_shape=None, seed=None)(x)
  x1 = keras.layers.Dense(18, activation="relu",kernel_initializer='uniform')(x)
  output=keras.layers.Dense(1, activation="sigmoid",kernel_initializer='uniform')(x1)
  model = keras.Model(inputs=inputs, outputs=output)
  model.compile(loss='binary_crossentropy', optimizer=opt , metrics=['accuracy'])
  model.summary()
  h=model.fit(X[train], Y[train], epochs=500, batch_size=128, verbose=2,validation_data=(X[test],Y[test]))
  costo[i]=h.history['loss']
  acc[i]=h.history['accuracy']
  acc_t[i]=h.history['val_accuracy']
  i+=1
  scores = model.evaluate(X[test], Y[test], verbose=0)
  cvscores.append(scores[1] * 100)
print(np.mean(cvscores), np.std(cvscores))
acc_m=np.mean(acc,axis=0)
acc_std=np.std(acc,axis=0)
acc_t_m=np.mean(acc_t,axis=0)
acc_t_std=np.std(acc_t,axis=0)
c_m=np.mean(costo,axis=0)
c_std=np.std(costo,axis=0)
epocas=np.linspace(1,500,500)
with plt.style.context('seaborn-darkgrid'):
  plt.grid(True)

  ax=plt.subplot(311)
  ax.plot(epocas,acc_m,label="Media quitando 0's")
  ax.fill_between(epocas,acc_m-acc_std,acc_m+acc_std,alpha=0.5,label="Std quitando 0's")
  ax.set_ylabel(r'Accuracy con train')
  ax.set_xlabel(r'Epocas transcurridas')
  plt.legend()
  ax2 = plt.subplot(312)
  ax2.plot(epocas,acc_t_m,label="Media quitando 0's")
  ax2.fill_between(epocas,acc_t_m-acc_t_std,acc_t_m+acc_t_std,alpha=0.5,label="Std quitando 0's")
  ax2.set_ylabel(r'Accuracy con test')
  ax2.set_xlabel(r'Epocas transcurridas')
  plt.legend()
  ax3 = plt.subplot(313)
  ax3.plot(epocas,c_m,label="Media quitando 0's")
  ax3.fill_between(epocas,c_m-c_std,c_m+c_std,alpha=0.5,label="Std quitando 0's")
  ax3.set_xlabel(r'Epocas transcurridas')
  plt.legend()
plt.tight_layout()

plt.savefig('p6.pdf')
plt.show()