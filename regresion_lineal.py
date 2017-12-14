#creada por diego en handytech.mobi
#Importamos la librerias necesarias para el ejericio
import matplotlib #para graficos
matplotlib.use('GTKAgg')

import matplotlib.pyplot as plt
import numpy as np #Libreria para realizar cálculo y algebra lineal
from sklearn import datasets, linear_model #Librerías de machine learning
import pandas as pd #Análisis de Datos

import io
import requests #Para obtención de dataset alojado en una URL
url="https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Housing.csv"
s=requests.get(url).content

# Cargar las columnas del csv
df = pd.read_csv(io.StringIO(s.decode('utf-8')))

#Cargar las funciones
Y = df['price']
X = df['lotsize']

X=X.values.reshape(len(X),1)
Y=Y.values.reshape(len(Y),1)

# Dividir el dataset en un dataset de entrenamiento y prueba
X_train = X[:-250]
X_test = X[-250:]

# Dividir el dataset en un dataset de entrenamiento y prueba
Y_train = Y[:-250]
Y_test = Y[-250:]

# Pintar los resultados de los datos de Test
plt.scatter(X_test, Y_test,  color='black')
plt.title('Test Data')
plt.xlabel('Size')
plt.ylabel('Price')
plt.xticks(())
plt.yticks(())

plt.show()

# Crear un objeto de Regresión lineal
regr = linear_model.LinearRegression()

# Entrenar el modelo usando el dataset de entrenamiento
regr.fit(X_train, Y_train)

# Pintar los resultados
plt.plot(X_test, regr.predict(X_test), color='red',linewidth=3)

print( str(round(regr.predict(6000))) )
