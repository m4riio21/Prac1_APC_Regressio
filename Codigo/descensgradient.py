from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats
import seaborn as sns
import os


# Visualitzarem només 3 decimals per mostra
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Funcio per a llegir dades en format csv
def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    return dataset

# Carreguem dataset d'exemple
dataset = load_dataset('../BBDD/ysnp.csv')
dataset.drop('Year/Month/Day', inplace=True, axis=1)
#dataset = dataset.replace(np.nan, 0)
dataset = dataset.fillna(dataset.mean())

data = dataset.values
x = data[:, :20]
y = data[:, 0] #Escollim com a y l'atribut Recreation Visits

#calcul mse
def mse(v1, v2):
    return ((v1 - v2)**2).mean()

def regression(x, y):
    # Creem un objecte de regressió de sklearn
    regr = LinearRegression()

    # Entrenem el model per a predir y a partir de x
    regr.fit(x, y)

    # Retornem el model entrenat
    return regr

def standarize(x_train):
    mean = x_train.mean(0)
    std = x_train.std(0)
    x_t = x_train - mean[None, :]
    x_t /= std[None, :]
    return x_t

standarized_data = standarize(data)


def cost_fun(x, y, a, b):
    m = len(x)
    error = 0.0
    for i in range(m):
        tmp = a + b * x[i]
        error += (y[i] - tmp) ** 2
    return error / (2 * m)


def descensgradiente(x, y, a, b, alpha, iters):

    historic = []
    for i in range(iters):
        derivate_a = 0
        derivate_b = 0
        for i in range(len(x)):
            tmp = a + b * x[i]
            derivate_a += tmp - y[i]
            derivate_b += (tmp - y[i]) * x[i]
            historic.append(cost_fun(x, y, a, b))
        a -= (derivate_a / len(x)) * alpha
        b -= (derivate_b / len(x)) * alpha

    return a, b, historic

# Farem descens de gradient dels atributs 6 i 7

#Atribut 6
x = standarized_data[:,6]
y = standarized_data[:,0]

w0, w1, hist = descensgradiente(x,y,1,1,0.05,500)

recta = w1*x + w0
plt.scatter(x,y)
plt.plot(x, recta)

valors6 = []
for valor in data[:,6]:
    valors6.append(valor*w1 + w0)
error = mse(valors6, data[:,6])
with open("../Resultados/descensgradientAtribut6.txt",'w') as r:
    r.write(str(error))

plt.scatter(x,y)
plt.plot(x, recta)
plt.savefig("../Imagenes/descensgradient/atribut6.png")
plt.clf()

#Atribut 7
x = standarized_data[:,7]
y = standarized_data[:,0]

w0, w1, hist = descensgradiente(x,y,1,1,0.05,500)

recta = w1*x + w0
plt.scatter(x,y)
plt.plot(x, recta)

valors7 = []
for valor in data[:,7]:
    valors7.append(valor*w1 + w0)
error = mse(valors7, data[:,7])
with open("../Resultados/descensgradientAtribut7.txt",'w') as r:
    r.write(str(error))

plt.scatter(x,y)
plt.plot(x, recta)
plt.savefig("../Imagenes/descensgradient/atribut7.png")
plt.clf()