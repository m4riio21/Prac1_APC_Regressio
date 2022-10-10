import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


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


from sklearn.metrics import r2_score

standarized_data = standarize(data)

std_x = standarized_data[:,:20]
std_y = standarized_data[:,0]

mses = []
r2s = []
with open("../Resultados/R2_SCORE.txt",'w') as r:
    with open("../Resultados/MSE.txt",'w') as m:
        for i in range(x.shape[1]):
            atribut1 = std_x[:,i].reshape(x.shape[0], 1)
            regr = regression(atribut1, std_y)
            predicted = regr.predict(atribut1)
            # Mostrem la predicció del model entrenat en color vermell a la Figura anterior 1
            plt.figure()
            ax = plt.scatter(atribut1[:,0], std_y)
            plt.plot(atribut1[:,0], predicted, 'r')
            plt.savefig("../Imagenes/regresion/" + "Attribute_" + str(i) + ".png")
            plt.clf()

            MSE = mse(std_y, predicted)
            m.write("Attribute " + str(i) + ": " + str(MSE) +'\n')
            r2 = r2_score(std_y, predicted)
            r.write("Attribute " + str(i) + ": " + str(r2) + '\n')