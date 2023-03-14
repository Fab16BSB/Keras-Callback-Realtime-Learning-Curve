# -*- coding: utf-8 -*-
"""First_reseau_de_neuronne.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Kj516Ja11XOKnYd1WNa_PQnSdblWKDB0
"""

"""# Exercice 2 : Développer un perceptron multicouche (MLP) pour un problème de classification
Le but de cet exercice sera de développer en Python, à l'aide de la librairie Keras, un MLP capable de résoudre le problème du XOR à partir du jeu de données fourni sur Moodle.
"""

# === Génération du jeu de données séparé en 2 fichiers : train / validation ====

import csv

def generatedata(min_number, max_number, name):

  # Définition des noms des colonnes
  header = ['number1', 'number2', 'label']

  # Initialisation des données
  data = []

  # Génération de toutes les combinaisons de nombres possibles
  for i in range(min_number, max_number+1):
      for j in range(min_number, max_number+1):
          
          # Calcul de la valeur de label
          label = 1 if i == j else 0
          
          # Ajout de la combinaison de nombres et du label aux données
          data.append([i, j, label])

  # Écriture des données dans un fichier CSV
  with open(name+'.csv', mode='w', newline='') as file:
      writer = csv.writer(file)
      writer.writerow(header)
      writer.writerows(data)

generatedata(0, 6, "train")
generatedata(50, 56, "val")

#_______________IMPORTS_____________

# importation lié au graphe
import matplotlib.pyplot as plt 

# [Optionel] forcer les graphes à avoir un fond noir
plt.style.use('dark_background')

# Importation pour manipuler les fichier csv
import pandas as pd

# Importation permettant de séparé notre jeu de données
from sklearn.model_selection import train_test_split

# Importation des metriques d'évaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# importation lié aux divers calcul matricielle
import numpy as np

# Importation lié à la création du model (reseau de neurone)
from keras.models import Sequential
from keras.layers import Dense 

# Importation pour afficher le modèle
from keras.utils.vis_utils import plot_model


# Importation permettant de sauvegarder le modèle 
import  pickle

#_______________LOAD_DATA_____________

df_train = pd.read_csv("train.csv")
df_val = pd.read_csv("val.csv")

print("Columns : ", list(df_train.columns))
print("Total data lenght :", len(df_train), "\n")

print(df_train.info(), "\n")

print("Example :\n", df_train.head, "\n")

#_______________DATA_REPARTITION_____________

# delete labels columns of feature variables
features_train = df_train.copy()
del features_train["label"]

features_val = df_val.copy()
del features_val["label"]

print("Available features: ", features_train.shape[1])

labels_train = df_train["label"]
labels_val= df_val["label"]

# Count of classe in target variable
nbClasse = len(set(labels_train))
print("Available classes :", nbClasse, "->", set(labels_train))


#_______________DATA_SEPARATION_____________
# Split the dataset into 2 sets : train (80%) / test (20%)
# featureTrain, featureTest, labelTrain, labelTest = train_test_split (features, labels, test_size = 0.20, shuffle=False)

from keras.callbacks import CSVLogger
from callback import LearningCurves

#csv_logger = CSVLogger("perf.csv", append=True)
learning_curves = LearningCurves("", lastEpoch=0, show_graph = False)
callbackList = []
#callbackList.append(csv_logger)
callbackList.append(learning_curves)

# ____________________MODELES_CREATION____________

# Creation d'un modèle vide
model = Sequential() 

# add input layer
# dense / full connected
# 2 are the number of neurons
# input_dim = number of feature for a sample
model.add(Dense(2, activation = 'relu', input_dim = 2)) 
                
# add output layer
# use sigmoid for binary classification
model.add(Dense(1, activation = 'sigmoid'))

# Use loss function : binary_crossentropy because you have 2 classes
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Afficher un résumé de l'architecture du modèle
model.summary()

# Affichage du modèle
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

#____________________TRAINING____________
# Fit the model with the train features and train labels
# batch_size represente the number of sample it will see at the same time
# epochs represente the number of time he will see all data
history = model.fit(features_train, labels_train, batch_size=1, epochs=100, validation_data=(features_val, labels_val), callbacks=callbackList)

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# Predict all test data
prediction = model.predict(features_val)

# Take the best classe (hight score) for all prediction
bestPred = [1 if pred >= 0.5 else 0 for pred in prediction]

# Calculate accuracy : How many prediction are good ?
acc = accuracy_score(bestPred, labels_val)

# Calculate precision : Number of correct prediction for this class / total of predictions for this class
precision = precision_score(bestPred, labels_val)

# Calculate recall : Number of correct prediction  / total element of this class
recall = recall_score(bestPred, labels_val)

# Relation beetwen precision and recall
f1Score = f1_score(bestPred, labels_val)

print("\nAccuracy:", acc*100, "\nPrecision :", precision*100, "\nRecall", recall*100, "\nF1 score", f1Score*100)

#____________________SAVE____________
with open("deepModel.pkl", 'wb')  as saveFile:
    pickle.dump(model,  saveFile)
  
# Aussi possible via keras
# model.save('path/to/location')

#____________________LOAD____________
# pickle.load("name_of_your_file.pkl")

# Aussi possible via keras
# keras.models.load_model('path/to/location')