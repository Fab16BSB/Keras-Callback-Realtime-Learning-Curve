import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score
import keras.backend as K



class LearningCurves(Callback):

    #TODO revoir commentaire
    def __init__(self, savePath=".", show_graph=True, save_graph=True, override=True, blackBackground=False,  metric=["Loss"]):
        '''
        Objectif :
        Params :
        '''

        self.savePath = savePath
        self.show_graph = show_graph
        self.save_graph = save_graph
        self.metric = metric

        self.keyList = []
        self.figures = []
        self.axes = []
        self.data = {}

        # Set background
        self.setBackground(blackBackground)

        # Activer le mode interactif
        plt.ion()  

        # Create path if not exist and if graphe exist and override false add datetime on graph name
        self.extension = self.checkPath(override)

        # Information to the User
        print("\n[INFO] Des graphiques lié a la precision, recall ou f1-score peuvent etre généré pour cela ajouter les methodes statique : precision, recall et f1_score dans l'attribut metrique de votre methode compile.")
        print("Exemple : model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', LearningCurves.precision, LearningCurves.recall, LearningCurves.f1_score])\n")
        

        

    #TODO revoir commentaire
    # À chaque tour du training (train + validation)
    def on_epoch_end(self, epoch, logs={}):
        
        assert "val_loss" in logs, "[ERROR] Pas de validation set trouver"


        # Si c'est la premiere epoch
        if 'epoch' not in self.data:

            # Preparation structure de données
            self.data['epoch'] = []

            for key in logs :
                self.data[key] = []


            # Filtrage des metric
            keys = list(self.data.keys())
            self.keyList = [key.lower() for key in keys if key != "epoch" and not key.startswith('val_')]
            self.metric = self.keyList if self.metric == "*" else [key.lower() for key in self.metric if key.lower() in self.keyList]

            # Preparation du graphe
            self.setGraphe(self.metric)


        # Stockage des performance de l'epoch actuel
        for key in logs :
            self.data[key].append(logs[key])

        self.data['epoch'].append(epoch)

        
        # Desin sur le graphe
        self.plotGraphe(self.metric)

        
        # Sauvegarde du graphe
        self.saveGraphe()
                

        # Affichage du graphe    
        self.showGraphe(0.1)
            
        
    #TODO revoir commentaire
    def on_train_end(self, logs=None):
        plt.close()
        metrics = list(set(self.keyList) - set(self.metric))
        self.setGraphe(metrics, False)
        self.plotGraphe(metrics)
        self.saveGraphe()


    #TODO revoir commentaire
    def setGraphe(self, listeMetric, show=True):

        # Création des graphes individuels
        for key in self.data:
            if key != "epoch" and key in listeMetric:

                # Creation des figures
                fig = plt.figure()
                fig.suptitle('Learning Curves - ' + key + ' Graph')

                # Ajout de la légende
                red_patch = mpatches.Patch(color='red', label='Train')
                blue_patch = mpatches.Patch(color='blue', label='Validation')
                fig.legend(handles=[red_patch, blue_patch], loc=1)
                self.figures.append(fig)

                # Creation des axes
                ax = fig.add_subplot()
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Value')
                self.axes.append(ax)

                # Empèche les graphes d'etre affiché
                if show == False:
                    matplotlib.use('agg')


    #TODO revoir commentaire
    def plotGraphe(self, metricList):
        for index in range(0, len(metricList)) :
            key = self.keyList[index]
            ax = self.axes[index]
            
            ax.plot(self.data['epoch'], self.data[key], color='r')
            ax.plot(self.data['epoch'], self.data['val_'+key], color='b')


    #TODO revoir commentaire
    def showGraphe(self, time):
        if self.show_graph == True:
            plt.show()

            # Ajouter une pause entre les mises à jour
            plt.pause(time)



    #TODO revoir commentaire
    def saveGraphe(self):
        if self.save_graph == True:

            for indexFigure in range(0, len(self.figures)) :
                print(indexFigure, self.keyList[indexFigure])
                path = os.path.join(self.savePath, self.keyList[indexFigure] + self.extension)

                self.figures[indexFigure].savefig(path)


    #TODO revoir commentaire
    def setBackground(self, blackBackground):
        if blackBackground == True :
            plt.style.use('dark_background')
        else :
            plt.style.use('classic') 



    #TODO revoir commentaire  
    def checkPath(self, override):
        extension = ".png"

        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)

        else:
            if override == False and os.path.exists(os.path.join(self.savePath, "loss.png")): 
                current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                extension = "_" + current_time + ".png"

        return extension



    # TODO A COMMENTER
    # Fonctions de métriques personnalisées
    # https://aakashgoel12.medium.com/how-to-add-user-defined-function-get-f1-score-in-keras-metrics-3013f979ce0d
    @staticmethod
    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    @staticmethod
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    @staticmethod
    def f1_score(y_true, y_pred):
        precision_val = LearningCurves.precision(y_true, y_pred)
        recall_val = LearningCurves.recall(y_true, y_pred)
        f1_val = 2 * ((precision_val * recall_val) / (precision_val + recall_val + K.epsilon()))
        return f1_val
            