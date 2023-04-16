import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import csv
from keras.callbacks import Callback


class LearningCurves(Callback):
    def __init__(self, save_path="", graph_name="LearningCurve.png", save_graph=True, show_graph=False, black_background=False, csv_to_load=""):

        self.save_path = save_path
        self.graphe_name = graph_name
        self.save_graph = save_graph
        self.show_graph = show_graph
        

        self.data = {
            'epoch': [],
            'accuracy': [],
            'loss': [],
            'val_accuracy': [],
            'val_loss': []
        }

        # Charger les données d'un fichier CSV existant
        if csv_to_load != "" and os.path.exists(csv_to_load):
            self.__read_csv(csv_to_load)
           
        '''
        # Override graphe si nécessaire
        if self.csv_to_load != "" and os.path.exists(self.csv_to_load) and os.path.exists(os.path.join(save_path, graph_name)):
            os.remove(os.path.join(savePath, fileName))
        '''

        # Configuration du graphe
        if black_background:
            plt.style.use('dark_background')
        else:
            plt.style.use('classic')


        self.figure, self.axs = plt.subplots(2)
        self.figure.suptitle('Learning Curves')
        self.axs[0].set_ylabel("Loss")
        self.axs[1].set_ylabel("Accuracy")

    def on_epoch_end(self, epoch, logs={}):
        self.data['epoch'].append(epoch)
        self.data['accuracy'].append(logs.get('accuracy'))
        self.data['loss'].append(logs.get('loss'))
        self.data['val_accuracy'].append(logs.get('val_accuracy'))
        self.data['val_loss'].append(logs.get('val_loss'))

        self.axs[0].plot(self.data['epoch'], self.data['loss'], color='r')
        self.axs[1].plot(self.data['epoch'], self.data['accuracy'], color='r')

        if 'val_loss' in logs:
            self.axs[0].plot(self.data['epoch'], self.data['val_loss'], color='b')
            self.axs[1].plot(self.data['epoch'], self.data['val_accuracy'], color='b')

        red_patch = mpatches.Patch(color='red', label='Train')
        blue_patch = mpatches.Patch(color='blue', label='Validation')
        self.axs[0].legend(handles=[red_patch, blue_patch], loc=1)
        self.axs[1].legend(handles=[red_patch, blue_patch], loc=4)

        if self.show_graph:
            plt.show()

        if self.save_graph:
            self.figure.savefig(os.path.join(self.save_path, self.graphe_name))


    def __read_csv(self, path_to_csv):
        with open(path_to_csv, 'r') as f:
            reader = csv.DictReader(f)

            for row in reader:
                self.data['epoch'].append(int(row['epoch']))
                self.data['accuracy'].append(float(row['accuracy']))
                self.data['loss'].append(float(row['loss']))

                if 'val_accuracy' in row:
                    self.data['val_accuracy'].append(float(row['val_accuracy']))

                if 'val_loss' in row:
                    self.data['val_loss'].append(float(row['val_loss']))
