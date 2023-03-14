import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from keras.callbacks import Callback


# TODO remove duplicate epoch entry ?
# TODO accuracy first graphe ?

class LearningCurves(Callback):

    def __init__(self, save_path="", graph_name="LearningCurve.png", save_graph=True, show_graph=True, black_background=False, csv_to_load=""):
        self.save_path = save_path
        self.graphe_name = graph_name
        self.save_graph = save_graph
        self.show_graph = show_graph

        # Configuration de la structure pour stockage des données
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
        if self.lastEpoch == 0 and os.path.exists(os.path.join(savePath, fileName)):
            os.remove(os.path.join(savePath, fileName))
        '''
        
        # Configuration du graphe

        # Set background
        if black_background == True :
            plt.style.use('dark_background')
        else :
            plt.style.use('classic')

        # Séparation du graphe en 2 section : loss / accuracy
        self.figure, self.axs = plt.subplots(2)
        self.figure.suptitle('Learning Curves')
        self.axs[0].set_ylabel("Loss")
        self.axs[1].set_ylabel("Accuracy")

       

    # À chaque tour du training (train + validation)
    def on_epoch_end(self, epoch, logs={}):
        
        # Stockage des performance de l'epoch actuel
        self.data['epoch'].append(epoch)
        self.data['loss']['train'].append(logs.get('loss'))
        self.data['accuracy']['train'].append(logs.get('accuracy'))

        # Plot Accuracy / Loss performance
        self.axs[0].plot(self.data['epoch'], self.data['loss']['train'], color='r')
        self.axs[1].plot(self.data['epoch'], self.data['accuracy']['train'], color='r')

        # Si validation dispo plot validation accuracy / loss performance
        if 'val_loss' in logs:
            self.data['loss']['val'].append(logs.get('val_loss'))
            self.data['accuracy']['val'].append(logs.get('val_accuracy'))

            self.axs[1].plot(self.data['epoch'], self.data['accuracy']['val'], color='b')
            self.axs[0].plot(self.data['epoch'], self.data['loss']['val'], color='b')

        # Ajout de la légende
        red_patch = mpatches.Patch(color='red', label='Train')
        blue_patch = mpatches.Patch(color='blue', label='Validation')
        self.axs[0].legend(handles=[red_patch, blue_patch], loc=1)
        self.axs[1].legend(handles=[red_patch, blue_patch], loc=4)

        # Affichage du graphe
        if self.show_graph == True:
            plt.show()

        # Sauvegarde du graphe
        if self.save_graph == True:
            self.figure.savefig(os.path.join(self.savePath,  self.fileName))
            
            
            
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
