#__IMPORTATION DES LIBRARIES__
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sn
import pandas as pd
from keras.callbacks import Callback




# Callback permettant de faire des matrices de confusion pour chaque epoque de la phase de test
class ConfusionMatrixPlotter(Callback):

	def __init__(self, testData, testLabel, classes, analyseMetric="loss", title='Confusion Matrix', show_graph = False, save_graph=True, classificationReportFile = True, pathSave = "../Result/"):

		self.pathSave = pathSave
		self.classificationReportFile = classificationReportFile
		
		self.save_graph = save_graph
		self.show_graph = show_graph
	
		self.testData = testData
		self.testLabel = testLabel
		self.classes = classes

		self.analyseMetric = analyseMetric
		self.baseMetric = None
		self.epoch = 0

       

    
	def on_epoch_end(self, epoch, logs={}):    
		epoch +=1

		# Check si Matrice nécéssaire
		if self.baseMetric == None or (self.analyseMetric == "accuracy" and logs.get(self.analyseMetric) > self.baseMetric) or (self.analyseMetric == "loss" and logs.get(self.analyseMetric) < self.baseMetric):

			# Recupération new seuil
			self.baseMetric = logs.get(self.analyseMetric)

			# Draw matrice
			pred = self.model.predict_classes(np.array(self.testData))
			#print(len(self.testLabel),len(pred),pred)
			matrix = confusion_matrix(self.testLabel, pred)

			df_cm = pd.DataFrame(matrix, index = self.classes, columns = self.classes)
			plt.figure(figsize = (12,9))
			dessin = sn.heatmap(df_cm, annot=True)

			# Save Matrice
			if self.save_graph :
				plt.savefig(self.pathSave + "_Confusion_Matrice_" + str(epoch) + ".png")


			# classification report to csv
			if self.classificationReportFile == True:
				report = classification_report(self.testLabel, pred, output_dict=True)
				df = pd.DataFrame(report).transpose()
				df.to_csv(self.pathSave + "classification_report_" + str(epoch) + ".csv")


			# Show Matrice
			if self.show_graph:
				print(matrix)
				plt.show()



# TODO a adapter au niveau 3D (predict_generator)
# Callback permettant de faire des matrices de confusion à l'aide d'un générator pour chaque epoque de la phase de test 
class ConfusionMatrixPlotter_Generator(Callback):
	
	def __init__(self, testData, analyseMetric="loss", title='Confusion Matrix', show_graph = False, save_graph=True, classificationReportFile = True, pathSave = "../Result/"):

		self.pathSave = pathSave
		self.classificationReportFile = classificationReportFile
		
		self.save_graph = save_graph
		self.show_graph = show_graph
	
		self.testData = testData

		self.analyseMetric = analyseMetric
		self.baseMetric = None
		self.epoch = 0

       

    
	def on_epoch_end(self, epoch, logs={}):    
		epoch +=1

		# Check si Matrice nécéssaire
		if self.baseMetric == None or (self.analyseMetric == "accuracy" and logs.get(self.analyseMetric) > self.baseMetric) or (self.analyseMetric == "loss" and logs.get(self.analyseMetric) < self.baseMetric):

			# Recupération new seuil
			self.baseMetric = logs.get(self.analyseMetric)

			# Draw matrice
			pred = self.model.predict_generator(self.testData, steps=1)
			
			print("taille prediction ",len(pred), "taille data", len(self.testData.filenames))
			#pred = np.argmax(pred, axis=1)

			matrix = confusion_matrix(self.testData.classes, pred)

			
			df_cm = pd.DataFrame(matrix, index = list(self.testData.class_indices.keys()), columns = list(self.testData.class_indices.keys()))
			plt.figure(figsize = (12,9))
			dessin = sn.heatmap(df_cm, annot=True)

			# Save Matrice
			if self.save_graph :
				plt.savefig(self.pathSave + "_Confusion_Matrice_" + str(epoch) + ".png")


			# classification report to csv
			if self.classificationReportFile == True:
				report = classification_report(self.testData.classes, pred, target_names=list(self.testData.class_indices.keys()))
				df = pd.DataFrame(report).transpose()
				df.to_csv(self.pathSave + "classification_report_" + str(epoch) + ".csv")


			# Show Matrice
			if self.show_graph:
				print(matrix)
				plt.show()



# Todo faire module callback avec un callback par fichier
# TODO Mieux commenter
# TODO ATTRIBUTS PERIODE
# Callback permettant de faire des graphes representant les courbes d'apprentissage (train / test) en temps réel
class LearningCurves(Callback):

	def __init__(self, savePath, filePath, epoch, metrics = ['accuracy', 'loss'], periode = 1, startZero = False, save_graph = True, show_graph = True):
		self.filePath = filePath
		self.savePath = savePath
		self.save_graph = save_graph
		self.show_graph = show_graph

		self.epoch = 0
		self.startZero = startZero
		self.periode = periode if periode > 0 else 1

		# Configuration du graphe
		metrics = [element.lower() for element in metrics]
		self.figure, self.axs = plt.subplots(len(metrics))
		self.figure.suptitle('Learning Curves')

		# Definir les axis
		# Set accuracy limite to 0-1
		if "acc" in metrics or "accuracy" in metrics:
			indexMetric = metrics.index("accuracy") if "accuracy" in metrics else metrics.index("acc")
			self.axs[indexMetric].set_ylim(0,1)
			self.axs[indexMetric].set_xlim(0,epoch)

		if "loss" in metrics:
			indexMetric = metrics.index("loss")
			self.axs[indexMetric].set_xlim(0,epoch)
		

		# Configuration de la structure pour stockage des données
		self.data = []

		for metric in metrics:
			self.data.append({metric: [], "val_" + metric: []})		


	def on_epoch_end(self, epoch, logs={}):

		# If file existe (re-train) or csv callback
		if os.path.exists(self.filePath):

			# Recuperation des données dans le fichier
			dataFrame = pd.read_csv(self.filePath) 

			# Add zero point only for accuracy
			if self.startZero:
				for col in dataFrame.columns:
					if "acc" in col:
						dataFrame.at[0,col]=0
			

			# Construction du graphe
			for index in range(0,len(self.data)):

				self.axs[index].set_ylabel(list(self.data[index].keys())[0].replace("val_",""))

				self.axs[index].plot(dataFrame["epoch"], dataFrame[list(self.data[index].keys())[0]], color='r')
				self.axs[index].plot(dataFrame["epoch"], dataFrame[list(self.data[index].keys())[1]], color='b')
					
				red_patch = mpatches.Patch(color='red', label='Train')	
				blue_patch = mpatches.Patch(color='blue', label='Test')
				self.axs[index].legend(handles=[red_patch, blue_patch], loc=4)


		# TODO VOIR ERROR
		# File not found (train)
		else:
			
			# Add zero point # TODO REVOIR
			if self.epoch == 0 and self.startZero :
				for key in logs:
					logs[key] = 0.0
			

			# Construction de l'axes
			epochs = [x for x in range(self.epoch, self.periode)]


			# Récupèration des données
			[dico[key].append(logs.get(key)) for dico in self.data for key in dico]


			# Construction du graphe
			for index in range(0,len(self.data)):
				self.axs[index].set_ylabel(list(self.data[index].keys())[0].replace("val_",""))

				self.axs[index].plot(epochs,self.data[index][list(self.data[index].keys())[0]], color='r')
				self.axs[index].plot(epochs,self.data[index][list(self.data[index].keys())[1]], color='b')
	
				red_patch = mpatches.Patch(color='red', label='Train')	
				blue_patch = mpatches.Patch(color='blue', label='Test')
				self.axs[index].legend(handles=[red_patch, blue_patch], loc=4)

		
		# Affichage du graphe
		if self.show_graph == True : 
			plt.show()

		# Sauvegarde du graphe
		if self.save_graph == True : 
			self.figure.savefig(os.path.join(self.savePath, "LearningCurve.png"))


		# Incrementation du nombre de tour
		self.epoch += 1



# TODO A AMELIORER
# TODO COMMENTER
class MemoryUse(Callback):


	def __init__(self, path):
		self.path = path
		self.memory = []

		# On ecrase le fichier s'il existait déjà
		fichier = open(self.path,"w")
		fichier.close()


	def on_train_batch_end(self, batch, logs=None):
		self.memory.append(psutil.virtual_memory()._asdict()["percent"])

	def on_test_batch_end(self, batch, logs=None):
		self.memory.append(psutil.virtual_memory()._asdict()["percent"])


	def on_epoch_end(self, epoch, logs={}):  
		fichier = open(self.path,"a")
		fichier.write(str(self.memory) + "\n")
		fichier.close() 

		self.memory = []


# TODO A AMELIORER
# TODO COMMENTER
class GPU_NVIDIA_Consomation(Callback):


	def __init__(self, path):
		self.path = path


	def on_train_batch_end(self, batch, logs=None):
		os.system("nvidia -smi >> " + self.path)

	def on_test_batch_end(self, batch, logs=None):
		os.system("nvidia -smi >> " + self.path)

	
