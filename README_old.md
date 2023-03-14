# DynamiqueLearningCurve
Pour utiliser commencer par import :  
from keras.callbacks import CSVLogger  

Puis importez mon callback en le placant dans le même dossier  
from callback import LearningCurve. 

Une fois importer dans votre training avant le fit construiser ces 2 callback. 

csv_logger = CSVLogger(path_save, append=True).  
learning_curves = LearningCurves(savePath, pathCSV, NBepochs, metrics = ['acc', 'loss'], show_graph = False)  
 
Ajouter ces callback a une liste            
callbackList.append(csv_logger)  
callbackList.append(learning_curves).  


Puis ajouter les dans votre fit en temps que param callbacks et vous verrez des learning curve faite en temps réel ;-)  
histo = self.model.fit(trainData, TestData, verbose=1, epochs=int(epochs), callbacks=callbackList)  

Combine ces callback avec celui de keras ModelCheckpoint afin de pouvoir kill votre training a tout moment  
plus dinfo ici -> https://keras.io/api/callbacks/model_checkpoint/  
            
