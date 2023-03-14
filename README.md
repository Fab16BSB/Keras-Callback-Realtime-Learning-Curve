<h1> V1.0 : </h1>
<h2> Objectif </h2>
Concevoir un callback keras créant une learning curve en temps réel.
Bien commenté le code
Rendre le projet publique 
Documenté le repo en francais, espagnol et anglais.

<h2> Cas possible </h2>
On créer une courbe en live en utilisant uniquement un jeu de train
On créer une courbe en live en utilisant uniquement un jeu de train + validation
On reprend le graphe la ou on l'a laissé

<h2> Descriptif du graphe </h2>
Le graphique est composé de 2 sous graphe :
- L'un deux affiche les courbe de loss pour pouvoir les comparer
- L'autre affiche les courbe d'accuracy

<h2> Contenu </h2>

<ol>
  <li> 
    first_reseau_de_neuronne est un code permettant de testé le callback.
    Il représente un exo de classification du probleme de xor à l'aide d'un MLP fait en keras.
  </li>
  <li> 
    Callback contient le code pour effectuer une real time learning curve.
    Elle possède une methode init et une methode on_epoch_end qui se lance une fois la tour de training + validation terminé afin de plot les new point de la courbe.
    Elle a uniquement besoin du chemin pour etre sauvegarde (pourrait etre mis en optionnel pour save à l'endroit de lancement du code)
    Et a des params optionnel comme:
    - le nom du schema
    - la derniere epoch qui permet de reprendre le dessin en cas de re-training
    - save_graph
    - show graph
    - blackBackgrounf : pour effectuer un graphe sous fond noir ou blanc.
  </li>
  <li>
    L'ancien readme + l'ancien callback : version de test fait à la vas vite pour comprendre comment les callback keras marche.
  </li>
</ol>


