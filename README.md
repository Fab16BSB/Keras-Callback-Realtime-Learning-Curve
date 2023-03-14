<h1> V1.0 : </h1>
<h2> Objectif </h2>
Concevoir un callback keras créant une learning curve en temps réel. <br>
Bien commenté le code <br>
Rendre le projet publique <br>
Documenté le repo en francais, espagnol et anglais <br>
Fournir les installation necessaire <br>

<h2> Cas possible </h2>
On créer une courbe en live en utilisant uniquement un jeu de train -> OK <br>
On créer une courbe en live en utilisant uniquement un jeu de train + validation -> OK <br>
On reprend le graphe la ou on l'a laissé -> Aucune erreur mais pas ok <br>
On demande de reprendre mais le graphe n'existe pas ou n'est pas trouvé => erreur de l'utilisateur mais le code marche <br>
On demande de reprendre mais on se trompe dans le numero de la der epoch et on compte -1 => erreur de l'utilisateur le code ne plante pas mais ne semble pas marcher correctement.
On demande de reprendre mais on se trompe dans le numero de la der epoch et on compte +1 => erreur de l'utilisateur le code ne plante pas mais ne semble pas marcher correctement.

<h2> Descriptif du graphe </h2>
Le graphique est composé de 2 sous graphe : <br>
- L'un deux affiche les courbe de loss pour pouvoir les comparer <br>
- L'autre affiche les courbe d'accuracy <br>
Exemple dispo dans le repo : LearningCurve_exemple.png

<h2> Contenu </h2>

<ol>
  <li> 
    first_reseau_de_neuronne est un code permettant de testé le callback. <br>
    Il représente un exo de classification du probleme de xor à l'aide d'un MLP fait en keras. <br> <br>
  </li>
  <li> 
    Callback contient le code pour effectuer une real time learning curve. <br>
    Elle possède une methode init et une methode on_epoch_end qui se lance une fois la tour de training + validation terminé afin de plot les new point de la courbe. <br>
    Elle a uniquement besoin du chemin pour etre sauvegarde (pourrait etre mis en optionnel pour save à l'endroit de lancement du code) <br>
    Et a des params optionnel comme: <br>
    - le nom du schema <br>
    - la derniere epoch qui permet de reprendre le dessin en cas de re-training <br>
    - save_graph <br>
    - show graph <br>
    - blackBackgrounf : pour effectuer un graphe sous fond noir ou blanc. <br><br>
  </li>
  <li>
    L'ancien readme + l'ancien callback : version de test fait à la vas vite pour comprendre comment les callback keras marche.
  </li>
</ol>


