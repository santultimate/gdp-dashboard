Prédiction de Séries Temporelles avec ARIMA, SARIMA et Prophet
Bienvenue dans ce projet de prédiction de séries temporelles, où nous explorons trois modèles principaux : ARIMA, SARIMA et Prophet. Ce projet combine des techniques de statistiques classiques et de machine learning pour prédire des données temporelles avec précision, tout en offrant une interface utilisateur interactive grâce à Streamlit.

Table des Matières
Introduction
Prérequis
Installation
Utilisation
Structure du Projet
Modèles Utilisés
Contributions et Formations
Auteur
Licence
Introduction
Ce projet a pour objectif de prédire des séries temporelles en utilisant trois modèles :

ARIMA : Un modèle statistique pour les séries non stationnaires.
SARIMA : Une version saisonnière d'ARIMA pour gérer des données avec des tendances périodiques.
Prophet : Un modèle de machine learning développé par Facebook, adapté à des séries plus complexes avec des vacances ou autres effets externes.
Le projet comprend également une application Streamlit qui permet aux utilisateurs d'interagir facilement avec les données et les prédictions, et offre un outil pédagogique pour ceux qui souhaitent apprendre à construire leurs propres applications de Data Science.

Prérequis
Avant de commencer, assurez-vous d'avoir les éléments suivants :

Python 3.7+
Les bibliothèques suivantes installées :
pandas
numpy
matplotlib
seaborn
statsmodels
fbprophet (Prophet)
streamlit
sklearn
Environnement virtuel (optionnel mais recommandé)
Pour éviter les conflits de dépendances, il est conseillé de créer un environnement virtuel :

bash
Copy code
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
Installation
Clonez le dépôt GitHub et installez les dépendances.

bash
Copy code
git clone https://github.com/votre-utilisateur/prediction-series-temporelles.git
cd prediction-series-temporelles
pip install -r requirements.txt
Utilisation
Streamlit Application
Pour lancer l'application interactive Streamlit qui permet d'explorer les modèles et de faire des prédictions en temps réel, exécutez simplement la commande suivante dans le terminal :

bash
Copy code
streamlit run app.py
Jupyter Notebook
Si vous préférez explorer et ajuster le code dans un Jupyter Notebook, ouvrez-le avec la commande suivante :

bash
Copy code
jupyter notebook
Exécution des Modèles
Vous pouvez tester individuellement les modèles ARIMA, SARIMA, et Prophet en utilisant le fichier model_comparison.ipynb pour ajuster les paramètres et observer les résultats.

Structure du Projet
bash
Copy code
├── data/                   # Dossier pour les données d'entrée
├── app.py                  # Application Streamlit
├── model_comparison.ipynb   # Comparaison des modèles ARIMA, SARIMA et Prophet
├── utils.py                # Fonctions utilitaires pour prétraitement des données
├── requirements.txt        # Dépendances Python
└── README.md               # Ce fichier
Modèles Utilisés
ARIMA : Analyse des valeurs passées pour prédire des séries non stationnaires.
SARIMA : Gestion des séries saisonnières pour capturer les effets périodiques.
Prophet : Utilisation de Prophet pour modéliser les séries temporelles complexes avec des composants additifs.
Le projet inclut une fonction de grid search pour trouver les hyperparamètres optimaux des modèles ARIMA et SARIMA en fonction du RMSE le plus bas.

Contributions et Formations
Ce projet est également un outil pédagogique conçu pour aider ceux qui souhaitent apprendre les bases de la modélisation de séries temporelles et le développement d’applications interactives en Streamlit.

Les collègues ou étudiants intéressés par la Data Science pourront non seulement comprendre les concepts théoriques, mais aussi apprendre à déployer des outils d’analyse interactifs. N'hésitez pas à contribuer ou à organiser des sessions de formation à partir de ce projet.

Auteur
Nom : SANTARA
Email : yacoubasantara@yahoo/fr

