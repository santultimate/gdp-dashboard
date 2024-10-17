import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from PIL import Image
from sklearn.preprocessing import StandardScaler


# Charger et afficher le logo
logo_path = "/workspaces/gdp-dashboard/data/Le_0.jpg"
logo = Image.open(logo_path)

# Appliquer le style Seaborn pour l'esthétique des graphiques
sns.set(style="whitegrid")

# Titre du projet
st.image(logo, width=300)  # Affiche le logo en haut de la page
st.title("Analyse Prédictive des Séries Temporelles : Comparaison entre ARIMA, SARIMA, et Prophet (ASAP)")
st.subheader("Projet Libre 41")

# Fonction pour charger les données de consultation d'une page Wikipedia via l'API Pageviews
def load_pageviews_from_wikipedia(page_title, start_date, end_date):
    headers = {
        'User-Agent': 'MyApp/1.0 (https://mywebsite.com/contact; myemail@example.com)'  # Adapt to your needs
    }

    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{page_title}/daily/{start_date}/{end_date}"
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        try:
            data = response.json()
            items = data['items']
            dates = [item['timestamp'][:8] for item in items]  # Extraction des dates (au format YYYYMMDD)
            views = [item['views'] for item in items]  # Extraction des vues
            df = pd.DataFrame({'date': dates, 'views': views})  # Création du DataFrame
            df['date'] = pd.to_datetime(df['date'])  # Conversion des dates en format datetime
            return df
        except ValueError:
            st.error("Erreur lors de la décodification des données JSON.")
            return None
    else:
        st.error(f"Erreur lors de la récupération des données (Code {response.status_code}): {response.text}")
        return None

# Charger un jeu de données
st.subheader("Téléchargez votre fichier de données pour l'analyse ou utilisez l'API Wikipedia")
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")

# Initialisation de la variable data
data = None

def preprocess_data(data):
    """
    Prétraiter les données en effectuant des opérations comme :
    - Renommage des colonnes
    - Gestion des valeurs manquantes
    - Conversion de types de données

    :param data: DataFrame contenant les données brutes
    :return: DataFrame prétraité
    """

    # Vérifier si 'date' est une colonne et la mettre comme index
    if 'date' in data.columns:
        data.set_index('date', inplace=True)
    
    # Gestion des valeurs manquantes (exemple : remplissage avec la moyenne)
    data.fillna(method='ffill', inplace=True)  # Remplir les valeurs manquantes par propagation
    data.fillna(data.mean(), inplace=True)  # Remplir les valeurs manquantes par la moyenne, si nécessaire

    # Assurez-vous que les types de données sont corrects
    # Ici, nous supposons que toutes les colonnes restantes doivent être des numériques
    data = data.apply(pd.to_numeric, errors='coerce')

    # Supprimer les lignes avec des valeurs manquantes restantes
    data.dropna(inplace=True)

    return data

# Sélection entre fichier CSV ou données Wikipedia
use_wikipedia = st.checkbox("Utiliser les données d'une page Wikipedia")

if uploaded_file is not None:
    # Chargement des données CSV locales
    data = pd.read_csv(uploaded_file, parse_dates=True, index_col=0)
    st.write("Aperçu des données:", data.head())

elif use_wikipedia:
    # Entrée pour le titre de la page Wikipedia et sélection des dates
    page_title = st.text_input("Entrez le titre de la page Wikipedia (ex: 'Python_(programming_language)')")

    start_date = st.date_input("Sélectionnez la date de début", value=pd.to_datetime("2023-01-01"))
    end_date = st.date_input("Sélectionnez la date de fin", value=pd.to_datetime("2023-12-31"))

    if page_title and start_date and end_date:
        start_date_str = start_date.strftime("%Y%m%d")
        end_date_str = end_date.strftime("%Y%m%d")

        # Récupération des données de la page Wikipedia
        data = load_pageviews_from_wikipedia(page_title, start_date_str, end_date_str)
        
        if data is not None:
            # Prétraitement des données : interpolation des valeurs manquantes
            data = preprocess_data(data)
            st.write("Données de consultation de la page (après prétraitement) :", data.head())

else:
    st.warning("Veuillez télécharger un fichier CSV ou utiliser les données d'une page Wikipedia.")

if data is not None:
    # Affichage de la matrice de corrélation
    st.subheader("Matrice de corrélation des données")
    correlation_matrix = data.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    

    # Suggestions de paramètres basées sur la matrice de corrélation
    st.subheader("Suggestions de paramètres pour ARIMA et SARIMA")

    # Test de stationnarité sur la série temporelle
    def test_stationarity(timeseries):
        dftest = adfuller(timeseries, autolag='AIC')
        return dftest[1]  # Retourner la p-value

    # Choix de la colonne à analyser
    column = st.selectbox("Sélectionnez la colonne pour l'analyse de séries temporelles", data.columns)
    
    # Test ADF (Augmented Dickey-Fuller) pour la stationnarité
    p_value = test_stationarity(data[column])

    if p_value > 0.05:
        st.write(f"La série '{column}' semble non stationnaire (p-value = {p_value:.4f}).")
        st.write("Il est recommandé d'appliquer une différenciation (d=1).")
        d_suggestion = 1
    else:
        st.write(f"La série '{column}' semble stationnaire (p-value = {p_value:.4f}).")
        st.write("Aucune différenciation n'est nécessaire (d=0).")
        d_suggestion = 0
    
    # Proposer des paramètres initiaux pour ARIMA et SARIMA
    st.write("**Suggestions de paramètres pour ARIMA :**")
    p_suggestion = st.slider("AR (p) : Autoregressive order", 0, 5, value=1)
    d_suggestion = st.slider("Différence (d)", 0, 2, value=d_suggestion)
    q_suggestion = st.slider("MA (q) : Moving Average order", 0, 5, value=1)

    st.write("**Suggestions de paramètres pour SARIMA :**")
    seasonal_period = st.slider("Période saisonnière (s)", 0, 24, value=12)
    P_suggestion = st.slider("AR saisonnier (P)", 0, 3, value=1)
    D_suggestion = st.slider("Différence saisonnière (D)", 0, 2, value=1)
    Q_suggestion = st.slider("MA saisonnier (Q)", 0, 3, value=1)

    st.write(f"Les paramètres proposés sont : p={p_suggestion}, d={d_suggestion}, q={q_suggestion} pour ARIMA.")
    st.write(f"Pour SARIMA : P={P_suggestion}, D={D_suggestion}, Q={Q_suggestion}, s={seasonal_period}.")
    
    # Visualisation des données
    st.subheader("Évolution graphique des données")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=data[column], ax=ax)
    ax.set_title("Évolution des données", fontsize=16)
    ax.set_xlabel("Date")
    ax.set_ylabel("Valeur")
    st.pyplot(fig)

# Prétraitement des données avec interpolation linéaire
def preprocess_data(df):
    df.set_index('date', inplace=True)  # Utiliser la date comme index
    df['views'] = df['views'].replace(0, np.nan)  # Remplacer les valeurs 0 par NaN
    df['views'].interpolate(method='linear', inplace=True)  # Interpolation linéaire des NaN
    return df


# Modèle ARIMA
def model_arima(data, p=5, d=1, q=0, steps=10):
    model = ARIMA(data, order=(p, d, q))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast, model_fit

# Modèle SARIMA
def model_sarima(data, p=1, d=1, q=1, P=1, D=1, Q=1, s=12, steps=10):
    model = SARIMAX(data, order=(p, d, q), seasonal_order=(P, D, Q, s))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast, model_fit

# Modèle Prophet
def model_prophet(data, steps=12):
    data.reset_index(inplace=True)
    data.columns = ['ds', 'y']
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=steps, freq='D')
    forecast = model.predict(future)
    return forecast, model

# Calcul du RMSE
def calculate_rmse(actual, predicted):
    min_len = min(len(actual), len(predicted))
    actual = actual[-min_len:]  # Adapter la taille des données réelles
    predicted = predicted[:min_len]  # Adapter la taille des prédictions
    return np.sqrt(mean_squared_error(actual, predicted))

# Interface Streamlit
st.title("Prédiction des vues d'une page Wikipedia")

# Entrée de l'utilisateur pour le titre de la page
page_title = st.text_input("Entrez le titre de la page Wikipedia (ex: 'Python_(programming_language)')")

# Sélection des dates de début et de fin
start_date = st.date_input("Sélectionnez la date de début", value=pd.to_datetime("2023-01-01"))
end_date = st.date_input("Sélectionnez la date de fin", value=pd.to_datetime("2023-12-31"))

# Si l'utilisateur entre une page et des dates valides, récupérer les données de consultation
if page_title and start_date and end_date:
    # Conversion des dates au format YYYYMMDD pour l'API Wikipedia
    start_date_str = start_date.strftime("%Y%m%d")
    end_date_str = end_date.strftime("%Y%m%d")
    
    # Récupérer les données de consultation de la page Wikipedia
    data = load_pageviews_from_wikipedia(page_title, start_date_str, end_date_str)
    
    if data is not None:
        # Prétraitement des données : interpolation des valeurs manquantes
        data = preprocess_data(data)

        # Afficher un aperçu des données
        st.write("Données de consultation de la page (après prétraitement) :", data.head())

        # Afficher l'évolution des données
        st.write("Évolution des données de consultation :")
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=data.index, y=data['views'], label="Données historiques", color='#3498db')
        plt.xlabel("Date")
        plt.ylabel("Vues")
        plt.title(f"Évolution des vues pour {page_title}")
        plt.legend()
        st.pyplot(plt)
        # Suggestions de paramètres basées sur la matrice de corrélation
    st.subheader("Suggestions de paramètres pour ARIMA et SARIMA")

    # Test de stationnarité sur la série temporelle
    def test_stationarity(timeseries):
        dftest = adfuller(timeseries, autolag='AIC')
        return dftest[1]  # Retourner la p-value

    # Choix de la colonne à analyser
    column = st.selectbox("Sélectionnez la colonne pour l'analyse de séries temporelles", data.columns)
    
    # Test ADF (Augmented Dickey-Fuller) pour la stationnarité
    p_value = test_stationarity(data[column])

    if p_value > 0.05:
        st.write(f"La série '{column}' semble non stationnaire (p-value = {p_value:.4f}).")
        st.write("Il est recommandé d'appliquer une différenciation (d=1).")
        d_suggestion = 1
    else:
        st.write(f"La série '{column}' semble stationnaire (p-value = {p_value:.4f}).")
        st.write("Aucune différenciation n'est nécessaire (d=0).")
        d_suggestion = 0
    
    # Proposer des paramètres initiaux pour ARIMA et SARIMA
    st.write("**Suggestions de paramètres pour ARIMA :**")
    p_suggestion = st.slider("AR (p) : Autoregressive order", 0, 5, value=1)
    d_suggestion = st.slider("Différence (d)", 0, 2, value=d_suggestion)
    q_suggestion = st.slider("MA (q) : Moving Average order", 0, 5, value=1)

    st.write("**Suggestions de paramètres pour SARIMA :**")
    seasonal_period = st.slider("Période saisonnière (s)", 0, 24, value=12)
    P_suggestion = st.slider("AR saisonnier (P)", 0, 3, value=1)
    D_suggestion = st.slider("Différence saisonnière (D)", 0, 2, value=1)
    Q_suggestion = st.slider("MA saisonnier (Q)", 0, 3, value=1)

    st.write(f"Les paramètres proposés sont : p={p_suggestion}, d={d_suggestion}, q={q_suggestion} pour ARIMA.")
    st.write(f"Pour SARIMA : P={P_suggestion}, D={D_suggestion}, Q={Q_suggestion}, s={seasonal_period}.")
    
        # Sélection des paramètres des modèles avant la prédiction
    st.subheader("Paramètres des modèles")

    # Description des paramètres pour l'utilisateur
    st.write("""
            **ARIMA** : (p, d, q) où :
            - p : le nombre de termes autorégressifs
            - d : le nombre de différences non saisonnières
            - q : le nombre de termes de moyenne mobile

            **SARIMA** : (p, d, q) * (P, D, Q, s) où :
            - P, D, Q : termes saisonniers (autorégressif, différenciation, moyenne mobile)
            - s : période saisonnière
        """)

        # Paramètres pour ARIMA
    st.write("Paramètres ARIMA")
    p = st.number_input("Paramètre p (ARIMA)", value=5)
    d = st.number_input("Paramètre d (ARIMA)", value=1)
    q = st.number_input("Paramètre q (ARIMA)", value=0)
    steps = st.number_input("Pas de prédiction", value=10)

        # Paramètres pour SARIMA
    st.write("Paramètres SARIMA")
    p_sarima = st.number_input("Paramètre p (SARIMA)", value=1)
    d_sarima = st.number_input("Paramètre d (SARIMA)", value=1)
    q_sarima = st.number_input("Paramètre q (SARIMA)", value=1)
    P_sarima = st.number_input("Paramètre P (saisonnier)", value=1)
    D_sarima = st.number_input("Paramètre D (différenciation saisonnière)", value=1)
    Q_sarima = st.number_input("Paramètre Q (erreur saisonnière)", value=1)
    s_sarima = st.number_input("Saisonnalité (SARIMA)", value=12)

        # Bouton pour générer les prédictions
    if st.button("Générer les prédictions"):
            # ARIMA
            arima_forecast, arima_model_fit = model_arima(data['views'], p, d, q, steps)
            st.write("Résumé du modèle ARIMA :")
            st.text(arima_model_fit.summary())  # Affichage du summary ARIMA

            # SARIMA
            sarima_forecast, sarima_model_fit = model_sarima(data['views'], p_sarima, d_sarima, q_sarima, P_sarima, D_sarima, Q_sarima, s_sarima, steps)
            st.write("Résumé du modèle SARIMA :")
            st.text(sarima_model_fit.summary())  # Affichage du summary SARIMA

            # Prophet
            prophet_forecast, prophet_model = model_prophet(data[['views']], steps)
            st.write("Prédiction Prophet :")
            st.write(prophet_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

            # Comparaison des modèles avec Seaborn
            plt.figure(figsize=(10, 6))
            sns.lineplot(x=data.index, y=data['views'], label="Données historiques", color='#3498db')
            sns.lineplot(x=pd.date_range(start=data.index[-1], periods=steps), y=arima_forecast, label="Prédiction ARIMA", color='#e74c3c')
            sns.lineplot(x=pd.date_range(start=data.index[-1], periods=steps), y=sarima_forecast, label="Prédiction SARIMA", color='#2ecc71')
            sns.lineplot(x=prophet_forecast['ds'][-steps:], y=prophet_forecast['yhat'][-steps:], label="Prédiction Prophet", color='#9b59b6')
            plt.xlabel("Date")
            plt.ylabel("Vues")
            plt.title(f"Comparaison des modèles pour {page_title}")
            plt.legend()
            st.pyplot(plt)

            # Calcul des RMSE pour chaque modèle
            rmse_arima = calculate_rmse(data['views'], arima_forecast)
            rmse_sarima = calculate_rmse(data['views'], sarima_forecast)
            rmse_prophet = calculate_rmse(data['views'], prophet_forecast['yhat'])

            # Affichage des RMSE
            st.write(f"RMSE ARIMA : {rmse_arima}")
            st.write(f"RMSE SARIMA : {rmse_sarima}")
            st.write(f"RMSE Prophet : {rmse_prophet}")


            # Conclusion
            if min(rmse_arima, rmse_sarima, rmse_prophet) == rmse_arima:
                st.write("Le modèle ARIMA semble offrir la meilleure performance en termes de précision (basé sur le RMSE).")
            elif min(rmse_arima, rmse_sarima, rmse_prophet) == rmse_sarima:
                st.write("Le modèle SARIMA semble offrir la meilleure performance en termes de précision (basé sur le RMSE).")
            else:
                st.write("Le modèle Prophet semble offrir la meilleure performance en termes de précision (basé sur le RMSE).")
