import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os import listdir
import pandas as pd
import time


liste=pd.read_csv("../data/liste_entreprises_banque.csv")
df = pd.read_csv("../data/avis/general_df.csv")
df_cleaned = pd.read_csv("../data/avis/df_cleaned.csv")

st.title("Projet d'analse des avis et verbatim")
st.sidebar.title("Sommaire")
pages=["I.	Introduction et objectif du projet", "II.	Préparations des données ", "III.	Analyses descriptives des données", "IV.	Modélisations"]
page=st.sidebar.radio("Aller vers", pages)


if page == pages[0] : 
    st.write("### Introduction et objectif")
    st.write("Faire des modélisations à partir des avis et verbatim à partir du site 'fr.trustpilot.com/categories/bank' afin d'analyser les verbatim et le lien entre les notations et les verbatim")
    st.write("- Prédire la satisfaction d'un client : problème de régression et Entraînement supervisé possible.")
    st.write("- Identifier les points clés des avis : localisation, nom d'entreprise... ")
    st.write("- Extraire les propos du commentaire et trouver les mots importants : problème de livraison, article défectueux... avec l'approche non supervisée avec CamemBert")
    st.write("- Trouver une réponse rapide adaptée pour répondre au commentaire, par exemple sur les reviews Google")

    st.write("### Aperçu de la base de données téléchargée")

    st.dataframe(liste.head())
    st.dataframe(df.head())
    st.write(df.shape)
    st.dataframe(df.describe())

    if st.checkbox("Afficher les NA") :
        st.dataframe(df.isna().sum())



if page == pages[1] : 
    st.write("### Préparations des données")
    st.write("Nous avons retiré du dataset les avis liés à l'une des banques qui semblait émaner de bots. Cette société a environs 60 000 avis avec la grande majorité de 5 étoiles. Nous avons fait le split des données train et test avant de faire une sélection équilibrée des étoiles (pour éviter un leaking de la structure des résultats attendus en test dans l'entrainement) en se basant sur un tirage aléatoire égal au nombre de messages présents dans la classe la plus minoritaire. Le dataset retenu faisait 15 000 avis, dont 30% du jeu destiné au jeu de test. Nous avons fait un benchmark pour évaluer la performance de différents modèles : SVM, Random Forests, XGBOOST, KNN, SVC, Logistic Regression et CAMEMBERT.")
    st.write("Pour chaque modèle, nous avons testé les features numériques uniquement d'une part : le nombre d'avis, le sentiment (score inféré par CAMEMEMBERT) et la longueur de l'avis, et d'autre part les features numériques et le texte de l'avis (concaténation du titre et de l'avis). ")
    st.write("Pour chaque modèle (excepté CAMEMBERT) nous avons lancé un entrainement avec les hyper paramètres par défaut puis lancé une grille de recherche des meilleurs paramètres.")

    st.write("### Aperçu de la base de données nettoyées")

    st.dataframe(df_cleaned.head())


if page == pages[2] :
    st.write("###    Analyses descriptives des données")
    
    fig = plt.figure()
    top_20_banques = df_cleaned["Société"].value_counts().head(20).index
    filtered_df = df_cleaned[df_cleaned['Société'].isin(top_20_banques)]
    sns.countplot(y='Société', data=filtered_df, order=top_20_banques)
    plt.title("Top 20 des banques par nombre d'avis")  
    st.pyplot(fig)

    fig = plt.figure()
    filtered_avis = df_cleaned[(df_cleaned['n_avis'] >= 1) & (df_cleaned['n_avis'] <= 10)]
    sns.countplot(x='n_avis', data=filtered_avis)
    plt.title("Distribution du nombre d'avis donnés par utilisateur (1 à 10 avis)")
    st.pyplot(fig) 

    fig = plt.figure()
    sns.countplot(y='localisation', data=df_cleaned, order=df_cleaned['localisation'].value_counts().iloc[:10].index)  # top 10 localisations
    plt.title('Top 10 des localisations des utilisateurs')
    st.pyplot(fig) 

    fig = plt.figure()
    df_cleaned['length_avis'] = df_cleaned['text_total'].apply(len)
    sns.boxplot(x='etoiles', y='length_avis', data=df_cleaned)
    plt.title('Longueur des avis par notes')
    st.pyplot(fig)

if page == pages[3] :
    st.write("###   Résultat du benchmark")
    st.write("Pour chaque modèle, nous avons testé les features numériques uniquement d'une part : le nombre d'avis, le sentiment (score inféré par CAMEMEMBERT) et la longueur de l'avis, et d'autre part les features numériques et le texte de l'avis (concaténation du titre et de l'avis).")
    st.write("Le modèle de deep learning Camembert a donné les meilleurs résultats. Sur les données d'entrainement il a atteint une précision, un recall et un f1 de 0.63 chacun. Le deuxième modèle le plus performant a été Random Forest avec un f1 de 0.55. Ce score a été obtenu sur les features numériques uniquement et par une grille qui a retenu les paramètres suivants : 'max_depth': None, 'min_samples_leaf': 2, 'min_samples_split': 20, 'n_estimators': 100. Le troisième modèle le plus performant suit de près le deuxième, il s’agit de XGBOOST avec un score de 0.54 obtenu sur les features numériques et les hyperparamètres par défaut, la recherche par grille a donné le même score f1.")
    bench_list = listdir("../reports/benchmark")
    # bench_list.remove(".DS_Store")

    benchmark = pd.DataFrame()

    for file in bench_list:
        df = pd.read_csv(f'../reports/benchmark/{file}', index_col=0)
        benchmark = pd.concat([benchmark, df])

    new_cols = ['model', 'grid search', 'score', 'precision', 'recall', 'f1', 'time_taken_mns', 'run_date', 'used/best params']
    benchmark = benchmark.reindex(columns=new_cols)
    benchmark = benchmark.sort_values("f1", ascending = False)
    benchmark= benchmark[benchmark.score!='na']
    benchmark

    st.write("###   Interprétation des résultats")
    st.write("Le score f1 de 0.63 sur 5 classes est trois fois plus performant qu'une classification au hasard. Sans surprise c'est le modèle de deep learning basé sur l'architecture Transformers qui atteint le meilleur score. Dans l'absolu cependant ce score n'est pas optimal, idéalement notre score aurait dû se situer au-dessus de 0.75. Cependant la prédiction d'étoile est par nature très délicate, d’une part parce qu’il s'agit d'interpréter des données non structurées (du texte) et d'autre part car l'appréciation des étoiles peut varier d'une personne à l'autre. Par exemple certains usagers peuvent estimer, selon l'adage scolaire, qu'un score parfait (20/20 ou 5 étoiles sur 5) n'existe pas, et vont donc donner 4 étoiles alors que d'autres utilisateurs pour une satisfaction similaire en mettraient 5. De même la différence dans le « ventre mou », entre 2 et 3 ; 3 et 4 étoiles peut être sujette à des variations interpersonnelles importantes. Dans l'ensemble et malgré un score non optimal, nous sommes satisfaits de la performance du modèle Camembert. Nous devons aussi noter ici que les modèles de machine learning utilisés ont également bénéficié de la puissance de Camembert puisqu'ils utilisaient un score de sentiment inféré par ce modèle, mais même dans ces cas-là l'inférence de Camembert sur le texte a donné de meilleurs résultats.")
    st.write("###   Labelisation")
    st.write("Nous avons fait une classification des sentiments des utilisateurs concernant la communication, l'efficacité et la valeur ajoutée par similarité sémantique. Par la suite nous avons entrepris de caractériser les arguments que les usagers invoquent pour expliquer leur notation, afin de dégager les aspects positifs et négatifs des services, qui pourraient être utiles pour augmenter leur qualité et la satisfaction des clients.")
    df = pd.read_excel("../data/labelisation.xlsx", header=0, index_col=0)
    st.dataframe(df.head(7))

    st.write("###   Résultat des différents traitements")
    st.write()

    bench= pd.read_csv("../reports/similarity/best_validation_params.csv", index_col=0)
    st.dataframe(bench.head(7))    

    #test camembert 