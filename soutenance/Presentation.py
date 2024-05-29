import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os import listdir
#python -m spacy download fr_core_news_lg
import spacy
#pip install time
#pip install warnings
import time
import warnings 
import platform

if platform.system() == "Darwin":  # macOS
    key_path = "/Users/lheyerda/Documents/GitHub/juil23_cds_supply_chain"
else:
    key_path = ".." 

# cache

@st.cache_data
def load_df():
    df = pd.read_csv(f"{key_path}/data/avis/general_df.csv", index_col=0)
    return df

@st.cache_data
def load_clean_df():
    df_cleaned = pd.read_csv(f"{key_path}/data/avis/df_cleaned.csv", index_col=0)
    return df_cleaned

@st.cache_data
def load_nlp():
    nlp = spacy.load("fr_core_news_lg")
    return nlp

@st.cache_data
def load_entreprises():
    ent = pd.read_csv(f"{key_path}/data/liste_entreprises_banque.csv", index_col=0)
    return ent

entreprises = load_entreprises()
df = load_df()
#df=df.iloc[:,1:]

df_cleaned = load_clean_df()
#tab=pd.read_csv(f"{key_path}/soutenance/describe_avis.csv")

nlp = load_nlp()

st.title("Projet d'analse des avis et verbatim")
st.sidebar.title("Sommaire")



pages=["I.	Introduction et objectif du projet", "II.	Préparations des données ", "III.	Analyses descriptives des données", "IV.	Modélisations I", "V.	Modélisations II","VI.	Prédiction"]
page=st.sidebar.radio("Aller vers", pages)
st.sidebar.write("### équipe:")
st.sidebar.write("[Léonard Heyerdahl](https://www.linkedin.com/in/leonardo-heyerdahl/)")
st.sidebar.write("Alexis Garatti")
st.sidebar.write("Huazhen Hou")
st.sidebar.write("Alexandre PRZYBYLSKI")



# PAGE 1 
if page == pages[0] : 
    st.write("### Introduction et objectif")
    st.write("Faire des modélisations à partir des avis et verbatim à partir du site 'fr.trustpilot.com/categories/bank' afin d'analyser les verbatim et le lien entre les notations et les verbatim")
    st.write("- Prédire la satisfaction d'un client : problème de régression et Entraînement supervisé possible.")
    st.write("- Identifier les points clés des avis : localisation, nom d'entreprise... ")
    st.write("- Extraire les propos du commentaire et trouver les mots importants : problème de livraison, article défectueux... avec l'approche non supervisée avec CamemBert")
    st.write("- Trouver une réponse rapide adaptée pour répondre au commentaire, par exemple sur les reviews Google")


    st.write('### Aperçu des banques listées sur Trust Pilot (scrap)')
    st.write(entreprises)

    st.write("### Aperçu de la base de données des avis scrappés")
    st.write(f"Nombre d'avis scrappés: **{len(df)}**")
    st.dataframe(df)
    #st.dataframe(df.etoiles.describe())

    if st.checkbox("Afficher les NA") :
        st.dataframe(df.isna().sum())



if page == pages[1] : 
    st.write("### Préparations des données")
    st.write("""
- **Retrait des avis liés à une banque suspectée d'utiliser des bots**:
  - Environ 60 000 avis, majorité de 5 étoiles.
- **Nettoyage des données**:
  - Retrait des na et des entrées de type incohérent (str là ou données numériques attendues)
  - Conversion des dates au format datetime
- **Renforcement du dataframe**:
  - Ajout d'une colone contenant la longueur de l'avis en charactères
  - Ajout d'une colone du score de sentiment (probabilité x label postif (1) ou négatif (-1)) inféré par CAMEMBERT
  - Concaténation du *titre de l'avis* et de *l'avis*, pour regrouper les données textuelles existantes
""")
    st.write("### Aperçu de la base de données nettoyée")

    st.dataframe(df_cleaned.head())


if page == pages[2] :
    st.write("###    Analyses descriptives des données")
    
    fig = plt.figure()
    top_20_banques = df_cleaned["page"].value_counts().head(20).index
    filtered_df = df_cleaned[df_cleaned['page'].isin(top_20_banques)]
    sns.countplot(y='page', data=filtered_df, order=top_20_banques)
    plt.title("Top 20 des banques par nombre d'avis")  
    st.pyplot(fig)

    fig = plt.figure()
    filtered_avis = df_cleaned[(df_cleaned['n_avis'] >= 1) & (df_cleaned['n_avis'] <= 10)]
    sns.countplot(x='n_avis', data=filtered_avis)
    plt.title("Distribution du nombre d'avis donnés par utilisateur (1 à 10 avis)")
    st.pyplot(fig) 

    #fig = plt.figure()
    #sns.countplot(y='localisation',
                #data=df_cleaned,
                #order=df_cleaned['localisation'].value_counts().iloc[:10].index)  # top 10 localisations
    #plt.title('Top 10 des localisations des utilisateurs')
    #st.pyplot(fig) 
    

    fig = plt.figure()
    df_cleaned['length_avis'] = df_cleaned['text_total'].apply(len)
    sns.boxplot(x='etoiles', y='length_avis', data=df_cleaned)
    plt.title('Longueur des avis par notes')
    st.pyplot(fig)

if page == pages[3] :
    st.write("###   Modélisation")
    st.write ("""
              - **Split des données en train et test avant la sélection équilibrée des étoiles**:
  - Basé sur un tirage aléatoire égal au nombre de messages dans la classe la plus minoritaire.
- **Dataset final** : 15 000 avis.
  - 30% destinés au jeu de test.
- **Benchmark pour évaluer différents modèles** :
  - SVM
  - Random Forests
  - XGBOOST
  - KNN
  - SVC
  - Logistic Regression
  - CAMEMBERT
- **Tests pour chaque modèle** :
  - Features numériques uniquement : nombre d'avis, sentiment (score inféré par CAMEMBERT), longueur de l'avis.
  - Features numériques et texte de l'avis (concaténation du titre et de l'avis).
- **Entraînement avec hyperparamètres par défaut pour chaque modèle, suivi d'une grille de recherche des meilleurs paramètres (sauf CAMEMBERT)**.
""")
    st.write("Le modèle de deep learning d'architecture Transformer Camembert a donné les meilleurs résultats. Sur les données de test il a atteint une précision, un recall et un f1 de 0.63 chacun. Le deuxième modèle le plus performant a été Random Forest avec un f1 de 0.55. Ce score a été obtenu sur les features numériques uniquement et par une grille qui a retenu les paramètres suivants : 'max_depth': None, 'min_samples_leaf': 2, 'min_samples_split': 20, 'n_estimators': 100. Le troisième modèle le plus performant suit de près le deuxième, il s’agit de XGBOOST avec un score de 0.54 obtenu sur les features numériques et les hyperparamètres par défaut, la recherche par grille a donné le même score f1.")
    st.dataframe(pd.read_csv(f"{key_path}/soutenance/benchmark_modelisationI.csv", index_col=0).iloc[0:3,:])
    all_models = st.checkbox("visualiser tout le benchmark")
    if all_models:
        st.dataframe(pd.read_csv(f"{key_path}/soutenance/benchmark_modelisationI.csv", index_col=0))


    
    st.write("###   Interprétation des résultats")
    st.write("Le score f1 de 0.63 sur 5 classes est trois fois plus performant qu'une classification au hasard. Sans surprise c'est le modèle de deep learning basé sur l'architecture Transformers qui atteint le meilleur score. Dans l'absolu cependant ce score n'est pas optimal, idéalement notre score aurait dû se situer au-dessus de 0.75. Cependant la prédiction d'étoile est par nature très délicate, d’une part parce qu’il s'agit d'interpréter des données non structurées (du texte) et d'autre part car l'appréciation des étoiles peut varier d'une personne à l'autre. Par exemple certains usagers peuvent estimer, selon l'adage scolaire, qu'un score parfait (20/20 ou 5 étoiles sur 5) n'existe pas, et vont donc donner 4 étoiles alors que d'autres utilisateurs pour une satisfaction similaire en mettraient 5. De même la différence dans le « ventre mou », entre 2 et 3 ; 3 et 4 étoiles peut être sujette à des variations interpersonnelles importantes. Dans l'ensemble et malgré un score non optimal, nous sommes satisfaits de la performance du modèle Camembert. Nous devons aussi noter ici que les modèles de machine learning utilisés ont également bénéficié de la puissance de Camembert puisqu'ils utilisaient un score de sentiment inféré par ce modèle, mais même dans ces cas-là l'inférence de Camembert sur le texte a donné de meilleurs résultats.")
    st.write("### Focus sur le résultat Random Forest")
    st.write("""
Nous utilisons aussi la technique TF-IDF pour vectoriser le texte des avis (limité à 1000 caractéristiques pour simplifier) et préparer les données pour un modèle de forêt aléatoire. Il divise les données en ensembles d'entraînement et de test, entraîne le modèle sur l'ensemble d'entraînement, puis prédit les résultats sur l'ensemble de test. Nous obtenons les résultats suivants :
""")
    st.image(f"{key_path}/soutenance/crosstab_RF.png")
    st.write("""Le modèle semble bien performer pour les classes extrêmes (1 et 5 étoiles), mais il a du mal avec les classes intermédiaires.
    La matrice de confusion normalisée affichée ci-dessus indique les performances du modèle Random Forest sur l'ensemble de test. Les valeurs de la matrice sont normalisées par le nombre d'observations réelles pour chaque classe, ce qui nous permet de voir la proportion des prédictions correctes par rapport au total des cas pour chaque classe réelle.

             """)
    


if page == pages[4] :

    st.write("### Modélisation II classification des sentiments (Communication, Efficacité, Valeur économique)")
    st.write("""
- **Classification des sentiments des utilisateurs concernant** :
  - Communication
  - Efficacité
  - Valeur ajoutée
  - **Méthode** : Similarité sémantique.
- **Caractérisation des arguments des usagers pour expliquer leur notation** :
  - Identification des aspects positifs et négatifs des services.
  - Objectif : Augmenter la qualité des services et la satisfaction des clients.
""")
   
    tableau_labels = st.dataframe(pd.read_excel(f"{key_path}/soutenance/tableau_labels.xlsx"))
    
    st.write("""
- **Notre benchmark évaluera plusieurs stratégies** :
  - **Granularité des éléments comparés** :
    - Phrases de l'avis contre phrase des exemples de référence
    - Phrases de l'avis vs liste entière des exemples
    - Avis complet vs liste entière des exemples
  - **Utilisation de l'avis** :
    - Sans filtre de mots vides
    - Avec un filtre de mots vides
    - Avec un filtre sur les phrases positives/négatives
    - Sans filtre de sentiment
- **Seuil optimal de similarité**:
    - Tester un seuil de similarité entre 0.65 et .99
""")
    st.write("Un premier test sur les données d'entrainement montre de meilleures performances en utilisant l'avis sans filtre de mots vides et avec un filtre de sentiment")
    baseline_banchmark_sem= pd.read_csv(f"{key_path}/soutenance/baseline_semantic_bench.csv")
    fig_sem_bench_test = pd.read_csv(f"{key_path}/soutenance/benchmark_sem_test.csv", index_col=0)


    plt_1 = plt.figure(figsize=(12,10))
    plt.plot(baseline_banchmark_sem.threshold,baseline_banchmark_sem.Stop_words, label ="stop words", color ="indianred")
    plt.plot(baseline_banchmark_sem.threshold,baseline_banchmark_sem.Stop_words_sents, label ="stop words + sentiment", color ="indianred",linestyle="--")
    plt.plot(baseline_banchmark_sem.threshold, baseline_banchmark_sem.All_words, label = "all words", color="seagreen")
    plt.plot(baseline_banchmark_sem.threshold, baseline_banchmark_sem.All_words_sents, label ="all words + sentiment", linestyle="--", color="seagreen")
    plt.grid(True, which='both', linestyle=':', linewidth=0.5)
    plt.xlabel("similarity threshold used")
    plt.ylabel("f1 score")
    plt.legend()

    st.pyplot(plt_1)

    
    def heat(train_test):
        # pivot table heatmap
        benchmark = pd.read_csv(f"{key_path}/soutenance/benchmark_sem_{train_test}.csv")
        pivot = benchmark.pivot_table(index='code_mode', columns='test', values='f1')
        fig_sem_bench = plt.figure(figsize=(12, 16))  
        heatmap = sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0, linecolor='grey')

        # horizontal lines 
        for i in range(0, len(pivot.index), 4):
            plt.axhline(i, color='white', linestyle=':', linewidth=2)

        # Label the color bar
        color_bar = heatmap.collections[0].colorbar
        color_bar.set_label('f1')

        plt.title(f'Heatmap of best {train_test} F1 Scores')
        heatmap.xaxis.tick_top() 
        plt.xticks(rotation=45, ha='left')
        return fig_sem_bench 
    
    st.write("#### Résultats benchmark sur données d'entrainement")
    st.write("nous avons testé l'ensemble des stratégies pour les 6 labels sur les données d'entrainement:")
    st.image(f"{key_path}/soutenance/heatmap_train.png", caption='') 

    #fig_sem_bench_train = heat("train")
    #st.pyplot(fig_sem_bench_train)

    st.write("#### Résultats benchmark sur données de validation")
    #fig_sem_bench_val = heat("validation")
    #st.pyplot(fig_sem_bench_val)
    st.write("Puis nous avons testé nos stratégies sur des données de validation, qui incluent des messages inédits")
    st.image(f"{key_path}/soutenance/heatmap_validation.png", caption='')

    val_best_params = pd.read_csv(f"{key_path}/soutenance/best_params_validation.csv")
    st.write('Pour la validation, la meilleure combinaison était:')
    st.dataframe(val_best_params)

    st.write("#### Evaluation des résultats avec les données test")
    st.write('Nous avons repris les meilleurs paramètres puis testé leur efficacité sur un dernier jeu de test:')
    st.dataframe(fig_sem_bench_test)

    st.write("""
- **Bien que l'utilisation des mots vides et le filtrage par sentiment aient été les moins efficaces lors de la phase d'entraînement** :
  - Devenus des atouts précieux lors des phases de validation et de test face à des données inédites.
- **Nos tests finaux montrent des résultats contrastés** :
  - **Scores F1 élevés** :
    - bad_com : 0.80
    - bad_efficacy : 0.71
  - **Scores moyens** :
    - good_com : 0.60
    - good_efficacy : 0.60
  - **Scores bas** :
    - good_value : 0.57
    - bad_value : 0.54
  - Tous ces scores sont nettement meilleurs que le hasard (0.2), mais devraient idéalement être plus élevés.
- **Interprétation des faibles scores de good_value et bad_value** :
  - Manque de cohérence des espaces sémantiques auxquels ils se rapportent.
  - Regroupement de sujets variés (frais, taux d'intérêt, prêts) sous les étiquettes good_value et bad_value.
  - **Suggestion rétrospective** : Garder des étiquettes séparées pour ces différents sujets aurait pu donner des scores de précision individuellement plus élevés.
""")
    st.write("###   Classification sur l'ensemble des avis")
    st.write("Nous avons effectué l'étiquetage sur l'ensemble du jeu de données (105000 avis) et nous avons veillé à retirer l'étiquetage des avis où deux étiquettes opposées existaient, pour ne tenir compte que des opinions clairement tranchées. Sur la page suivante, vous trouverez la représentation des entreprises bancaires ayant plus de mille avis.")

if page == pages[5]:
    st.write("### Prédiction")
    st.write("Nous avons effectué l'étiquetage sur l'ensemble du jeu de données (environs 100 000 avis) et nous avons veillé à retirer l'étiquetage des avis où deux étiquettes opposées existaient, pour ne tenir compte que des opinions clairement tranchées.")
    st.write("En utilisant la proximité sémantique et des références issues d'une labelisation à la main, nous pouvons prédire les sentiment des avis concernant la Communication, la Valeur ajoutée et l'Efficacité du service bancaire.")


    df = pd.read_csv(f"{key_path}/soutenance/df_sim_small.csv")
    references = pd.read_csv(f"{key_path}/soutenance/references_bag.csv")


    def allocate_lab(sim_score, y_pred, thresh):
        #print("thresh", thresh, "sim_score", sim_score)
        try:
            if max(sim_score) > thresh:
                y_pred.append(1)
                #print(max(sim_score), "over threshold", round(thresh,2), "y_pred = 1")
            else:
                y_pred.append(0) 
                #print(max(sim_score), "under threshold", round(thresh,2), "y_pred = 0")
        except:
            y_pred.append(0)
            #print("error, y_pred = 0")
        return y_pred

    # review sentence vs reference sentence
    def decision_sentiment(sentiment,label_of_interest, word_mode):
        # determining if using stop words or full version of reference texts
        if "stop" in word_mode:
            ref_bag = label_of_interest+"_stop"
        elif "all" in word_mode:
            ref_bag = label_of_interest
        else:
            print("error in word_mode?", word_mode)

        # determining if using sentiment filtered review sentences or all reviews sentences
        if sentiment == True:
            if "good" in label_of_interest:
                if "stop" in word_mode:
                    reviews= "text_clean_sentences_pos_stop"
                else:
                    reviews =  "text_clean_sentences_pos"
            elif "bad" in label_of_interest:
                if "stop" in word_mode:
                    reviews =  "text_clean_sentences_neg_stop"
                else:
                    reviews =  "text_clean_sentences_neg"
            else:
                print("mispelling in label_of_interest?", label_of_interest)

        elif sentiment == False:
            # for some reason the original, uncleaned punctuation versions give much better accuracy
            if "stop" in word_mode:
                reviews = "text_stop"
            else:
                reviews = "text_total"
                
        else :
            print("error: no sentiment / granularity found")

        return reviews, ref_bag

    def review_sentence_vs_ref_sentence(i,df,label_of_interest, thresh,word_mode, sentiment=True):
        test_name = "review_sentence_vs_ref_sentence"
        #print("passing",test_name) 
        reviews,ref_bag = decision_sentiment(sentiment,label_of_interest, word_mode)
        ref_bag = nlp(references[ref_bag].iloc[0])
        y_pred = [] 
        sim_score = []
        try:
            review = nlp(df[reviews].iloc[i])
            review_sentences = review.sents
            for review_sentence in review_sentences: # take each review sentence
                if len(review_sentence) > 1:
                    for sentence_exemple in ref_bag.sents:
                        if len(sentence_exemple) > 1:
                            temp_sim_score = round(sentence_exemple.similarity(review_sentence),2) # test similarity with whole reference bag
                            sim_score.append(temp_sim_score)
        except:
            #print("message null")
            sim_score.append(0.0)
        y_pred = allocate_lab(sim_score, y_pred, thresh) # if similarity failed, mark as 0 (empty review or not sentence aligned with label sentiment if sentiment = True)     
        return y_pred

    # Comparing whole review vs whole reference bag, returns y_pred series
    def review_vs_whole_ref_bag(i,df,label_of_interest, thresh,word_mode, sentiment=True):
        # establish the correct review granularity, review and reference pieces to integrate based on sentiment bool, stop_word (word_mode) bool and label of interest
        reviews,ref_bag = decision_sentiment(sentiment,label_of_interest, word_mode) 
        #print("for label", label_of_interest)
        #print("review colname", reviews)
        #print("reference bag colname", ref_bag)
        ref_bag = nlp(references[ref_bag][0]) # we convert the references once to nlp before the loop
        y_pred = []
        sim_score = []
        try:
            review = nlp(df[reviews].iloc[i])
            sim_score.append(round(ref_bag.similarity(review),2)) # compare it with the full reference bag
            allocate_lab(sim_score, y_pred, thresh) # if similarity score is > than threshold we append 1, otherwise: 0
        except: # if similarity test fails, we put 0 (in this case na or non text review)
            print("nan or error on message:", review)
            y_pred.append(0)
        return y_pred 
        
    sim = pd.DataFrame()
    def predi (i,sim=sim):
        sim['c_bad_com'] = review_sentence_vs_ref_sentence(i,df,"c_bad_com", 0.73,"stopwords", sentiment=True)
        sim['c_bad_efficacy'] = review_vs_whole_ref_bag(i,df,"c_bad_efficacy", 0.86, "allwords", sentiment=False)
        sim['c_good_efficacy'] = review_sentence_vs_ref_sentence(i,df, "c_good_efficacy", 0.71,"stopwords", sentiment=True)
        sim['c_good_com'] = review_sentence_vs_ref_sentence(i,df, "c_good_com", 0.68,"stopwords", sentiment=True)
        sim['c_good_value'] = review_sentence_vs_ref_sentence(i,df, "c_good_value", 0.68,"stopwords", sentiment=True)
        sim['c_bad_value'] = review_sentence_vs_ref_sentence(i,df, "c_bad_value", 0.69,"stopwords", sentiment=True)

        sim = sim.astype("bool")
        
        if sim['c_bad_com'][0]:
            if sim['c_good_com'][0]:
                b_com = "😐 une communication mitigée"
            else:
                b_com ="👎 une mauvaise communication"
        else:
            if sim['c_good_com'][0]:
                b_com = "👍 une bonne communication"
            else:
                b_com= ""#"😐 une communication mitigée"

        if sim['c_bad_value'][0]:
            if sim['c_good_value'][0]:
                b_value = "😐 un rendement économique mitigé"
            else:
                b_value = "👎 un mauvais rendement économique"
        else:
            if sim['c_good_value'][0]:
                b_value = "👍 un bon rendement économique"
            else:
                b_value = ""#"😐 un rendement économique mitigé"

        if sim['c_bad_efficacy'][0]:
            if sim['c_good_efficacy'][0]:
                b_efficacy = "😐 une efficacité mitigée"
            else:
                b_efficacy ="👎 est inefficace"
        else:
            if sim['c_good_efficacy'][0]:
                b_efficacy = "👍 est efficace"
            else:
                b_efficacy =""# "😐 a une efficacité mitigée"
        
        if sim['c_bad_com'][0] + sim['c_good_com'][0] + sim['c_bad_value'][0] + sim['c_good_value'][0] + sim['c_bad_efficacy'][0] + sim['c_good_efficacy'][0]:
            st.write(f"l'avis considère que la banque {df.Société[i]} a:")
            st.write(b_com)
            st.write(b_value)
            st.write(b_efficacy)
        else:
            st.write("pas de sentiment particulier détécté")
        b_com,b_value,b_efficacy = "","",""
   
    st.write("#### 1) Prédiction par avis (échantillon de 200 avis choisis au hasard)")
    i = st.slider(
        "## Choisir un avis:",
        1, 199, 1)
    st.write(df.text_total[i])

    if st.button("Prédiction"):
        predi(i,sim)

    # visu banques

    from math import pi

    df_banks = pd.read_csv(f"{key_path}/soutenance/tabs_banques.csv")
    n = len(df_banks)

    def pyramid(df, one_all, bank="all banks",ax=None):
        if bank != "all banks":
            #print(f'Focusing on bank {bank}')
            df = df[df.Société == bank]
        else:
            print("Matched all banks")
            df = df
        
        ratio_efficacy = round(df.abs_good_efficacy.iloc[0]/(df.abs_good_efficacy.iloc[0]+df.abs_bad_efficacy.iloc[0]),2)
        ratio_com = round(df.abs_good_com.iloc[0]/(df.abs_good_com.iloc[0]+ df.abs_bad_com.iloc[0]),2)
        ratio_value = round(df.abs_good_value.iloc[0]/(df.abs_good_value.iloc[0] + df.abs_bad_value.iloc[0]),2)
        n = df.n.iloc[0]
        # Define the number of variables and their ratings
        num_vars = 3
        ratings = [ratio_com, ratio_efficacy, ratio_value]
        ratings_sum = sum(ratings)
   
        if ratings_sum > 2:
            data_color = "ForestGreen"
        elif ratings_sum >1.5:
            data_color = "DarkOrange"
        else:
            data_color = "IndianRed"
        # Compute angle for each axis
        angles = np.linspace(0, 2 * pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the loop

        # Rotation to level the triangle
        angles = [angle + pi / 2 for angle in angles]

        # Ratings need to be repeated at the end to close the plot
        ratings = np.concatenate((ratings, [ratings[0]]))

        # Perfect ratings for reference
        perfect_ratings = np.array([1, 1, 1, 1])

        # Check if an axis is provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        # Set the facecolor and plot
        ax.set_facecolor('white')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(['Communication\n', '\nEfficacité', '\nValeur'], fontsize=18, color=data_color)
        
        # Hide circular lines and y-axis labels
        ax.set_yticklabels([])
        ax.yaxis.grid(False)

        # Set the limit for the radial axis
        ax.set_ylim(0, 1)

        # Define a small offset for labels to avoid overlap with high scores
        label_offset = 1.1  # You can adjust this value as needed

        ax.plot(angles, perfect_ratings, linewidth=1, linestyle=':', color='grey', alpha=0.5)
        ax.fill(angles, perfect_ratings, color='Ivory', alpha=0.5)
        # plotting the perfect triangle for visual reference
        ax.plot(angles, ratings, linewidth=2, linestyle='solid', color=data_color)
        ax.fill(angles, ratings, data_color, alpha=0.25)
        ax.spines['polar'].set_visible(False)
        ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.yticks([], [], color="black", size=8)
        plt.ylim(0, 1)

        # Adding values at triangle tips
        for angle, rating in zip(angles[:-1], ratings[:-1]):
            ax.text(angle, rating + 0.10, f'{rating:.2f}', ha='center', va='center', fontsize=16, color=data_color)
        
        # Adding the sum of ratings in bold at the center of the triangle
        ax.text(0, 0, f'{ratings_sum:.2f}', ha='center', va='center', fontsize=25, color= data_color, fontweight='bold')
        ax.set_title(f"Scores pour {bank} (n={n} avis)", size=25)
        if one_all =="one":
            return fig,ax

    st.write ("#### 2) Scores globaux pour les banques")
    banque = st.selectbox("",df_banks.Société,index = None,
                          placeholder="choisissez une banque")
    if banque:
        figue, axe = pyramid(df_banks, "one",banque )
        st.pyplot(figue)

    all_banques_check = st.checkbox("Visualiser toutes les banques")
    if all_banques_check:
        # plotting the pyramids
        cols = 3  
        rows = n // cols + (n % cols > 0)  
        fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows), subplot_kw=dict(polar=True))
        axs = axs.flatten()  

        fig.suptitle("Ratios de sentiment utilisateur positif concernant la communication, l'efficacité et la valeur des services bancaires pour les banques ayant plus de 1000 avis", fontsize=20)  # Add an overall title

        for idx, bank in enumerate(df_banks.Société):
            ax = axs[idx]
            pyramid(df_banks, "all", bank, ax)
            ax.grid(False)  
            ax.set_yticklabels([]) 
            ax.set_title(f'{bank}', fontsize=20) 

        for ax in axs[len(df_banks):]:
            ax.set_visible(False)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 

        plt.show()
        st.pyplot(fig)

    st.write("""
- **Cette labélisation permet de tester les facteurs économiques** :
  - Analyse de sentiment associée aux avis émis sur la **valeur économique**.
  - Association de ces sentiments à des notions financières telles que :
    - Compte
    - Solde
    - Crédit
    - Débit
    - Prêt
    - Hypothèque
    - Intérêt
    - Investissement
    - Épargne
    - Transaction
    - Dépôt
    - Retrait
    - Frais
    - Charge
    - Budget
    - Finance
    - Argent
    - Monnaie
    - Échange
    - Taux
    - Action
    - Obligation
    - Marché
    - Économique
    - Économie
    - Financier
    - Fiscal
    - Inflation
    - Déflation
    - Taxe
    - Revenu
    - Profit
    - Perte
    - Évaluation
    - Actif
    - Passif
    - Équité
    - Dividende
    - Portefeuille
- **Observations** :
  - Polarisation des opinions négatives non pas sur les services habituels liés aux échanges ou des prêts.
  - Avis positifs en net pour les échanges d'informations et les prêts.
  - Opinions négatives concentrées sur la notion de "perte" ou de compte.
  - Suggère qu’un facteur conjoncturel serait à l’œuvre.
""")    
    st.image(f"{key_path}/references/Net sentiment.jpg", caption='')  
    st.write("Pour vérifier cela, nous détaillons la fréquence des opinions exprimées en fonction des mois durant lesquels ces dernières sont exprimées. Nous constatons une hausse brutale de ces opinions à la fin de notre échantillon aux alentours de Octobre 2023. Les entreprises françaises ont effectivement commencé à rencontrer de plus grandes difficultés vers la fin de l'année 2023.")
    st.image(f"{key_path}/references/timeserie.png", caption='')    
    st.write("Les défaillances d'entreprises ont augmenté significativement, reflétant une situation économique tendue marquée par l'inflation, la hausse des taux d'intérêt et le ralentissement économique. Ces facteurs ont combiné leurs effets avec le début du remboursement des dettes accumulées pendant la crise du Covid-19, comme les Prêts Garantis par l'État (PGE), exacerbant les difficultés pour de nombreuses entreprises. Ces données montrent clairement que la fin de l'année 2023 a marqué un tournant pour les entreprises françaises, avec une augmentation significative des faillites et des procédures collectives, signalant des défis accrus en matière de liquidité et de solvabilité pour les entreprises de toutes tailles, et notamment pour les plus petites d’entre elles ou les auto-entrepreneurs. ")

    st.write("### Perspectives")
    st.write("""des travaux futurs pourraient:
  - Utiliser une labelisation plus étendue, notamment sur les aspects economiques
  - Utiliser des modèles de generative AI pour:
    - Labéliser les messages
    - Produire des résumés des principaux pain points              
             """)