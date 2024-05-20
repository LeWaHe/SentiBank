import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# cache?

st.title("Projet de classification ")
st.sidebar.title("Sommaire")
pages=["Introduction", "Exploration/DataVizualization", "Modélisation I","Modélisation II", "Prédiction"]
page=st.sidebar.radio("Aller vers", pages)

if page == pages [0]:
    st.write("## Introduction")
    st.write("""Extraire des informations ainsi que les sentiments contenus dans des commentaires du secteurs financier ou bancaire sur le site trustpilot.com.

Objectifs spécifiques :
""")
    st.write("•	Prédire la satisfaction d’un client : problème de régression (prédire le nombre d'étoiles). Entraînement supervisé possible.")
    st.write(" •	Identifier les points clés des avis : localisation, nom d’entreprise... ")
    st.write("•	Extraire les propos du commentaire et trouver les mots importants (problème de livraison, article défectueux...) : approche non supervisée avec CamemBert.")
    st.write("•	Trouver une réponse rapide adaptée pour répondre au commentaire (par exemple sur les reviews Google).")

if page == pages[1]:
    st.write("### Exploration/DataVizualization")
    st.write("""Nous avons collecté 165 000 avis sur les banques du site trust pilot en utilisant BeautifulSoup. Nous avons retiré du dataset les avis liés à l’une des banques qui semblait émaner de bots (60000 avis, la grande majorité de 5 étoiles). Nous avons fait le split des données train et test avant de faire une sélection équilibrée des étoiles (pour éviter un leaking de la structure des résultats attendus en test dans l’entrainement) en se basant sur un tirage aléatoire égal au nombre de messages présents dans la classe la plus minoritaire. Le dataset retenu faisait 15 000 avis, dont 30% du jeu destiné au jeu de test.
    Nous avons fait un benchmark pour évaluer la performance de différents modèles : SVM, Random Forests, XGBOOST, KNN, SVC, Logistic Regression et CAMEMBERT.
    """)


if page == pages[2]:
    st.write("### Modélisation")
    model_benchmark = pd.read_csv('/Users/lheyerda/Library/CloudStorage/OneDrive-InstitutPasteurParis/github/juil23_cds_supply_chain/reports/classification_benchmark_2024-02-08.csv', index_col= 0)
    st.write("""
             Pour la prédiction des étoiles des avis, ,Le modèle de deep learning Camembert a donné les meilleurs résultats. 
             Sur les données d’entrainement il a atteint une précision, un recall et un f1 de 0.63 chacun. 

             Le deuxième modèle le plus performant a été Random Forest avec un f1 de 0.55. 
             Ce score a été obtenu sur les features numériques uniquement et par une grille qui a retenu les paramètres suivants
              : 'max_depth': None, 'min_samples_leaf': 2, 'min_samples_split': 20, 'n_estimators': 100.
              
             Le troisième modèle le plus performant suit de près le deuxième, il s’agit de XGBOOST avec un score de 0.54 obtenu sur les features numériques et les hyperparamètres par défaut,
              la recherche par grille a donné le même score f1.""")
    
    st.dataframe(model_benchmark)

if page == pages[3]:
    st.write("### Modélisation II classification des sentiments (Communication, Efficacité, Valeur économique)")
    
    st.write("""Nous avons fait une classification des sentiments des utilisateurs concernant la communication, l’efficacité et la valeur ajoutée par similarité sémantique.
    Par la suite nous avons entrepris de caractériser les arguments que les usagers invoquent pour expliquer leur notation, afin de dégager les aspects positifs et négatifs des services, qui pourraient être utiles pour augmenter leur qualité et la satisfaction des clients.
    """)
   
    tableau_labels = st.dataframe(pd.read_excel("/Users/lheyerda/Library/CloudStorage/OneDrive-InstitutPasteurParis/github/juil23_cds_supply_chain/soutenance/tableau_labels.xlsx"))
    
    st.write("""Nous avons fait une classification des sentiments des utilisateurs concernant la communication, l’efficacité et la valeur ajoutée par similarité sémantique.
    Par la suite nous avons entrepris de caractériser les arguments que les usagers invoquent pour expliquer leur notation, afin de dégager les aspects positifs et négatifs des services, qui pourraient être utiles pour augmenter leur qualité et la satisfaction des clients.
    """)
    baseline_banchmark_sem= pd.read_csv('/Users/lheyerda/Library/CloudStorage/OneDrive-InstitutPasteurParis/github/juil23_cds_supply_chain/soutenance/baseline_semantic_bench.csv')

    import matplotlib.pyplot as plt

    plt_1 = plt.figure(figsize=(10,6))
    plt.plot(baseline_banchmark_sem.threshold,baseline_banchmark_sem.Stop_words, label ="stop words", color ="indianred")
    plt.plot(baseline_banchmark_sem.threshold,baseline_banchmark_sem.Stop_words_sents, label ="stop words + sentiment", color ="indianred",linestyle="--")
    plt.plot(baseline_banchmark_sem.threshold, baseline_banchmark_sem.All_words, label = "all words", color="seagreen")
    plt.plot(baseline_banchmark_sem.threshold, baseline_banchmark_sem.All_words_sents, label ="all words + sentiment", linestyle="--", color="seagreen")
    plt.grid(True, which='both', linestyle=':', linewidth=0.5)
    plt.xlabel("similarity threshold used")
    plt.ylabel("f1 score")
    plt.legend()
    plt.show();
    st.pyplot(plt_1)


    def heat(train_test):
        # pivot table heatmap
        benchmark = pd.read_csv(f"/Users/lheyerda/Library/CloudStorage/OneDrive-InstitutPasteurParis/github/juil23_cds_supply_chain/soutenance/benchmark_sem_{train_test}.csv")
        pivot = benchmark.pivot_table(index='code_mode', columns='test', values='f1')
        fig_sem_bench = plt.figure(figsize=(7, 10))  
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
    
    fig_sem_bench_train = heat("train")
    st.pyplot(fig_sem_bench_train)

    st.write("#### Résultats benchmark sur données de validation")
    fig_sem_bench_val = heat("validation")
    st.pyplot(fig_sem_bench_val)

    st.write("#### évaluation des résultats avec les données test")
    fig_sem_bench_test = pd.read_csv("/Users/lheyerda/Library/CloudStorage/OneDrive-InstitutPasteurParis/github/juil23_cds_supply_chain/soutenance/benchmark_sem_test.csv", index_col=0)
    st.dataframe(fig_sem_bench_test)

    #plt.show()


if page == pages[4]:
    st.write("### Prédiction")
    st.write("En utilisant la proximité sémantique et des références issues d'une labelisation à la main, nous pouvons prédire les sentiment des avis concernant la Communication, la Valeur ajoutée et l'Efficacité du service bancaire.")
    import pandas as pd
    import spacy

    df = pd.read_csv("/Users/lheyerda/Library/CloudStorage/OneDrive-InstitutPasteurParis/github/juil23_cds_supply_chain/soutenance/df_sim_small.csv")
    references = pd.read_csv('/Users/lheyerda/Library/CloudStorage/OneDrive-InstitutPasteurParis/github/juil23_cds_supply_chain/soutenance/references_bag.csv')

    nlp = spacy.load("fr_core_news_lg")

    import numpy as np
    import warnings

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
            st.write(f"l'avis considère que la banque {df.Société[i]} a {b_com} {b_value} {b_efficacy}")
        else:
            st.write("pas de sentiment particulier détécté")
        b_com,b_value,b_efficacy = "","",""
   
    st.write("#### 1) Prédiction par avis")
    i = st.slider(
        "## Choisir un avis:",
        1, 199, 1)
    st.write(df.text_total[i])

    if st.button("Prédiction"):
        predi(i,sim)

    # visu banques
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from math import pi

    df_banks = pd.read_csv("/Users/lheyerda/Library/CloudStorage/OneDrive-InstitutPasteurParis/github/juil23_cds_supply_chain/soutenance/tabs_banques.csv")
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
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        
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




    
