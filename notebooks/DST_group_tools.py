def review_vector(df):
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # 1000 pour garder l'essentiel, plus?
    vec_text = tfidf_vectorizer.fit_transform(df['text_lemma'])
    print(vec_text[0:5])
    # liste des variables à ajouter
    df = df.drop("text_lemma")
    # Ajouter les variables en format dense, comme le texte vectorisé
    df_tf = hstack([vec_text, csr_matrix(df)])
    from scipy.sparse import csr_matrix, hstack
    return df_tf

def model_report(model,X_test):
    score = round(model.score(X_test, y_test),2)
    print("train score: ", score)
    y_pred = model.predict(X_test)
    print(pd.crosstab(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return(score)