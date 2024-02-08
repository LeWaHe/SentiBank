# Fonctions de vectorisation et de reporting
def model_report():
    import time
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    # measuring time taken to train the model
    t1 = time.time()
    delais = round((t1-t0)/60,2)
    # test score
    try:
        score = round(model.score(X_test, y_test),2)
    except:
        score =  "na"
    print("train score: ", score)

    # predictiong on test set, accomodating to dm matrix in except (test contains X and y)
    try:
        y_pred = model.predict(X_test)
    except:
        y_pred = model.predict(test)
    
    # saving results in the benchmark file
    try:
        extra_features_rep = X_train.columns.values
    except:
        extra_features_rep = extra_features
       
    if "Grid" in type(model).__name__:
        bool_grid = "yes"
    else:
        bool_grid = "no"
  
    if bool_grid == "yes":
        used_params = model.best_params_
    else:
        used_params = model.get_params()

    model_name = type(model).__name__
    report =classification_report(y_test, y_pred, output_dict=True)
    macro_precision =  round(report['macro avg']['precision'],2) 
    macro_recall = round(report['macro avg']['recall'],2)    
    macro_f1 = round(report['macro avg']['f1-score'],2)  
    tempdf = pd.DataFrame({"model":[model_type],
                            "grid search": [bool_grid],
                            "used/best params":[used_params],
                            "features": [extra_features_rep],
                            "score":[score],
                            "precision": [macro_precision],
                            "recall": [macro_recall],
                            "f1":[macro_f1],
                            "time_taken_mns":[delais],
                            "run_date": [time.strftime('%Y-%m-%d', time.localtime())]
                        })
    # reports: classification report and crosstab heatmap 
    print(classification_report(y_test, y_pred))
    # Generate and normalize the confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred)
    conf_mat_normalized = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    # Create a heatmap for the confusion matrix
    plt.figure(figsize=(4, 4))
    sns.heatmap(conf_mat_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title(f'Normalized Confusion Matrix for {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # load and append results to the benchmark, save
    bench = pd.read_csv('../reports/benchmark/model_benchmark.csv', index_col=0)
    bench = pd.concat([bench, tempdf])
    bench.to_csv('../reports/benchmark/model_benchmark.csv')

def review_vector(df,raw_extra_features):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy.sparse import csr_matrix, hstack
    extra_features = raw_extra_features
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # 1000 pour garder l'essentiel, plus?
    vec_text = tfidf_vectorizer.fit_transform(df['text_lemma'])
    #print(vec_text[0:5])
    # Ajouter les variables en format dense, comme le texte vectorisé
    df_tf = hstack([vec_text, csr_matrix(df[extra_features])])
    return df_tf