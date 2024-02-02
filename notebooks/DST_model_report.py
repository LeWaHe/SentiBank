def DST_model_report(model, X_test):
    score = round(model.score(X_test, y_test),2)
    print("train score: ", score)
    y_pred = model.predict(X_test)
    print(pd.crosstab(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return(score)