SentiBank
==============================
This repo is a fork from a Data Science training project @ DataScientest / La Sorbonne university.

Team:
Leonardo Heyerdahl (lead)
Alexis Garatti
Huazhen Hou
Alexandre PRZYBYLSKI

Objectives:
- Collect banking reviews 
- Predict stars using review texts and meta data
- Characterize user opinions regarding banking services

Results:
- We collected 170k banking reviews in French from Truspilot
- We evaluated the performance of classic ML models (KNN, lr, Random Forests, XGBoost) and one Transformer based model (CAMEMBERT), which outperformed (f1 = .64)
- We created a 3 axis dichotomic taxonomy to characterize user opinions about banks:
    - Communication Good/Bad
    - Value Good / Bad
    - Efficacy Good / Bad
- We hand labelled 200 reviews using these 6 labels
- We evaluated different strategies for automatic labelling using
    - Different granularities (sentence based review, whole review, sentence based label references,..)
    - CamemBERT infered sentiment filtering
    - All words vs tailored stop words filtering
- Our semantic labeler performed between .54 and .80 on test results.
- We labelled the whole data set and produced global notations on the 3 axis for all banking actors in France
