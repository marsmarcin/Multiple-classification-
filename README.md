Multiple-classification-
Multiple-classification with a small amount of data

Problem statement
====
    
 The first question is about the multi-classification of heart disease detection, with a data size of 452x279 and sample labels from 1 to 16, of which samples with labels 11, 12, and 13 are missing. In addition, the label 1 sample is unaffected, the label 2 to 15 is a defined heart disease, and the label 16 is an uncertain heart disease.

problem analysis
====
    
  The first thing that comes to mind about this type of problem is the traditional machine learning method SVM, and the more novel ones are lasso and xgboost. For the SVM classifier, the feature is directly sent to the model, trained with no more than 70% of the data, and tested with the remaining data. For xgboost you can handle it directly. For lasso, I will prepare for feature dimension reduction and feature selection, and send the selected features to xgboost for learning.
  
