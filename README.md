Multiple-classification-
Multiple-classification with a small amount of data

Problem statement
====
    
 The first question is about the multi-classification of heart disease detection, with a data size of 452x279 and sample labels from 1 to 16, of which samples with labels 11, 12, and 13 are missing. In addition, the label 1 sample is unaffected, the label 2 to 15 is a defined heart disease, and the label 16 is an uncertain heart disease.

problem analysis
====
    
 The first thing that comes to mind about this type of problem is the traditional machine learning method SVM, and the more novel ones are lasso and xgboost. For the SVM classifier, the feature is directly sent to the model, trained with no more than 70% of the data, and tested with the remaining data. For xgboost you can handle it directly. For lasso, I will prepare for feature dimension reduction and feature selection, and send the selected features to xgboost for learning.
  
SVM three classification
====
    
 Because it is a two-category problem from the point of view of illness and disease, but considering that there is an uncertain type of heart disease, the problem can be seen as a three-category problem. First, the data set is divided into a training set and a test set, wherein the number of samples in the training set is 310, and the number of samples in the test set is 142.

![Image text](https://github.com/marsmarcin/Multiple-classification-/blob/master/pictures/Table1.png)
    
 Set the null value to 0 during data processing, place the label in the first column of the data, and save it as a csv format file. The training set file is named data_two_test.csv and the test set file is data_two_test.csv.
  
Training process and test results
===
    
  The whole framework is carried out under tensorflow. The training and testing are written in a project. The svm_3_c weight file is saved in model1_1. Batch is set to 100 and the number of iterations is also 100. Optimizer is a gradient descent method.
  The results learned after 100 iterations are as follows:
  
![Image text](https://github.com/marsmarcin/Multiple-classification-/blob/master/pictures/pic1.png)
    
  The test is still taking 100 sets of data for testing each time, taking 10 times and finally taking the average. Four tests were continued with stability in mind, with test accuracy around 0.9. It can be considered that the three classifications have higher precision.
  
![Image text](https://github.com/marsmarcin/Multiple-classification-/blob/master/pictures/pic2.png)
    
 Although the three classifications have higher precision, this is not the result we want. Imagine that the patient knows that he has a certain heart disease, but he still doesn't know which type of heart disease he has. It doesn't help the mood or the whole treatment. However, the three classifications provide an effective and highly accurate reference for doctors to further diagnose. If the patient has a certain type of heart disease, manual intervention is required to obtain an accurate diagnosis.
  
Xgboost + Lasso six classification
====
    
  Considering the good performance of xgboost, try to use xgboost for multi-classification. Xgboost data needs to be preprocessed. Another point is the problem of label imbalance. In order to solve this problem, six classifications have been carried out. Six classification master
  
![Image text](https://github.com/marsmarcin/Multiple-classification-/blob/master/pictures/pic3.png)


Figure 3. svm three-category, six-category data distribution map
    
  If only 22 samples were considered for the type of heart disease that was uncertain, I regrouped the 185 samples of the identified type of heart disease to get the results of the six classifications.
![Image text](https://github.com/marsmarcin/Multiple-classification-/blob/master/pictures/Table2.png)

Training process and test results
===
    
  
 The relevant experiments of xgboost were done in four groups. The first set of experiments is the experiment of source data under featureless dimension reduction and selection. The second group is classified after feature selection under the lasso two classification label. The third group is selected after the feature selection under the lasso three classification label. Classification, the fourth group is the experiment after the feature selection under the lasso six classification label.
  These experiments are in the test_exp1, test_exp2, test_exp3, and test_exp4 folders.
  
![Image text](https://github.com/marsmarcin/Multiple-classification-/blob/master/pictures/Table3.png)
    
  In the second set of experiments, the use of features from 279 to 25 dimensional accuracy was reduced compared to when no feature selection was used. In the third set of experiments, the feature was reduced from 279 to 30, and the accuracy was improved. The fourth set of experiments reduced the feature dimension to 36 dimensions, and the accuracy rate was already higher than without reducing the feature dimension. Therefore, the use of feature selection can effectively improve the accuracy of classification.
  
SVM thirteen classification
====
    
  The feature reduction and feature selection can effectively improve the classification results. I have also sorted the 279 features by sorting, using the random forest regression in ensemble. Based on this, the svm classifier is optimized.
  
![Image text](https://github.com/marsmarcin/Multiple-classification-/blob/master/pictures/Table4.png)

  Since each score is a three-dimensional vector, my operation is to perform an exponential operation after averaging, and the base is 100000. This number has no meaning, just to distinguish each feature score very well. As a result of this operation, the score conversion range of these 279 features becomes between 0.27 and 4.13. The feature with the highest score is the ‘V2 R wave amplitude’ feature, and the lowest score is the ‘Height’ feature.
  
![Image text](https://github.com/marsmarcin/Multiple-classification-/blob/master/pictures/pic4.png)

  
  The experiment in this section will be divided into two parts, one is the svm classifier without feature selection, and the other is the svm classifier for feature selection.
  
Training process and test results
===
experiment one
==
    
  The training of the svm13 classifier without feature selection is similar to the three classifications. It is still iterated 100 times in the tensorflow framework. The choice of 100 times here is because the laboratory computer is older. In the programming implementation, you should pay attention to changes in several dimensions.
  
![Image text](https://github.com/marsmarcin/Multiple-classification-/blob/master/pictures/pic5.png)

  The test result accuracy rates are 0.86100006, 0.796, 0.88500005, and 0.80200005, respectively.
  
Experiment 2
==
    
  Only the 20 features of the pre-name are extracted and classified. The same iteration is performed 100 times, and the training results are obtained four times as shown in the following figure.
  
![Image text](https://github.com/marsmarcin/Multiple-classification-/blob/master/pictures/pic6.png)
    
  The test results were 0.8549999, 0.8809999, 0.82, and 0.857, respectively. Personal feeling is not obvious, I will increase the number of iterations from 100 to 1000. No feature selection was made, and the results of the two sets of experiments were 0.897 and 0.866, respectively. The results of feature selection were 0.916 and 0.896, respectively. I increased the number of iterations to 10,000. The test accuracy without feature selection was 0.872 and 0.77, respectively, and the feature selection was 0.878 and 0.866.
  
to sum up
====
    
  The implementation and completion of the entire experiment is carried out in the process of continuous learning and improvement. From the previous model argumentation to the feature engineering is the process of continuous experimental argumentation and code change. Especially recently, the lab changed the computer back and forth to delay a lot of work, and the idea was transferred from svm to xgboost, and laasso. The reason why I returned to svm from xgboost is because some experiments have found that the accuracy is difficult to improve, especially under the premise that svm still maintains high precision. When the accuracy of xgboost is still around 0.6, I resolutely give up using the xgboost method. .
  
![Image text](https://github.com/marsmarcin/Multiple-classification-/blob/master/pictures/pic7.png)
 
  After thinking about it, one reason is very important, that is, the data set is not big enough. The traditional svm method performs very well for small dataset problems, and the xgboost related method has been eye-catching in major data mining competitions since 2014. A key issue is that the data set in the game is basically millions. Level, so the accuracy is not excusable.
