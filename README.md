# Credit_Risk_Analysis

  In this challenge, we built and evaluated several machine learning models to assess credit risk, using data from LendingClub; a peer-to-peer lending services company.

The goals of this analysis were:

  a. Implement machine learning models.
  b. Use resampling to attempt to address class imbalance.
  c. Evaluate the performance of machine learning models.
  
  
  ## Results
  
  We conducted six machine learning models:
  
    1. Naive Random Sampling: the accuracy score of this model is 0.65, meaning this model accurately predicts 65% of the times. Precision for this model is 0.99 and recall 0.67.
    
    
    2. SMOTE Oversampling: the accuracy score of this model is 0.65, precision is 0.99 and recall 0.66. It is important to note that The precision score are 1.00 for predicting low-risk and 0.01 for predicting high-risk.
    
    
    3. Undersampling: the accuracy for this model is 0.51, precision is 0.99 and recall is 0.44. Once again, it is important to note that The precision score are 1.00 for predicting low-risk and 0.01 for predicting high-risk.
    
    4. Combination sampling: the accuracy for this model is 0.62, precision is 0.99 and recall is 0.55. It is important to note that The precision score are 1.00 for predicting low-risk and 0.01 for predicting high-risk.
    
    5. Balanced Random Forest Classifier: the accuracy for this model is 0.78, precision is 0.99 and recall is 0.87. It is important to bring that the precision score are 1.00 for predicting low-risk and 0.03 for predicting high-risk.
    6. Easy Ensemble AdaBoost Classifier: the accuracy for this model is 0.93, precision is 0.99 and recall is 0.94. And again it is important to note that the precision score are 1.00 for predicting low-risk and 0.09 for predicting high-risk. 
    
    
## Summary: 
  While we tested the six models above, we observe that Easy Ensemble AdaBoost Classifier performs better and in a more balanced way than other models so we suggest that this model be used for this project. The testing proves that the recall score for this model has the highest coefficients for predicting both low-risk and high risk loan status, meaning this would be our best model for moving forward.
