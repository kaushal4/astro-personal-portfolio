---
title: Modeling Financial Risk For Loan Approval
publishDate: 2025-01-28 00:00:00
img: /assets/loan/loan-main.webp
img_alt: stock image
type: project
description: |
    We analysed loan data to create a loan predictor. 
tags:
  - Machine Learning
  - Regression
  - Data Visualization
---

# Introduction

Loan approval is a pivotal process in the financial sector, requiring a careful balance between granting credit access and mitigating risks. Approving high-risk applications can lead to financial losses, while overly restrictive criteria may exclude qualified borrowers, eroding customer trust and reducing revenue. To address this challenge, this study explores the automation of loan approval decisions using machine learning techniques. The primary goal is to develop a binary classification model capable of predicting loan approval outcomes based on diverse demographic and financial features. By identifying the key factors influencing these decisions, the study aims to provide actionable insights that help financial institutions enhance risk assessment strategies and improve overall customer satisfaction.

# Dataset

The dataset analyzed in this study consists of 20,000 records with 33 features, including 28 numerical and 5 categorical variables. This dataset is specifically designed to support the development of predictive models for risk assessment. It includes a diverse range of features such as demographic details, credit history, employment status, income levels, existing debt, and other critical financial metrics. Together, these attributes provide a comprehensive foundation for sophisticated, data-driven analysis and informed decision-making in loan approval processes.
[Data Source](https://www.kaggle.com/datasets/lorenzozoppelletto/financial-risk-for-loan-approval/data)

# Data Analysis

![image](/assets/loan/loanvedu.png)

We used a bar plot to study the distribution of the target variable. We see that the data is skewed towards loans not being approved (0.73%). Hence when we do a dev-test-split we stratify the data. Furthermore we used bar plots to plot the distribution of categorical variables against the target variable. From these graphs we noticed loan approval changes with Education Level. To Visualize this further we use donut plot to plot Education level against Loan approval (check the figure above). Additionally, we plotted box and Kernel Density Estimator plots for numerical features in our dataset.

![image](/assets/loan/b-plot.png)

Out of all numerical features, we have shown some features that show clear distinction between approved vs not approved (check the figure above). For Annual Income, the median annual income is lower for individuals whose loans were not approved. The range of annual incomes for non-approved loans is narrower, and there are significant outliers in the upper range. On the other hand, Approved loans correspond to a higher median annual income, indicating that income positively influences loan approval likelihood. There are still outliers in the high-income range, but overall, the distribution skews higher. For Credit Score, The median interest rate is higher for non-approved loans, reflecting higher risk for the lender. This indicates that individuals facing higher interest rates are more likely to have their loans denied. On the other hand, Approved loans have a slightly higher median credit score, and the distribution covers a broader range. This suggests a higher credit score positively correlates with loan approval. For Interest Rate, The median interest rate is higher for non-approved loans, reflecting higher risk for the lender. This indicates that individuals facing higher interest rates are more likely to have their loans denied. On the other hand, Approved loans are associated with a lower median interest rate. The distribution of interest rates is narrower, and the lower rates likely improve approval odds. For total debt to income ratio, The median debt-to-income ratio is significantly higher for non-approved loans. The distribution indicates that high ratios negatively affect approval, as lenders perceive greater financial risk. On the other hand, Approved loans correspond to a lower debt-to-income ratio, reflecting a better balance between income and financial obligations.

![image](/assets/loan/kde-plot.png)

Similarly, we show most KDE plots for these numerical features. For annual income, loan-approved individuals generally have highGr annual incomes. The density shifts towards larger income values for approvals compared to denials. For Credit Scores, approved loans tend to cluster at higher credit scores, indicating creditworthiness as a key factor. For interest rates, loans not approved are more associated with higher interest rates, while approved loans show a density shift towards lower rates. For Debt to income ratio, approved loans show a peak at lower debt-to-income ratios, highlighting financial stability as critical for approval. There are significant outliers for all features in both approved and non-approved categories. These outliers indicate that exceptions exist but follow general trends. Overall, better financial metrics are strongly associated with loan approval.

To visualize the variance of Loan Approval with our features we use Principal Component Analysis.

![image](/assets/loan/PCA.png)

Figure above provides a comprehensive view of the dataset's structure and the relative importance of the different principal components. The PCA plot allows you to visualize the relationships between observations, while the Scree plot helps determine the optimal number of principal components and the variance distribution.

# Data Preprocessing
The dataset used in this study is complete, containing no `nan` values, eliminating the need for data imputation.
However, an imbalance was observed, with 73% of samples indicating loan approval and 27% indicating no approval. To address this imbalance during model evaluation, stratified sampling was employed to split the dataset into development and test sets. The dataset includes 28 numerical features and 5 categorical features, of which 4 are nominal and 1 is ordinal. An analysis of numerical features using a correlation matrix revealed three highly correlated pairs, identified using a threshold of 0.9. To reduce redundancy, we removed three numerical features: ‘Experience,’ ‘MonthlyIncome,’ and ‘NetWorth.’
For preprocessing, numerical features were scaled using a standard scaler, while nominal categorical features were encoded with a one-hot encoder, and the ordinal feature was processed using an ordinal encoder. The resulting preprocessed dataset served as the foundation for running experiments across all models.

# Experiments and Results
The table below presents the models evaluated in this study, assessed using metrics including Accuracy, Precision, Recall, F1-Score, and AUROC. Due to space constraints, only Accuracy and AUROC results are displayed, with additional metrics available in the accompanying Jupyter notebooks. The models were evaluated using 10-fold cross-validation on the development dataset, along with hyperparameter tuning to identify optimal parameters. Detailed descriptions of the hyperparameter search process and the selected best configurations for each model are also provided in the notebooks. Model interpretability was analyzed using LIME and Shapley value-based methods, providing insights into feature contributions for each prediction. Among the models tested, CatBoost, AdaBoost, and HistGradientBoosting demonstrated the best performance, as measured by both Accuracy and AUROC.

| Model                | Accuracy | AUROC   |
|----------------------|----------|---------|
| AdaBoost             | 97       | 99.35   |
| CatBoost             | 97       | 99.37   |
| DecisionTree         | 90       | 92.90   |
| ExtraTrees           | 92       | 98.1    |
| GradientBoosting     | 96       | 99.23   |
| HistGradientBoosting | 97       | 99.27   |
| LightGBM             | 96       | 99.16   |
| LogisticRegression   | 95       | 99.06   |
| MLP                  | 95       | 99.98   |
| RandomForest         | 93       | 97.771  |
| SVM                  | 95       | 93      |
| XGBoost              | 96       | 99.25   |

# Model Interpretability

We have used Model Interpretability techniques to interpret the results for Catboost and Adaboost models. We have done Shapley and LIME analysis for all models we have trained. Due to space constraints we have only included a few models in this report.

## Cat Boost

![image](/assets/loan/catboost-1.png)

From the above figure we can see that the most important features for the CatBoost model are TotalDebtToIncomeRatio, InterestRate, MonthlyIncome, NetWorth, AnnualIncome, CreditScore, LoanAmount.

![image](/assets/loan/catboost-2.png)

The Shapley plot in the figure above provides a clear visualization of the relative importance and impact of each feature on the CatBoost model's output.

## Ada Boost

![image](/assets/loan/adaboost-1.png)

Figure above shows the variation in difference feature importances for adaboost model. We see that the features we found important in the data exploration stage are some of the most important features in this model.

![image](/assets/loan/adaboost-1.png)

Figure above Shows LIME analysis of Adaboost model. Using this analysis we can see what factors push the results to being approved or not approved. For the instance used for this analysis we observe that Income to debt ratio places a major role in pushing the result toward the loan not being approved.

# Appendix

You can find methodology and code at [Github](https://github.com/kaushal4/group22Project)