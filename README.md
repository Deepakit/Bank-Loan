# [Bank Loan Default Case](https://github.com/Deepakit/Bank-Loan/blob/main/Bank_loan%20Default.ipynb)
Loan default will cause huge loss for the banks,so they pay a lot of attention on the same 
issue and apply various method to detect and predict default behaviours of their customers.

## Problem Statement
The loan default dataset has 8 variables and 850 records, each record being loan default status for each customer. Each applicant was rated as “Defaulted(1)” or “Non-defaulted(0)”. New applicants for loan application can also be evaluated on these 8 predictor variables and classified as a default or non-default based on predictor variables.

## Motivation
Loans are one of the important aspect of banking industry. All the banks try out effective business strategies to persuade customers to apply their loans. But some customers behave different after the application are approved. So to prevent the same, banks have to find some methods to predict customers behaviours. ML algos. Have a pretty good performance on this purpose.

## Data
We are provided with a single data sheet containing 850 records and 8 features.
A look at the data

![](/Image/data.png)

## EDA
We take a look at the data provided to us by organizing,plotting and summarizing data. By doing this, we can get a idea about the distribution and pattern present in the data.

### Target Distribution
![](/Image/target.png)
We see that there is a imbalance data present but here we are classifying for loan default which is meant to be low.

### Missing Value
![](/Image/missing.png)
 We see that our target variable have 150 missing values. Here we cannot fill these values by our traditional method because if target value is assigned a wrong value then it will effect our model predictions.
To resolve this we will create a new dataframe where we will store only the null entries and remove those entries from our main dataframe.

### Outlier Analysis
We will perform box plot on our features to estimate the number of outliers present in our data.
![](/Image/outlier.png)
We found that apart from ‘age’, every feature have got outliers. But for education(ed) and emply feature which are although numeric seems to categorical in nature removing outliers for them does not seems to be best idea.
Also, for features like income,debtinc and similar we cannot remove their ouliers because for our particular scenario bank would like to have the different records of different users. So our model will require to handle different data altogether so we will not be removing outliers here.

### Data distribution
We plot a distplot to observe the distribution of our data.
![](/Image/dist1.png)![](/Image/dist2.png)
![](/Image/dist3.png)![](/Image/dist4.png)
![](/Image/dist5.png)![](/Image/dist6.png)
![](/Image/dist7.png)![](/Image/dist.png)
All independent variables are rightly skewed.

### Independent Relation
We plot a crosstab to visualize the relation ship of ,’ed’,’employ’,’age’ features with our target variable.
![](/Image/ind_relation.png)![](/Image/ind_rel2.png)
![](/Image/ind_rel3.png)![](/Image/ind_rel4.png)
![](/Image/ind_rel5.png)
We observe that  after value of employ reach to 17 absolutely no loan is defaulted. Also, chance of loan being defaulted is greater when employ is at initial stage.
Similarly, here for a person with educational qualification as 1 has very less chance of loan getting defaulted.Also, other groups have similar distribution.For employee with highest employee experience we have zero records of loan getting defaulted.
Chances of loan getting defalut is greater when a person is younger. This also emphasise our previous assumption for employee experience because when a person has high number of years of experience his/her age will also tend to be greater.

### Scatter plots
We plot scatter plots for ‘debtinc’,’income’,’creddebt’,’othdebt’ with our target variable to understand whether there is any pattern involved.
![](/Image/scatter_1.png)![](/Image/scatter_2.png)
![](/Image/scatter_3.png)
For income we observe that as income increase less loan gets defaulted and as all other debt gets increase more loans gets defaulted, from which we can infer that there might be a linear relationship.
To make sure our assumption is correct we calculate the mean of our independent features by grouping on the target variable


### Feature Selection
 We use .corr() method to find the correlation between our variable and see whether there is any string correlation with the target variable or whether there is any multicollinearity and plot the same on a heatmap.
 ```python
 df.corr()
 ```
 ![](/Image/multi_coll.png)
 
 We will droping the, 'address' feature because it is not revealing as much information also it is not strongly correlated.
One idea was to combine debtinc and othdebt features but the correlation is not that strong between them.


### PCA
We will using PCA also to check whether we further reduce the dimension of our dataset. First we will standarise the data using StandardScaler() and the use PCA() on it.
```python
ratio ={}
for i in rang(4,7):
  pca = PCA(n_components=i).fit(bank_df[columns])
  ratio[i] = sum(pca.explained_variance_ratio_)
pd.Series(ratio).plot()
plt.plot()
```

We did not find any strong evidence to reduce the dimension so we will continue with our remaining features only.
 
## Modelling
After all early stages of preprocessing, then model the data. So, we have to select best model for this project with the help of some metrics.
In modelling we first have to split the clean dataset to train-set and test-set and then develop different models and evaluate them by metrics.
Models used are:
1) Logistic Regression
2) Decision Tree
3) Random Forest
4) Naive Bayes

## Metrics
Now, we have a three models for predicting the target variable, but we need to decide which model better for this project. There are many metrics used for model evaluation. Classification accuracy may be misleading if we have an imbalanced dataset or if we have more than two classes in dataset.  
In this project, we are using two metrics for model evaluation as follows:
1)	Confusion Matrix:
In machine learning, confusion matrix is one of the easiest ways to summarize the performance of your algorithm. At times, it is difficult to judge the accuracy of a model by just looking at the accuracy because of problems like unequal distribution. So, a better way to check how good your model is, is to use a confusion matrix.
![](/Image/conf.png)

2)	Receiver operating characteristics (ROC)_Area under curve(AUC) Score:
It is a metric that computes the area under the Roc curve and also used metric for imbalanced data. Roc curve is plotted true positive rate or Recall on y axis against false  positive rate or specificity on x axis. The larger the area under the roc curve better the performance of the model.

## [Threshold](https://github.com/Deepakit/Bank-Loan/blob/main/Logistic%20Regression_threshold_selection.ipynb)
* By default, logistic regression have the threshold of 0.5 to assign the probabilities to 1 or 0.To improve the recall of the model, we can use the probabilities predicted by the the model and set threshold ourselves.The threshold is based on different factors and objectives. Here for bank, they want to control the loss to an acceptable level so the threshold can be set to low value,so more customers will be grouped as “potential bad customers” and their profile can be checked so that bank does not incur any losses.
* To set the threshold we will use .predict_proba() function, also our aim would be to find a balance between sensitivity and specificity. For this purpose we will use F1 score which is a harmonic mean of them and will try to maximize the F1 score.
* We will test the F1 score with different threshold value and store them in a list and find the maximum F1 score and its corresponding index.
F1 score = (2 * Precision * Recall) / (Precision + Recall)

```python
def to_labels(prob,t):
    return(prob>=t).astype('int')
    
from numpy import arange,argmax
y_pred_prob = log_model.predict_proba(test_x)
probs = y_pred_prob[:,1]
thresholds = arange(0,1,0.001)

scores = [f1_score(test_y,to_labels(probs,t))for t in thresholds]
ix = argmax(scores)
print("Threshold=%.3f , F-score=%.5f"%(thresholds[ix],scores[ix]))
```

The threshold value found is 0.561 which is very close to our default threshold value 0.5. So here we will not be changing our threshold value.

### Summary
This project can help the company to help us identify which customers have higher chance to get their loan defaulted and which have low.
This can help banks to check the applications with more scrutiny and save themselves from huge losses. This can also help them in understanding the potential good customer which can be targeted for the loans.
They can also help the potential bad customers to realize their application fault and maybe work with them together to improve their portfolio.
