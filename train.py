
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px 
import seaborn as sns
import pickles

from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.metrics import auc

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

C= 1.0
n_splits = 10

output_file = f'model_C={C}.bin'

data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.zcsv')

data.head()

data.shape

data.Churn.value_counts()

data.nunique()

data.isnull().sum()

# Data Preparation

data.head().T

data.dtypes

# transform column titles to lower case
data.columns = data.columns.str.lower().str.replace(' ', '_')

# transform the values for all categorical features
categorical_columns = list(data.dtypes[data.dtypes=='object'].index)
for c in categorical_columns:
  data[c] = data[c].str.lower().str.replace(' ', '_')

data.head().T

# the totalcharges feature shoud be numeric
data.totalcharges = pd.to_numeric(data.totalcharges, errors='coerce')
data.totalcharges = data.totalcharges.fillna(data.totalcharges.median())

data.churn = data.churn.replace({'yes': 1, 'no': 0})

target = data.churn.value_counts().to_frame()
target = target.reset_index()
target = target.rename(columns={'index': 'category'})
fig = px.pie(target, values='churn', names='category', title='Churn Distribution')
fig.show()

"""*The data is largely imbalanced

**Setting up the Validation Framework**
"""

# split the data into train data and test data
df_full_train, df_test = train_test_split(data, test_size=0.2, random_state=1)
# split the train data into train and validation
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1) # here, the validation size is 20% of the 80% in df_full_train
print(len(df_full_train), len(df_train) , len(df_val), len(df_test))

y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values

# define a function that resets index, deletes the target value from the data 
def reset_index(df):
  df = df.reset_index(drop=True)
  return df

df_train = reset_index(df_train) 
df_val = reset_index(df_val) 
df_val = reset_index(df_test)

y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values

del df_train['churn']
del df_val['churn']
del df_test['churn']

"""**EDA**"""

# EDA is done on the entire train data
df_full_train = df_full_train.reset_index(drop=True)

df_full_train.isnull().sum()

df_full_train.churn.value_counts(normalize=True)

overall_churn_rate = df_full_train.churn.mean()
overall_churn_rate

numerical = ['tenure', 'monthlycharges', 'totalcharges']
categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
        'phoneservice', 'multiplelines', 'internetservice',
       'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
       'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
       'paymentmethod']

"""**Note:** *The customerID column isn't a needed feature for building the model*

"""

df_full_train[categorical].nunique()

# Defining the stacked plot plotting function for categorical features
def stack_plot(data):
  for c in categorical:
    sns.set()
    # cross tab
    tab = pd.crosstab(data[c], data.churn, normalize='index')
    tab.plot(kind='bar', figsize=(12, 7))
    print(tab)
    plt.title('Churn rate by' + ' ' + str(c))
    plt.show()

# Defining the histogram plotting function for numerical features
def hist(data):
  for n in numerical:
    group_df = data.groupby([n, 'churn']).size().reset_index()
    group_df = group_df.rename(columns={0: 'Count'})
    fig = px.histogram(group_df, x=n, y='Count', color='churn', marginal='box', title=f'Churn rate frequency to {n} distribution', color_discrete_sequence=["green", "red"])
    fig.show()

stack_plot(data=df_full_train)

df_full_train.tenure.value_counts()

hist(data=df_full_train)

# To check for feature importance, I will be adopting 2 techniques
# First, I will look at the difference between the overall churn rate and churn rate within each group in a feature. 
# The closer the difference between the overall churn rate and the churn rate for the individual groups in a feature is to zero, the less important the feature is in predicting churn
# For each group in a feature, if the difference is higher than 0, it means that the group being considered is less likely to churn
# Else, if the difference is less than 0, then the group being considered is more likely to churn

# Another way to check for feature importance is to check the risk ratio i.e the ratio between the churn rate for each group in a feature and the overall churn rate
# For each group in a feature, if the ratio is greater than 1, then the customers in that group are more likely to churn
# Else, if the ratio is less than 1, then the customers in that group are less likely to churn
# Also, if there is not a lot of differences in the risk ratio for each group in a feature, it might mean that that feature is not very important in predicting churn
# For example, the phoneservices feature shown below

# from IPython.display import display
# for c in categorical:
#   print(c)
#   df_group = df_full_train.groupby([c]).churn.agg(['mean', 'count'])
#   df_group['diff'] = df_group['mean'] - overall_churn_rate
#   df_group['risk_ratio'] = df_group['mean'] / overall_churn_rate
#   display(df_group)
#   print()
#   print()

"""**Feature Importance : Mutual Information**

Mutual Information Concept from information technology tells us how much we can learn about one variable if we know the value of another. In this case, it'd tell us how much we can know about churn by observing all the other variables. The closer the mutual info. score is to 1, the more important the feature might be.
"""

# define a function that takes in the features and returns the mutual information score
def mutual_info_churn_score(series):
  return mutual_info_score(series, df_full_train.churn)

# apply the function to all categorical features
mutual_info = df_full_train[categorical].apply(mutual_info_churn_score)
mutual_info.sort_values(ascending=False)

"""From the above, the most important feature in predicting churn is contract, while the least important is gender

**Feature Importance for numerical features: Correlation**

I'll be checking the correlation coefficient to measure the degree of dependencies between two variables. When correlation is -ve, it means that an increase in one variable (x) leads to a decrease in another (y). While a +ve correlation means that an increase in one variable leads to an increase in another
"""

df_full_train[numerical].corrwith(df_full_train.churn)

"""The above results shows that the higher the customer tenure, the less likely they churn. Also, the more people pay (totalcharges) the less likely they are to churn. Also, the higher the monthly charges the more likely they are to churn. This corresponds to the result of the EDA

**Feature Scaling for Continous Variables**
"""

#feature scaling
sc = MinMaxScaler()
df_train['tenure'] = sc.fit_transform(df_train[['tenure']])
df_train['monthlycharges'] = sc.fit_transform(df_train[['monthlycharges']])
df_train['totalcharges'] = sc.fit_transform(df_train[['totalcharges']])

"""**One-Hot Encoding**"""

dv = DictVectorizer(sparse=False)

train_dicts = df_train[categorical + numerical].to_dict(orient='records')
train_dicts[0]

X_train = dv.fit_transform(train_dicts)
# dv.get_feature_names()
# X_train.head()

X_train.shape

val_dicts = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dicts)
X_val.shape

"""**Training Logistic Regression**"""

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(X_train, y_train)

# print the weights
model.coef_[0]

# print the bias charm/intercept
model.intercept_[0]

# print hard predictions (i.e predictions that have the exact labels)
model.predict(X_train)

# print soft predictions (i.e the probability)
model.predict_proba(X_train)

# we are only interested in the probability of a customer churning
y_pred = model.predict_proba(X_val)[:, 1]
y_pred

# considering customers with 50% or more probability of churning
churn_decision = (y_pred >= 0.5)

churn_customers = df_val[churn_decision].customerid # this way, we can select all customers with a higher chance of churning and maybe send promotional emails to them
churn_customers.to_dict()

# check the number of correct predictions that was made
df_pred = pd.DataFrame()
df_pred['probability'] = y_pred
df_pred['prediction'] = churn_decision.astype('int')
df_pred['actual'] = y_val
df_pred['correct'] = df_pred['prediction'] == df_pred['actual'] 
df_pred

# get the percentage of correct predictions
accuracy_score(y_val, y_pred>=0.5)

"""**Using the model**"""

dict_full_train = df_full_train[categorical + numerical].to_dict(orient="records")
X_full_train = dv.fit_transform(dict_full_train)
y_full_train = df_full_train.churn.values

model.fit(X_full_train, y_full_train)

dict_test = df_test[categorical + numerical].to_dict(orient="records")
X_test = dv.transform(dict_test)

y_pred = model.predict_proba(X_test)[:, 1]
churn_decision = (y_pred > 0.5)

"""**Evaluation**"""

accuracy_score(y_test, y_pred>=0.5)

confusion_matrix(y_test, y_pred>=0.5)

precision_score(y_test, y_pred>=0.5)

recall_score(y_test, y_pred>=0.5)

fpr, tpr, thresholds = roc_curve(y_test, y_pred)

plt.figure(figsize=(5,5))
plt.plot(fpr, tpr, label='Model')
plt.plot([1, 0], [1, 0], label='Random', linestyle='--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()

auc(fpr, tpr)

auc = roc_auc_score(y_test, y_pred)

"""**Cross Validation**"""

print(f'doing validation with C={C}')

kfold = KFold(n_splits=10, shuffle=True, random_state=1)
scores = []
fold = 0

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    print(f'auc on fold {fold} is {auc}')
    fold = fold + 1

print('validation results:')
print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))

# training the final model

print('training the final model')

train_dicts = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)
model = LogisticRegression(C=1.0, max_iter=1000)
model.fit(X_train, y_train)

dicts = df_val[categorical + numerical].to_dict(orient='records')
X = dv.transform(dicts)
y_pred = model.predict_proba(X)[:, 1]

auc = roc_auc_score(y_val, y_pred)
auc

scores.append(auc)

dict_full_train = df_full_train[categorical + numerical].to_dict(orient="records")
X_full_train = dv.fit_transform(dict_full_train)
y_full_train = df_full_train.churn.values

model.fit(X_full_train, y_full_train)
dict_test = df_test[categorical + numerical].to_dict(orient="records")
X_test = dv.transform(dict_test)
y_pred = model.predict_proba(X_test)[:, 1]
churn_decision = (y_pred > 0.5)

auc = roc_auc_score(y_test, y_pred)
print(f'auc={auc}')

"""**Saving the model**"""

output_file

model_file = 'model_C=1.0.bin'

f_out = open(output_file, 'wb')
pickle.dump((dv, model), f_out)
f_out.close

with open(output_file, 'wb') as f_out: 
    pickle.dump((dv, model), f_out)

print(f'the model is saved to {output_file}')

"""**Load the model**"""
with open(model_file, 'rb') as f_in: 
    dv, model = pickle.load(f_in)

dv, model

customer = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': 29.85
}

