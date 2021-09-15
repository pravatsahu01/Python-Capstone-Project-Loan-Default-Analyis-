#!/usr/bin/env python
# coding: utf-8

# ## Capstone Project - Vehicle Loan Default Prediction
# ### Pravat Kumar Sahu
# #### Simplilearn - PGP DA FEB 2021 Cohort 1
The data is about Finance sector and related to vehicle loan taken by customers and loan account status. The data has 233154 observations and 41 variables. The dataframe is mix of interger, object, datetype. To begin with the study of the data, let's first import all necessary packages and import the dataset to the jupyter notebook.
# In[1]:


import pandas as pd
import numpy as np
import os
import re
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.model_selection import train_test_split as split
from sklearn.linear_model import LinearRegression


# In[2]:


#Load the data using pandas.
importdata = pd.read_excel('/Users/priya/Pravat/Simplilearn Data Analytics/Class 5/project/project 2/data.xlsx')
importdata


# In[3]:


loandata = importdata


# ### 1. Preliminary Data analysis

# In[4]:


#find the structure of the data
loandata.info()


# In[5]:


loandata.head(5)


# In[6]:


loandata.columns


# In[7]:


#Check for null values in the data. Get the number of null values for each column.
loandata.isnull().sum()


# In[8]:


#Missing values found in Employement type column only. As it is catagorial data, fill the missing values with Mode value using pandas. 
loandata['Employment.Type'].fillna(loandata['Employment.Type'].mode()[0], inplace=True)


# In[9]:


#verify after fill NA.
loandata.isnull().sum()

Alternatively the we can also drop missing records if number of missing data points are less and dont make any impact if ignored.
To do this user can follow below code in pandas.
loandata.dropna(axis = 0, inplace = True)
# In[10]:


#Now let's check number of rows and coloumns in dataframe
loandata.shape


# In[11]:


#Let's check the if any duplicates in dataframe
loandata.duplicated().any()

Result : As the result is False, we can conclude no duplicates in the dataframe
# In[12]:


loandata.info()


# In[13]:


#Variable names in the data may not be in accordance with the identifier naming in Python, so let's change the variable names accordingly.


# In[14]:


loandata = loandata.rename(columns={"Date.of.Birth":"Date_of_Birth",
                                     "Employment.Type":"Employment_Type",
                                     "PERFORM_CNS.SCORE":"PERFORM_CNS_SCORE",
                                     "PERFORM_CNS.SCORE.DESCRIPTION":"PERFORM_CNS_SCORE_DESCRIPTION",
                                     "PRI.NO.OF.ACCTS":"PRI_NO_OF_ACCTS",
                                     "PRI.ACTIVE.ACCTS":"PRI_ACTIVE_ACCTS",
                                     "PRI.OVERDUE.ACCTS":"PRI_OVERDUE_ACCTS",
                                     "PRI.CURRENT.BALANCE":"PRI_CURRENT_BALANCE",
                                     "PRI.SANCTIONED.AMOUNT":"PRI_SANCTIONED_AMOUNT",
                                     "PRI.DISBURSED.AMOUNT":"PRI_DISBURSED_AMOUNT",
                                     "SEC.NO.OF.ACCTS":"SEC_NO_OF_ACCTS",
                                     "SEC.ACTIVE.ACCTS":"SEC_ACTIVE_ACCTS",
                                     "SEC.OVERDUE.ACCTS":"SEC_OVERDUE_ACCTS",
                                     "SEC.CURRENT.BALANCE":"SEC_CURRENT_BALANCE",
                                     "SEC.SANCTIONED.AMOUNT":"SEC_SANCTIONED_AMOUNT",
                                     "SEC.DISBURSED.AMOUNT":"SEC_DISBURSED_AMOUNT",
                                     "PRIMARY.INSTAL.AMT":"PRIMARY_INSTAL_AMT",
                                     "SEC.INSTAL.AMT":"SEC_INSTAL_AMT",
                                     "NEW.ACCTS.IN.LAST.SIX.MONTHS":"NEW_ACCTS_IN_LAST_SIX_MONTHS",
                                     "DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS":"DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS",
                                     "AVERAGE.ACCT.AGE":"AVERAGE_ACCT_AGE",
                                     "CREDIT.HISTORY.LENGTH":"CREDIT_HISTORY_LENGTH",
                                     "NO.OF_INQUIRIES":"NO_OF_INQUIRIES"})


# In[15]:


#Verify the Variable names after rename
loandata.info()


# ### 2. Performing EDA

# In[16]:


#Check the statistical description of the quantitative data variables
loandata.describe()

Above stats shows that the dataframe has 233154 unique customer id. 
Maximum loan disbursed amount is 990572, minimum is 13320 and averge loan disbursed amount is 54356. The standard devaition in loan disbursed amount is 12971.
# In[17]:


#How is the target variable distributed overall?

Target variables in the dataframe is loan default. The data type is a binary which has either 1 or 0 value. 
1 indicates customer who are defaulted in loan repaymemt and 0 indicates who are not.
so, to see how target variables are distributed we can use value_counts function as below.
# In[18]:


print(loandata.loan_default.value_counts())
loandata.loan_default.value_counts().plot.bar()
plt.title('Default Count')

From the above result, we can say that majority that is roughly 78% customers are non defaulters while less portion that is 22% are defaulters.
# In[19]:


#Study the distribution of the target variable across the various categories such as branch, city, state, branch, supplier, manufacturer, etc. 

To study this, let's use for loop function and plot various stacked bar chart to check how defaulters and non defaulers related to various categorial variables are distribuited accross all branch, city, state, suppliers, manufaturers etc.
# In[20]:


for i in ['branch_id','Current_pincode_ID','State_ID','supplier_id','manufacturer_id']:
    ct = pd.crosstab(loandata[i], loandata['loan_default'])
    ct.plot.bar(stacked = True,figsize=(20,7))
    plt.legend(labels=['Not Defaulted','Defaulted'])
    plt.show()

From the stacked bar chart, Branch id 2, 36 and 67 are among the list of branch which have highest defaulters.
State id 3, 4, 6 and 13 are among the list of states which have highest defaulters.
Manufacturer id 86 have highest defaulters.
For City and suppliers, it is difficult to infer how the target variables are distributed.We can draw a heat map to know the relationship between the target and independent variables.
# In[21]:


plt.figure(figsize=(12,8))
sns.heatmap(loandata.corr())


# In[22]:


#What are the different employment types given in the data? Can a strategy be developed to fill in the missing values (if any)? 

To get this let's use value count function on employment type and draw a bar chart to see how are the employment types in the dataframe.
# In[23]:


loandata['Employment_Type'].value_counts()


# In[24]:


loandata['Employment_Type'].value_counts().plot(kind='bar')


# In[25]:


# pie chart
labels = ['Self employed', 'Salaried']
sizes = loandata['Employment_Type'].value_counts()

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True)
ax1.axis('equal')
plt.show()

Since we have already filled the mising value with Mode Value previously there is no missing value now so we can say that there are 2 levels or category in Employed Type variable, they are Self employment and Salaried.let's check the data and plot bar chart and pie chart to express how different types of employment defines defaulter and non-defaulters.
# In[26]:


pct_loan_default = loandata['loan_default'].value_counts(normalize=True)*100
pct_loan_default

Total Defaulters are 21% and non defaulters are 78%. Let's first check the % defaulted customer for each employment category.
# In[27]:


print('% of salaried customer only who have defaulted:',
     np.round(loandata[(loandata['Employment_Type']=='Salaried') 
                       & (loandata['loan_default']==1)].shape[0]/(loandata[loandata['Employment_Type']=='Salaried'].shape[0])*100,2))

print('% of self employed customer only who have defaulted:',
     np.round(loandata[(loandata['Employment_Type']=='Self employed') 
                       & (loandata['loan_default']==1)].shape[0]/(loandata[loandata['Employment_Type']=='Self employed'].shape[0])*100,2))


# In[28]:


# Let's Bar Chart to draw the employment vs loan default chart
sns.countplot(x='Employment_Type',hue='loan_default',data=loandata)
plt.title('Employment Bar Graph')

Now, check and express how both types of employment defines defaulter and non-defaulters, lets create a cross tab and plot pie chart.
# In[29]:


emp_loan=pd.crosstab(loandata['Employment_Type'],loandata['loan_default'])
emp_loan


# In[30]:


emp_loan.groupby(['Employment_Type']).sum().plot(kind='pie', subplots=True, shadow = True,startangle=90,
figsize=(15,10), autopct='%1.1f%%')

Above pie chart shows that arround 57.3% of the customers self employed and 42.7% of the customers salaried by profession are non-defaulters while 60.7% of the customers who are self employed and 39.3% of the customers salaries by profession are defaulters.
# In[31]:


#Has age got something to do with defaulting? What is the distribution of age w.r.t. to defaulters and non-defaulters?

We have Date of Birth of the customer and the date of disbursal of loan from which we can to calculate the age of the customer at the the time of disbursement of loan. Let's draw a histogram to observe the distribution of the ages in the dataframe.
# In[32]:


loandata['age'] = pd.DatetimeIndex(loandata['DisbursalDate']).year - pd.DatetimeIndex(loandata['Date_of_Birth']).year
loandata['age'].plot.hist()
plt.title('Age Histogram')

Histogram plot - Noticed that customers age are between 18-69.
# In[33]:


loandata.age.describe()

We can plot a box plot to see the distribution of the age groups as well.
# In[34]:


loandata.boxplot('age')
plt.title('Age BoxPlot')

From the box plot, it is obervserd that avergae age is 34 and lowest age of customer taking loan is 18 while highest is 69. Age above 60 are outliers.Now, lets draw the relation between between age and loan default variable.
# In[35]:


sns.boxplot(x='loan_default', y='age',data=loandata)
plt.title('Age BoxPlot')

From the box plot, observed that age are almost same for defaulters and non defaulters.
# In[36]:


#What type of ID was presented by most of the customers as proof?

To find out this which document was presented most, we can do value_counts for each document type variables.
# In[37]:


print(loandata['Aadhar_flag'].value_counts())
print(loandata['PAN_flag'].value_counts())
print(loandata['VoterID_flag'].value_counts())
print(loandata['Driving_flag'].value_counts())
print(loandata['Passport_flag'].value_counts())


# In[38]:


print(loandata['Aadhar_flag'].value_counts(normalize=True)*100)
print(loandata['PAN_flag'].value_counts(normalize=True)*100)
print(loandata['VoterID_flag'].value_counts(normalize=True)*100)
print(loandata['Driving_flag'].value_counts(normalize=True)*100)
print(loandata['Passport_flag'].value_counts(normalize=True)*100)

From the above result, we can infer that Aadhar card is the document which was presented most by customer (195924 customers) followed by Voter id (33794 customer)
# In[39]:


#Study the credit bureau score distribution. How is the distribution for defaulters vs non-defaulters? Explore in detail.

PERFORM_CNS_SCORE is the variable which indicates Bureau or CIBIL Score for each customer. Let's study this by checking basic stat function.
# In[40]:


loandata['PERFORM_CNS_SCORE'].describe()


# In[41]:


loandata['PERFORM_CNS_SCORE'].plot(kind='hist')
plt.show()

We can see that in the distribution, the avergae score is 289.46 and maximum score is 890. The minimum score is 0 which indicates there is no score available for that customer.Now let's filter defaulter and non defaulters and separetely study their score.
# In[42]:


cibil_non_default = loandata[loandata['loan_default']==0]['PERFORM_CNS_SCORE']
cibil_default = loandata[loandata['loan_default']==1]['PERFORM_CNS_SCORE']


# In[43]:


pd.DataFrame([cibil_non_default.describe(), cibil_default.describe()], index=['non_defaulters','defaulters'])

We can observe a difference in the mean and median cibil scores among the defaulters and non defaulters. 
The mean and median in cibil scores are higher for non defaulters.
# In[44]:


sns.distplot( a = cibil_non_default, color='blue', label = 'Non Defaulter')
sns.distplot(a = cibil_default, color='red', label = 'Defaulter')

plt.legend()
plt.show()

Above displot indicates the CIBIL score distribution is looking almost similar for defaulters and non defaulters customers.
# In[45]:


#Explore the primary and secondary account details. Is the information in some way related to loan default probability ?

To study this, let's check the histo plot for primary and secondary account.
# In[46]:


loandata['PRI_NO_OF_ACCTS'].plot(kind='hist')
plt.show()


# In[47]:


loandata['SEC_NO_OF_ACCTS'].plot(kind='hist')
plt.show()


# In[48]:


#Checking the correlation between primary and loan deafult vairable
sns.heatmap(loandata[['PRI_NO_OF_ACCTS','loan_default']].corr(),annot=True)
plt.show()


# In[49]:


#Checking the correlation between seondary and loan deafult vairable
sns.heatmap(loandata[['SEC_NO_OF_ACCTS','loan_default']].corr(),annot=True)
plt.show()

There is no correlation between no of primary or no of secondary account with loan default variable.Now, let's find out if this information is some way related to loan default probability.
# In[50]:


pri_acct_non_default = loandata[loandata['loan_default']==0]['PRI_NO_OF_ACCTS']
pri_acct_default = loandata[loandata['loan_default']==1]['PRI_NO_OF_ACCTS']


# In[51]:


pd.DataFrame([pri_acct_non_default.describe(), pri_acct_default.describe()], index=['non_defaulters','defaulters'])


# In[52]:


sec_acct_non_default = loandata[loandata['loan_default']==0]['SEC_NO_OF_ACCTS']
sec_acct_default = loandata[loandata['loan_default']==1]['SEC_NO_OF_ACCTS']


# In[53]:


pd.DataFrame([sec_acct_non_default.describe(), sec_acct_default.describe()], index=['non_defaulters','defaulters'])

Observed that for customers having primary accounts are maximum defaulters and customers having secondary accounts are less defaulters.
# In[54]:


#Is there a difference between the sanctioned and disbursed amount of primary & secondary loans. Study the difference by providing apt statistics and graphs.

We value count function and find oout the % of the data for the sanctioned and disbursed amount for both primary and secondary account and see if there are any differnces.
# In[55]:


pri_sanc_amt_counts = loandata['PRI_SANCTIONED_AMOUNT'].value_counts()
pri_sanc_amt_counts_percent = loandata['PRI_SANCTIONED_AMOUNT'].value_counts(normalize=True)*100

pd.DataFrame({'counts':pri_sanc_amt_counts,'percent_of_data':pri_sanc_amt_counts_percent})


# In[56]:


pri_disb_amt_counts = loandata['PRI_DISBURSED_AMOUNT'].value_counts()
pri_disb_amt_counts_percent = loandata['PRI_DISBURSED_AMOUNT'].value_counts(normalize=True)*100

pd.DataFrame({'counts':pri_disb_amt_counts,'percent_of_data':pri_disb_amt_counts_percent})


# In[57]:


pri_acct_loan_amt =['PRI_SANCTIONED_AMOUNT', 'PRI_DISBURSED_AMOUNT']


# In[58]:


count = 1
plt.figure(figsize=(20,10))
for i in pri_acct_loan_amt:
    plt.subplot(2,2,count)
    sns.distplot(loandata[i])
    count += 1
plt.show()

For primary account holder customers there is not much differnce between sanctioned amount and disbursal amount as arround 59% of the accounts no amount was sanctioned or disbursed.
# In[59]:


sec_sanc_amt_counts = loandata['SEC_SANCTIONED_AMOUNT'].value_counts()
sec_sanc_amt_counts_percent = loandata['SEC_SANCTIONED_AMOUNT'].value_counts(normalize=True)*100

pd.DataFrame({'counts':sec_sanc_amt_counts,'percent_of_data':sec_sanc_amt_counts_percent})


# In[60]:


sec_disb_amt_counts = loandata['SEC_DISBURSED_AMOUNT'].value_counts()
sec_disb_amt_counts_percent = loandata['SEC_DISBURSED_AMOUNT'].value_counts(normalize=True)*100

pd.DataFrame({'counts':sec_disb_amt_counts,'percent_of_data':sec_disb_amt_counts_percent})


# In[61]:


sec_acct_loan_amt =['SEC_SANCTIONED_AMOUNT', 'SEC_DISBURSED_AMOUNT']


# In[62]:


count=1
plt.figure(figsize=(20,10))
for i in sec_acct_loan_amt:
    plt.subplot(2,2,count)
    sns.distplot(loandata[i])
    count+=1
plt.show()

For secondary account holder customers too there is not much differnce between sanctioned amount and disbursal amount as arround 98% of the accounts no amount was sanctioned or disbursed.Hypothesis Testing : Is there a any differnce between sanctioned and disbursed amount for Primary account and secondary account?

Alternate Hypthoesis (Ha) : mu(sanctioned) - mu(disbursed)  = 0, (there is no differnce between sanctioned and disbursed amount for Primary and secondary account)
Null Hypothesis (H0) : mu(sanctioned) - mu(disbursed) <= 0, (there is differnce between sanctioned and disbursed amount for Primary and secondary account)
# In[63]:


stats.ttest_ind(loandata.PRI_SANCTIONED_AMOUNT, loandata.PRI_DISBURSED_AMOUNT)


# In[64]:


stats.ttest_ind(loandata.SEC_SANCTIONED_AMOUNT, loandata.SEC_DISBURSED_AMOUNT)

From the above T test result, it is oberved that for both primary and secondary account case the P value is greater than alpha 0.05, so we can conclude that we failed to reject null hypothesis that means there are differnces in sanctioned and disbursed amount for both primary and secondary account.
# In[65]:


#Do customer who make higher no. of enquiries end up being higher risk candidates? 

first let's identify how many customers or % of customers have made an inquiry before taking a loan.
# In[66]:


enquiries_counts = loandata['NO_OF_INQUIRIES'].value_counts()
enquiries_counts_percent = loandata['NO_OF_INQUIRIES'].value_counts(normalize=True)*100

pd.DataFrame({'counts':enquiries_counts,'percent_of_data':enquiries_counts_percent})

Above statistics shows that most (approx 87%) of the customers have not made any enquiries regarding loans.
# In[67]:


no_of_loan_inquiries = pd.crosstab(index=loandata['NO_OF_INQUIRIES'], columns=loandata['loan_default'])
no_of_loan_inquiries['pct_default'] = (no_of_loan_inquiries[1]/no_of_loan_inquiries.sum(axis=1))*100
no_of_loan_inquiries

From the above result, we can infer that except for majority cases, as the number of enquires increase, there is an increase in the default percentage of customers and so being end up being higher risk candidates for the bank.
# In[68]:


#Is credit history, i.e. new loans in last six months, loans defaulted in last six months, time since first loan, etc., a significant factor in estimating probability of loan defaulters?

Before we start our exploration, let's first create a function to replace alpha numeric values in CREDIT_HISTORY_LENGTH vairaible and change them to months. This will also change the data type to float type from object type.
# In[69]:


def duration(dur):
    yrs = int(dur.split(' ')[0].replace('yrs',''))
    mon = int(dur.split(' ')[1].replace('mon',''))
    return yrs*12+mon


# In[70]:


loandata['CREDIT_HISTORY_LENGTH'] = loandata['CREDIT_HISTORY_LENGTH'].apply(duration)


# In[71]:


#verify to check the data type after function apply
loandata['CREDIT_HISTORY_LENGTH'].describe()

Now, let's see how the target variable is related to CREDIT_HISTORY_LENGTH variable.
# In[72]:


credit_non_default = loandata[loandata['loan_default'] == 0]['CREDIT_HISTORY_LENGTH']
credit_default = loandata[loandata['loan_default'] == 1]['CREDIT_HISTORY_LENGTH']


# In[73]:


pd.DataFrame([credit_non_default.describe(), credit_default.describe()], index=['non_defaulters','defaulters'])

Above stats shows that the mean and standard deviation are higher for non defaulters customers.
# In[74]:


sns.distplot(loandata['CREDIT_HISTORY_LENGTH'])
plt.show()

From the displot observed that CREDIT_HISTORY_LENGTH is Highly right skewed.
# In[75]:


new_acct_counts = loandata['NEW_ACCTS_IN_LAST_SIX_MONTHS'].value_counts()
new_acct_counts_percent = loandata['NEW_ACCTS_IN_LAST_SIX_MONTHS'].value_counts(normalize=True)*100

pd.DataFrame({'counts':new_acct_counts,'percent_of_data':new_acct_counts_percent})

Most of customers have not opened any new account in the last 6 months.Now, lets check loans defaulted in last six months.
# In[76]:


delinquent_acct_counts = loandata['DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS'].value_counts()
delinquent_acct_counts_percent = loandata['DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS'].value_counts(normalize=True)*100

pd.DataFrame({'counts':delinquent_acct_counts,'delinquent_acct_counts':delinquent_acct_counts_percent})

We can obseved that can see that 92% of customers are not defaulted in last six months and 8% of customers are deafulted at least once or more than once in last 6 month.
# ### Perform Model Building and Predict

# In[77]:


#Perform logistic regression modelling, predict the outcome for the test data, and validate the results using the confusion matrix.


# In[78]:


#dropping unnecessary columns
# MobileNo_Avl_Flag - All values are 1
# Date_of_Birth , DisbursalDate -  Already used to compute age
# PERFORM_CNS_SCORE_DESCRIPTION - Score is already in dataset

loandata = loandata.drop(['MobileNo_Avl_Flag','Date_of_Birth','AVERAGE_ACCT_AGE','DisbursalDate','PERFORM_CNS_SCORE_DESCRIPTION'],axis=1)


# In[79]:


loandata_new = pd.get_dummies(loandata,drop_first=True) 
print(loandata_new.columns)


# In[80]:


#train test split
X = loandata_new.drop('loan_default',axis=1)
y = loandata_new['loan_default']


# In[81]:


X.shape


# In[82]:


y.shape


# In[83]:


from sklearn.preprocessing import StandardScaler


# In[84]:


sc = StandardScaler()


# In[85]:


sc.fit(X)


# In[86]:


Xt_z = pd.DataFrame(sc.transform(X), columns =X.columns)


# In[87]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xt_z, y, stratify=y, random_state=12)


# In[88]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
fit=lr.fit(X_train, y_train)


# In[89]:


y_pred= lr.predict(X_test)


# In[90]:


pred_vs_actual_outcome = pd.crosstab(index = y_pred, columns = y_test)
pred_vs_actual_outcome


# In[91]:


from sklearn.metrics import confusion_matrix
from sklearn import metrics
print(confusion_matrix(y_pred,y_test))


# In[92]:


cm = confusion_matrix(y_pred,y_test)


# In[93]:


#accuracy = (TN+TP)/(ALL)
accuracy_lr = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])*100
print('accuracy' ,accuracy_lr)

#precision = (TP)/(TP+FP)
precision_lr = (cm[1,1])/(cm[1,1]+cm[1,0])*100
print('precision' ,precision_lr)


#recall or sensitivity(TPR) for class1 = (TP)/(TP+FN)
recall_lr_class_1 = (cm[1,1])/(cm[1,1]+cm[0,1])*100
print('class1-recall' ,recall_lr_class_1)

#recall or specificity(TNR) for class 0 = (TN)/(TN+FP)
recall_lr_class_0 = (cm[0,0])/(cm[0,0]+cm[0,1])*100
print('class0-recall' ,recall_lr_class_0)

#F1_Score or Harmonic mean(HM) of precision and recall  = 2*precision*recall/(precision + recall)
F1_Score_lr = (2*precision_lr*recall_lr_class_1*recall_lr_class_0)/(precision_lr+recall_lr_class_1+recall_lr_class_0)
print('F1_Score' ,F1_Score_lr)

From above values, we can observe that accuracy of the model is 78% with precision that is the repeatatibilty of the predcited results is 43%.
Recall or sensistvity of the model for class 1 is 0.54 and for class 0 is 78.35%. The model is giving very good sensitivty for class 0.
The F1 score of the model is 30.20% and is too low to accpet the model's prediction.Now, export the data to excel for further vizualiation in Tableau.
# In[94]:


loandata_output = loandata_new

We can add and drop variables and can focus on selected variables for vizualiation in Tableau.
# In[95]:


loandata_output = pd.concat([loandata_output,importdata['PERFORM_CNS.SCORE.DESCRIPTION']], axis = 1)


# In[96]:


loandata_output = loandata_output.drop(['asset_cost','ltv','Employee_code_ID','PRI_ACTIVE_ACCTS',
                                        'PRI_OVERDUE_ACCTS','PRI_CURRENT_BALANCE','SEC_ACTIVE_ACCTS','SEC_OVERDUE_ACCTS','SEC_CURRENT_BALANCE',
                                       'PRIMARY_INSTAL_AMT','SEC_INSTAL_AMT'],axis=1)


# In[97]:


loandata_output.to_excel('/Users/priya/Pravat/Simplilearn Data Analytics/Class 5/project/project 2/loandata_output.xlsx', index = False)

