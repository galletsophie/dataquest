## 1. Recap ##

import pandas as pd

loans = pd.read_csv('filtered_loans_2007.csv')
null_counts = loans.isnull().sum()
print(null_counts.sort_values(ascending = False))

## 2. Handling missing values ##

loans = loans.drop('pub_rec_bankruptcies', axis=1)
loans = loans.dropna()
loans.dtypes.value_counts()

## 3. Text columns ##

object_columns_df = loans.select_dtypes(include=['object'])
object_columns_df.iloc[0]

## 5. First 5 categorical columns ##

cols = ['home_ownership', 'verification_status', 'emp_length', 'term', 'addr_state']

for col in cols:
    print(loans[col].value_counts())

## 6. The reason for the loan ##

print(loans["purpose"].value_counts())
print(loans["title"].value_counts())

## 7. Categorical columns ##

mapping_dict = {
    "emp_length": {
        "10+ years": 10,
        "9 years": 9,
        "8 years": 8,
        "7 years": 7,
        "6 years": 6,
        "5 years": 5,
        "4 years": 4,
        "3 years": 3,
        "2 years": 2,
        "1 year": 1,
        "< 1 year": 0,
        "n/a": 0
    }
}

loans = loans.drop(['last_credit_pull_d', 'addr_state', 'title', 'earliest_cr_line'], axis=1)
int_rate = loans['int_rate'].str.rstrip('%').astype('float')
revol_util = loans['revol_util'].str.rstrip('%').astype('float')
loans['int_rate'] = int_rate
loans['revol_util'] = revol_util
loans = loans.replace(mapping_dict)

## 8. Dummy variables ##

dummy_df = pd.get_dummies(loans[['home_ownership','verification_status','purpose','term']])
loans = pd.concat([loans, dummy_df], axis=1)
loans = loans.drop(['home_ownership','verification_status','purpose','term'], axis=1)


