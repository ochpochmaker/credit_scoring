import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

data_scoring = pd.read_csv(r'C:\Users\Админ\Desktop\Папки\кредитный скоринг\credit_train.csv')
data_scoring_test = pd.read_csv(r'C:\Users\Админ\Desktop\Папки\кредитный скоринг\credit_test.csv')

data_scoring['Loan ID'] = pd.factorize(data_scoring['Loan ID'])[0]
data_scoring['Customer ID'] = pd.factorize(data_scoring['Customer ID'])[0]
data_scoring.drop(['Customer ID', 'Loan ID'], axis=1, inplace=False)

data_scoring['Years in current job'].fillna('10+ years', inplace=True)
data_scoring['Credit Score'].fillna(data_scoring['Credit Score'].median(), inplace=True)
data_scoring['Annual Income'].fillna(data_scoring['Annual Income'].median(), inplace=True)
data_scoring['Months since last delinquent'].fillna(data_scoring['Months since last delinquent'].median(), inplace=True)
data_scoring['Maximum Open Credit'].fillna(data_scoring['Maximum Open Credit'].median(), inplace=True)
data_scoring['Bankruptcies'].fillna(data_scoring['Bankruptcies'].median(), inplace=True)
data_scoring['Tax Liens'].fillna(data_scoring['Tax Liens'].median(), inplace=True)
data_scoring = pd.concat([data_scoring,
                          pd.get_dummies(data_scoring['Years in current job'], prefix="Years in current job"),
                          pd.get_dummies(data_scoring['Term'], prefix="Term"),
                          pd.get_dummies(data_scoring['Home Ownership'], prefix="Home Ownership"),
                          pd.get_dummies(data_scoring['Purpose'], prefix="Purpose"),
                          ],
                         axis=1)
data_scoring.drop(['Years in current job', 'Term', 'Home Ownership', 'Purpose'], axis=1, inplace=True)
data_scoring['Loan Status']= data_scoring['Loan Status'].factorize()[0]


X_train = data_scoring.drop(('Loan Status'), axis=1)
Y_train = data_scoring['Loan Status']
X_test = data_scoring_test


#X_train.info()
#print(X_train)
X_train, Y_train = make_classification(n_samples=10000)
X_test, Y_test = make_classification(n_samples=10000)
#print(X_train)

rfc = RandomForestClassifier(n_estimators=10000, bootstrap=False, criterion='entropy', max_depth=3, max_features='sqrt', min_samples_leaf=4)
rfc.fit(X_train,Y_train)
Y_test_rfc = rfc.predict(X_test)
X_test_rfc = X_test[:, 0]

itog = pd.DataFrame({"Loan ID": X_test_rfc,"Loan Status": Y_test_rfc})

itog.loc[(itog["Loan Status"] == 0), "Loan Status"] = "Fully Paid"
itog.loc[(itog["Loan Status"] == 1), "Loan Status"] = "Charged Off"
itog.to_csv("itog.csv", index=False)