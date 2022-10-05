import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

data_scoring = pd.read_csv(r'C:\Users\Админ\Desktop\Папки\кредитный скоринг\credit_train.csv')
data_scoring_test = pd.read_csv(r'C:\Users\Админ\Desktop\Папки\кредитный скоринг\credit_test.csv')

X_train = data_scoring.drop(('Loan Status'), axis=1)
Y_train = data_scoring['Loan Status']

X_test = data_scoring_test


X_train, Y_train = make_classification(n_samples=10000)
X_test, Y_test = make_classification(n_samples=10000)


rfc = RandomForestClassifier(n_estimators=10000, bootstrap=False, criterion='entropy', max_depth=3, max_features='sqrt', min_samples_leaf=4)
rfc.fit(X_train,Y_train)
Y_test_rfc = rfc.predict(X_test)
X_test_rfc = X_test[:, 0]


itog = pd.DataFrame({"Loan ID": X_test_rfc,
                   "Loan Status": Y_test_rfc
                   })

itog.loc[(itog["Loan Status"] == 0), "Loan Status"] = "Fully Paid"
itog.loc[(itog["Loan Status"] == 1), "Loan Status"] = "Charged Off"
itog.to_csv("itog.csv", index=False)