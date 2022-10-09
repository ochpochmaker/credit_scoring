import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn import datasets, ensemble
from sklearn.preprocessing import OneHotEncoder
import sklearn as sk
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder

Video_Games = pd.read_csv(r'C:\Users\Админ\Desktop\Папки\findata-videogames\Video_Games.csv')

#категорриальные признаки заменяем  на моду
Video_Games['Genre'].fillna(Video_Games['Genre'].mode(), inplace=True)
Video_Games['Rating'].fillna(Video_Games['Rating'].mode(), inplace=True)
#числовые признаки заменяем  на медиану
Video_Games['NA_Sales'].fillna(Video_Games['NA_Sales'].median(), inplace=True)
Video_Games['EU_Sales'].fillna(Video_Games['EU_Sales'].median(), inplace=True)
Video_Games['Other_Sales'].fillna(Video_Games['Other_Sales'].median(), inplace=True)
# остальное заменяем на нули
Video_Games['Publisher'].fillna(-1000)
Video_Games['Developer'].fillna(-1000)
Video_Games['Critic_Score'].fillna(-1000)
Video_Games['Critic_Count'].fillna(-1000)
Video_Games['User_Score'].fillna(-1000)
Video_Games['User_Count'].fillna(-1000)

#добавляем хуеву гору новых флагов
ohe = OneHotEncoder(sparse=False)

Genre_flg = ohe.fit_transform(Video_Games['Genre'].values.reshape(-1, 1))
tmp = pd.DataFrame(Genre_flg, columns=['Genre=' + str(i) for i in range(Genre_flg.shape[1])])
Video_Games = pd.concat([Video_Games, tmp], axis=1)

Platform_flg = ohe.fit_transform(Video_Games['Platform'].values.reshape(-1, 1))
tmp = pd.DataFrame(Platform_flg, columns=['Platform=' + str(i) for i in range(Platform_flg.shape[1])])
Video_Games = pd.concat([Video_Games, tmp], axis=1)

Publisher_flg = ohe.fit_transform(Video_Games['Publisher'].values.reshape(-1, 1))
tmp = pd.DataFrame(Publisher_flg, columns=['Publisher=' + str(i) for i in range(Publisher_flg.shape[1])])
Video_Games = pd.concat([Video_Games, tmp], axis=1)

Developer_flg = ohe.fit_transform(Video_Games['Developer'].values.reshape(-1, 1))
tmp = pd.DataFrame(Developer_flg, columns=['Developer=' + str(i) for i in range(Developer_flg.shape[1])])
Video_Games = pd.concat([Video_Games, tmp], axis=1)

Rating_flg = ohe.fit_transform(Video_Games['Rating'].values.reshape(-1, 1))
tmp = pd.DataFrame(Rating_flg, columns=['Rating=' + str(i) for i in range(Rating_flg.shape[1])])
Video_Games = pd.concat([Video_Games, tmp], axis=1)

#причесываем числовые признаки
le = LabelEncoder()
le.fit(Video_Games.Year_of_Release)
Video_Games['Year_of_Release_le'] = le.transform(Video_Games.Year_of_Release)
le.fit(Video_Games.NA_Sales)
Video_Games['NA_Sales_le'] = le.transform(Video_Games.NA_Sales)
le.fit(Video_Games.EU_Sales)
Video_Games['EU_Sales_le'] = le.transform(Video_Games.EU_Sales)
le.fit(Video_Games.Other_Sales)
Video_Games['Other_Sales_le'] = le.transform(Video_Games.Other_Sales)
le.fit(Video_Games.Critic_Score)
Video_Games['Critic_Score_le'] = le.transform(Video_Games.Critic_Score)
le.fit(Video_Games.Critic_Count)
Video_Games['Critic_Count_le'] = le.transform(Video_Games.Critic_Count)
le.fit(Video_Games.User_Score)
Video_Games['User_Score_le'] = le.transform(Video_Games.User_Score)
le.fit(Video_Games.User_Count)
Video_Games['User_Count_le'] = le.transform(Video_Games.User_Count)

#убираем ненужные признаки
Video_Games = Video_Games.drop(('Name'), axis=1)
Video_Games = Video_Games.drop(('Platform'), axis=1)
Video_Games = Video_Games.drop(('Genre'), axis=1)
Video_Games = Video_Games.drop(('Publisher'), axis=1)
Video_Games = Video_Games.drop(('Developer'), axis=1)
Video_Games = Video_Games.drop(('Rating'), axis=1)



#делим датасет на параметры и таргеты
X = Video_Games.drop(('JP_Sales'), axis=1)
y = Video_Games['JP_Sales']
#кросс-валидация
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)

#используем GradientBoostingRegressor, потому что препод использовал GradientBoostingRegressor
params = {
    "n_estimators": 500,
    "max_depth": 4,
    "min_samples_split": 5,
    "learning_rate": 0.01,
    "loss": "squared_error",
}

X_train, Y_train = make_classification(n_samples=len(X_train))
gbr = ensemble.GradientBoostingRegressor(**params)
gbr.fit(X_train, y_train)


Y_test_gbr = gbr.predict(X_test)
X_test_gbr = X_test[:, 0]

Video_Games_Itog = pd.DataFrame({"Id": X_test_gbr, "JP_Sales": Y_test_gbr})
print(Video_Games_Itog)