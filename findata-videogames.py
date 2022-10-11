import pandas as pd
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn import datasets, ensemble
from sklearn.preprocessing import OneHotEncoder
import sklearn as sk
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder

Video_Games_train = pd.read_csv(r'C:\Users\Админ\Desktop\Папки\findata-videogames\Video_Games.csv')
Video_Games_test = pd.read_csv(r'C:\Users\Админ\Desktop\Папки\findata-videogames\Video_Games_Test.csv')


#категорриальные признаки заменяем  на моду
Video_Games_train['Genre'].fillna(Video_Games_train['Genre'].mode(), inplace=True)
Video_Games_train['Rating'].fillna(Video_Games_train['Rating'].mode(), inplace=True)

Video_Games_test['Genre'].fillna(Video_Games_test['Genre'].mode(), inplace=True)
Video_Games_test['Rating'].fillna(Video_Games_test['Rating'].mode(), inplace=True)


#числовые признаки заменяем  на медиану
Video_Games_train['NA_Sales'].fillna(Video_Games_train['NA_Sales'].median(), inplace=True)
Video_Games_train['EU_Sales'].fillna(Video_Games_train['EU_Sales'].median(), inplace=True)
Video_Games_train['Other_Sales'].fillna(Video_Games_train['Other_Sales'].median(), inplace=True)

Video_Games_test['NA_Sales'].fillna(Video_Games_test['NA_Sales'].median(), inplace=True)
Video_Games_test['EU_Sales'].fillna(Video_Games_test['EU_Sales'].median(), inplace=True)
Video_Games_test['Other_Sales'].fillna(Video_Games_test['Other_Sales'].median(), inplace=True)


# остальное заменяем на нули
Video_Games_train['Year_of_Release'].fillna(Video_Games_train['Year_of_Release'].median(), inplace=True)
Video_Games_train['Publisher'].fillna('0', inplace=True)
Video_Games_train['Developer'].fillna('0', inplace=True)
Video_Games_train['Critic_Score'].fillna(0, inplace=True)
Video_Games_train['Critic_Count'].fillna(0, inplace=True)
Video_Games_train['User_Score'].fillna(0, inplace=True)
Video_Games_train['User_Count'].fillna(0, inplace=True)

Video_Games_test['Year_of_Release'].fillna(Video_Games_test['Year_of_Release'].median(), inplace=True)
Video_Games_test['Publisher'].fillna('0', inplace=True)
Video_Games_test['Developer'].fillna('0', inplace=True)
Video_Games_test['Critic_Score'].fillna(0, inplace=True)
Video_Games_test['Critic_Count'].fillna(0, inplace=True)
Video_Games_test['User_Score'].fillna(0, inplace=True)
Video_Games_test['User_Count'].fillna(0, inplace=True)


#добавляем хуеву гору новых флагов
ohe = OneHotEncoder(sparse=False)

Genre_flg = ohe.fit_transform(Video_Games_train['Genre'].values.reshape(-1, 1))
tmp = pd.DataFrame(Genre_flg, columns=['Genre=' + str(i) for i in range(Genre_flg.shape[1])])
Video_Games_train = pd.concat([Video_Games_train, tmp], axis=1)

Platform_flg = ohe.fit_transform(Video_Games_train['Platform'].values.reshape(-1, 1))
tmp = pd.DataFrame(Platform_flg, columns=['Platform=' + str(i) for i in range(Platform_flg.shape[1])])
Video_Games_train = pd.concat([Video_Games_train, tmp], axis=1)

Publisher_flg = ohe.fit_transform(Video_Games_train['Publisher'].values.reshape(-1, 1))
tmp = pd.DataFrame(Publisher_flg, columns=['Publisher=' + str(i) for i in range(Publisher_flg.shape[1])])
Video_Games_train = pd.concat([Video_Games_train, tmp], axis=1)

Developer_flg = ohe.fit_transform(Video_Games_train['Developer'].values.reshape(-1, 1))
tmp = pd.DataFrame(Developer_flg, columns=['Developer=' + str(i) for i in range(Developer_flg.shape[1])])
Video_Games_train = pd.concat([Video_Games_train, tmp], axis=1)

Rating_flg = ohe.fit_transform(Video_Games_train['Rating'].values.reshape(-1, 1))
tmp = pd.DataFrame(Rating_flg, columns=['Rating=' + str(i) for i in range(Rating_flg.shape[1])])
Video_Games_train = pd.concat([Video_Games_train, tmp], axis=1)


Genre_flg = ohe.fit_transform(Video_Games_test['Genre'].values.reshape(-1, 1))
tmp = pd.DataFrame(Genre_flg, columns=['Genre=' + str(i) for i in range(Genre_flg.shape[1])])
Video_Games_test = pd.concat([Video_Games_test, tmp], axis=1)

Platform_flg = ohe.fit_transform(Video_Games_test['Platform'].values.reshape(-1, 1))
tmp = pd.DataFrame(Platform_flg, columns=['Platform=' + str(i) for i in range(Platform_flg.shape[1])])
Video_Games_test = pd.concat([Video_Games_test, tmp], axis=1)

Publisher_flg = ohe.fit_transform(Video_Games_test['Publisher'].values.reshape(-1, 1))
tmp = pd.DataFrame(Publisher_flg, columns=['Publisher=' + str(i) for i in range(Publisher_flg.shape[1])])
Video_Games_test = pd.concat([Video_Games_test, tmp], axis=1)

Developer_flg = ohe.fit_transform(Video_Games_test['Developer'].values.reshape(-1, 1))
tmp = pd.DataFrame(Developer_flg, columns=['Developer=' + str(i) for i in range(Developer_flg.shape[1])])
Video_Games_test = pd.concat([Video_Games_test, tmp], axis=1)

Rating_flg = ohe.fit_transform(Video_Games_test['Rating'].values.reshape(-1, 1))
tmp = pd.DataFrame(Rating_flg, columns=['Rating=' + str(i) for i in range(Rating_flg.shape[1])])
Video_Games_test = pd.concat([Video_Games_test, tmp], axis=1)


# #причесываем числовые признаки
# le = LabelEncoder()
# le.fit(Video_Games_train.Year_of_Release)
# Video_Games_train['Year_of_Release_le'] = le.transform(Video_Games_train.Year_of_Release)


#убираем ненужные признаки
Video_Games_train = Video_Games_train.drop(('Name'), axis=1)
Video_Games_train = Video_Games_train.drop(('Platform'), axis=1)
Video_Games_train = Video_Games_train.drop(('Genre'), axis=1)
Video_Games_train = Video_Games_train.drop(('Publisher'), axis=1)
Video_Games_train = Video_Games_train.drop(('Developer'), axis=1)
Video_Games_train = Video_Games_train.drop(('Rating'), axis=1)

Video_Games_test = Video_Games_test.drop(('Name'), axis=1)
Video_Games_test = Video_Games_test.drop(('Platform'), axis=1)
Video_Games_test = Video_Games_test.drop(('Genre'), axis=1)
Video_Games_test = Video_Games_test.drop(('Publisher'), axis=1)
Video_Games_test = Video_Games_test.drop(('Developer'), axis=1)
Video_Games_test = Video_Games_test.drop(('Rating'), axis=1)


#поиск нулов
val = Video_Games_train.isna().any()[lambda x: x]
print(val)
val2 = Video_Games_test.isna().any()[lambda x: x]
print(val2)

#делим датасет на параметры и таргеты
X_train = Video_Games_train.drop(('JP_Sales'), axis=1)
Y_train = Video_Games_train['JP_Sales']
X_test = Video_Games_test.drop(('Id'), axis=1)
Test_Id = Video_Games_test['Id']
print('X_train',X_train.shape)
print('Y_train',Y_train.shape)
print('X_test',X_test.shape)
print('Test_Id',X_train.shape)
# #кросс-валидация
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X_train, Y_train, test_size=0.33, random_state=33)
print('X_train',X_train.shape)
print('Y_train',Y_train.shape)
print('X_test',X_test.shape)
print('Test_Id',X_train.shape)

print('X_train1',X_train.shape)
print('Y_train1',Y_train.shape)
print('X_test1',X_test.shape)
print('Y_test1',X_train.shape)

#используем GradientBoostingRegressor, потому что препод использовал GradientBoostingRegressor
params = {
    "n_estimators": 500,
    "max_depth": 4,
    "min_samples_split": 5,
    "learning_rate": 0.01,
    "loss": "squared_error",
}

print(X_train.shape)
gbr = ensemble.GradientBoostingRegressor(**params)
gbr.fit(X_train1, Y_train1)
Y_test_gbr = gbr.predict(X_test1)
error = mean_absolute_error(Y_test1, Y_test_gbr)
print(error)

gbr.fit(X_train, Y_train)
Y_test_gbr = gbr.predict(X_test)

itog = pd.DataFrame({"Id": Test_Id,"JP_Sales": Y_test_gbr})
itog.to_csv("itog2.csv", index=False)