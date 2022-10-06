import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn import datasets, ensemble

#для кросс-валидации
import sklearn as sk
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC

Video_Games = pd.read_csv(r'C:\Users\Админ\Desktop\Папки\findata-videogames\Video_Games.csv')

#анализ датасета
print(Video_Games.head())#Что есть наши данные
print(Video_Games.shape) # мы увидим информацию о размерности нашего датафрейма
print(Video_Games.info()) # покажет информацию о размерности данных
print(Video_Games.describe()) # показывает статистики count,mean, std, min, 25%-50%-75% percentile, max
print(Video_Games.nunique()) # количество уникальных значений для каждого столбца
#Также было бы неплохо увидеть информацию о количестве каждого уникального значения для каждого столбца в наборе данных
feature_names = Video_Games.columns.tolist()
for column in feature_names:
    print (column)
    print (Video_Games[column].value_counts(dropna=False))

#категорриальные признаки заменяем  на моду
Video_Games['Genre'].fillna(Video_Games['Genre'].mode(), inplace=True)
Video_Games['Publisher'].fillna(Video_Games['Publisher'].mode(), inplace=True)
Video_Games['Developer'].fillna(Video_Games['Developer'].mode(), inplace=True)
Video_Games['Rating'].fillna(Video_Games['Rating'].mode(), inplace=True)
#числовые признаки заменяем  на моду
Video_Games['NA_Sales'].fillna(Video_Games['NA_Sales'].median(), inplace=True)
Video_Games['EU_Sales'].fillna(Video_Games['EU_Sales'].median(), inplace=True)
Video_Games['Other_Sales'].fillna(Video_Games['Other_Sales'].median(), inplace=True)
Video_Games['Critic_Score'].fillna(Video_Games['Critic_Score'].median(), inplace=True)
Video_Games['Critic_Count'].fillna(Video_Games['Critic_Count'].median(), inplace=True)
Video_Games['User_Score'].fillna(Video_Games['User_Score'].median(), inplace=True)
Video_Games['User_Count'].fillna(Video_Games['User_Count'].median(), inplace=True)

#убираем ненужные признаки
Video_Games = Video_Games.drop(('Year_of_Release'), axis=1)

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

#reg = ensemble.GradientBoostingRegressor()
#reg.fit(X_train, y_train)