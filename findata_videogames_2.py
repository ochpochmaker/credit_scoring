import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
Video_Games = pd.read_csv(r'C:\Users\Админ\Desktop\Папки\findata-videogames\Video_Games.csv')

# ohe = OneHotEncoder(sparse=False)
#
# Genre_flg = ohe.fit_transform(Video_Games['Genre'].values.reshape(-1, 1))
# tmp = pd.DataFrame(Genre_flg, columns=['Genre=' + str(i) for i in range(Genre_flg.shape[1])])
# Video_Games = pd.concat([Video_Games, tmp], axis=1)
#
# Platform_flg = ohe.fit_transform(Video_Games['Platform'].values.reshape(-1, 1))
# tmp = pd.DataFrame(Platform_flg, columns=['Platform=' + str(i) for i in range(Platform_flg.shape[1])])
# Video_Games = pd.concat([Video_Games, tmp], axis=1)
#
# Publisher_flg = ohe.fit_transform(Video_Games['Publisher'].values.reshape(-1, 1))
# tmp = pd.DataFrame(Publisher_flg, columns=['Publisher=' + str(i) for i in range(Publisher_flg.shape[1])])
# Video_Games = pd.concat([Video_Games, tmp], axis=1)
# print(Video_Games)
#
# Developer_flg = ohe.fit_transform(Video_Games['Developer'].values.reshape(-1, 1))
# tmp = pd.DataFrame(Developer_flg, columns=['Developer=' + str(i) for i in range(Developer_flg.shape[1])])
# Video_Games = pd.concat([Video_Games, tmp], axis=1)
#
# Rating_flg = ohe.fit_transform(Video_Games['Rating'].values.reshape(-1, 1))
# tmp = pd.DataFrame(Rating_flg, columns=['Rating=' + str(i) for i in range(Rating_flg.shape[1])])
# Video_Games = pd.concat([Video_Games, tmp], axis=1)
#
#
# print(Video_Games)
# # le = LabelEncoder()
# # #преобразовываем категориальные данные в циферки
# # le.fit(Video_Games.Platform)
# # Video_Games['Platform_le'] = le.transform(Video_Games.Platform)
# # le.fit(Video_Games.Year_of_Release)
# # Video_Games['Year_of_Release_le'] = le.transform(Video_Games.Year_of_Release)
# # le.fit(Video_Games.Genre)
# # Video_Games['Genre_le'] = le.transform(Video_Games.Genre)
# # le.fit(Video_Games.Publisher)
# # Video_Games['Publisher_le'] = le.transform(Video_Games.Publisher)
# # le.fit(Video_Games.Developer)
# # Video_Games['Developer_le'] = le.transform(Video_Games.Developer)
# # le.fit(Video_Games.Rating)
# # Video_Games['Rating_le'] = le.transform(Video_Games.Rating)
# # print(Video_Games)
#
# # #анализ датасета
# # print(Video_Games.head())#Что есть наши данные
print(Video_Games.shape) # мы увидим информацию о размерности нашего датафрейма
print(Video_Games.info()) # покажет информацию о размерности данных
# # print(Video_Games.describe()) # показывает статистики count,mean, std, min, 25%-50%-75% percentile, max
# # print(Video_Games.nunique()) # количество уникальных значений для каждого столбца
# # #Также было бы неплохо увидеть информацию о количестве каждого уникального значения для каждого столбца в наборе данных
# feature_names = Video_Games.columns.tolist()
# for column in feature_names:
#     print(column)
#     print(Video_Games[column].value_counts(dropna=False))
#
# #поиск нулов
# val = Video_Games.isna().any()[lambda x: x]
# print(val)