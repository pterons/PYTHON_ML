# -*- coding: utf-8 -*-
"""1.tmdb_5000.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mN-i3MxJJekedbEPb09tkkZxBLDLiWlv
"""

# Recommending system : Content-based filtering
import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Commented out IPython magic to ensure Python compatibility.
# 학습 데이터를 읽어온다.
# %cd '/content/drive/My Drive/Colab Notebooks'
movie = pd.read_csv('data/tmdb_5000_movies.csv')
df = movie[['id', 'title', 'genres', 'vote_average', 'vote_count', 'popularity', 'keywords', 'overview']]
 
# 'genres'와 'keywords'의 단순 string을 [dict1, dict2, ..] 객체로 변환.
pd.options.mode.chained_assignment = None  # SettingWithCopyWarning 표시하지 않음.
df['genres'] = df['genres'].apply(literal_eval)
df['keywords'] = df['keywords'].apply(literal_eval)
# df['genres'][0][0]['name'] = 'Action'
 
# 'genres'와 'keywords'의 dictionary에서 'name'만 추출
df['genres'] = df['genres'].apply(lambda x: [y['name'] for y in x])
df['keywords'] = df['keywords'].apply(lambda x: [y['name'] for y in x])
df[['genres', 'keywords']][:5]

df['overview'][1]

# 장르가 유사한 영화 
# 'genres'를 문자열로 변환 -> count vector -> 코사인 유사도 측정
# -------------------------------------------------------------
# 장르의 공백을 '_'으로 변환 : ex) 'Science Fiction' -> 'Science_Fiction'으로 변환
df['genres'] = df['genres'].apply(lambda x: [s.replace(' ', '_') for s in x])
 
# 장르 리스트를 문자열로 변환
df['genres_literal'] = df['genres'].apply(lambda x: ' '.join(x))
df['genres_literal'][:5]

# 장르 문자열을 count vector 행렬로 변환
count_vect = CountVectorizer(min_df = 0)
genre_mat = count_vect.fit_transform(df['genres_literal'])

print(genre_mat)

# 코사인 유사도
genre_sim = cosine_similarity(genre_mat)
 
# 유사도가 높은 순서로 정렬
np.fill_diagonal(genre_sim, 0)  # 자기 자신과의 유사도는 0으로 설정
genre_sort = genre_sim.argsort()[:, ::-1]

import pdb
# 장르 콘텐츠 필터링을 이용한 영화 추천
def find_sim_movie(df, idx, title, top_n = 10):
    title_movie = df[df['title'] == title]
    title_index = title_movie.index.values
    similar_indexs = idx[title_index, : top_n].reshape(-1)
    return df.iloc[similar_indexs]

title = 'Avatar'
recommend = find_sim_movie(df, genre_sort, title)
print(recommend[['title', 'vote_average']])

# 가중 평점이 높은 순서로 추천
# 가중 평점 = R * v / (v + m) + C * m / (v + m)
#
# v : 개별 영화에 평점을 투표한 횟수
# m : 평점을 부여하기 위한 최소 투표 횟수
# R : 개별 영화에 대한 평균 평점
# C : 전체 영화에 대한 평균 평점
percentile = 0.6  # m을 위한 값. 상위 60%에 해당하는 횟수를 기준으로 정함.
C = df['vote_average'].mean()
m = df['vote_count'].quantile(percentile)

def weighted_vote_average(record):
    v = record['vote_count']
    R = record['vote_average']
    
    return R * v / (v + m) + C * m / (v + m)

df['weighted_vote'] = df.apply(weighted_vote_average, axis=1)

df[['vote_average', 'vote_count', 'weighted_vote']].head()

# 장르 콘텐츠 필터링을 이용한 영화 추천
def find_sim_movie2(df, idx, title, top_n = 10):
    title_movie = df[df['title'] == title]
    title_index = title_movie.index.values
    
    # top_n * 2 만큼 추출
    similar_indexs = idx[title_index, : (top_n * 2)].reshape(-1)
    
    # weighted_vote가 높은 순으로 top_n 만큼 추출
    return df.iloc[similar_indexs].sort_values('weighted_vote', ascending=False)[:top_n]

title = 'Avatar'
recommend = find_sim_movie2(df, genre_sort, title)
print(recommend[['title', 'vote_average', 'weighted_vote']])

