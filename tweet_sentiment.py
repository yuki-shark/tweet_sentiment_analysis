import requests
from requests_oauthlib import OAuth1Session
import json
import csv
from google.cloud import language
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from dateutil.tz import gettz
import pandas as pd
import numpy as np
import re
import os

#取得したkeyを以下で定義する
access_token        = os.getenv('ACCESS_TOKEN')
access_token_secret = os.getenv('ACCESS_TOKEN_SECRET')
consumer_key        = os.getenv('CONSUMER_KEY')
consumer_key_secret = os.getenv('CONSUMER_KEY_SECRET')

# タイムライン取得用のURL
url = "https://api.twitter.com/1.1/statuses/user_timeline.json"

#パラメータの定義
params = {'screen_name':'yuki_shar18',
          'exclude_replies':True,
          'include_rts':False,
          'count':200}

#APIの認証
twitter = OAuth1Session(consumer_key, consumer_key_secret, access_token, access_token_secret)

# get tweet
f_out = open('tweet_data.csv','w')
times = []
for j in range(50):
    #リクエストを投げる
    res = twitter.get(url, params = params)

    if res.status_code == 200:

        # API残り
        limit = res.headers['x-rate-limit-remaining']
        print ("API remain: " + limit)
        if limit == 1:
            sleep(60*15)

        n = 0
        timeline = json.loads(res.text)
        # 各ツイートの本文を表示
        for i in range(len(timeline)):
            if i != len(timeline)-1:
                f_out.write(re.sub(r'[^ぁ-んァ-ンー-龥a-xA-Z0-9_]', "", timeline[i]['text']) + '\n')
            else:
                f_out.write(re.sub(r'[^ぁ-んァ-ンー-龥a-xA-Z0-9_]', "", timeline[i]['text']) + '\n')
                #一番最後のツイートIDをパラメータmax_idに追加
                params['max_id'] = timeline[i]['id']-1
            time = datetime.datetime.strptime(timeline[i]['created_at'],
                '%a %b %d %H:%M:%S %z %Y').astimezone(gettz('Asia/Tokyo'))
            times.append(time.strftime('%Y-%m-%d'))
            if time < datetime.datetime(2018,1,1).astimezone(gettz('Asia/Tokyo')):
                break

f_out.close()

# remove pictures and read tweets
with open('tweet_data.csv','r') as f:
    reader = csv.reader(f, delimiter='\t')
    texts = []
    for row in reader:
        if row:
            text = row[0].split('http')[0]
            texts.append(text)

len(texts)
len(times)

df = pd.DataFrame({'time' : times, 'text' : texts, 'score' : np.zeros(len(times))})

num = 0
for i in range(150):
    num += len(df.iloc[i].text)

df = df[:150]

language_client = language.Client()
itr = 0
scores = []
while(itr < len(df)):
    str_len = 0
    text = ""
    while(itr < len(df) and str_len < 1000):
        tweet = df.iloc[itr].text
        if len(tweet) < 1:
            itr += 1
            continue
        # if tweet[-1] != '.':
        #     tweet += '.'
        tweet += '.\n'
        if len(text) + len(tweet) < 1000:
            text += tweet
            itr += 1
        else:
            print(len(text))
            print(text)
            break

    # sentiment analysis
    document = language_client.document_from_text(text)
    response = document.analyze_sentiment()
    sentiment = response.sentiment
    sentences = response.sentences

    # 各段落の感情スコアを出力
    for sentence in sentences:
        # print('=' * 20)
        # print('Text: {}'.format(sentence.content))
        # print('Sentiment: {}, {}'.format(sentence.sentiment.score, sentence.sentiment.magnitude))
        scores.append(sentence.sentiment.score)

for i in range(len(scores)):
    df.score[i] = scores[i]

df.to_csv("tweet_sentiment.csv")

df.time = pd.to_datetime(df.time)
hoge = df[['time', 'score']].groupby('time').mean()
fuga = hoge['score'].groupby(hoge.index.week).mean()

# plt.plot(hoge.index, hoge.score)
plt.plot(fuga.index, fuga.values)
plt.grid(True)
plt.title("Sentiment Analysis")


# =============================================================== #

# リクエストのデータを格納
document = language_client.document_from_text(text)
# 感情分析のレスポンスを格納
response = document.analyze_sentiment()
# ドキュメント全体の感情が含まれたオブジェクト
sentiment = response.sentiment
# 各段落の感情が含まれたオブジェクトのリスト
sentences = response.sentences

# 全体の感情スコアを出力
print('Text全体')
print('Text: {}'.format(text))
print('Sentiment: {}, {}'.format(sentiment.score, sentiment.magnitude))

scores = []
# 各段落の感情スコアを出力
for sentence in sentences:
    print('=' * 20)
    print('Text: {}'.format(sentence.content))
    print('Sentiment: {}, {}'.format(sentence.sentiment.score, sentence.sentiment.magnitude))
    scores.append(sentence.sentiment.score)

df.time = pd.to_datetime(df.time)
hoge = df[['time', 'score']].groupby('time').mean()

plt.plot(hoge.index[-15:], hoge.score[-15:])
plt.grid(True)
