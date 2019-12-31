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

#APIの認証
twitter = OAuth1Session(consumer_key, consumer_key_secret, access_token, access_token_secret)

def get_tweets(user_id):
    #パラメータの定義
    # user_id = 2366743729
    params = {'user_id':user_id,
              'exclude_replies':True,
              'include_rts':False,
              'count':200}

    # get tweet
    fn = 'tweets/' + str(user_id) + '.csv'
    f_out = open(fn,'w')
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
                time = datetime.datetime.strptime(timeline[i]['created_at'],
                    '%a %b %d %H:%M:%S %z %Y').astimezone(gettz('Asia/Tokyo'))
                if time < datetime.datetime(2019,1,1).astimezone(gettz('Asia/Tokyo')):
                    break
                times.append(time.strftime('%Y-%m-%d'))
                if i != len(timeline)-1:
                    f_out.write(re.sub(r'[^ぁ-んァ-ンー-龥a-xA-Z0-9_]', "", timeline[i]['text']) + '\n')
                else:
                    f_out.write(re.sub(r'[^ぁ-んァ-ンー-龥a-xA-Z0-9_]', "", timeline[i]['text']) + '\n')
                    #一番最後のツイートIDをパラメータmax_idに追加
                    params['max_id'] = timeline[i]['id']-1
            if time < datetime.datetime(2019,1,1).astimezone(gettz('Asia/Tokyo')):
                break

    f_out.close()

    # remove pictures and read tweets
    with open(fn,'r') as f:
        reader = csv.reader(f, delimiter='\t')
        texts = []
        for row in reader:
            if row:
                text = row[0].split('http')[0]
                texts.append(text)

    print('user_id : ' + str(user_id))
    print('tweets : ' + str(len(texts)))
    df = pd.DataFrame({'time' : times, 'text' : texts,
                        'score' : np.zeros(len(times)),
                        'magnitude' : np.zeros(len(times))})
    fn = 'tweets_df/' + str(user_id) + '.csv'
    df.to_csv(fn, index=False)

def sentiment_analysis(user_id, num_tweets=500):
    fn = 'tweets_df/' + str(user_id) + '.csv'
    df = pd.read_csv(fn)
    if len(df) < num_tweets:
        return
    use_index = np.sort(np.random.choice(df.index,num_tweets,replace=False))
    df = df.iloc[use_index]

    language_client = language.Client()
    itr = 0
    scores = []
    magnitudes = []
    while(itr < len(df)):
        text = ""
        while(itr < len(df) and len(text) < 1000):
            tweet = df.iloc[itr].text
            if len(tweet) < 1:
                itr += 1
                continue
            tweet += '.\n\n'
            if len(text) + len(tweet) < 1000:
                text += tweet
                itr += 1
            else:
                break
        print(len(text))
        # print(text)

        # sentiment analysis
        document = language_client.document_from_text(text)
        response = document.analyze_sentiment()
        sentences = response.sentences

        # 各段落の感情スコアを出力
        for sentence in sentences:
            scores.append(sentence.sentiment.score)
            magnitudes.append(sentence.sentiment.magnitude)

    df.loc[:, 'score'] = scores
    df.loc[:, 'magnitude'] = magnitudes

    fn = 'sentiment/' + str(user_id) + '.csv'
    df.to_csv(fn, index=False)


if __name__ == "__main__":
    # test
    user_id = 2366743729
    get_tweets(user_id)
    sentiment_analysis(user_id, num_tweets=50)

    fn = 'sentiment/' + str(user_id) + '.csv'
    df = pd.read_csv(fn)

    # plot
    df.time = pd.to_datetime(df.time)
    df['sm'] = df.score * df.magnitude
    # hoge = df[['time', 'score']].groupby('time').mean()
    hoge = df[['time', 'sm']].groupby('time').mean()
    # fuga = hoge['score'].groupby(hoge.index.week).mean()
    fuga = hoge['sm'].groupby(hoge.index.week).mean()

    # plt.plot(hoge.index, hoge.score)
    plt.plot(fuga.index, fuga.values)
    plt.grid(True)
    plt.title("Sentiment Analysis")


    # # =============================================================== #
    #
    # # リクエストのデータを格納
    # document = language_client.document_from_text(text)
    # # 感情分析のレスポンスを格納
    # response = document.analyze_sentiment()
    # # ドキュメント全体の感情が含まれたオブジェクト
    # sentiment = response.sentiment
    # # 各段落の感情が含まれたオブジェクトのリスト
    # sentences = response.sentences
    #
    # # 全体の感情スコアを出力
    # print('Text全体')
    # print('Text: {}'.format(text))
    # print('Sentiment: {}, {}'.format(sentiment.score, sentiment.magnitude))
    #
    # scores = []
    # # 各段落の感情スコアを出力
    # for sentence in sentences:
    #     print('=' * 20)
    #     print('Text: {}'.format(sentence.content))
    #     print('Sentiment: {}, {}'.format(sentence.sentiment.score, sentence.sentiment.magnitude))
    #     scores.append(sentence.sentiment.score)
    #
    # df.time = pd.to_datetime(df.time)
    # hoge = df[['time', 'score']].groupby('time').mean()
    #
    # plt.plot(hoge.index[-15:], hoge.score[-15:])
    # plt.grid(True)
