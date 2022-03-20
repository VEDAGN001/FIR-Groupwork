import tweepy

api_key = "gHAOLzjoQiLLMWsPEGbGTFxmE"
api_secret_key = "LmEsRXJQgAwJyeDTrLO9KjY58RdK9BaOFKdOkmM3A1omiNaACn"
access_token = "1485536788177432580-xLfcNtubaqmCeHdBvawyfmHXXSCDtU"
access_token_secret = "wFS6P13xjgfc3NISB9VGLYWCmxugsVscpq7bWVdexVoNE"

# Create The Authenticate Object
authenticate = tweepy.OAuthHandler(api_key, api_secret_key)

# Set The Access Token & Access Token Secret
authenticate.set_access_token(access_token, access_token_secret)

# Create The API Object
api = tweepy.API(authenticate, wait_on_rate_limit=True)

import pandas as pd

keywords = '$BABA'
limit = 24740
tweets = tweepy.Cursor(api.search_tweets, q=keywords, count=200, tweet_mode='extended').items(limit)

columns = ['User', 'Tweet']
data = []

for tweet in tweets:
    data.append([tweet.user.screen_name, tweet.full_text])

df = pd.DataFrame(data, columns=columns)

print(df)
df.to_csv('$BABA tws.csv')