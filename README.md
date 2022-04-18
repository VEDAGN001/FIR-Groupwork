# Stock trend prediction based on NLP sentiment analysis and deep learning techniques
Here is the brief desciption of our project.<br>
***For more information, please see：https://yuey3816.gitbook.io/stock-trend-prediction-based-on-nlp-and-dl-1-1***
## Introduction
In the report, we will use Twitter Application Programming Interface (API) to obtain tweets about Alibaba through keyword searches. We will then deploy NLP to analyze the sentiment of the collected tweets and derive people's sentiment on social media about the company, thus predicting the company's stock price movement in the short term. At the same time, we also deploy a deep neural network (DNN) to predict future short-term share price movements using a dataset of historical closing prices of Alibaba shares. The two methods will complement each other to improve the quality of the forecasts.
## Collecting data
Apply Tweepy API to get tweets data of keywords<br>
Apply yfinance API to get the stock price from YahooFinance<br>
## Natural Language Process
```python 
import re
import nltk
from collections import Counter
from wordcloud import WordCloud 
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
# STOPWORDS = ["an", "a", "the"] + NEGWORDS
STOPWORDS = ["an", "a", "the", "or", "and", "thou", "must", "that", "this", "self", "unless", "behind", "for", "which",
             "whose", "can", "else", "some", "will", "so", "from", "to", "by", "within", "of", "upon","rt", "th", "with",
             "it","baba","today","yesterday","0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", 
             "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", 
             "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "affects", "after", "afterwards", "ag", 
             "again", "against", "ah", "ain", "ain't", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", 
             "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce",
             "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "ao", 
             "ap", "apart", "apparently", "appear", "appreciate", "appropriate", "approximately", "ar", "are", "aren", "arent", 
             "aren't", "arise", "around", "as", "a's", "aside", "ask", "asking", "associated", "at", "au", "auth", "av", 
             "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", 
             "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", 
             "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", 
             "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", 
             "bt", "bu", "but", "bx", "by", "c", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "can't", 
             "cause", "causes", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl", 
             "clearly", "cm", "c'mon", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", 
             "considering", "contain", "containing", "contains", "corresponding", "could", "couldn", "couldnt", "couldn't", 
             "course", "cp", "cq", "cr", "cry", "cs", "c's", "ct", "cu", "currently", "cv", "cx", "cy", "cz", "d", "d2", "da", 
             "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", 
             "didn't", "different", "dj", "dk", "dl", "do", "does", "doesn", "doesn't"]


def _remove_stopwords(txt):
    """Delete from txt all words contained in STOPWORDS."""
    words = txt.split()
    # words = txt.split(" ")
    for i, word in enumerate(words):
        if word in STOPWORDS:
            words[i] = " "
    return (" ".join(words))


with open('tweets.txt', 'r', encoding='utf-8') as tweets_read:

    tweets_string = tweets_read.read()

tweets_split = str.split(tweets_string, sep=',')
print(tweets_split)
len(tweets_split)

doc_out = []
for k in tweets_split:
    cleantextprep = str(k)
       
        
        # Regex cleaning
    expression = "[^a-zA-Z ]"  # keep only letters, numbers and whitespace
    cleantextCAP = re.sub(expression, '', cleantextprep)  # apply regex
    cleantext = cleantextCAP.lower()  # lower case
    cleantext = _remove_stopwords(cleantext)
    bound = ''.join(cleantext)
    doc_out.append(bound)       # a list of sentences

print(doc_out)
print(tweets_split)
```
***Result***:

![清洗1.png](https://s2.loli.net/2022/04/17/C9tvZFesVp2ij6z.png)

***Calculate the number and weights of positive and negative words***

``` python
# decompose a list of sentences into words by self-defined function
tokens = decompose_word(doc_out)
# decompose a list of sentences into words from NLTK module
tokens_nltk = nltk.word_tokenize(str(doc_out))

# Number of words in article
nwords = len(tokens)

nwc = wordcount(tokens, ndct)   

pwc = wordcount(tokens, pdct)

# Total number of positive/negative words
ntot, ptot = 0, 0
for i in range(len(nwc)):
    ntot += nwc[i][1]

for i in range(len(pwc)):
    ptot += pwc[i][1]


# Print results
print('Positive words:')
for i in range(len(pwc)):
    print(str(pwc[i][0]) + ': ' + str(pwc[i][1]))
print('Total number of positive words: ' + str(ptot))
print('\n')
print('Percentage of positive words: ' + str(round(ptot / nwords, 4)))
print('\n')
print('Negative words:')
for i in range(len(nwc)):
    print(str(nwc[i][0]) + ': ' + str(nwc[i][1]))
print('Total number of negative words: ' + str(ntot))
print('\n')
print('Percentage of negative words: ' + str(round(ntot / nwords, 4)))
```

***Positive results***：

![积极结果.png](https://s2.loli.net/2022/04/17/7GXf8gnirmI1M5a.png)

***Negative results***：

![消极结果.png](https://s2.loli.net/2022/04/17/uaPKnWf1Zmr5Q3I.png)

***Weights***：

![图片 1.png](https://s2.loli.net/2022/04/17/ZqwkrNlSI3M54nV.png)

## Sentiment/Textual analysis

### Wordcloud

![Wordcloud.png](https://s2.loli.net/2022/04/17/QIGmv1XcwxVPnRT.png)

***High Frenquency Keywords list***<br>1. China Chinese stock<br>2. Seguidores(Spainish)<br>3. mskvsk(web celebrity)<br>4. IPO(Initial Public Offering)<br>5. $aapl(Apple)

### Sentiment analysis by Vader and Testblob

#### Textblob

``` python
from textblob import TextBlob
def fetch_sentiment_using_textblob(text):
    analysis = TextBlob(text)
    return 'pos' if analysis.sentiment.polarity >= 0 else 'neg'
from pandas_datareader import data
import pandas as pd
tweets_df = pd.read_csv(tweets#abc.csv)
sentiments_using_textblob = tweets_df.Tweet.apply(lambda tweet: fetch_sentiment_using_textblob(tweet))

pd.DataFrame(sentiments_using_textblob.value_counts())
```

***Positive tweets VS Negative tweets***

![Textblob 结果.png](https://s2.loli.net/2022/04/17/9jPZetyNlkvRDSb.png)

***Tweets Polarity Analysis***

```python
tweets_df['sentiment'] = sentiments_using_textblob
tweets_df.head()
```

![Textblob结果2.png](https://s2.loli.net/2022/04/17/k5pdKlx7Ww9fbjZ.png)

*****

#### Vader

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer
def fetch_sentiment_using_SIA(text):
    sid = SentimentIntensityAnalyzer()
    polarity_scores = sid.polarity_scores(text)
    return 'neg' if polarity_scores['neg'] > polarity_scores['pos'] else 'pos'
tweets_df = pd.read_csv(tweets.csv)
sentiments_using_SIA = tweets_df.Tweet.apply(lambda tweet: fetch_sentiment_using_SIA(tweet))
pd.DataFrame(sentiments_using_SIA.value_counts())
```

***Positive tweets VS Negative tweets***

![Vader 1 .png](https://s2.loli.net/2022/04/17/k63R1UzvjV789d4.png)

***Tweets Polarity Analysis***

``` python
tweets_df['sentiment'] = sentiments_using_SIA
tweets_df.head()
```

***Results***

![Vader 2.png](https://s2.loli.net/2022/04/17/pULOJH7A5zDo3CW.png)

## Deep learning model
**TensorFlow**, developed by the Google Brain team, is a free open-source software library for machine learning and AI. It can automatically calculate the gradients of the parameters in the model and is easy to optimise for deep learning with the Adam optimiser. Adam can iteratively update the neural network weights based on the training data, making it very simple and efficient. This makes TensorFlow ideal for the training and inference of deep neural networks. For example, AlphaGo was trained using TensorFlow, which is named after how it works: tensor means tensor (i.e. a multidimensional array) and flow means a stream. Therefore, TensorFlow means that the multidimensional arrays flow from one end of the data flow graph to the other. The diagram below illustrates the flow of data.

![textblob.gif](https://tva1.sinaimg.cn/large/e6c9d24egy1h1c8w4bmang20700cgdpd.gif)

**Keras** is a deep learning library based on TensorFlow and Theano (a machine learning framework developed by the University of Montreal, Canada). It is a high-level neural network API written in python and only supports python development.

Keras can run with [TensorFlow](https://github.com/tensorflow/tensorflow),  [CNTK](https://github.com/Microsoft/cntk) or [Theano](https://github.com/Theano/Theano) as the backend. Keras is designed to support fast experiments, being able to translate your ideas into experimental results with minimal latency.

Use Keras if you need a deep learning library in the following situations:

- Allows simple and fast prototyping (due to user-friendliness, high modularity, scalability).
- Supports both convolutional and recurrent neural networks, as well as combinations of both.
- Runs seamlessly on CPUs and GPUs.

This report uses the **Keras API** operating with Tensorflow as the backend: **Tensorflow.keras**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.utils import Sequence
```
Dataset: download from **Yahoo Finance**

90% as the training sets, 10% as the testing sets

Construct and Train model
In order to find the optimal model, the hyperparameter collocation and MSE calculation will be carried out by coding.
Activation function: relu
epochs = 80
n_lstm: (32,64,128,256)
n_layer: (3,6,9,20)
n_neuron: (10,25,50,100)

## Conclusion
Machine learning models based on historical data for stock price prediction inevitably face the problem of overfitting, which can lead to model failure in unstable markets and therefore biased investment decisions. Some examples are the quant meltdown in 2008 and the quant crash during the covid pandemic.  

In addition, quantitative analysis models based on mean reversion theory are meaningless in the face of a diverse investment market, especially in speculative markets such as Bitcoin and the NFT market. This is because those markets are immature and do not yet have a well-established market mechanism and a rational investor base. In such markets, machine learning cannot form effective models and reliable price predictions. 

Social media, on the other hand, is the perfect representation of public sentiment and opinion about the current time. Twitter has attracted a large number of researchers working on public sentiment (Pagolu, Reddy, Panda and Majhi, 2016). Sentiment analysis for social media is characterized by timeliness and low acquisition costs. Sentiment analysis can be considered as a multiclassification problem, while polarity analysis for tweet sentiment is a binary classification. By analyzing whether the text is positive or negative in relation to keywords, investors can use it to decide whether to buy or sell. This is the advantage and value of sentiment analysis in the investment field compared to traditional machine learning. 

## Reference

Abadi, M. (2016). TensorFlow: learning functions at scale. *Proceedings of the 21st ACM SIGPLAN International Conference on Functional Programming - ICFP 2016*. 

Abid, F., Yasir, M. and Li, C. (2019). Sentiment analysis through recurrent variants latterly on convolutional neural network of Twitter", Future Gener. Comput. Syst., vol. 95, pp. 292-308. 

Akerlof, G. A., & Shiller, R. J. (2010). *Animal spirits: How human psychology drives the economy, and why it matters for global capitalism*. Princeton university press.  

Behera, R., Jena, M., Rath, S. and Misra, S., 2021. Co-LSTM: Convolutional LSTM model for sentiment analysis in social big data. *Information Processing & Management*, 58(1), p.102435. 

Bollen, J., Mao, H. and Zeng, X. (2011b) 'Twitter mood predicts the stock market', *Journal of Computational Science*, 2(1), pp. 1–8. doi:[10.1016/j.jocs.2010.12.007](https://doi.org/10.1016/j.jocs.2010.12.007).  

Dickinson, B. and Hu, W. (2015) 'Sentiment Analysis of Investor Opinions on Twitter', *Social Networking*, 04(03), p. 62. doi:[10.4236/sn.2015.43008](https://doi.org/10.4236/sn.2015.43008).  

Fama, E.F. (1970). Efficient Capital Markets: a Review of Theory and Empirical Work. *The Journal of Finance*, 25(2), pp.383–417. 

Ghosh, I. and Chaudhuri, T., 2021. FEB-Stacking and FEB-DNN Models for Stock Trend Prediction: A Performance Analysis for Pre and Post Covid-19 Periods. *Decision Making: Applications in Management and Engineering*, 4(1), pp.51-86. 

Heaton, J.B., Polson, N.G. and Witte, J.H. (2016). Deep learning for finance: deep portfolios. *Applied Stochastic Models in Business and Industry*, 33(1), pp.3–12. 

Heimerl, F. *et al.* (2014) 'Word Cloud Explorer: Text Analytics Based on Word Clouds', in *2014 47th Hawaii International Conference on System Sciences*. *2014 47th Hawaii International Conference on System Sciences*, pp. 1833–1842. doi:[10.1109/HICSS.2014.231](https://doi.org/10.1109/HICSS.2014.231).  

Hu, Z., Zhao, Y. and Khushi, M. (2021). A Survey of Forex and Stock Price Prediction Using Deep Learning. *Applied System Innovation*, 4(1), p.9. 

Hutto, C. and E. Gilbert (2014). Vader: A parsimonious rule-based model for sentiment analysis of social media text. Proceedings of the international AAAI conference on web and social media.  

Hyun, D., Park, C., Yang, M.-C., Song, I., Lee, J.-T. and Yu, H. (2019). Target-aware convolutional neural network for target-level sentiment analysis. *Information Sciences*, 491, pp.166–178. 

Jia, H. (2016). Investigation into the effectiveness of long short term memory networks for stock price prediction. arXiv preprint arXiv:1603.07893. 

Khare, K., Darekar, O., Gupta, P. and Attar, V.Z. (2017). 、Gupta and V. Z. Attar, ‘Short term stock price prediction using deep learning,’ 2017 2nd IEEE International Conference on Recent Trends in Electronics, Information & Communication Technology (RTEICT), pp. 482-486. 

Kumar, D., Sarangi, P. K., & Verma, R. (2021). A systematic review of stock market prediction using machine learning and statistical techniques. *Materials Today: Proceedings*.  

Li, J., Luong, M., Jurafsky, D. and Hovy, E., 2022. *When Are Tree Structures Necessary for Deep Learning of Representations?*. [online] arXiv.org. Available at: <https://doi.org/10.48550/arXiv.1503.00185> 

Mehta, Y., Malhar, A. and Shankarmani, R. (2021) 'Stock Price Prediction using Machine Learning and Sentiment Analysis', in *2021 2nd International Conference for Emerging Technology (INCET)*. *2021 2nd International Conference for Emerging Technology (INCET)*, pp. 1–4. doi:[10.1109/INCET51464.2021.9456376](https://doi.org/10.1109/INCET51464.2021.9456376).  

Neelakandan, S., & Paulraj, D. (2020). An automated learning model of conventional neural network based sentiment analysis on Twitter data. *Journal of Computational and Theoretical Nanoscience*, *17*(5), 2230-2236.  

Pagolu, V., Reddy, K., Panda, G. and Majhi, B., 2016. Sentiment analysis of Twitter data for predicting stock market movements. *2016 International Conference on Signal Processing, Communication, Power and Embedded System (SCOPES)*,. 

Ranco, G. *et al.* (2015) 'The Effects of Twitter Sentiment on Stock Price Returns', *PLOS ONE*, 10(9), p. e0138441. doi:[10.1371/journal.pone.0138441](https://doi.org/10.1371/journal.pone.0138441).  

Statista (2021a) *Most famous social media sites in the U.S. 2021*, *Statista*. Available at: https://www.statista.com/statistics/265773/market-share-of-the-most-popular-social-media-websites-in-the-us/ (Accessed: 30 March 2022).  

Statista (2021b) *Twitter: most users by country*, *Statista*. Available at: https://www.statista.com/statistics/242606/number-of-active-twitter-users-in-selected-countries/ (Accessed: 30 March 2022).  

Statista (2021c) *Twitter: number of profiles in Spain 2014-2020*, *Statista*. Available at: https://www.statista.com/statistics/751083/twitter-number-of-profiles/ (Accessed: 30 March 2022).  

Zhang, L., Fu, S. and Li, B., 2018. Research on Stock Price Forecast Based on News Sentiment Analysis—A Case Study of Alibaba. *Lecture Notes in Computer Science*, pp.429-442. 

Zucco, C., et al. (2020). "Sentiment analysis for mining texts and social networks data: Methods and tools." Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery 10(1): e1333.  
