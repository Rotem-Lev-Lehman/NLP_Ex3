
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, auc, roc_curve
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchnlp
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from torchnlp.word_to_vector import GloVe
import re

df = pd.read_csv('trump_train.tsv', header=None, sep='\t', names=['tweet id', 'user handle', 'tweet text', 'time stamp', 'device'])
df.head()

df.drop(labels=['tweet id'], axis=1, inplace=True)

df.head()

def time_stamp_to_features(df):
  df['time stamp'] = pd.to_datetime(df['time stamp'])
  df["year"] = df["time stamp"].dt.year
  df["month"] = df["time stamp"].dt.month
  df["day"] = df["time stamp"].dt.day
  df["dayOfWeek"] = df["time stamp"].dt.dayofweek
  df["dayOfYear"] = df["time stamp"].dt.dayofyear
  df["hour"] = df["time stamp"].dt.hour
  df["minute"] = df["time stamp"].dt.minute
  df.drop(['time stamp'], axis=1, inplace=True)

time_stamp_to_features(df)

df.head()

def contains_URL(text): 
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex,text)       
    return int(len([x[0] for x in url]) > 0)

def count_caps_locked_words(text):
    caps_locked_num = sum(map(str.isupper, text.split()))
    return caps_locked_num

def count_hashtags(text):
    hashtags_num = text.count('#')
    return hashtags_num

def count_mentions(text):
    mentions_num = text.count('@')
    return mentions_num

df['contains_URL'] = df['tweet text'].apply(lambda text: contains_URL(text))
df['caps_locked_count'] = df['tweet text'].apply(lambda text: count_caps_locked_words(text))
df['count_hashtags'] = df['tweet text'].apply(lambda text: count_hashtags(text))
df['count_mentions'] = df['tweet text'].apply(lambda text: count_mentions(text))

df['device'] = (df['device'] != 'android').astype(int)

df

dummies = pd.get_dummies(df['user handle'])
df = pd.concat([df, dummies], axis=1)
df.drop(labels=['user handle'], axis=1, inplace=True)

df.head()

punctuation = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
stop_words = stopwords.words('english')
porter = PorterStemmer()

df['tweet text'] = df['tweet text'].apply(lambda tweet: tweet.lower())
df['tweet text'] = df['tweet text'].apply(lambda tweet: "".join([char for char in tweet if char not in punctuation]))
df['tweet text'] = df['tweet text'].apply(lambda tweet: word_tokenize(tweet))
df['tweet text'] = df['tweet text'].apply(lambda words: [word for word in words if word not in stop_words])
#df['tweet text'] = df['tweet text'].apply(lambda filtered_words: [porter.stem(word) for word in filtered_words])

df.head()



df2 = pd.read_csv('../../../Desktop/second_degree/קורסים/סמסטר ג/עיבוד שפה טבעית/עבודות/עבודה 3/trump_test.tsv', header=None, sep='\t', names=['user handle', 'tweet text', 'time stamp'])
df2.head()



#vectors = GloVe('twitter.27B')
#vectors['hello']

legal_words = set()
for tweet_array in df['tweet text']:
    legal_words |= set(tweet_array)

len(legal_words)

legal_lines = []
with open('/content/embedding_stuff/glove/glove.6B.300d.txt') as file1:
    lines = file1.readlines()
    for l in lines:
        split_line = l.rstrip().split()
        #word = porter.stem(split_line[0])
        word = split_line[0]
        if word in legal_words:
            legal_lines.append(l)

with open('embedded.txt', 'w') as write_file:
    write_file.writelines(legal_lines)

columns_to_scale = ['year', 'month', 'day', 'dayOfWeek', 'dayOfYear', 'hour', 'minute']

scaled_features = StandardScaler().fit_transform(df[columns_to_scale].values)

df[columns_to_scale] = scaled_features

df

"""Add the embedding of the text: (Text embedding will be represented as the mean of all of the words embedding in the text)"""

import numpy as np

embedding_dict = {}
with open('/content/embedded.txt') as file1:
    all_lines = file1.readlines()
    for line in all_lines:
        split_line = line.strip().split()
        embedding_dict[split_line[0]] = np.array(split_line[1:]).astype(np.float32)



embedding_dict['hey']

"""Now to convert the text into an embedded vector:"""

df['tweet embedding'] = df['tweet text'].apply(lambda words: np.apply_along_axis(np.mean, 0, np.array([embedding_dict[word] for word in words if word in embedding_dict.keys()])))

df

df = df.dropna()

for i in range(len(df['tweet embedding'][0])):
    df[f'embedding_{i}'] = df['tweet embedding'].apply(lambda vector: vector[i])

df

df.drop(labels=['tweet text', 'tweet embedding'], axis=1, inplace=True)

df

y = df['device']
X = df.iloc[:,df.columns != 'device']

train_x

train_y

lr = MLPClassifier()
lr.fit(train_x, train_y)

y_pred = lr.predict(train_x)

print(classification_report(train_y, y_pred))

print(f'accuracy = {accuracy_score(train_y, y_pred)}')

fpr, tpr, thresholds = roc_curve(train_y, y_pred)
print(f'auc = {auc(fpr, tpr)}')

from sklearn.model_selection import cross_val_score
print(cross_val_score(lr, train_x, train_y, cv=5, scoring='roc_auc'))

class FF_NN(nn.Module):
    def __init__(self, input_size, classes_num):
        super(FF_NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        #self.fc3 = nn.Linear(256, 128)
        self.out = nn.Linear(64, classes_num)
    
    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.softmax(self.out(x))
        return x

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_torch = torch.FloatTensor(X_train.values)
X_test_torch = torch.FloatTensor(X_test.values)
y_train_torch = torch.LongTensor(y_train.values)
y_test_torch = torch.LongTensor(y_test.values)

model = FF_NN(len(X_train.columns), 2)
model

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 1000
loss_arr = []

for i in range(epochs):
    y_hat = model.forward(X_train_torch)
    loss = criterion(y_hat, y_train_torch)
    loss_arr.append(loss)
    
 
    if i % 10 == 0:
        print(f'Epoch: {i} Loss: {loss}')
 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



prob_preds = []
preds = []
with torch.no_grad():
    for val in X_test_torch:
        y_hat = model.forward(val)
        prob_preds.append(y_hat[1].item())
        preds.append(y_hat.argmax().item())

accuracy_score(y_test, preds)

fpr, tpr, thresholds = roc_curve(y_test, prob_preds)
print(f'auc = {auc(fpr, tpr)}')