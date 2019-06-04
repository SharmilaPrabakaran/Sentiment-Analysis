import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

df = pd.read_csv('amazon_baby.csv')

df.dropna(inplace=True)
df[df['rating'] != 3]
df['Positivity'] = np.where(df['rating'] > 3, 1, 0)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['Positivity'], random_state = 0)

vect = CountVectorizer(min_df = 5, ngram_range = (1,2)).fit(X_train)
X_train_vectorized = vect.transform(X_train)

model = LogisticRegression()
model.fit(X_train_vectorized, y_train)
predictions = model.predict(vect.transform(X_test))
print 'AUC: ', roc_auc_score(y_test, predictions)

print (model.predict(vect.transform(['The candy is not good, I would never buy them again','The candy is not bad, I will buy them again'])))
