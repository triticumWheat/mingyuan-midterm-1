import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import gc
nltk.download('vader_lexicon')
use_small_dataset = False  ### PLEASE MAKE SURE THIS IS FALSE
train_full = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
train = train_full[train_full['Score'].notna()].reset_index(drop=True)
test_ids = test['Id'].unique()
test = train_full[train_full['Id'].isin(test_ids)].reset_index(drop=True)
test = test[test['Score'].isna()].reset_index(drop=True)
if use_small_dataset:
    train_sample = train.sample(n=100000, random_state=1).reset_index(drop=True)
    test_sample = test.sample(n=20000, random_state=1).reset_index(drop=True)
else:
    train_sample = train
    test_sample = test
del train_full, train, test
gc.collect()
def add_features_to(df):
    df['HelpfulnessNumerator'] = df['HelpfulnessNumerator'].fillna(0)
    df['HelpfulnessDenominator'] = df['HelpfulnessDenominator'].fillna(0)
    df['Text'] = df['Text'].fillna('')
    df['Summary'] = df['Summary'].fillna('')

    df['HelpfulnessDenominator'] = df['HelpfulnessDenominator'].replace(0, np.nan)
    df['Helpfulness'] = df['HelpfulnessNumerator'] / df['HelpfulnessDenominator']
    df['Helpfulness'] = df['Helpfulness'].fillna(0)

    df['CleanedText'] = df['Text'].apply(clean_text)
    df['CleanedSummary'] = df['Summary'].apply(clean_text)

    df['TextLength'] = df['CleanedText'].str.len()
    df['SummaryLength'] = df['CleanedSummary'].str.len()

    df['TextWordCount'] = df['CleanedText'].apply(lambda x: len(x.split()))
    df['SummaryWordCount'] = df['CleanedSummary'].apply(lambda x: len(x.split()))

    sid = SentimentIntensityAnalyzer()
    df['TextSentiment'] = df['CleanedText'].apply(lambda x: sid.polarity_scores(x)['compound'])
    df['SummarySentiment'] = df['CleanedSummary'].apply(lambda x: sid.polarity_scores(x)['compound'])
    return df

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Processing training data")
train_processed = add_features_to(train_sample)

print("Processing test data")
test_processed = add_features_to(test_sample)

features = ['Helpfulness', 'TextLength', 'SummaryLength', 'TextWordCount', 'SummaryWordCount',
            'TextSentiment', 'SummarySentiment']
X_train = train_processed[features]
Y_train = train_processed['Score']
X_test = test_processed[features]
label_encoder = LabelEncoder()
Y_train_encoded = label_encoder.fit_transform(Y_train.astype(int))
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2), stop_words='english')
train_processed['CombinedText'] = train_processed['CleanedText'] + ' ' + train_processed['CleanedSummary']
test_processed['CombinedText'] = test_processed['CleanedText'] + ' ' + test_processed['CleanedSummary']
print("Vectorizing text data...")
X_train_text_tfidf = tfidf.fit_transform(train_processed['CombinedText'])
X_test_text_tfidf = tfidf.transform(test_processed['CombinedText'])
X_train_features = csr_matrix(X_train.values)
X_test_features = csr_matrix(X_test.values)
X_train_combined = hstack([X_train_features, X_train_text_tfidf]).tocsr()
X_test_combined = hstack([X_test_features, X_test_text_tfidf]).tocsr()

del X_train_features, X_train_text_tfidf, X_test_features, X_test_text_tfidf
gc.collect()



import xgboost as xgb

dtrain = xgb.DMatrix(X_train_combined, label=Y_train_encoded)
print("Training")
params = {
    'objective': 'multi:softmax',
    'num_class': len(label_encoder.classes_),
    'max_depth': 8,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',
    'device': 'cuda',
    'eval_metric': 'mlogloss',
}

model = xgb.train(params, dtrain, num_boost_round=200)
del dtrain, X_train_combined
gc.collect()
print("Predicting on test data")
dtest = xgb.DMatrix(X_test_combined)
Y_test_pred_encoded = model.predict(dtest)
Y_test_pred_encoded = Y_test_pred_encoded.astype(int)
test_processed['Score'] = label_encoder.inverse_transform(Y_test_pred_encoded)
submission = test_processed[['Id', 'Score']]
submission = test_sample[['Id']].merge(submission, on='Id', how='left')



submission.to_csv("./data/submission.csv", index=False)
print("Submission file saved to ./data/submission.csv")
