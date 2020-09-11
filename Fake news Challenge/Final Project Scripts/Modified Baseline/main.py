
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
from score import report_score, LABELS, score_submission
import s

def create_dataframe(filename):
    #Read file into a pandas dataframe
    df = pd.read_csv(filename)
    df.columns = [c.replace(' ', '_') for c in df.columns]
    return df

#Create dataframes for both training and testing sets
train_df_temp = create_dataframe("C:\\Users\\jeswi\\train_stances.csv")
train_bodies_df = create_dataframe('C:\\Users\\jeswi\\train_bodies.csv')

train_df=pd.merge(train_df_temp,train_bodies_df[['Body_ID', 'articleBody']],on='Body_ID')

competetion_bodies = pd.read_csv('C:\\Users\\jeswi\\competition_test_bodies.csv')
competetion_stances = pd.read_csv('C:\\Users\\jeswi\\competition_test_stances.csv')

comp = competetion_stances.join(competetion_bodies.set_index('Body ID'), on='Body ID')
print('tst_df')
print(comp.head(10))
train_df = train_df.rename(columns={'articleBody': 'Body_Text'})
test_df = comp.rename(columns={'articleBody': 'Body_Text'})

#Split training data into training and validation set
X_train, X_val, y_train, y_val = train_test_split(train_df[['Body_Text','Headline']], train_df['Stance'], test_size=0.1, random_state=42)

#Sanity Check
print("lenxtrain",len(X_train))
print(X_train[1:10])
print("lenxval",len(X_val))
print("lenYtrain",len(y_train))
print(y_train[1:10])
print("y val",len(y_val))


#Apply Scikit Learn TFIDF Feature Extraction Algorithm
body_text_vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, stop_words='english',max_features=1024)
headline_vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, stop_words='english',max_features=1024)

#Create vocabulary based on training data
train_body_tfidf = body_text_vectorizer.fit_transform(X_train['Body_Text'])
train_headline_tfidf = headline_vectorizer.fit_transform(X_train['Headline'])

val_body_tfidf = body_text_vectorizer.fit_transform(X_val['Body_Text'])
val_headline_tfidf = headline_vectorizer.fit_transform(X_val['Headline'])

#Use vocabulary for testing data
test_body_tfidf = body_text_vectorizer.transform(test_df['Body_Text'])
test_headline_tfidf = headline_vectorizer.transform(test_df['Headline'])

def get_cosine_similarity(body_tfidf,headline_tfidf):
    cosine_features = []

    for i in tqdm(range(body_tfidf.shape[0])):
        cosine_features.append(cosine_similarity((body_tfidf.A[0].reshape(1,-1)),(headline_tfidf.A[0].reshape(1,-1)))[0][0])
    return np.array(cosine_features).reshape(body_tfidf.shape[0],1)

# Train data
train_cosine_features = get_cosine_similarity(train_body_tfidf,train_headline_tfidf)

#Validate data
val_cosine_features = get_cosine_similarity(val_body_tfidf,val_headline_tfidf)
#
#Test data
#
test_cosine_features = get_cosine_similarity(test_body_tfidf,test_headline_tfidf)
# pickle.dump(train_cosine_features,open('train_cosine_features.p','wb'))
# pickle.dump(val_cosine_features,open('validate_cosine_features.p','wb'))

#pickle.dump(test_cosine_features,open('test_cosine_features.p','wb'))
#
# train_cosine_features = pickle.load(open('train_cosine_features.p','rb'))
# test_cosine_features = pickle.load(open('test_cosine_features.p','rb'))
# val_cosine_features =pickle.load(open('validate_cosine_features.p','rb'))


# '''''''''
# train_hand_features = s.hand_features(X_train['Headline'],X_train['Body_Text'])
# val_hand_features = s.hand_features(X_val['Headline'],X_val['Body_Text'])
# #
# # #val_hand_features = s.hand_features(test_df['Headline'],test_df['Body_Text'])
# # #
# # # import pickle
# # #
# pickle.dump(train_hand_features,open('train_hand_features.p','wb'))
# pickle.dump(val_hand_features,open('val_hand_features.p','wb'))
test_hand_features = s.hand_features(test_df['Headline'],test_df['Body_Text'])
print("Hand Features")
train_hand_features = pickle.load(open('train_hand_features.p','rb'))
#test_hand_features = pickle.load(open('test_hand_features.p','rb'))
val_hand_features = pickle.load(open('val_hand_features.p','rb'))

train_hand_features = np.array(train_hand_features)
test_hand_features = np.array(test_hand_features)
val_hand_features = np.array(val_hand_features)


print("Over Lap Features")

train_overlap_features = s.word_overlap_features(X_train['Headline'],X_train['Body_Text'])
val_overlap_features = s.word_overlap_features(X_val['Headline'],X_val['Body_Text'])
test_overlap_features = s.word_overlap_features(test_df['Headline'],test_df['Body_Text'])
# #
# pickle.dump(train_overlap_features,open('train_overlap_features.p','wb'))
# pickle.dump(val_overlap_features,open('val_overlap_features.p','wb'))


# train_overlap_features = pickle.load(open('train_overlap_features.p','rb'))
# #test_overlap_features = pickle.load(open('test_overlap_features.p','rb'))
# val_overlap_features = pickle.load(open('val_overlap_features.p','rb'))

train_overlap_features = np.array(train_overlap_features)
val_overlap_features=np.array(val_overlap_features)
test_overlap_features = np.array(test_overlap_features)

#
#Polarity Features (Baseline Feature)
print("Polarity Features")
train_polarity_features = s.polarity_features(X_train['Headline'],X_train['Body_Text'])
val_polarity_features = s.polarity_features(X_val['Headline'],X_val['Body_Text'])
test_polarity_features = s.polarity_features(test_df['Headline'], test_df['Body_Text'])
# pickle.dump(train_polarity_features,open('train_polarity_features.p','wb'))
# pickle.dump(val_polarity_features,open('val_polarity_features.p','wb'))


# train_polarity_features = pickle.load(open('train_polarity_features.p','rb'))
# #test_polarity_features = pickle.load(open('test_polarity_features.p','rb'))
# val_polarity_features=pickle.load(open('val_polarity_features.p','rb'))

train_polarity_features = np.array(train_polarity_features)
test_polarity_features = np.array(test_polarity_features)
val_polarity_features=np.array(val_polarity_features)

#
#Refuting Features (Baseline)
train_refuting_features = s.refuting_features(X_train['Headline'],X_train['Body_Text'])
val_refuting_features = s.refuting_features(X_val['Headline'],X_val['Body_Text'])
test_refuting_features = s.refuting_features(test_df['Headline'],test_df['Body_Text'])
# # pickle.dump(train_refuting_features,open('train_refuting_features.p','wb'))
# # pickle.dump(val_refuting_features,open('val_refuting_features.p','wb'))
# # print("Refuting Features")
#
# train_refuting_features = pickle.load(open('train_refuting_features.p','rb'))
# #test_refuting_features = pickle.load(open('test_refuting_features.p','rb'))
# val_refuting_features=pickle.load(open('val_refuting_features.p','rb'))

train_refuting_features = np.array(train_refuting_features)
test_refuting_features = np.array(test_refuting_features)
val_refuting_features=np.array(val_refuting_features)


test_labels = list(test_df['Stance'])

train_features = hstack([
    train_body_tfidf,
    train_headline_tfidf,
    train_hand_features,
    train_overlap_features,
    train_polarity_features,
    train_refuting_features,
    train_cosine_features
])

val_features = hstack([
    val_body_tfidf,
    val_headline_tfidf,
    val_hand_features,
    val_overlap_features,
    val_polarity_features,
    val_refuting_features,
    val_cosine_features
])
test_features = hstack([
    test_body_tfidf,
    test_headline_tfidf,
    test_hand_features,
    test_overlap_features,
    test_polarity_features,
    test_refuting_features,
    test_cosine_features
])

test_labels = list(test_df['Stance'])
test_lables=np.array(test_labels)

import xgboost

from sklearn.ensemble import AdaBoostClassifier

classifiers =AdaBoostClassifier(n_estimators=200,learning_rate=0.5)

print("training the model")


#clf = GridSearchCV(AdaBoostClassifier(), parameters, cv=10, verbose=2)

#clf.fit(train_features, y_train)

#print('Best Score: ', clf.best_score_)
#print('Best Params: ', clf.best_params_)

classifiers.fit(train_features,y_train)
print("Prediction on dev dataset Ada")

predicted={}
predicted=classifiers.predict(val_features)

print(report_score(y_val, predicted))

print("Prediction on test dataset")
predicted=classifiers.predict(test_features)

competetion_unlabeled = pd.read_csv('C:\\Users\\jeswi\\competition_test_stances_unlabeled.csv')
Predicted = pd.DataFrame({'Stance': predicted})
result = pd.concat([competetion_unlabeled, Predicted], axis=1, sort=False)
result.to_csv('C:\\Users\\jeswi\\Desktop\\submission_LSTM_Glove.csv', index=False, encoding='utf-8')

print(report_score(test_labels, predicted))

print('ADA:')
# acc=accuracy_score(test_labels, predicted)
# print(acc)
# f1=f1_score(test_labels, predicted, average='weighted')
# print('f1score',f1)
# r=recall_score(test_labels, predicted, average='weighted')
# print('recall',r)
# p=precision_score(test_labels, predicted, average='weighted')
# print('prescision',p)

classifiers=xgboost.XGBClassifier(n_estimators=200,learning_rate=0.5)

classifiers.fit(train_features,y_train)
print("Prediction on dev dataset")
predicted=classifiers.predict(val_features)

print(report_score(y_val, predicted))
print("Prediction on test dataset")
predicted=classifiers.predict(test_features)

print(report_score(test_labels, predicted))

from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score
#
# print('xg:')
# acc=accuracy_score(test_labels, predicted   )
# print(acc)
# f1=f1_score(test_labels, predicted, average='weighted')
# print('f1score',f1)
# r=recall_score(test_labels, predicted, average='weighted')
# print('recall',r)
# p=precision_score(test_labels, predicted, average='weighted')
# print('prescision',p)
