#!pip install simpletransformers
import pandas as pd

from simpletransformers.classification import ClassificationModel
from sklearn.model_selection import train_test_split

model = ClassificationModel('bert', 'bert-base-cased', 4, use_cuda = False)


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

X_train, X_val, y_train, y_val = train_test_split(train_df[['Body_Text','Headline']], train_df['Stance'], test_size=0.1, random_state=42)
df1 = X_train['Body_Text']+ X_train['Headline']
train = pd.concat([df1, y_train], axis=1, sort=False)
train.replace('unrelated',1,True)
train.replace('agree',2,True)
train.replace('disagree',3,True)
train.replace('discuss',4,True)
test_df = competetion_stances.join(competetion_bodies.set_index('Body ID'), on='Body ID')
model.train_model(train_df)

model.eval_model(test_df)

