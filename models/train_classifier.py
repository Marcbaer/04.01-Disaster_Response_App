import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import re
import pickle

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix, classification_report, hamming_loss
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.svm import SVC

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    '''
    Function to load the desaster response data for classification.
    
    Input:
    database filepath
    
    Output:
    Dataset split into Features (X) and Labels (Y) and the category names
    '''
    # load data from database
    engine =create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('desaster_messages', con=engine)  

    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    category_names=list(Y.columns.values)
    
    # Fill NaN values with 0
    for x in Y.columns:
        Y[x]=Y[x].fillna(0)
        
        
    return X,Y,category_names

def tokenize(text):
    """
    Function to pre-process text messages for classification.
    
    Input:
    Text messages
       
    Output:
    Tokenized text
    """
    #remove URLs
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    #stemming
    tokens = [PorterStemmer().stem(w) for w in tokens]
    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words("english")]
    tokens=pd.DataFrame(tokens)
    #tokens.replace([np.inf, -np.inf], '', inplace=True)
    #tokens.replace(np.nan, '', inplace=True)
    tokens=tokens.values.tolist()
    tokens=[x[0] for x in tokens]
    return tokens


def build_model():
    '''
    Build the classification Model
    Input: 
    None
    
    Output:
    GridSearch Model
    '''    
    #Define Pipeline
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
    
    #Define Grid Search Parameter range
    parameters = {'clf__estimator__n_estimators': [30],
                      'clf__estimator__min_samples_split': [2, 3],
                      'clf__estimator__criterion': ['entropy']
                 }
    #Build Model
    cv = GridSearchCV(pipeline, param_grid=parameters,verbose=4)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate model performance using test data
    
    Input: 
    model: Model to be evaluated
    X_test: Test data (features)
    Y_test: True labels for Test data
    category_names: Category Labels
    
    Output:
    Print accuracy and classfication report for each category
    '''
    Y_pred = model.predict(X_test)
    
    # Calculate the accuracy for each of them.
    for i in range(len(category_names)):
        print("Category:", category_names[i],"\n", classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))
        print('Accuracy of %25s: %.2f' %(category_names[i], accuracy_score(Y_test.iloc[:, i].values, Y_pred[:,i])))



def save_model(model, model_filepath):
    '''
    Function to save the trained model.
    
    Input:
    Trained model and the destination filepath
    
    Output:
    None, saves the model
    '''
    # Pickle best model
    pickle.dump(model, open('disaster_model.sav', 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()