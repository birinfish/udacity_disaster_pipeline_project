import sys
import nltk
nltk.download(['omw-1.4','punkt', 'wordnet', 'stopwords'])

import re
import pandas as pd
import numpy as npr
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


stop_words = set(stopwords.words("english"))

def load_data(database_filepath):
    """
    function to pull data from database "database_filepath" (arg)
    argument:
            database_filepath(str)
    return:
            X (np.array): input variable for the model
            Y (np.array): output variables for the model
            category_names (list)
    """
    # create engine with the databse file path
    engine = create_engine(f'sqlite:///{database_filepath}')

    # create a dataframe with the data
    df = pd.read_sql_table(table_name='DISASTER_MSG', con= engine)

    # define target variables X and Y 
    X = df["message"].values
    Y = df.iloc[:,4:].values

    # get categiry names
    category_names = list(df.iloc[:,4:].columns)

    return X, Y, category_names


def tokenize(text):
    """
    function to process text data to be used as input variable
    """
    # remove punctuation, convert to the lowercase strings
    # and remove spaces
    text = re.sub(r'[^\w\s]',' ', text).lower().strip()
    
    # divide text into a list of substrings
    tokens = word_tokenize(text)
    
    # lemmatize and remove stopwords from the list
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    
    return tokens

    
def build_model():
    """
    function to build a ML model with pipeline
    with two transformers 'CountVectorizer' and 'TfidfTransformer'
    and one predictor 'RandomForestClassifier' with the help of 
    'MultiOutputClassifier' for multiple target variables 
    """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
       "clf__estimator__n_estimators" : [50, 100, 150],
       "clf__estimator__min_samples_split": [2, 3, 4],
       "clf__estimator__max_depth": range(2,20,1)
       }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    function to test a model with classification_report
    The test results are printed with 'precision', 'recall'
    and 'f1-score' for each category
    """
    # get y_pred results
    y_pred = model.predict(X_test)

    for i in range(len(category_names)):
        cat_name = category_names[i]
        cr_y = classification_report(Y_test[:,i], y_pred[:,i])
    
        print("---------------------------")
        print(f"'{i}' column :", cat_name)
        print(cr_y)
    
    print(model.best_params_)


def save_model(model, model_filepath):
    """
    pickle a model with "pickled file path" (arg 2)
    """
    pickle.dump(model, open(f"{model_filepath}", "wb"))


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