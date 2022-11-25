import json
import pickle
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
from sqlalchemy import create_engine

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DISASTER_MSG', engine)

# load model
model = pickle.load(open("../models/classifier.pkl", "rb"))

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # graph 2: count number of values for each category
    cat_count = []
    for column in df.columns[4:]:
        cat_num = df.loc[df[column]==1][column].count()
        cat_count = np.append(cat_count,cat_num)
    
    cat_names = list(df.columns[4:])

    # graph 3: for the genre "news", count number of values for each category
    # and make a pie chart with percentages of each category to total number of news message
    df_news = df.loc[df["genre"]=="news"]

    news_cat_count = {}
    for column in df_news.columns[4:]:
        news_cat_num = df_news.loc[df_news[column]==1][column].count()
        news_cat_count[column] =  news_cat_num
    
    # list up five categories having largest count number
    news_cat_list = sorted(news_cat_count.items(), key=lambda x:x[1], reverse = True)
    five_large_cat = news_cat_list[:5]

    # get the rest of values as "others" and add to the five_large_cat tuple list
    res = news_cat_list[5:]
    othr_sum = sum([item[1] for item in res])
    five_large_cat.append(("others", othr_sum))

    # get the rest of values as "others"
    labels = [item[0] for item in five_large_cat]
    values = [item[1] for item in five_large_cat]

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre",
                }
            }
        },
         {
            'data': [
                Bar(
                    x=cat_names,
                    y=cat_count,
                    marker=dict(color ='goldenrod')
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
            'data': [
                Pie(
                    labels=labels,
                    values=values,
                    textinfo="label+percent"
                )
            ],

            'layout': {
                'title': 'Top 5 categories found for the genre "news"'
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()