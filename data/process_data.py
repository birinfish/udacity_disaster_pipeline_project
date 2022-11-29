import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath:str, categories_filepath:str):
    """
    function to load two dataframes from csv files
    and to merge them into one big data frame
    
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath) 
    
    # merge the two datasets
    df = pd.merge(left= messages, right= categories, how= "outer", on= "id")
    
    return df
    
    
def clean_data(df:pd.DataFrame):
    """
    function to clean the merged datasets 
    0 split the column "categories" into separate columns
    1 rename the column names
    2 keep only the integer values for the category columns 
    3 remerge the separated columns into message dataset
    4 remove duplicates
    """
    # split the column "categories" into separate columns
    categories = df["categories"].str.split(";", expand=True)
    
    # select the first row of the categories dataframe
    # extract strings of each category
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])

    # remanem the columns of df with the extracted strings
    categories.columns = category_colnames
    
    # iterate each column to extract the last character of the string
    for column in categories:
        categories[column] = categories[column].apply(
            lambda x: x[-1:]).astype(str)
        
        # convert dtype of the columns with extracted strings 
        # from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # drop the column "categories" and remerge the data frame
    # with the separate columns restructured with a new dataframe "categories"
    df = df.drop(columns= "categories")
    df = pd.concat([df, categories], axis =1)

    # for columns having values which are not 0 or 1, replace the values
    # with the given data, the column "related" has some values "2"
    df["related"].replace(2,1,inplace=True) 
    
    # remove duplicates
    df = df.drop_duplicates()
    
    return df
    

def save_data(df:pd.DataFrame, database_filename:str):
    """
    function to upload the dataframe(arg) into database 
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql(name = 'DISASTER_MSG', con= engine,if_exists='replace', index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()