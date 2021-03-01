import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Function to load and combine the csv files
    
    Input:
    filepaths to messages and categories files
    
    Output:
    Pandas Dataframe with combined dataset
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, left_on='id', right_on='id', how='inner')

    # create a dataframe of the 36 individual category columns
    categories_exp = categories.categories.str.split(';',expand=True)
    categories_exp=pd.DataFrame(categories_exp)

    # create categories columnames & rename the  columns
    row = categories_exp.iloc[0,:]
    category_colnames = row.apply(lambda x: x[:-2])
    categories_exp.columns = category_colnames

    #Process columns to only keep the indicator value 0/1
    for column in categories_exp:
        # set each value to be the last character of the string
        categories_exp[column] = categories_exp[column].astype(str).str[-1]
        # convert column from string to numeric
        categories_exp[column] = categories_exp[column].astype(int)
        
    # drop the original categories column from `df`
    df.drop('categories',axis=1,inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df_concat = pd.concat([df,categories_exp],axis=1)
    return df_concat


def clean_data(df):
    '''
    Function to clean the dataset and drop duplicates
    
    Input:
    DataFrame
    
    Output:
    Cleaned DataFrame
    '''
    # drop duplicates
    df=df.drop_duplicates()
    return df


def save_data(df, database_filename):
    '''
    Function to save a dataframe into a sqllite database
    
    Input:
    Dataframe
    database filename
    
    Output:
    None, creates the database
    '''
    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('desaster_messages', engine, index=False)


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
              'DisasterResponseMarc.db')


if __name__ == '__main__':
    main()