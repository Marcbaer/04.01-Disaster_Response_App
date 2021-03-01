# Disaster Response Pipeline Project
This project builds and trains a Random Forest Classifier to categorize disaster response messages.
An ETL pipeline is built to process and transform the messages and the data is stored in a SQLite database.
The final model is deployed on a WebApp using Flask that allows to categorize new disaster response messages and tweets.

### Environment and Setup
The following package versions are used:


* python=3.8.5
* flask=1.1.2
* pandas=1.2.2
* scikit-learn=0.23.2
* scipy=1.6.1
* nltk=3.5
* sqlalchemy=1.3.23

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - Run ETL pipeline to clean and process data and store it using SQLite:
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - Run ML pipeline to train the classifier and save the trained model:
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    - `python run.py`

3. Navigate to http://0.0.0.0:3001/
