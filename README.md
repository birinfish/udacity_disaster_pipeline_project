# Disaster Response Pipeline Project

### Instructions:
1. Step 1: run process_data.py

    - /data/process_data.py
      In this python script, two csv files are cleaned and transformed before used for train a ML model
      And the transformed data is stored in database table.(i.e. "DisasterResponse.db")
      
      To run the script, input syntax as shown below. It requires three additional arguments
      as shown the example below
      (two input csv file names, a database path setup by you)
      
      Ex:
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

2. Step 2: run train_classifier.py

    - /models/train_classifier.py
      In this script, the input data(text) is tokenized, tansformed into a meaningful representation of numbers
      before being used to train a ML model. The input data will be taken from the database which is saved
      in step 1. Finally the ML model is stored with pickle

      To run the script, input syntax as shown below. It requires two additional arguments
      as shown the example below
      (a database path setup by you, a pickle file path for your model)
      
      Ex:
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Step 3: run run.py
    
    - /app/run.py
     In this script, you upload your data from the database(step 1) and the pickle file for the model(step 2)
     into the script and create visualizations based on the data. 

     To run the script, simply input syntax "python run.py" then you can see the results in a web address 
     http://0.0.0.0:3001/ or an address you will find from your terminal
