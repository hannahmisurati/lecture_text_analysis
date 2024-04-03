import pandas

import dill as pickle

# Load the trained model, CountVectorizer instance, lemmatizer, stopwords, string module, and preprocessing function using pickle
with open("text_analysis_machine.pickle", "rb") as f:
  machine = pickle.load(f) # Load the trained machine learning model
  count_vectorize_transformer = pickle.load(f)  # Load the CountVectorizer instance
  lemmatizer = pickle.load(f)  # Load the lemmatizer
  stopwords = pickle.load(f) # Load the stopwords
  string = pickle.load(f) # Load the string module
  pre_processing = pickle.load(f) # Load the preprocessing function
  

# Load new reviews data from CSV file
new_reviews = pandas.read_csv("new_reviews.csv") 

# Transform new reviews data using the CountVectorizer instance
new_reviews_tranformed = count_vectorize_transformer.transform(new_reviews.iloc[:,0])

# Make predictions and predict probabilities for new reviews
prediction = machine.predict(new_reviews_tranformed)
prediction_prob = machine.predict_proba(new_reviews_tranformed)

# Print predictions and prediction probabilities
print(prediction)
print(prediction_prob)


# Add predictions and prediction probabilities to the new reviews DataFrame
new_reviews['prediction'] = prediction
prediction_prob_dataframe = pandas.DataFrame(prediction_prob)
prediction_prob_dataframe = prediction_prob_dataframe.rename(columns={
  prediction_prob_dataframe.columns[0]: "prediction_prob_1",
  prediction_prob_dataframe.columns[1]: "prediction_prob_3",
  prediction_prob_dataframe.columns[2]: "prediction_prob_5"
  })

new_reviews = pandas.concat([new_reviews,prediction_prob_dataframe], axis=1)

# Print new reviews with predictions
print(new_reviews)

# Rename columns and round prediction probabilities
new_reviews = new_reviews.rename(columns={
  new_reviews.columns[0]: "text"
  })
new_reviews['prediction'] = new_reviews['prediction'].astype(int)
new_reviews['prediction_prob_1'] = round(new_reviews['prediction_prob_1'],4)
new_reviews['prediction_prob_3'] = round(new_reviews['prediction_prob_3'],4)
new_reviews['prediction_prob_5'] = round(new_reviews['prediction_prob_5'],4)

# Save new reviews with predictions to CSV file
new_reviews.to_csv("new_reviews_with_prediction.csv", index=False)

#This code loads the trained model, CountVectorizer instance, lemmatizer, stopwords, string module, 
#and preprocessing function from the "text_analysis_machine.pickle" file using pickle. 
#Then, it loads new reviews data from a CSV file, transforms the new reviews data using the CountVectorizer instance, 
#makes predictions and predicts probabilities for the new reviews, adds predictions and prediction probabilities 
#to the new reviews DataFrame, renames columns and rounds prediction probabilities, and finally saves 
#the new reviews with predictions to a CSV file.
















