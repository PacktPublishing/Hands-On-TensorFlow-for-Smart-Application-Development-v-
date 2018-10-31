import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import json
import pickle
import urllib
from sklearn.preprocessing import MultiLabelBinarizer

""" Load the dataset """

# Download the data from GCS
# wget 'https://storage.googleapis.com/movies_data/movies_metadata.csv'
data = pd.read_csv('movies_metadata.csv')

""" Preparing the data for our model """

urllib.request.urlretrieve(
    'https://storage.googleapis.com/bq-imports/descriptions.p', 'descriptions.p')
urllib.request.urlretrieve(
    'https://storage.googleapis.com/bq-imports/genres.p', 'genres.p')

descriptions = pickle.load(open('descriptions.p', 'rb'))
genres = pickle.load(open('genres.p', 'rb'))

""" Splitting our data """

train_size = int(len(descriptions) * .8)

train_descriptions = descriptions[:train_size].astype('str')
train_genres = genres[:train_size]

test_descriptions = descriptions[train_size:].astype('str')
test_genres = genres[train_size:]

""" Formatting our labels """

encoder = MultiLabelBinarizer()
encoder.fit_transform(train_genres)
train_encoded = encoder.transform(train_genres)
test_encoded = encoder.transform(test_genres)
num_classes = len(encoder.classes_)

# Print all possible genres and the labels for the first movie in our training dataset
print(encoder.classes_)
print(train_encoded[0])

""" Create our TF Hub embedding layer """

description_embeddings = hub.text_embedding_column(
    "descriptions",
    module_spec="https://tfhub.dev/google/universal-sentence-encoder/2",
    trainable=False)

""" Instantiating our DNNEstimator Model """

multi_label_head = tf.contrib.estimator.multi_label_head(
    num_classes,
    loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE
)

features = {
  "descriptions": np.array(train_descriptions).astype(np.str)
}
labels = np.array(train_encoded).astype(np.int32)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    features, labels, shuffle=True, batch_size=32, num_epochs=25)
estimator = tf.contrib.estimator.DNNEstimator(
    head=multi_label_head,
    hidden_units=[64,10],
    feature_columns=[description_embeddings])

""" Training and serving our model """

estimator.train(input_fn=train_input_fn)

# Define our eval input_fn and run eval
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"descriptions": np.array(test_descriptions).astype(np.str)},
    test_encoded.astype(np.int32),
    shuffle=False)
estimator.evaluate(input_fn=eval_input_fn)

""" Generating predictions on new data """

# Test our model on some raw description data
raw_test = [
    "An examination of our dietary choices and the food we put in our bodies. Based on Jonathan Safran Foer's memoir.", # Documentary
    "After escaping an attack by what he claims was a 70-foot shark, Jonas Taylor must confront his fears to save those trapped in a sunken submersible.", # Action, Adventure
    "A teenager tries to survive the last week of her disastrous eighth-grade year before leaving to start high school.", # Comedy
]

# Generate predictions
predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"descriptions": np.array(raw_test).astype(np.str)},
    shuffle=False)
results = estimator.predict(predict_input_fn)

# Display predictions
for movie_genres in results:
  top_2 = movie_genres['probabilities'].argsort()[-2:][::-1]
  for genre in top_2:
    text_genre = encoder.classes_[genre]
    print(text_genre + ': ' + str(round(movie_genres['probabilities'][genre] * 100, 2)) + '%')
  print('')
