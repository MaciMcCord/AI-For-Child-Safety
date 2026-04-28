import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
import nltk
import string
import warnings
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

MAX_WORDS     = 10000
MAX_LEN       = 100
RANDOM_STATE  = 42

# Load the dataset
df = pd.read_csv('Hatespeech-data.csv')
print(df.head())

df['tweet'] = df['tweet'].str.lower()

class_0 = df[df['class'] == 0]  # Hate Speech
class_1 = df[df['class'] == 1].sample(n=3500, random_state=42)  # Offensive Language
class_2 = df[df['class'] == 2]  # Neutral

balanced_df = pd.concat([class_0, class_0, class_0, class_1, class_2], axis=0)

punctuations_list = string.punctuation
def remove_punctuations(text):
    temp = str.maketrans('', '', punctuations_list)
    return text.translate(temp)

df['tweet']= df['tweet'].apply(lambda x: remove_punctuations(x))

features = balanced_df['tweet']
target = balanced_df['class']
X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.2, random_state=42)

# One-hot encode the labels
Y_train = pd.get_dummies(Y_train)
Y_val = pd.get_dummies(Y_val)

# Tokenization
tokenizer = Tokenizer(num_words=MAX_WORDS, lower=True, split=' ')
tokenizer.fit_on_texts(X_train)

# Convert text to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)

# Pad sequences
X_train_padded = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
X_val_padded = pad_sequences(X_val_seq, maxlen=MAX_LEN, padding='post', truncating='post')


model = keras.Sequential([
    layers.Embedding(input_dim=MAX_WORDS, output_dim=64, input_length=MAX_LEN),
    layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dense(64, activation='relu', kernel_regularizer='l2'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(3, activation='softmax')
])

model.build(input_shape=(None, MAX_LEN))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


es = EarlyStopping(patience=3, monitor='val_accuracy', restore_best_weights=True)
lr = ReduceLROnPlateau(patience=2, monitor='val_loss', factor=0.5, verbose=0)

history = model.fit(X_train_padded, Y_train,
                    validation_data=(X_val_padded, Y_val),
                    epochs=20,
                    batch_size=10,
                    callbacks=[es, lr])

test_loss, test_acc = model.evaluate(X_val_padded, Y_val)
print(f"Validation Accuracy: {test_acc:.2f}")

model.save("hatespeech_model.keras")

with open("tokenizer.json", "w") as f:
    f.write(tokenizer.to_json())

print("Model and tokenizer saved.")
