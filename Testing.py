import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import string
import warnings
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('Hatespeech-data.csv')
print(df.head())

# Lowercase
df['tweet'] = df['tweet'].str.lower()

# Remove punctuation
punctuations_list = string.punctuation
def remove_punctuations(text):
    temp = str.maketrans('', '', punctuations_list)
    return text.translate(temp)

df['tweet'] = df['tweet'].apply(remove_punctuations)

# Balance dataset
class_0 = df[df['class'] == 0]
class_1 = df[df['class'] == 1].sample(n=3500, random_state=42)
class_2 = df[df['class'] == 2]
balanced_df = pd.concat([class_0, class_0, class_0, class_1, class_2], axis=0)

# Features and target
features = balanced_df['tweet']
target = balanced_df['class']

# Split
X_train, X_val, Y_train, Y_val = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# One-hot encode
Y_train = pd.get_dummies(Y_train)
Y_val = pd.get_dummies(Y_val)

# Tokenization
max_words = 10000
max_len = 100

tokenizer = Tokenizer(num_words=max_words, lower=True, split=' ')
tokenizer.fit_on_texts(X_train)

X_train_padded = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len, padding='post', truncating='post')
X_val_padded = pad_sequences(tokenizer.texts_to_sequences(X_val), maxlen=max_len, padding='post', truncating='post')

# Model
model = keras.Sequential([
    layers.Embedding(input_dim=max_words, output_dim=32, input_length=max_len),
    layers.Bidirectional(layers.LSTM(16)),
    layers.Dense(512, activation='relu', kernel_regularizer='l1'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(3, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Callbacks
es = EarlyStopping(patience=3, monitor='val_accuracy', restore_best_weights=True)
lr = ReduceLROnPlateau(patience=2, monitor='val_loss', factor=0.5, verbose=0)

# Train
history = model.fit(
    X_train_padded, Y_train,
    validation_data=(X_val_padded, Y_val),
    epochs=20,
    batch_size=32,  # ✅ increased from 10 for faster training
    callbacks=[es, lr]
)

# Evaluate
test_loss, test_acc = model.evaluate(X_val_padded, Y_val)
print(f"Validation Accuracy: {test_acc:.2f}")