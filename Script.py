import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import string
import re
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from tensorflow.keras import models

df_train = pd.read_csv('Train.csv', lineterminator='\n')
df_valid = pd.read_csv('Valid.csv', lineterminator='\n')


# Dataset preprocessing
def clean(text):
    tokens = text.split()
    # Prepare punctuation filtering regular
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # Remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # Remove all non-letter tokens
    tokens = [word for word in tokens if word.isalpha()]
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # Remove tokens whose length is less than or equal to 1
    tokens = [w for w in tokens if len(w) > 1]
    tokens = ' '.join(tokens)
    return tokens


df_train.Text = df_train.Text.apply(clean)
df_valid.Text = df_valid.Text.apply(clean)
print(df_train.Text.map(lambda x: len(x)).describe())

# Split the dataset
x_train = df_train.iloc[:, 0:1].values
y_train = df_train.iloc[:, 1:].values
x_valid = df_valid.iloc[:, 0:1].values
y_valid = df_valid.iloc[:, 1:].values
y_train = tf.keras.utils.to_categorical(y_train)
y_valid = tf.keras.utils.to_categorical(y_valid)

# Create a vocabulary dictionary
word = []
for i in x_train:
    for j in i:
        word.append(j.split(' '))
for i in x_valid:
    for j in i:
        word.append(j.split(' '))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(word)
# Convert text to a sequence of integers
x_tr_seq = [tokenizer.texts_to_sequences(x_train[i]) for i in range(len(x_train))]
x_val_seq = [tokenizer.texts_to_sequences(x_valid[i]) for i in range(len(x_valid))]
# Integer sequence format (delete [])
x_tr_sequence = []
x_val_sequence = []
for i in x_tr_seq:
    for j in i:
        x_tr_sequence.append(j)
for i in x_val_seq:
    for j in i:
        x_val_sequence.append(j)
# Padding to prepare sequences of the same length
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_tr_sequence, padding='post', truncating='post', maxlen=1000)
x_valid = tf.keras.preprocessing.sequence.pad_sequences(x_val_sequence, padding='post', truncating='post', maxlen=1000)

# Model building
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(output_dim=32, input_dim=500000, input_length=1000))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(units=2, activation='softmax'))

print(model.summary())

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=20, batch_size=128, verbose=1)


# Visualization
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.show()


print(plot_learning_curves(history))

test_loss, test_acc = model.evaluate(x_valid, y_valid, verbose=2)
print('Test accuracy：', test_acc)

# save model
tf.saved_model.save(model, r"model\Model_2_01")

# Function for predicting
sentiment_dict = [1, 0]


def display_text_sentiment(text):
    input_seq = tokenizer.texts_to_sequences([text])
    pad_input_seq = tf.keras.preprocessing.sequence.pad_sequences(input_seq, padding='post', truncating='post',
                                                                  maxlen=1000)
    pred = model.predict(pad_input_seq)
    return sentiment_dict[np.argmax(pred)]


# Model Loading
loaded_model = models.load_model('model\Model_2_01')
df_test = pd.read_csv('testset.csv', lineterminator='\n', header=None, encoding='iso-8859-1')
df_test.columns = ['text']

df_test['pre'] = df_test.text.apply(display_text_sentiment)
df_test.to_csv("output.csv", columns=["text", "pre"], header=False, index=False)
