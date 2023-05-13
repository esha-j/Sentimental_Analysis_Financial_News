#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


#INSTALL THE REQUIRED LIBRARIES
#pip install pandas
#pip install tensorflow
#pip install matplotlib 


# In[2]:


# DATA PREPROCESSING
# Read the dataset

import pandas as pd
df = pd.read_csv("Financial_News.csv")


# In[3]:


df.head(5)


# In[4]:


df.columns


# In[5]:


# Clean the Dataset by keeping only Positive and Negative Comments

df = df[df['Sentiment'] != 'neutral']

print(df.shape)
df.head(5)


# In[6]:


# Check the values of the Sentiment column

df["Sentiment"].value_counts()


# In[7]:


# Convert Categorical value into Numerical value by using Factorize() method

Sentiment_Label = df.Sentiment.factorize()
Sentiment_Label


# In[8]:


# Get all text values 

Sentence = df.Sentence.values


# In[9]:


Sentence


# In[10]:


# Tokenise the words [All words/sentences are broken into smaller parts (tokens)]

from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=6000)
tokenizer.fit_on_texts(Sentence)
vocab_size = len(tokenizer.word_index) + 1

tokenizer


# In[11]:


# Replace words with their assigned numbers

Assigned_num = tokenizer.texts_to_sequences(Sentence)


# In[12]:


# Make all sentences of equal length

from tensorflow.keras.preprocessing.sequence import pad_sequences

padded_sequence = pad_sequences(Assigned_num, maxlen=200)


# In[13]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding

vocab_size = len(tokenizer.word_index) + 1
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(vocab_size, embedding_vector_length, input_length=200))
model.add(SpatialDropout1D(0.25))
model.add(LSTM(50,dropout=0.5,recurrent_dropout=0.5))
model.add(Dropout(0.5))
                  
                
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

print(model.summary())


# In[14]:


# Downgrade your Numpy if the above code results in error.
# pip install -U numpy==1.18.5


# In[36]:


# Train the model 
# Validation Split = 30%
# Epoch = 10
# Batch Size = 30
# For quick processing, alter the above inputs.

fit_model = model.fit(padded_sequence,Sentiment_Label[0],validation_split=0.3, epochs=5, batch_size=80)


# In[37]:


# Plot the accuracy 

import matplotlib.pyplot as plt

plt.plot(fit_model.history['accuracy'], label='Acc')
plt.plot(fit_model.history['val_accuracy'], label='Val_acc')
plt.legend()
plt.show()

plt.savefig("Accuracy plot.jpg")


# In[38]:


# Plot loss

plt.plot(fit_model.history['loss'], label='loss')
plt.plot(fit_model.history['val_loss'], label='val_loss')

plt.legend()
plt.show()

plt.savefig("Loss plt.jpg")


# In[41]:


#Execute the model

def predict_sentiment(text):
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw,maxlen=200)
    prediction = int(model.predict(tw).round().item())
    print("Predicted label: ",Sentiment_Label[1][prediction])

test_sentence = input('Enter your statement: ')
predict_sentiment(test_sentence)


# In[ ]:




