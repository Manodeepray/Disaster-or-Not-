# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 13:07:06 2024

@author: Manodeep ray

"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Embedding, GlobalMaxPooling1D, Dropout, Bidirectional, Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import f1_score, mean_squared_error


def preprocess(data):
    df = data
    df = df.drop(['id','keyword','place'] , axis =  1)
    df.columns = ['data','label']
    y = df.iloc[:,1].values
    x = df.iloc[:,0].values
    
    x_train , x_test , y_train , y_test = train_test_split(x,y,test_size =0.2 , random_state = 42)
    
    checkpoint_callback = ModelCheckpoint(filepath='/best_weights.keras',
                                              monitor='val_accuracy',
                                              verbose=1,
                                              save_best_only=True,
                                              mode='max')
    
    MAX_VOCAB_SIZE = 20000
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
    
    # Fit tokenizer on training data
    tokenizer.fit_on_texts(x_train)
    
    # Convert text to sequences of word indices
    sequences_train = tokenizer.texts_to_sequences(x_train)
    sequences_test = tokenizer.texts_to_sequences(x_test)
    
    data_train = pad_sequences(sequences_train)
    print(data_train.shape)
    T = data_train.shape[1]
    
    data_test = pad_sequences(sequences_test , maxlen = T)
    print(data_test.shape)
    
    
    word2idx = tokenizer.word_index
    v = len(word2idx)
    
    D = 20
    M = 15
    
    return D , v , M , T , data_train , data_test ,x_train , x_test , y_train , y_test
    

def load_model(T,v,D):
    i = Input(shape = (T,))
    x = Embedding(v +1 ,D)(i)
    x = LSTM(128, return_sequences=True)(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1 , activation  = 'sigmoid')(x)
    model = Model(i , x)
    return model


def callback():
    checkpoint_callback = ModelCheckpoint(filepath='/best_weights.keras',
                                  monitor='val_accuracy',
                                  verbose=1,
                                  save_best_only=True,
                                  mode='max')
    return checkpoint_callback

def evaluation(data_test, y_test , model):
    loss, accuracy = model.evaluate(data_test, y_test)
    
    # Make predictions on the test data
    y_pred = model.predict(data_test)
    
    # Round the predictions to get binary values
    y_pred_binary = (y_pred > 0.5).astype(int)
    f1 = f1_score(y_test, y_pred_binary)
    
    # Compute mean squared error
    mse = mean_squared_error(y_test, y_pred)
    
    print("Test loss:", loss)
    print("Test accuracy:", accuracy)
    print("F1 score:", f1)
    print("Mean squared error:", mse)
    
    return 0

if __name__ == "__main__":
    data = pd.read_csv("/train.csv")
    D , v , M , T , data_train , data_test , x_train , x_test , y_train , y_test = preprocess(data)
    checkpoint_callback = callback()
    model = load_model(T , v, D)
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )    
    history = model.fit(
        data_train,
        y_train,
        epochs=20,  # Increase epochs to allow more training
        validation_data=(data_test, y_test),
        callbacks=[checkpoint_callback]
    )
    
    model.load_weights("/best_weights.keras")
    evaluation(data_test , y_test , model)
    
    
    
    
    
    
    
    
    