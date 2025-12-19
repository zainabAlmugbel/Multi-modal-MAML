# FastText Word Embeddings Example
import os
import numpy as np
import fasttext
import fasttext.util

class FastTextclass():
    """ Usage:
    """
    def __init__(self, model_name='cc.en.300.bin'):

    # Download pre-trained model if needed (only run this once)
    # This downloads the 300-dimensional English model (about 6.8 GB)
        #fasttext.util.download_model('en', if_exists='ignore')
        #print("Model downloaded successfully")
        #print("Loading pre-trained model")
        self.model=fasttext.load_model(model_name)


    # Function to get word embeddings
    def get_word_embedding(self, word):
        
        return self.model.get_word_vector(word)

    # Function to get sentence embedding (average of word vectors)
    def get_sentence_embedding(self, sentences):
        
        words = sentences.lower().split()
        word_vectors = [self.model.get_word_vector(word) for word in words]
        return np.mean(word_vectors, axis=0)
    






    



