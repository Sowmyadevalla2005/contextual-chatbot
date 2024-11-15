import nltk
nltk.download('punkt')
import numpy as np

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    """
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bag = [ 0 ,  1 ,  0 ,  1 ,  0 ,  0 ,  0 ]
    
    """
    tokenized_sentence = [stem(word) for word in tokenized_sentence]
    
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx,w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
            
    return bag

sentence = ["hello", "how", "are", "you"]
words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
bag = bag_of_words(sentence , words)
print(bag)
    #bag = [1 if word in tokenized_sentence else 0 for word in all_words]
    #return bag

# Example usage:
a = "Hey, how are you?"
print("Original sentence:", a)    

# Tokenize the sentence
a_tokens = tokenize(a)    
print("Tokens:", a_tokens)

# Stem each token
stemmed_words = [stem(word) for word in a_tokens]
print("Stemmed words:", stemmed_words)

# Example all words for bag of words
all_words = ["hey", "how", "are", "you", "hello"]
bag = bag_of_words(a_tokens, all_words)
print("Bag of words:", bag)
