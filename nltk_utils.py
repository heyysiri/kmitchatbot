import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.stem.porter import PorterStemmer
import numpy as np
nltk.download('punkt')
nltk.download('stopwords')
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag

def lemmatize(word, pos):
    if pos.startswith('N'):
        return lemmatizer.lemmatize(word, pos='n')
    elif pos.startswith('V'):
        return lemmatizer.lemmatize(word, pos='v')
    elif pos.startswith('R'):
        return lemmatizer.lemmatize(word, pos='r')
    elif pos.startswith('J'):
        return lemmatizer.lemmatize(word, pos='a')
    else:
        return lemmatizer.lemmatize(word)
    
def pos_tagging(tokenized_sentence):
    return pos_tag(tokenized_sentence)

def remove_stopwords(tokenized_sentence):
    return [word for word in tokenized_sentence if word.lower() not in stop_words]


