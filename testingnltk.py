import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import numpy as np

# Download NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

stemmer = nltk.PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

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

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag

# Example usage
sentence = "Please google something"

# Tokenization
tokenized_sentence = tokenize(sentence)

# Part-of-Speech (POS) tagging
pos_tags = pos_tagging(tokenized_sentence)

# Removing stop words
filtered_sentence = remove_stopwords(tokenized_sentence)

# Lemmatization
lemmatized_sentence = [lemmatize(word, pos) for word, pos in pos_tags]

# Stemming (as per your original implementation)
stemmed_sentence = [stem(word) for word in tokenized_sentence]

print("Original Sentence:", sentence)
print("Tokenized Sentence:", tokenized_sentence)
print("POS Tags:", pos_tags)
print("Filtered Sentence (Stopwords Removed):", filtered_sentence)
print("Lemmatized Sentence:", lemmatized_sentence)
print("Stemmed Sentence:", stemmed_sentence)
