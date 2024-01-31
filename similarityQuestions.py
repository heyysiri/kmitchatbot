print("Just got in to similarityQuestions")
import nltk
import numpy as np
from gensim.models import KeyedVectors
from questionGen import result
from questionGen import document
# Download nltk resources (if not already downloaded)
# nltk.download('punkt')
print("Imported stuff from questionGen")
# Load the pre-trained Google News Word2Vec model (replace 'path/to/GoogleNews-vectors-negative300.bin' with the actual path)
word_model = KeyedVectors.load_word2vec_format(
    'C:/Users/sirik/kmitchatbot/GoogleNews-vectors-negative300.bin.gz', binary=True
)

# Example generated questions and webpage content
generated_questions = result
webpage_content = document.content

# Tokenize the questions and webpage content
question_tokens = [nltk.word_tokenize(question) for question in generated_questions]
webpage_tokens = nltk.word_tokenize(webpage_content)

# Handle missing embeddings (optional, depending on your needs)
def handle_missing_embeddings(tokens, model):
    known_tokens = [token for token in tokens if token in model]
    if len(known_tokens) == 0:
        return None
    return np.mean([model[token] for token in known_tokens], axis=0)

# Get embeddings for questions and webpage content
question_embeddings = [handle_missing_embeddings(tokens, word_model) for tokens in question_tokens]
webpage_embedding = handle_missing_embeddings(webpage_tokens, word_model)

# Compute cosine similarity scores
similarity_scores = [
    np.dot(webpage_embedding, q_embedding)
    / (np.linalg.norm(webpage_embedding) * np.linalg.norm(q_embedding))
    for q_embedding in question_embeddings
]
# print("Similarity Scores:", similarity_scores)

# Set a threshold for relevance
threshold = 0.65

# Filter out relevant questions
relevant_questions = [
    generated_questions[i] for i, score in enumerate(similarity_scores) if score >= threshold
]
# print("Relevant Questions:", relevant_questions)
# print("Number of Q in Result List: ", len(result))
# print("Number of Q in Relavant List: ", len(relevant_questions))
