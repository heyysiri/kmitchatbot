# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS 
import torch
import nltk
from model import NeuralNet  # Import your PyTorch model definition
from nltk_utils import bag_of_words, tokenize, stem

app = Flask(__name__)
CORS(app)

# Load the saved PyTorch model and necessary data
data = torch.load("data1.pth")
model_state = data["model_state"]
all_words = data["all_words"]
tags = data["tags"]

# Use the modified model architecture
model = NeuralNet(input_size=len(all_words), hidden_size=data["hidden_size"], num_classes=len(tags))
model.load_state_dict(model_state)
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input text from the frontend
        message = request.json['message']

        # Tokenize and stem the input message
        tokenized_message = tokenize(message)
        stemmed_message = [stem(word) for word in tokenized_message]

        # Create a bag of words
        X = bag_of_words(stemmed_message, all_words)

        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Make prediction using the loaded model
        with torch.no_grad():
            # Reshape the tensor to match the expected input shape
            output = model(X_tensor.view(1, -1))
            predicted_tag = tags[torch.argmax(output).item()]

        return jsonify({'tag': predicted_tag})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
