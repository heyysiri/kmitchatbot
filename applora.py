from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from trainwlora import load_data
print("hi this is applora")
app = Flask(__name__)
CORS(app)

# Load the fine-tuned model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained("./trained_model")
tokenizer = DistilBertTokenizerFast.from_pretrained("./trained_model")

# Load label mapping from the training data
data = load_data("C:/Users/sirik/kmitchatbot/my-react-app/src/intents.json")
label_mapping = {idx: intent['tag'] for idx, intent in enumerate(data['intents'])}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input text from the frontend
        message = request.json['message']

        # Tokenize and encode the input message
        encoded_data = tokenizer(message, return_tensors='pt', truncation=True, padding=True)
        input_ids = encoded_data['input_ids']
        attention_mask = encoded_data['attention_mask']

        # Make prediction using the loaded model
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            predicted_label = torch.argmax(output.logits, dim=1).item()

        # Map numerical label back to the original tag
        predicted_tag = label_mapping[predicted_label]

        return jsonify({'tag': predicted_tag})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=True, use_reloader=False)
