from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import json
from transformers import DistilBertTokenizer, DistilBertModel
from peft import get_peft_model, LoraConfig

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Load label mapping
with open('label_mapping.json', 'r') as f:
    label_mapping = json.load(f)
reverse_label_mapping = {i: label for label, i in label_mapping.items()}
output_size = len(label_mapping)

# Load DistilBERT tokenizer and base model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
base_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Define the classifier model with LoRA
class IntentClassifier(nn.Module):
    def __init__(self, base_model, input_size, output_size):
        super(IntentClassifier, self).__init__()
        self.distilbert = base_model
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        logits = self.fc(pooled_output)
        return logits

# Setup LoRA configuration
lora_config = LoraConfig(
    r=16, 
    lora_alpha=32, 
    target_modules=['q_lin', 'v_lin'],
    lora_dropout=0.1, 
    bias='lora_only'
)

# Integrate LoRA with the model
classifier = IntentClassifier(base_model, base_model.config.hidden_size, output_size)
classifier = get_peft_model(classifier, lora_config)

# Load the trained model weights
classifier.load_state_dict(torch.load('distilbert_intent_classifier_lora.pth', map_location=torch.device('cpu')))
classifier.eval()

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
            output = classifier(input_ids, attention_mask)
            predicted_label = torch.argmax(output, dim=1).item()

        # Map numerical label back to the original tag
        predicted_tag = reverse_label_mapping[predicted_label]

        return jsonify({'tag': predicted_tag})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=True, use_reloader=False)
