from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from train import label_mapping, output_size


app = Flask(__name__)
CORS(app)

# Load DistilBERT model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')
print("after bert import")
# Load your trained classifier
class IntentClassifier(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(IntentClassifier, self).__init__()
        self.distilbert = model
        self.fc = torch.nn.Linear(input_size, output_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        logits = self.fc(pooled_output)
        return logits

# Instantiate the model
input_size = model.config.hidden_size
  # Replace with the actual number of output classes
classifier = IntentClassifier(input_size, output_size)
print("after intent classifier")
classifier.load_state_dict(torch.load('distilbert_intent_classifier.pth'))
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
            predicted_label = torch.argmax(output).item()

        # Map numerical label back to the original tag
        reverse_label_mapping = {i: label for label, i in label_mapping.items()}
        predicted_tag = reverse_label_mapping[predicted_label]

        return jsonify({'tag': predicted_tag})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=True, use_reloader=False)
    print("after app run 1")
