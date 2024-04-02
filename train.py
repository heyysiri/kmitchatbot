import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel


# Load intents from the JSON file
with open('my-react-app/src/intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

# Extract sentences and labels from intents
sentences = []
labels = []
for intent in intents['intents']:
    tag = intent['tag']
    for pattern in intent['patterns']:
        sentences.append(pattern)
        labels.append(tag)

# Initialize DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Tokenize and encode the sentences
encoded_data = tokenizer(sentences, truncation=True, padding=True, return_tensors='pt', max_length=512)

# Convert labels to numerical values
label_mapping = {label: i for i, label in enumerate(set(labels))}
# print("label_mapping", label_mapping)
numerical_labels = [label_mapping[label] for label in labels]

# Create a custom dataset
class ChatDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create dataset and dataloader
dataset = ChatDataset(encoded_data, numerical_labels)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Define a simple classifier on top of DistilBERT
class IntentClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(IntentClassifier, self).__init__()
        self.distilbert = model
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        logits = self.fc(pooled_output)
        return logits

# Instantiate the model
input_size = model.config.hidden_size
output_size = len(set(labels))
# print("output_size", output_size)
classifier = IntentClassifier(input_size, output_size)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.00022)

# Train the model (customize this part based on your specific requirements)
num_epochs = 5
for epoch in range(num_epochs):
    for batch in train_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        optimizer.zero_grad()
        outputs = classifier(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the model
torch.save(classifier.state_dict(), 'distilbert_intent_classifier.pth')
print('Training complete. Model saved.')
