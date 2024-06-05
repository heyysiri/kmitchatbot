import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments, TrainerCallback, default_data_collator
from peft import LoraConfig, get_peft_model

print("hi just started")

# Load and preprocess the dataset
class IntentsDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = self.prepare_samples()

    def prepare_samples(self):
        samples = []
        label_to_idx = {intent["tag"]: idx for idx, intent in enumerate(self.data["intents"])}
        for intent in self.data["intents"]:
            tag = intent["tag"]
            label_idx = label_to_idx[tag]
            for pattern in intent["patterns"]:
                inputs = self.tokenizer(
                    pattern,
                    max_length=self.max_len,
                    padding='max_length',
                    truncation=True
                )
                inputs['labels'] = torch.tensor(label_idx, dtype=torch.long)
                samples.append(inputs)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# Custom callback to print epochs and loss
class PrintLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        epoch = state.epoch
        loss = logs.get("loss")
        if loss is not None:
            print(f"Epoch: {epoch:.0f}, Loss: {loss:.4f}")

# Load data
data = load_data("C:/Users/sirik/kmitchatbot/my-react-app/src/intents.json")

# Initialize tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Create dataset and dataloader
dataset = IntentsDataset(data, tokenizer)

# Initialize DistilBERT model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(data["intents"]))

# Setup LoRA configuration and model
lora_config = LoraConfig(
    r=16, 
    lora_alpha=32, 
    target_modules=['q_lin', 'v_lin'],  # Correct target modules for DistilBERT
    lora_dropout=0.1, 
    bias='lora_only'
)
model = get_peft_model(model, lora_config)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results', 
    num_train_epochs=5, 
    per_device_train_batch_size=8, 
    per_device_eval_batch_size=8, 
    warmup_steps=500, 
    weight_decay=0.001, 
    logging_dir='./logs', 
    logging_steps=10,
    evaluation_strategy="steps",
    save_total_limit=1,
    save_steps=500,
    eval_steps=500
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,  # In practice, you should have a separate validation set
    tokenizer=tokenizer,
    data_collator=default_data_collator,
    callbacks=[PrintLossCallback()],  # Add the custom callback here
    compute_metrics=None  # Disable computing and logging additional metrics
)


# Train the model
trainer.train()

# Save the model
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")
