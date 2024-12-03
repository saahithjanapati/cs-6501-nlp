import pandas as pd
import random
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Load the CSV file
data = pd.read_csv("your_file.csv")  # Replace with your file name
assert "text" in data.columns and "label" in data.columns

# Shuffle the dataset
random.seed(42)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Manually split the dataset
train_data = data.iloc[:3500]
val_data = data.iloc[3500:4000]
test_data = data.iloc[4000:]

# Convert to Hugging Face Dataset
def preprocess_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, padding=True)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_dataset = Dataset.from_pandas(train_data).map(lambda x: preprocess_function(x, tokenizer), batched=True)
val_dataset = Dataset.from_pandas(val_data).map(lambda x: preprocess_function(x, tokenizer), batched=True)
test_dataset = Dataset.from_pandas(test_data).map(lambda x: preprocess_function(x, tokenizer), batched=True)

# Define the model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=data['label'].nunique())

# Define evaluation metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = (labels == preds).mean()
    return {"accuracy": acc}

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Fine-tune the model
trainer.train()

# Evaluate on the test set
test_results = trainer.evaluate(test_dataset)
print("Test results:", test_results)