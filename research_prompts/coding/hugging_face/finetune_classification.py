import random
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = np.mean(predictions == labels)
    return {"accuracy": acc}

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

datasets = load_dataset("imdb")
train_dataset, eval_dataset = datasets["train"], datasets["test"]

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="output",
    overwrite_output_dir=True,
    num_train_epochs=200,
    evaluation_strategy="epoch",
    logging_steps=5,
    eval_steps=5,
    save_strategy="epoch",
    save_total_limit=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    report_to="none",
    early_stopping_patience=5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
