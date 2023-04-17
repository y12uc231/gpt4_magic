from transformers import (AutoModelForCausalLM, AutoTokenizer, TextDataset,
                          DataCollatorForLanguageModeling, Trainer, TrainingArguments)
import torch
from datasets import load_dataset


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    perplexity = torch.exp(torch.tensor(eval_pred.metrics["loss"]))
    return {"perplexity": perplexity}


model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

train_dataset = TextDataset(tokenizer, file_path="train.txt", block_size=128)
eval_dataset = TextDataset(tokenizer, file_path="eval.txt", block_size=128)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

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
    metric_for_best_model="perplexity",
    greater_is_better=False,
    report_to="none",
    early_stopping_patience=5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
