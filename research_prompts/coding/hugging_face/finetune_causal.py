import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset


def preprocess_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, max_length=1024, padding='max_length')

def filter_empty_short_texts(example):
    return len(example["text"].strip()) > 0  # You can change 0 to any desired minimum length

def main():
    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Add this line
    config = GPT2Config.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name, config=config)

    # Load dataset
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    dataset = dataset.filter(filter_empty_short_texts)
    dataset = dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)

    train_dataset = dataset['train']
    test_dataset = dataset['test']


    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], return_special_tokens_mask=True)

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Remove unnecessary columns
    tokenized_train_dataset = tokenized_train_dataset.remove_columns(['text', 'special_tokens_mask'])
    tokenized_test_dataset = tokenized_test_dataset.remove_columns(['text', 'special_tokens_mask'])

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        perplexity = torch.exp(torch.tensor(eval_pred.metrics["loss"]))
        return {"perplexity": perplexity}

    # Training arguments
    training_args = TrainingArguments(
        output_dir='output',
        num_train_epochs=200,
        evaluation_strategy='epoch',
        save_strategy='epoch',  # Add this line to match the evaluation_strategy
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        logging_dir='logs',
        logging_steps=10,
        eval_steps=5,
        load_best_model_at_end=True,
        metric_for_best_model='perplexity',
        greater_is_better=False,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Training
    trainer.train()

    # Evaluate the final model
    eval_metrics = trainer.evaluate()
    print(f'Perplexity: {eval_metrics["perplexity"]}')


if __name__ == '__main__':
    main()
