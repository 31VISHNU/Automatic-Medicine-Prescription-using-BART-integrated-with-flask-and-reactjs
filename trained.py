import transformers
from datasets import load_dataset, load_metric, load_from_disk
import numpy as np
import nltk
nltk.download('punkt')
import pandas as pd
metric = load_metric('rouge')
df = pd.read_excel('C:\\Users\\vishn\\finalyear\\project\\project\\src\\Datasets\\ready_dataset.xlsx')
data = df.to_dict(orient='records')
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

import torch
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
model_name = "facebook/bart-large-cnn"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)
collator = transformers.DataCollatorForSeq2Seq(tokenizer, model=model)
def compute_rouge(pred):
  predictions, labels = pred
  decode_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)

  decode_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

  res = metric.compute(predictions=decode_predictions, references=decode_labels, use_stemmer=True)

  res = {key: value.mid.fmeasure * 100 for key, value in res.items()}

  pred_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
  res['gen_len'] = np.mean(pred_lens)

  return {k: round(v, 4) for k, v in res.items()}

training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=100,
    do_train=True,
)


from datasets import Dataset

inputs = tokenizer([item['Description'] for item in train_data], max_length=1024, padding="max_length", truncation=True)
labels = tokenizer([item['Medicine'] for item in train_data], max_length=1024, padding="max_length", truncation=True)


test_inputs = tokenizer([item['Description'] for item in test_data], max_length=1024, padding="max_length", truncation=True)
test_labels = tokenizer([item['Medicine'] for item in test_data], max_length=1024, padding="max_length", truncation=True)

train_dataset = Dataset.from_dict({
    "input_ids": torch.tensor(inputs['input_ids']),
    "attention_mask": torch.tensor(inputs['attention_mask']),
    "labels": torch.tensor(labels['input_ids'])
})
test_dataset = Dataset.from_dict({
    "input_ids": torch.tensor(test_inputs['input_ids']),
    "attention_mask": torch.tensor(test_inputs['attention_mask']),
    "labels": torch.tensor(test_labels['input_ids'])
})

training_args = Seq2SeqTrainingArguments(
    'conversation-summ',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=10,
    predict_with_generate=True,
    eval_accumulation_steps=1,
    fp16=False
)

trainer = transformers.Seq2SeqTrainer(
    model,
    training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=collator,
    tokenizer=tokenizer,
    compute_metrics=compute_rouge
)
trainer.train()
save_directory = "C:/Users/vishn/Downloads/fine_tuned_bart_best_10epoch-20240208T140224Z-001/fine_tuned_bart_best_10epoch"

trainer.save_model(save_directory)

tokenizer.save_pretrained(save_directory)

model.save_pretrained("C:/Users/vishn/Downloads/fine_tuned_bart_best_10epoch-20240208T140224Z-001/fine_tuned_bart_best_10epoch")
tokenizer.save_pretrained("C:/Users/vishn/Downloads/fine_tuned_bart_best_10epoch-20240208T140224Z-001/fine_tuned_bart_best_10epoch")
