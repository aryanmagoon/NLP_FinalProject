import pandas as pd
import numpy as np
import transformers
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset, ClassLabel
from sklearn.metrics import log_loss
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import DataCollatorWithPadding
import datasets
from transformers import Trainer, TrainingArguments

train_df = pd.read_csv('drive/MyDrive/Quora_Pairs/train.csv')
train_df=train_df.dropna()

tokenizer=transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
model_initial=transformers.AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_initial.to(device)

train_dataset = datasets.Dataset.from_pandas(train_df, preserve_index=False)


def tokenize_texts(texts):
    global tokenizer
    q1rows=texts['question1']
    q2rows=texts['question2']
    return tokenizer(q1rows, q2rows, truncation=True)


tokenized_data = train_dataset.map(tokenize_texts, batched=True)

cast_features = tokenized_data.features.copy()
cast_features['is_duplicate'] = ClassLabel(num_classes=2, names=['not_duplicate', 'duplicate'], names_file=None, id=None)

tokenized_data = tokenized_data.cast(cast_features)

tokenized_data=tokenized_data.remove_columns(['question1','question2', 'id', 'qid1', 'qid2'])
tokenized_data=tokenized_data.rename_column('is_duplicate', 'labels')

for feature_name, feature_type in tokenized_data.features.items():
    print(f"{feature_name}: {feature_type}")

tokenized_data=tokenized_data.train_test_split(test_size=0.2)
print(tokenized_data)

def compute_metrics(preds):
    logits, labels = preds
    labels=preds.label_ids
    logits = torch.tensor(logits)
    probabilities = F.softmax(logits, dim=-1).numpy()
    probabilities = probabilities[:, 1]
    return {"log_loss": log_loss(y_pred=probabilities, y_true=labels, labels=[0,1])}

training_args_t1 = TrainingArguments("./quora-bert_t1", evaluation_strategy="epoch", save_strategy='yes', report_to='none', num_train_epochs=3, per_device_train_batch_size=32, per_device_eval_batch_size=32)


trainer = Trainer(
    model=model_initial,
    args=training_args_t1,
    train_dataset=tokenized_data['train'],
    eval_dataset=tokenized_data['test'],
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    tokenizer=tokenizer,
)

trainer.train()

tokenizer_tuned_t2=transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
model_tuned_t2=transformers.AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model_tuned_t2.to(device)

tuned_training_args = TrainingArguments("./quora-bert_tuned_t1", evaluation_strategy="epoch", save_strategy='Yes', report_to='none', num_train_epochs=5, per_device_train_batch_size=32, per_device_eval_batch_size=32)

optimizer_t2 = AdamW(model_tuned_t2.parameters(), lr=2e-5)

trainer_tuned_t2 = Trainer(
    model=model_tuned_t2,
    args=tuned_training_args,
    train_dataset=tokenized_data['train'],
    eval_dataset=tokenized_data['test'],
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer_tuned_t2),
    tokenizer=tokenizer_tuned_t2,
    optimizers=(optimizer_t2, None),
)

trainer_tuned_t2.train()

tokenizer_tuned_t3=transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
model_tuned_t3=transformers.AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model_tuned_t3.to(device)

tuned_training_args_t3 = TrainingArguments("./quora-bert_tuned", evaluation_strategy="epoch", save_strategy='no', report_to='none', num_train_epochs=3, per_device_train_batch_size=16, per_device_eval_batch_size=16)

optimizer = AdamW(model_tuned_t3.parameters(), lr=2e-5)

trainer_tuned_t3 = Trainer(
    model=model_tuned_t3,
    args=tuned_training_args_t3,
    train_dataset=tokenized_data['train'],
    eval_dataset=tokenized_data['test'],
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer_tuned_t3),
    tokenizer=tokenizer_tuned_t3,
    optimizers=(optimizer, None),
)

trainer_tuned_t3.train()

tokenizer_tuned_t4=transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
model_tuned_t4=transformers.AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model_tuned_t4.to(device)

tuned_training_args_t4 = TrainingArguments("./quora-bert_tuned", evaluation_strategy="epoch", save_strategy='no', report_to='none', num_train_epochs=3, per_device_train_batch_size=16, per_device_eval_batch_size=16)

optimizer_t4 = AdamW(model_tuned_t4.parameters(), lr=1e-6, weight_decay=.01)

trainer_tuned_t4 = Trainer(
    model=model_tuned_t4,
    args=tuned_training_args,
    train_dataset=tokenized_data['train'],
    eval_dataset=tokenized_data['test'],
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer_tuned_t4),
    tokenizer=tokenizer_tuned_t4,
    optimizers=(optimizer_t4, None),
)

trainer_tuned_t4.train()

tokenizer_tuned_t5=transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
model_tuned_t5=transformers.AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model_tuned_t5.to(device)

tuned_training_args_t5 = TrainingArguments("./quora-bert_tuned", evaluation_strategy="epoch", save_strategy='no', report_to='none', num_train_epochs=3, per_device_train_batch_size=16, per_device_eval_batch_size=16)

optimizer_t5 = AdamW(model_tuned_t5.parameters(), lr=2e-6, weight_decay=.01)

trainer_tuned_t5 = Trainer(
    model=model_tuned_t5,
    args=tuned_training_args,
    train_dataset=tokenized_data['train'],
    eval_dataset=tokenized_data['test'],
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer_tuned_t5),
    tokenizer=tokenizer_tuned_t5,
    optimizers=(optimizer_t5, None),
)

trainer_tuned_t5.train()

tokenizer_tuned_t6=transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
model_tuned_t6=transformers.AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model_tuned_t6.to(device)

tuned_training_args = TrainingArguments("./quora-bert_tuned", evaluation_strategy="epoch", save_strategy='no', report_to='none', num_train_epochs=3, per_device_train_batch_size=16, per_device_eval_batch_size=16)

optimizer_t6 = AdamW(model_tuned_t6.parameters(), lr=5e-6, weight_decay=.001)

trainer_tuned_t6 = Trainer(
    model=model_tuned_t6,
    args=tuned_training_args,
    train_dataset=tokenized_data['train'],
    eval_dataset=tokenized_data['test'],
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer_tuned_t6),
    tokenizer=tokenizer_tuned_t6,
    optimizers=(optimizer_t6, None),
)

trainer_tuned_t6.train()