import os
import json
import numpy as np
from datasets import Dataset, DatasetDict
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from sklearn.metrics import confusion_matrix, classification_report

# Configurações
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
MODEL_DIR = "Modelos"
RESULTS_DIR = "Resultados"
MAX_LENGTH = 512
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 5e-5

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Carregar dados
def load_jsonl_folder(folder_path):
    all_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json") or filename.endswith(".jsonl"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                for line in f:
                    all_data.append(json.loads(line))
    return all_data

jsonl_folder_path = "../Textos_Juridicos/texto_json_chunked"
data = load_jsonl_folder(jsonl_folder_path)
print(f"Total de segmentos carregados: {len(data)}")


# Labels
label_list = [
    "O",
    "B-JURISPRUDENCIA", "I-JURISPRUDENCIA",
    "B-LEGISLACAO", "I-LEGISLACAO",
    "B-LOCAL", "I-LOCAL",
    "B-ORGANIZACAO", "I-ORGANIZACAO",
    "B-PESSOA", "I-PESSOA",
    "B-TEMPO", "I-TEMPO"
]

label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for label, i in label_to_id.items()}


for example in data:
    example["labels"] = [label_to_id[label] for label in example["tags"]]

# Criar dataset
dataset = Dataset.from_list(data)

# Split
split = dataset.train_test_split(test_size=0.1, seed=42)
raw_datasets = DatasetDict({
    "train": split["train"],
    "validation": split["test"]
})

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

    all_labels = []
    for i, labels in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(labels[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

tokenized_datasets = raw_datasets.map(tokenize_and_align_labels, batched=True)


# Remover colunas
cols_to_remove = []
for col in tokenized_datasets["train"].column_names:
    if col not in ["input_ids", "attention_mask", "token_type_ids", "labels"]:
        cols_to_remove.append(col)

tokenized_datasets = tokenized_datasets.remove_columns(cols_to_remove)

# Modelo
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_list)
)

# Data collator
data_collator = DataCollatorForTokenClassification(tokenizer, padding=True)

# Métricas
metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [[id_to_label[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [id_to_label[pred] for (pred, lab) in zip(prediction, label) if lab != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)

    flat_preds = [item for sublist in true_predictions for item in sublist]
    flat_labels = [item for sublist in true_labels for item in sublist]
    cm = confusion_matrix(flat_labels, flat_preds, labels=label_list)
    report = classification_report(flat_labels, flat_preds, digits=4)

    np.save(os.path.join(RESULTS_DIR, "confusion_matrix.npy"), cm)
    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"]
    }

training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    remove_unused_columns=False,
    logging_dir=os.path.join(RESULTS_DIR, "logs"),
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model(MODEL_DIR)
print("\nTreino Concluído.")
