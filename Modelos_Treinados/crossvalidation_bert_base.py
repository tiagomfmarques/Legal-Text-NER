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
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt

# Configurações
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
MODEL_DIR = "Modelos/Bert-Base"
RESULTS_DIR = "Resultados/Bert-Base"
MAX_LENGTH = 512
BATCH_SIZE = 4
EPOCHS = 5
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

# Dataset
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

data_collator = DataCollatorForTokenClassification(tokenizer, padding=True)

metric = evaluate.load("seqeval")


# Métricas personalizadas
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Matriz de confusão
    def unify(label):
        if label == "O":
            return "O"
        return label.replace("B-", "").replace("I-", "")

    true_labels_cm = []
    pred_labels_cm = []

    for prediction, label in zip(predictions, labels):
        for p_id, l_id in zip(prediction, label):
            if l_id == -100:
                continue
            true_labels_cm.append(unify(id_to_label[l_id]))
            pred_labels_cm.append(unify(id_to_label[p_id]))

    labels_cm = sorted(list(set(true_labels_cm + pred_labels_cm)))
    if "O" in labels_cm:
        labels_cm.remove("O")
        labels_cm = ["O"] + labels_cm

    # Matriz de confusão
    cm = confusion_matrix(true_labels_cm, pred_labels_cm, labels=labels_cm)
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="red")
    ax.set_xticks(range(len(labels_cm)))
    ax.set_yticks(range(len(labels_cm)))
    ax.set_xticklabels(labels_cm, rotation=45, ha="right")
    ax.set_yticklabels(labels_cm)
    ax.set_xlabel("Previsão")
    ax.set_ylabel("Rótulos Verdadeiros")
    ax.set_title("Matriz de Confusão")
    fig.colorbar(cax)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plt.close()

    # Métricas Seqeval
    true_labels_seqeval = [
        [id_to_label[l_id] for l_id in label if l_id != -100]
        for label in labels
    ]
    pred_labels_seqeval = [
        [id_to_label[p_id] for p_id, l_id in zip(prediction, label) if l_id != -100]
        for prediction, label in zip(predictions, labels)
    ]

    seqeval_results = metric.compute(predictions=pred_labels_seqeval, references=true_labels_seqeval)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels_cm, pred_labels_cm, average="macro", zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        true_labels_cm, pred_labels_cm, average="micro", zero_division=0
    )

    # Relatório por Label
    report_dict = classification_report(
        true_labels_cm, pred_labels_cm, labels=labels_cm, output_dict=True, zero_division=0
    )

    # Salvar JSON
    metrics_json = {
        "overall": {
            "precision": seqeval_results["overall_precision"],
            "recall": seqeval_results["overall_recall"],
            "f1": seqeval_results["overall_f1"]
        },
        "macro_average": {
            "precision": float(precision_macro),
            "recall": float(recall_macro),
            "f1": float(f1_macro)
        },
        "micro_average": {
            "precision": float(precision_micro),
            "recall": float(recall_micro),
            "f1": float(f1_micro)
        },
        "per_label": report_dict
    }

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, ensure_ascii=False, indent=4)

    return {
        "precision": seqeval_results["overall_precision"],
        "recall": seqeval_results["overall_recall"],
        "f1": seqeval_results["overall_f1"]
    }


# Treino
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

print("\nTreino concluído.")
print("Métricas salvas em:", os.path.join(RESULTS_DIR, "metrics.json"))

