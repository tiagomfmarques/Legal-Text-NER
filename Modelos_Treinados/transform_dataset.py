import json
import os
import glob
from datasets import Dataset, Features, Sequence, Value
from transformers import (AutoTokenizer,AutoModelForTokenClassification,BertTokenizerFast,XLMRobertaTokenizerFast,pipeline)

# Tokenizer

#tokenizer = AutoTokenizer.from_pretrained("PORTULAN/albertina-ptpt")
#tokenizer = BertTokenizerFast.from_pretrained("neuralmind/bert-base-portuguese-cased")
#tokenizer = BertTokenizerFast.from_pretrained("neuralmind/bert-large-portuguese-cased")
tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-large")

nome_tokenizer = "xmlroberta" # "albertina" | "bert-base" | "bert-large" | "xmlroberta"

input_folder = "../Textos_Juridicos/texto_json_chunked"
output_folder = f"../Datasets/datasets_dadospessoais_ner_{nome_tokenizer}"
os.makedirs(output_folder, exist_ok=True)



# Mapas de labels
label2id = {
    "O": 0,
    "B-PERSONAL": 1,
    "I-PERSONAL": 2,
}
id2label = {v: k for k, v in label2id.items()}

# Features do HuggingFace Dataset
features = Features({
    "input_ids": Sequence(Value("int64")),
    "attention_mask": Sequence(Value("int64")),
    "labels": Sequence(Value("int64")),
    "ata_id": Value("string"),
})

# Ler todos os ficheiros JSON
ficheiros_json = glob.glob(os.path.join(input_folder, "*.jsonl"))

for ficheiro in ficheiros_json:
    exemplos = []

    with open(ficheiro, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            tokens = ex["tokens"]
            labels = ex.get("tags", [])

            if not tokens or not labels:
                continue

            # Tokenização
            tokenized = tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=512,
                padding=False,
                add_special_tokens=True
            )

            # Alinhamento de labels
            aligned_labels = []
            previous_word_idx = None
            for word_idx in tokenized.word_ids():
                if word_idx is None:
                    aligned_labels.append(-100)
                elif word_idx != previous_word_idx:
                    aligned_labels.append(label2id.get(labels[word_idx], 0))
                else:
                    aligned_labels.append(label2id["O"])
                previous_word_idx = word_idx

            exemplos.append({
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels": aligned_labels,
                "ata_id": str(ex.get("id", "unk"))
            })

    # Criação e salvamento do dataset
    if exemplos:
        dataset = Dataset.from_list(exemplos, features=features)
        nome_base = os.path.splitext(os.path.basename(ficheiro))[0]
        dataset.save_to_disk(os.path.join(output_folder, nome_base))

# Salvar mappings de labels
with open(os.path.join(output_folder, "label_mappings.json"), "w", encoding="utf-8") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2, ensure_ascii=False)

print("Datasets concluido.")
