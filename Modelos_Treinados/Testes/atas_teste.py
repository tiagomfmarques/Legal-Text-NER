from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch

# Caminho do modelo
MODEL_PATH = "../Modelos/Bert-Base "

# Lista de labels
label_list = [
    "O",
    "B-JURISPRUDENCIA", "I-JURISPRUDENCIA",
    "B-LEGISLACAO", "I-LEGISLACAO",
    "B-LOCAL", "I-LOCAL",
    "B-ORGANIZACAO", "I-ORGANIZACAO",
    "B-PESSOA", "I-PESSOA",
    "B-TEMPO", "I-TEMPO"
]

# Carrega tokenizer e modelo
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)

# Corrige o mapeamento de labels
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {v: k for k, v in id2label.items()}
model.config.id2label = id2label
model.config.label2id = label2id

device = 0 if torch.backends.mps.is_available() else -1

# Cria pipeline de NER
ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",
    device=device
)

# Texto de exemplo
texto = """O presidente da Câmara Municipal de Lisboa, João Silva, analisou a Lei nº 12/2020
durante a sessão do dia 10 de março de 2023, destacando jurisprudência relevante do Tribunal Constitucional, ano."""

# Executa NER
resultados = ner_pipeline(texto)

# Organiza resultados por categoria
categorias = ["JURISPRUDENCIA","LEGISLACAO","LOCAL","ORGANIZACAO","PESSOA","TEMPO"]
print("Estas são as tags:\n")

for cat in categorias:
    entidades = [r["word"] for r in resultados if cat in r["entity_group"]]
    print(f"{cat}:")
    if entidades:
        print(", ".join(entidades))
    else:
        print("Nenhuma entidade identificada")
    print()
