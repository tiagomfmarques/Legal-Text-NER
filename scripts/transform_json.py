import json
import os

# Extrai entidades do formato BIO
def extract_entities_from_bio(tokens, tags):

    entities = []
    entity_tokens = []
    entity_type = None
    entity_start = None
    position = 0

    for i, (token, tag) in enumerate(zip(tokens, tags)):
        if tag.startswith("B-"):
            # Finaliza entidade anterior se existir
            if entity_tokens:
                entity_text = " ".join(entity_tokens)
                entity_end = position - 1  # última posição
                entities.append({
                    "type": entity_type,
                    "begin": entity_start,
                    "end": entity_end,
                    "text": entity_text,
                    "position_value": None,
                    "result_value": None
                })
                entity_tokens = []

            entity_type = tag[2:]
            entity_start = position
            entity_tokens.append(token)
        elif tag.startswith("I-") and entity_tokens:
            entity_tokens.append(token)
        else:  # tag == "O" ou I- sem B- anterior
            if entity_tokens:
                entity_text = " ".join(entity_tokens)
                entity_end = position - 1
                entities.append({
                    "type": entity_type,
                    "begin": entity_start,
                    "end": entity_end,
                    "text": entity_text,
                    "position_value": None,
                    "result_value": None
                })
                entity_tokens = []
            entity_type = None
            entity_start = None

        position += len(token) + 1  # contando espaços entre tokens

    # Captura última entidade se existir
    if entity_tokens:
        entity_text = " ".join(entity_tokens)
        entity_end = position - 1
        entities.append({
            "type": entity_type,
            "begin": entity_start,
            "end": entity_end,
            "text": entity_text,
            "position_value": None,
            "result_value": None
        })

    return entities

# Converte texto CoNLL para JSON com entidades extraídas
def conll_to_json_with_entities(conll_text):
    tokens = []
    tags = []

    for line in conll_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        token, tag = parts[0], parts[-1]
        tokens.append(token)
        tags.append(tag)

    entities = extract_entities_from_bio(tokens, tags)

    entity_registry = {
        "entity_types": {},
        "total_entities": len(entities),
        "entity_details": entities
    }

    for ent in entities:
        ent_type = ent["type"]
        entity_registry["entity_types"][ent_type] = entity_registry["entity_types"].get(ent_type, 0) + 1

    return {
        "tokens": tokens,
        "tags": tags,
        "entities": entities,
        "entity_registry": entity_registry
    }

# Processa todos os arquivos .conll em uma pasta e salva como .json
def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".conll"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename.replace(".conll", ".json"))

            with open(input_path, "r", encoding="utf-8") as f:
                conll_text = f.read()

            data = conll_to_json_with_entities(conll_text)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            print(f"{filename} convertido para {output_path}")


if __name__ == "__main__":
    input_folder = "../Textos_Juridicos/texto_conll"
    output_folder = "../Textos_Juridicos/texto_json"

    process_folder(input_folder, output_folder)
