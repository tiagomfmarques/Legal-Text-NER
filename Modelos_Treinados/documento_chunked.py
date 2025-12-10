import json
import logging
import os
from typing import List, Optional

# Configura o logging
class JSONSegmentChunker:

    # Inicializa com parâmetros de chunking e pastas de entrada/saída
    def __init__(self,
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 input_folder: Optional[str] = None,
                 output_folder: Optional[str] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.input_folder = input_folder
        self.output_folder = output_folder

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    # Divide tokens em chunks com overlap
    def chunk_segments(self, tokens: List[str]) -> List[List[str]]:
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunks.append(tokens[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks

    # Ajusta entidades para o chunk atual
    def adjust_entities_for_chunk(self, entities: List[dict], start_idx: int, end_idx: int) -> List[dict]:

        adjusted = []
        for ent in entities:
            ent_start_token = ent.get("begin", 0)
            ent_end_token = ent.get("end", 0)
            if ent_start_token >= start_idx and ent_end_token < end_idx:
                new_ent = ent.copy()
                new_ent["begin"] = ent_start_token - start_idx
                new_ent["end"] = ent_end_token - start_idx
                adjusted.append(new_ent)
        return adjusted

    # Processa todos os arquivos na pasta de entrada
    def process_folder(self):
        if not self.input_folder or not self.output_folder:
            raise ValueError("input_folder and output_folder must be provided")

        for file_name in os.listdir(self.input_folder):
            if not file_name.endswith(".json"):
                continue

            input_path = os.path.join(self.input_folder, file_name)
            output_path = os.path.join(self.output_folder, file_name.replace(".json", ".jsonl"))

            with open(input_path, "r", encoding="utf-8") as fin, \
                 open(output_path, "w", encoding="utf-8") as fout:

                try:
                    data = json.load(fin)
                except Exception as e:
                    logging.warning(f"Erro ao carregar {input_path}: {e}")
                    continue

                tokens = data.get("tokens", [])
                tags = data.get("tags", [])
                entities = data.get("entities", [])
                linhas_escritas = 0

                if len(tokens) <= self.chunk_size:
                    chunks = [tokens]
                    chunk_ranges = [(0, len(tokens))]
                else:
                    chunks = self.chunk_segments(tokens)
                    # calcula intervalos de cada chunk para ajustar entidades
                    chunk_ranges = []
                    start = 0
                    for chunk in chunks:
                        end = start + len(chunk)
                        chunk_ranges.append((start, end))
                        start += len(chunk) - self.chunk_overlap

                for chunk_idx, (chunk, (start_idx, end_idx)) in enumerate(zip(chunks, chunk_ranges)):
                    new_data = {
                        "tokens": chunk,
                        "tags": tags[start_idx:end_idx],
                        "text": " ".join(chunk),
                        "entities": self.adjust_entities_for_chunk(entities, start_idx, end_idx),
                        "entity_registry": {},  # opcional: pode reconstruir se necessário
                        "id": f"{file_name.replace('.json','')}_{chunk_idx}",
                        "chunk": chunk_idx
                    }
                    fout.write(json.dumps(new_data, ensure_ascii=False) + "\n")
                    linhas_escritas += 1

                print(f"{file_name}: Total chunks escritos: {linhas_escritas}")
                print(f"Arquivo salvo em: {output_path}\n")

if __name__ == "__main__":
    input_folder = "../Textos_Juridicos/texto_json"
    output_folder = "../Textos_Juridicos/texto_json_chunked"

    chunker = JSONSegmentChunker(
        chunk_size=512,
        chunk_overlap=50,
        input_folder=input_folder,
        output_folder=output_folder
    )
    chunker.process_folder()
