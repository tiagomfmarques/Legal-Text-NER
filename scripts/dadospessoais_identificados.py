import os
import json

# Caminho da pasta onde estão os arquivos JSON
PASTA = "../Textos_Juridicos/texto_json"
ARQUIVO_SAIDA = "dados_identificados.txt"

# Dicionários para guardar dados
entidades_por_tipo = {}
labels_contagem = {}

# Percorrer todos os arquivos .json na pasta
for nome_arquivo in os.listdir(PASTA):
    if nome_arquivo.endswith(".json"):
        caminho_arquivo = os.path.join(PASTA, nome_arquivo)

        with open(caminho_arquivo, "r", encoding="utf-8") as f:
            try:
                dados = json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ Erro ao ler {nome_arquivo} (não é JSON válido)")
                continue

        # Contar tags labels
        tags = dados.get("tags", [])
        for tag in tags:
            tipo_base = tag.replace("B-", "").replace("I-", "")
            labels_contagem[tipo_base] = labels_contagem.get(tipo_base, 0) + 1

        # Agrupar entidades por tipo
        entidades = dados.get("entities", [])
        for ent in entidades:
            tipo = ent.get("type")
            texto = ent.get("text")
            if tipo and texto:
                entidades_por_tipo.setdefault(tipo, set()).add(texto.strip())

# Ordenar resultados
labels_ordenadas = dict(sorted(labels_contagem.items()))
entidades_ordenadas = dict(sorted(entidades_por_tipo.items()))


with open(ARQUIVO_SAIDA, "w", encoding="utf-8") as saida:
    print("===== RESUMO DAS ENTIDADES =====")
    saida.write("===== RESUMO DAS ENTIDADES =====\n")
    for tipo, qtd in labels_ordenadas.items():
        total_entidades = len(entidades_ordenadas.get(tipo, []))
        if total_entidades == 0:
            continue  # ignora tipos sem entidades
        linha = f"{tipo} ({total_entidades} Entidades)"
        print(linha)
        saida.write(linha + "\n")

    print("\n===== ENTIDADES POR TIPO =====")
    saida.write("\n\n===== ENTIDADES POR TIPO =====\n")
    for tipo, lista in entidades_ordenadas.items():
        if not lista:
            continue
        entidades_lista = sorted(lista)
        header = f"\n{tipo} ({len(entidades_lista)} Entidades)"
        print(header)
        saida.write(header + "\n")
        for ent in entidades_lista:
            print(f"{ent}")
            saida.write(f"{ent}\n")

print(f"\nResumo e entidades salvos em '{ARQUIVO_SAIDA}'")
