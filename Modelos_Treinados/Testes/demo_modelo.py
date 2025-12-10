import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
from html import escape

# Configuração do modelo
MODEL_PATH = "../Modelos/Bert-Base_89"

label_list = [
    "O",
    "B-JURISPRUDENCIA", "I-JURISPRUDENCIA",
    "B-LEGISLACAO", "I-LEGISLACAO",
    "B-LOCAL", "I-LOCAL",
    "B-ORGANIZACAO", "I-ORGANIZACAO",
    "B-PESSOA", "I-PESSOA",
    "B-TEMPO", "I-TEMPO"
]

# Carregamento do modelo e tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)

id2label = {i: label for i, label in enumerate(label_list)}
label2id = {v: k for k, v in id2label.items()}
model.config.id2label = id2label
model.config.label2id = label2id

device = 0 if torch.backends.mps.is_available() else -1

ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",
    device=device
)

# Cores para cada entidade
colors = {
    "JURISPRUDENCIA": "#FFD700",
    "LEGISLACAO": "#90EE90",
    "LOCAL": "#ADD8E6",
    "ORGANIZACAO": "#FFB6C1",
    "PESSOA": "#FFA07A",
    "TEMPO": "#D3D3D3"
}


# Interface Streamlit
st.title("Demo NER com Coloração de Entidades")
st.write("Insira um texto e veja as entidades reconhecidas pelo modelo.")

texto = st.text_area("Texto de entrada:", height=200)

if st.button("Executar NER") and texto.strip():
    resultados = ner_pipeline(texto)

    # Texto com cores
    highlighted_text = texto
    offset = 0
    for r in sorted(resultados, key=lambda x: x['start']):
        entity = r["entity_group"]
        word = r["word"]
        color = colors.get(entity, "#FFFFFF")
        start = r["start"] + offset
        end = r["end"] + offset
        span = f'<span style="background-color:{color};">{escape(word)}</span>'
        highlighted_text = highlighted_text[:start] + span + highlighted_text[end:]
        offset += len(span) - len(word)

    st.subheader("Texto com entidades destacadas")
    st.markdown(highlighted_text, unsafe_allow_html=True)

    # Entidades por tópicos
    st.subheader("Entidades identificadas por tópicos")

    categorias = ["PESSOA", "TEMPO", "JURISPRUDENCIA", "LEGISLACAO", "LOCAL", "ORGANIZACAO"]
    for cat in categorias:
        entidades = [r["word"] for r in resultados if r["entity_group"] == cat]
        st.markdown(f"**{cat.capitalize()}:**")
        if entidades:
            for ent in entidades:
                st.markdown(f"  * {ent}")
        else:
            st.markdown("  * Nenhuma")
