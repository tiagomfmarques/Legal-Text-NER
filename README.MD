# **_Extração de Entidades NER em Textos Jurídicos_**

## Descrição do Projeto
Este projeto tem como objetivo explorar a extração automática de entidades nomeadas (NER) em textos jurídicos. Para isso, implementamos e treinamos um modelo de NER capaz de identificar entidades como Pessoas, Organizações, Locais e Datas em decisões judiciais e outros documentos legais.

O trabalho permite:
- Identificar automaticamente entidades relevantes em textos jurídicos.
- Treinar modelos NER usando dados anotados no formato JSON.
- Avaliar o desempenho do modelo em termos de precision, recall e F1-score.

**Input:**  
> *Maria Sousa examinou o Decreto-Lei nº 45/2023, referenciando a súmula do Supremo Tribunal de Justiça.*

**Output do Modelo:**

> **Pessoas:** Maria Sousa  
> **Legislação:** Decreto-Lei nº 45/2023  
> **Local:** Supremo Tribunal de Justiça

### Elaborado por:
![Universidade](https://img.shields.io/badge/Universidade%20da%20Beira%20Interior-1E90FF?style=for-the-badge)  
![Curso](https://img.shields.io/badge/Curso-Intelig%C3%AAncia%20Artificial%20e%20Ci%C3%AAncia%20de%20Dados-1E90FF?style=for-the-badge)  
![Disciplina](https://img.shields.io/badge/Disciplina-Processamento%20de%20Linguagem%20Natural-1E90FF?style=for-the-badge)  
![Aluno](https://img.shields.io/badge/Tiago%20Miguel%20Fernandes%20Marques-51653-1E90FF?style=for-the-badge)

### Tecnologia Utilizada:
[![PyCharm](https://img.shields.io/badge/PyCharm-000000?style=for-the-badge&logo=PyCharm&logoColor=white)](https://www.jetbrains.com/pycharm/)
[![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)](https://www.python.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)](https://www.kaggle.com/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-ffffff?style=for-the-badge&logo=Matplotlib&logoColor=black)](https://matplotlib.org)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-000000?style=for-the-badge&logo=python&logoColor=white)](https://huggingface.co/transformers/)
[![BERTimbau](https://img.shields.io/badge/BERTimbau-1E90FF?style=for-the-badge&logo=python&logoColor=white)](https://huggingface.co/neuralmind/bert-base-portuguese-cased)
[![RoBERTa](https://img.shields.io/badge/RoBERTa-FF0000?style=for-the-badge&logo=python&logoColor=white)](https://huggingface.co/roberta-base)
