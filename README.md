# Estudo de Caso: Classificação de Avaliações

Projeto desenvolvido para a disciplina de Processamento de Linguagem Natural (PLN).

O objetivo do projeto é implementar uma **Rede Neural Perceptron de camada única** do zero em Python (sem o uso de bibliotecas de Machine Learning prontas) para classificar avaliações de clientes da loja de quadrinhos **HQMania** como **Positivas (1)** ou **Negativas (0)**.

## 🛠️ Etapas do Projeto

1. **Criação da Base de Dados**: Base fictícia contendo 20 avaliações de quadrinhos classificadas entre positivas e negativas.
2. **Pré-processamento**:
   - Conversão do texto para minúsculas;
   - Remoção de pontuações;
   - Tokenização (separação em palavras);
   - Remoção de *stopwords* em português.
3. **Vetorização (Bag of Words)**: Implementação manual da representação numérica BoW para transformar os textos em vetores binários.
4. **Implementação do Perceptron**: Rede Neural com pesos aleatórios, bias, taxa de aprendizagem de 0.1 e função de ativação do tipo degrau ($f(\text{soma}) = 1$ se $\text{soma} > 0.5$, senão $0$).
5. **Treinamento e Testes**: Avaliação do modelo na base de dados original e teste de generalização com novas frases.

## 💻 Como Executar

Certifique-se de ter o Python 3 instalado no seu ambiente.

1. Clone o repositório:
   ```bash
   git clone https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
   ```

2. Acesse a pasta do projeto:
   ```bash
   cd SEU_REPOSITORIO
   ```

3. Execute o script principal:
   ```bash
   python main.py
   ```

## 📊 Resultados do Modelo

* **Acurácia na base de treinamento**: 100.00% (convergência na época 5).
* **Resultados com novas avaliações**: O modelo obteve boa taxa de acerto em frases genéricas, porém apresentou limitações pontuais devido à ausência de análise de contexto e ao vocabulário limitado da base de treino.