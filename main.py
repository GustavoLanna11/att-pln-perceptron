import random
import string

# 1. CRIAÇÃO DA BASE DE DADOS (20 avaliações fictícias)
dados = [
    {"id": 1, "texto": "A história é incrível e os desenhos são excelentes", "classe": 1},
    {"id": 2, "texto": "A qualidade do quadrinho deixou a desejar e a história é fraca", "classe": 0},
    {"id": 3, "texto": "Personagens cativantes e enredo fantástico", "classe": 1},
    {"id": 4, "texto": "Muito cansativo e chato de ler", "classe": 0},
    {"id": 5, "texto": "Quadrinho divertido e muito bem desenhado", "classe": 1},
    {"id": 6, "texto": "História sem pé nem cabeça, perda de tempo", "classe": 0},
    {"id": 7, "texto": "Arte excelente e roteiro incrível", "classe": 1},
    {"id": 8, "texto": "Fraco e muito sem graça", "classe": 0},
    {"id": 9, "texto": "Adorei cada detalhe da edição, fantástico", "classe": 1},
    {"id": 10, "texto": "Desenhos feios e história cansativa", "classe": 0},
    {"id": 11, "texto": "Uma obra prima dos quadrinhos, super divertido", "classe": 1},
    {"id": 12, "texto": "Não gostei, roteiro muito fraco", "classe": 0},
    {"id": 13, "texto": "Excelente narrativa e personagens incríveis", "classe": 1},
    {"id": 14, "texto": "História previsível e cansativa", "classe": 0},
    {"id": 15, "texto": "Fantástico do começo ao fim", "classe": 1},
    {"id": 16, "texto": "Qualidade ruim e história fraca", "classe": 0},
    {"id": 17, "texto": "Muito divertido e arte incrível", "classe": 1},
    {"id": 18, "texto": "Cansativo e desinteressante", "classe": 0},
    {"id": 19, "texto": "História fantástica e excelente acabamento", "classe": 1},
    {"id": 20, "texto": "Péssimo quadrinho, história fraca", "classe": 0}
]

# Stopwords comuns em português
stopwords = {"a", "o", "as", "os", "um", "uma", "de", "do", "da", "em", "e", "é", "que", "com", "não"}

# 2. PRÉ-PROCESSAMENTO DOS TEXTOS
def preprocessar(texto):
    # 1. Converter para minúsculas e 2. Remover pontuação
    texto_lower = texto.lower()
    sem_pont = texto_lower.translate(str.maketrans('', '', string.punctuation))
    # 3. Separar em palavras
    palavras = sem_pont.split()
    # 4. Remover stopwords
    palavras_filtradas = [p for p in palavras if p not in stopwords]
    return palavras_filtradas

# Construir o vocabulário único com base em todas as avaliações
vocabulario = set()
for item in dados:
    palavras = preprocessar(item["texto"])
    vocabulario.update(palavras)
vocabulario = sorted(list(vocabulario))

# 3. TRANSFORMAÇÃO DOS TEXTOS EM NÚMEROS (Bag of Words Manual)
def texto_para_vetor(texto, vocab):
    palavras_texto = set(preprocessar(texto))
    # 1 se a palavra aparece no texto, 0 caso contrário
    return [1 if palavra in palavras_texto else 0 for palavra in vocab]

X = [texto_para_vetor(item["texto"], vocabulario) for item in dados]
y = [item["classe"] for item in dados]

# 4. IMPLEMENTAÇÃO DO PERCEPTRON
class Perceptron:
    def __init__(self, num_caracteristicas, taxa_aprendizagem=0.1):
        # Pesos iniciais aleatórios e bias
        self.pesos = [random.uniform(-0.5, 0.5) for _ in range(num_caracteristicas)]
        self.bias = random.uniform(-0.5, 0.5)
        self.taxa_aprendizagem = taxa_aprendizagem

    def funcao_ativacao(self, soma):
        # f(soma) = 1 se soma > 0.5, senão 0
        return 1 if soma > 0.5 else 0

    def prever(self, x):
        # Soma ponderada
        soma = sum(w * val for w, val in zip(self.pesos, x)) + self.bias
        return self.funcao_ativacao(soma)

    def treinar(self, X, y, max_epocas=100):
        for epoca in range(max_epocas):
            erros_total = 0
            for xi, yi in zip(X, y):
                soma = sum(w * val for w, val in zip(self.pesos, xi)) + self.bias
                predicao = self.funcao_ativacao(soma)
                
                # 4. Calcular o erro
                erro = yi - predicao
                
                if erro != 0:
                    erros_total += 1
                    # 5. Atualizar os pesos e bias
                    for i in range(len(self.pesos)):
                        self.pesos[i] += self.taxa_aprendizagem * erro * xi[i]
                    self.bias += self.taxa_aprendizagem * erro
            
            # Interrupção antecipada caso não ocorram erros
            if erros_total == 0:
                print(f"Treinamento concluído sem erros na época {epoca + 1}.")
                break

# Treinando o modelo
perceptron = Perceptron(len(vocabulario), taxa_aprendizagem=0.1)
perceptron.treinar(X, y, max_epocas=100)

# 6. TESTE DO MODELO E 7. CÁLCULO DA ACURÁCIA
corretas = 0
total = len(dados)

print("\n--- TESTE DO MODELO NA BASE DE TREINAMENTO ---")
for item, xi, yi in zip(dados, X, y):
    pred = perceptron.prever(xi)
    status = "Correto" if pred == yi else "Incorreto"
    if pred == yi:
        corretas += 1
    print(f"Texto: '{item['texto']}' | Real: {yi} | Previsto: {pred} | {status}")

acuracia = (corretas / total) * 100
print(f"\nTotal de avaliações: {total}")
print(f"Classificações corretas: {corretas}")
print(f"Classificações incorretas: {total - corretas}[cite: 5]")
print(f"Acurácia: {acuracia:.2f}%[cite: 5]")

# 8. TESTE COM NOVAS AVALIAÇÕES
novas_avaliacoes = [
    "História maravilhosa e traços fantásticos",
    "Muito chato, odiei a leitura",
    "Roteiro excelente e arte muito divertida",
    "História fraca e desenhos ruins",
    "Incrível e cativante do início ao fim"
]

print("\n--- TESTE COM NOVAS AVALIAções ---")
for texto in novas_avaliacoes:
    xi = texto_para_vetor(texto, vocabulario)
    pred = perceptron.prever(xi)
    classe_str = "Positiva (1)" if pred == 1 else "Negativa (0)"
    print(f"Novo Texto: '{texto}' -> Previsto: {classe_str}")