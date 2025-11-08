import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# print(f"TensorFlow version: {tf.__version__}")
# print(f"Keras version: {keras.__version__}")

fashion_mnist = keras.datasets.fashion_mnist
# Carregando os dados do Fashion MNIST, separados em treino/validação e teste
(train_and_valid_images, train_and_valid_labels), (test_images, test_labels) = (
    fashion_mnist.load_data()
)

largura_imagem = train_and_valid_images.shape[1]
altura_imagem = train_and_valid_images.shape[2]

print(
    f"Dados de treino e validação: {train_and_valid_images.shape},{train_and_valid_images.dtype}, Labels de treino e validação: {train_and_valid_labels.shape},{train_and_valid_labels.dtype}"
)

print(
    f"Dados de teste: {test_images.shape},{test_images.dtype}, Labels de teste: {test_labels.shape},{test_labels.dtype}"
)


print(f"Representação da imagem 1 (segunda imagem) antes da normalização")
# Imprimir a imagem train_and_valid_images[0][0-largura][0-altura] com cada valor representando o número do nível de cinza de cada pixel
for i in range(largura_imagem):
    for j in range(altura_imagem):
        print(f"{train_and_valid_images[1][i][j]:3}", end=" ")
    print()

# Normalizando os dados entre 0 e 1
train_and_valid_images = train_and_valid_images / 255.0
test_images = test_images / 255.0


# Dividir os dados de train_and_valid_images e train_and_valid_labels deixando 5000 amostras para X_valid_images e y_valid_labels e o restante para X_train_images e y_train_labels
X_valid_images = train_and_valid_images[:5000]
y_valid_labels = train_and_valid_labels[:5000]
X_train_images = train_and_valid_images[5000:]
y_train_labels = train_and_valid_labels[5000:]


print(
    f"Shape dos dados de treino: {X_train_images.shape}, Shape dos labels de treino: {y_train_labels.shape}"
)


print(
    f"Shape dos labels de validação: {X_valid_images.shape}, Shape dos labels de validação: {y_valid_labels.shape}"
)


# Nomes das classes do Fashion MNIST
class_names = [
    "Camiseta/top",
    "Calça",
    "Suéter",
    "Vestido",
    "Casaco",
    "Sandália",
    "Camisa",
    "Tênis",
    "Bolsa",
    "Bota",
]

# Imprimindo as 10 primeiras imagens e labels de treino
import matplotlib.pyplot as plt


def plot_images(
    images, labels, class_names, img_size=(4, 4), title_fontsize=5, num_images=10
):
    """Exibe uma grade de imagens com seus respectivos rótulos.

    Args:
        images (np.ndarray): Um array de imagens a serem exibidas.
        labels (np.ndarray): Um array com os rótulos correspondentes às imagens.
        class_names (list): Uma lista de strings com os nomes das classes para
                            mapear os rótulos numéricos.
        img_size (tuple, optional): O tamanho da figura (largura, altura) em polegadas.
                                    Defaults to (4, 4).
        title_fontsize (int, optional): O tamanho da fonte para o título de cada imagem.
                                        Defaults to 5.
        num_images (int, optional): O número de imagens a serem exibidas. Defaults to 10.
    """
    plt.figure(figsize=img_size)
    for i in range(num_images):
        # Criando um subplot para cada imagem (calcula automaticamente o número de linhas necessárias)
        plt.subplot(math.ceil(num_images / 5), 5, i + 1)
        # Imprimindo a imagem em escala de cinza
        plt.imshow(images[i], cmap="gray")
        # Imprimindo o título com o nome da classe correspondente utilizando fonte menor
        plt.title(class_names[labels[i]], fontsize=title_fontsize)
        plt.axis("off")

    plt.show()


# Imprimir as 10 primeiras imagens de treino
# plot_images(train_and_valid_images, train_and_valid_labels, class_names, num_images=10)

print(len(class_names))

model = keras.models.Sequential(
    [  # neurônios de entrada achatados em um vetor unidimencional com  (largura_imagem * altura_imagem) elementos
        keras.layers.Flatten(input_shape=(largura_imagem, altura_imagem)),
        keras.layers.Dense(300, activation="relu"),
        keras.layers.Dense(100, activation="relu"),
        # número neurônios de saída igual ao número de classes com ativação softmax
        keras.layers.Dense(len(class_names), activation="softmax"),
    ]
)

model.summary()
# importante:
# O número de parâmetros treináveis (trainable params) é dado por:
# Para cada camada densa: (n_entradas + 1) * n_saídas
# onde o +1 é para o bias
"""Exemplo:
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ flatten (Flatten)                    │ (None, 784)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 300)                 │         235,500 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 100)                 │          30,100 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (Dense)                      │ (None, 10)                  │           1,010 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 266,610 (1.02 MB)
 Trainable params: 266,610 (1.02 MB)
 Non-trainable params: 0 (0.00 B)
 """

# Explorando as funções de model:
print(f"Tipo do modelo: {type(model)}")
print(f"Camadas: {model.layers}")
print(f"Camada 3: {model.layers[2]}")
hidden1 = model.layers[1]
print(f"Primeira Camada densa: {hidden1.name}")
print(f"hidden1 tem o nome dense? {model.get_layer("dense") is hidden1}")

pesos, biases = hidden1.get_weights()
print(
    f"Numero de pesos da primeira camada densa: {pesos.shape}, Número de Biases da primeira camada densa: {biases.shape}"
)
print(
    f"Pesos da primeira camada densa: {pesos}, Biases da primeira camada densa: {biases}"
)

# Compilando o modelo (definindo a função de perda, otimizador e métricas)
model.compile(
    loss="sparse_categorical_crossentropy",  # porque os labels são inteiros, explicação no próximo comentário
    optimizer="sgd",  # Stochastic Gradient Descent
    metrics=[
        "accuracy"
    ],  # métricas para avaliar o desempenho do modelo durante o treinamento e validação
)

# Nota: As métricas como F1, Precision e Recall no Keras esperam labels no formato one-hot.
# Para usá-las com labels inteiros (sparse), seria necessário criar métricas customizadas
# ou usar soluções mais complexas. Para este estudo, a acurácia esparsa é suficiente e corrige o erro.

"""
Quando você usa loss="sparse_categorical_crossentropy", você está dizendo ao Keras:

"Ei, Keras, para cada imagem, eu vou te dar a resposta correta como um único número inteiro. A saída do seu modelo será um vetor de 10 probabilidades 
(uma para cada classe). Para calcular o erro (loss), pegue o número que eu te dei como resposta e use-o como o índice para encontrar a probabilidade 
correspondente no seu vetor de saída."

y_train_labels -> [4, 0, 7, 9, 9, ...] 
o 4 significa que a primeira imagem é um casaco, 
o 0 significa que a segunda imagem é uma camiseta/top, 
o 7 significa que a terceira imagem é um tênis, e assim por diante.

Exemplo:

Entrada: Uma imagem de um tênis.
Label Correto (seu y_train_labels): 7 (Tênis)
Saída do Modelo (última camada Dense): Um vetor de 10 probabilidades, por exemplo: [0.01, 0.02, 0.01, 0.01, 0.05, 0.05, 0.05, **0.8**, 0.0, 0.0]
Cálculo do Loss sparse: A função de perda olha para o label 7, vai até o índice 7 da saída do modelo (que tem o valor 0.8) 
e calcula o erro com base nessa probabilidade. O objetivo é fazer esse valor se aproximar de 1.0.

A Alternativa: Labels como Vetores (One-Hot Encoding)
E se usássemos a outra função, a loss="categorical_crossentropy"?
Para usá-la, você seria obrigado a transformar seus labels inteiros em vetores, um processo chamado One-Hot Encoding.
O label 7 (Tênis), em um universo de 10 classes, se tornaria o vetor: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0] 
(Um vetor com 0 em todas as posições, exceto na posição de índice 7, que tem 1).

Nesse caso, você diria ao Keras:
"Ei, Keras, para cada imagem, eu vou te dar a resposta como um vetor. A saída do seu modelo também será um vetor. 
Para calcular o erro, compare os dois vetores elemento por elemento."

Exemplo:

Entrada: Uma imagem de um tênis.
Label Correto (One-Hot Encoded): [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
Saída do Modelo: [0.01, 0.02, 0.01, 0.01, 0.05, 0.05, 0.05, 0.8, 0.0, 0.0]
Cálculo do Loss categorical: A função de perda compara os dois vetores e calcula o quão "distantes" eles estão.
Resumo da Comparação

sparse_categorical_crossentropy é a melhor escolha porque seus dados já vieram com labels inteiros. 
É a abordagem mais direta e eficiente para este caso, pois evita o passo extra de converter todos os seus labels para o formato one-hot.

"""

# Treinando o modelo
history = model.fit(
    X_train_images,
    y_train_labels,
    epochs=29,
    batch_size=32,  # No nosso caso, gerará 55.000 / 32 +-= 1718 batches por época
    validation_data=(X_valid_images, y_valid_labels),
)

# Plotando o gráfico do treino com as métricas
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)  # Definindo o limite do eixo y entre 0 e 1
plt.show()


print("Avaliação do modelo com os dados de teste:")
model.evaluate(test_images, test_labels)


# Fazendo previsões com os dados de teste com 10 amostras e arredondando para 2 casas decimais a probabilidade de cada item pertencer a cada classe
numero_de_amostras = 15
imagens_to_predict = test_images[:numero_de_amostras]
classes_reais = test_labels[:numero_de_amostras]

predictions = (model.predict(imagens_to_predict[:numero_de_amostras])).round(2)
print(
    f"Predições em probabilidade para as {numero_de_amostras} primeiras amostras de teste:\n{predictions}"
)

# Imprimindo as classes previstas para as primeiras amostras de teste (np.argmax para pegar o índice da maior probabilidade em cada predição)
predicted_classes = np.argmax(predictions, axis=1)

print(
    f"Classes previstas para as {numero_de_amostras} primeiras amostras de teste: {predicted_classes}"
)
print(
    f"Classes reais para as {numero_de_amostras} primeiras amostras de teste: {test_labels[:numero_de_amostras]}"
)


# Imprimindo predições detalhadas para as primeiras amostras de teste
plot_images(
    imagens_to_predict,
    predicted_classes,
    class_names,
    num_images=numero_de_amostras,
)
