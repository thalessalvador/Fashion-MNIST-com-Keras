from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt


import tensorflow as tf
from tensorflow import keras

housing = fetch_california_housing()
linha, coluna = housing.data.shape

print("Características (atributos):")
print(housing.feature_names)
print("\nDados (10 primeiras linhas):")

for i in range(10):
    for j in range(coluna):
        print(f"{housing.data[i][j]:3}", end=" ")
    print()

# separando os dados de teste
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target
)

# Em train_full estão os dados completos de treino, que serão divididos em treino e validação

# separando os dados de treino e validação
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

# normalizando os dados entre 0 e 1
scaler = StandardScaler()
# Nos dados de treino se usa o fit_transform para calcular a média e o desvio padrão e aplicar a normalização
X_train = scaler.fit_transform(X_train)
# Nos dados de validação e teste se usa apenas o transform para aplicar a normalização com os parâmetros do treino.
# Se aplicarmos o fit_transform aqui, estaremos "vazando" informações do conjunto de validação e teste para o conjunto de treino.
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# criando o modelo
print(X_train.shape[1:])
model = keras.models.Sequential(
    [
        # Primeira camada oculta com 30 neurônios e função de ativação ReLU, na entrada, o shape é igual ao número de atributos
        keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
        keras.layers.Dense(1),  # camada de saída com 1 neurônio (regressão)
    ]
)

# compilando o modelo
model.compile(
    loss="mean_squared_error",
    optimizer="sgd",
    metrics=["mae"],  # Usar apenas métricas de regressão
)


"""
# tentando um otimizador diferente (não foi bom)
model.compile(
    loss="mean_squared_error",
    optimizer="adam",
    metrics=["mse", "mae"],  # Usar apenas métricas de regressão
)
"""

# treinando o modelo
history = model.fit(
    X_train,
    y_train,
    epochs=175,
    validation_data=(X_valid, y_valid),
)

# Plotando o gráfico do treino com as métricas
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
# A linha abaixo foi removida pois a perda (MSE) pode ser maior que 1
# plt.gca().set_ylim(0, 1)
plt.show()

# avaliando o modelo no conjunto de teste
mse_mae_test = model.evaluate(X_test, y_test)
print("MSE/MAE no conjunto de teste:", mse_mae_test)

# fazendo previsões
X_new = X_test[
    :3
]  # novos dados de teste (aqui estamos usando os 3 primeiros dados do conjunto de teste)
y_pred = model.predict(X_new)
print("Previsões para os novos dados:", y_pred)
print("Valores reais dos novos dados:", y_test[:3])
