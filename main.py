# Implementacion del algoritmo de analisis de sentimiento utilizando el lexicon
# VADER.
#
# Para el lexicon VADER los valores utilizados para el calculo difuso son:
#   max = 1.0
#   min = 0
#   medio = 0.5
#
# El lexicon retorna un par de valores en formato tupla (positivo, negatitvo).
#
# Para mayor facilidad de implementacion se utilizo la libreria nltk, la cual ya
# incorpora el calculo de los valores de pertenencia utilizando dicho lexicon.
# Ademas se encarga de la limpieza de los datos de entrada.
#
# El programa genera un csv final con los siguientes resultados:
#   Oración original, label original, puntaje positivo, puntaje negativo,
#   Puntaje Positivo, Puntaje Negativo, resultado de inferencia, tiempo de
#   ejecución de la inferencia.
#
# b. Fuera del archivo csv, reportar el tiempo de ejecución promedio total.

# --------------------------------------------------------------------
# Importación de bibliotecas necesarias

import re
import time

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# NOTE: se puede comentar esta linea luego de la primera vez que se corre
nltk.download("vader_lexicon")

# --------------------------------------------------------------------
# Carga de datos y configuración inicial

inicio_tiempo = time.time()

# Cargar conjunto de datos de entrenamiento
conjunto_entrenamiento = pd.read_csv(
    "./dataset/semeval-2017-test.csv", sep="\t",encoding="ISO-8859-1"
)

# Extraer texto de los tweets y etiquetas de sentimiento
textos_tweets = conjunto_entrenamiento.text
etiquetas_sentimiento = conjunto_entrenamiento.label

numero_a_sentimiento = {1: "Positiva", -1: "Negativa", 0: "Neutra"}

print(f"Número total de tweets: {len(textos_tweets)}")

# --------------------------------------------------------------------
# Listas para guardar los resultados requeridos

sentimientos_esperados = []
sentimientos_calculados = []

analizador_sentimiento = SentimentIntensityAnalyzer()

# --------------------------------------------------------------------
# Generar variables del universo

# Rangos para positivo y negativo: [0, 1]
# Rango para salida: [0, 10] en puntos porcentuales
x_positivo = np.arange(0, 1, 0.1)
x_negativo = np.arange(0, 1, 0.1)
x_salida = np.arange(0, 10, 1)

# --------------------------------------------------------------------
# Generar las funciones de pertenencia difusa.

# Funciones de pertenencia difusa para positivo
positivo_bajo = fuzz.trimf(x_positivo, [0, 0, 0.5])
positivo_medio = fuzz.trimf(x_positivo, [0, 0.5, 1])
positivo_alto = fuzz.trimf(x_positivo, [0.5, 1, 1])

# Funciones de pertenencia difusa para negativo
negativo_bajo = fuzz.trimf(x_negativo, [0, 0, 0.5])
negativo_medio = fuzz.trimf(x_negativo, [0, 0.5, 1])
negativo_alto = fuzz.trimf(x_negativo, [0.5, 1, 1])

# Funciones de pertenencia difusa para salida (fijo para todos los lexicons)
salida_negativa = fuzz.trimf(x_salida, [0, 0, 5])  # Escala: Neg Neu Pos
salida_neutral = fuzz.trimf(x_salida, [0, 5, 10])
salida_positiva = fuzz.trimf(x_salida, [5, 10, 10])

# --------------------------------------------------------------------
# Preprocesamiento del texto y limpieza de los datos


def generar_puntuaciones(tweet) -> tuple[float, float, float]:
    # limpieza del tweett
    tweet = re.sub(r"won't", "will not", tweet)
    tweet = re.sub(r"can\'t", "can not", tweet)
    tweet = re.sub(r"@", "", tweet)  # removal of @
    tweet = re.sub(r"http\S+", "", tweet)  # removal of URLs
    tweet = re.sub(r"#", "", tweet)  # hashtag processing
    tweet = re.sub(r"n\'t", " not", tweet)
    tweet = re.sub(r"\'re", " are", tweet)
    tweet = re.sub(r"\'s", " is", tweet)
    tweet = re.sub(r"\'d", " would", tweet)
    tweet = re.sub(r"\'ll", " will", tweet)
    tweet = re.sub(r"\'t", " not", tweet)
    tweet = re.sub(r"\'ve", " have", tweet)
    tweet = re.sub(r"\'m", " am", tweet)

    puntuaciones = analizador_sentimiento.polarity_scores(tweet)

    puntuacion_positiva = puntuaciones["pos"]
    puntuacion_negativa = puntuaciones["neg"]
    puntuacion_neutral = puntuaciones["neu"]

    # Redondeo y ajuste de resultados (NOTE: el redondeo mejora la precision bastante)
    if puntuacion_positiva == 1:
        puntuacion_positiva = 0.9
    else:
        puntuacion_positiva = round(puntuacion_positiva, 1)

    if puntuacion_negativa == 1:
        puntuacion_negativa = 0.9
    else:
        puntuacion_negativa = round(puntuacion_negativa, 1)

    return (puntuacion_negativa, puntuacion_neutral, puntuacion_positiva)


# --------------------------------------------------------------------
# Calculo y procesamiento de los sentimientos de los tweets

prev_x = 0
for j in range(len(textos_tweets)):
    # Limpiar el tweet y guardarlo entre los tweets procesados
    tweet_original = conjunto_entrenamiento.text[j]

    # El dataset ya viene con la interpretacion esperada de los datos
    sentimiento = conjunto_entrenamiento.label[j]
    sentimientos_esperados.append(numero_a_sentimiento[sentimiento])

    t_fuzz = time.time()

    # Generar puntuaciones con el analizador (retorna una tupla con las puntuaciones)
    puntuacion_negativa, puntuacion_neutral, puntuacion_positiva = generar_puntuaciones(
        tweet_original
    )

    if j < 40:
        # Definir el ancho de cada barra y el espacio entre grupos
        width = 0.15
        x = prev_x + 1
        prev_x = x
        # Asegurarse de que las barras estén agrupadas sin espacios
        plt.bar(
            x - width,  # Ajustar la posición para la barra neutral
            puntuacion_neutral,
            color="blue",
            label="Neutral" if j == 0 else "",
            alpha=0.7,
            width=width,
        )
        plt.bar(
            x + width,  # Ajustar la posición para la barra negativa
            puntuacion_negativa,
            color="yellow",
            label="Negativo" if j == 0 else "",
            alpha=0.7,
            width=width,
        )
        plt.bar(
            x,  # Ajustar la posición para la barra positiva
            puntuacion_positiva,
            color="red",
            label="Positivo" if j == 0 else "",
            alpha=0.7,
            width=width,
        )

    # ---------------------------------------------------------------------
    # Calculo de niveles de pertenencia.

    # Calcular los niveles de pertenencia positiva (bajo, medio, alto) del tweet
    nivel_positivo_bajo = fuzz.interp_membership(
        x_positivo, positivo_bajo, puntuacion_positiva
    )
    nivel_positivo_medio = fuzz.interp_membership(
        x_positivo, positivo_medio, puntuacion_positiva
    )
    nivel_positivo_alto = fuzz.interp_membership(
        x_positivo, positivo_alto, puntuacion_positiva
    )

    # Calcular los niveles de pertenencia negativa (bajo, medio, alto) del tweet
    nivel_negativo_bajo = fuzz.interp_membership(
        x_negativo, negativo_bajo, puntuacion_negativa
    )
    nivel_negativo_medio = fuzz.interp_membership(
        x_negativo, negativo_medio, puntuacion_negativa
    )
    nivel_negativo_alto = fuzz.interp_membership(
        x_negativo, negativo_alto, puntuacion_negativa
    )

    # ---------------------------------------------------------------------
    # Aplicacion de las reglas de Mamdani utilizando los niveles de pert.

    # El operador OR significa que tomamos el máximo de estas dos.
    regla_activa_1 = np.fmin(nivel_positivo_bajo, nivel_negativo_bajo)
    regla_activa_2 = np.fmin(nivel_positivo_medio, nivel_negativo_bajo)
    regla_activa_3 = np.fmin(nivel_positivo_alto, nivel_negativo_bajo)
    regla_activa_4 = np.fmin(nivel_positivo_bajo, nivel_negativo_medio)
    regla_activa_5 = np.fmin(nivel_positivo_medio, nivel_negativo_medio)
    regla_activa_6 = np.fmin(nivel_positivo_alto, nivel_negativo_medio)
    regla_activa_7 = np.fmin(nivel_positivo_bajo, nivel_negativo_alto)
    regla_activa_8 = np.fmin(nivel_positivo_medio, nivel_negativo_alto)
    regla_activa_9 = np.fmin(nivel_positivo_alto, nivel_negativo_alto)

    # Aplicacion de las reglas de Mamdani
    n1 = np.fmax(regla_activa_4, regla_activa_7)
    n2 = np.fmax(n1, regla_activa_8)
    activacion_salida_bajo = np.fmin(n2, salida_negativa)

    neu1 = np.fmax(regla_activa_1, regla_activa_5)
    neu2 = np.fmax(neu1, regla_activa_9)
    activacion_salida_medio = np.fmin(neu2, salida_neutral)

    p1 = np.fmax(regla_activa_2, regla_activa_3)
    p2 = np.fmax(p1, regla_activa_6)
    activacion_salida_alto = np.fmin(p2, salida_positiva)

    salida_cero = np.zeros_like(x_salida)

    # Agregacion para calcular el sentimiento final.
    agregada = np.fmax(
        activacion_salida_bajo, np.fmax(activacion_salida_medio, activacion_salida_alto)
    )

    t_fuzz = time.time() - t_fuzz

    # Desfuzzificacion
    t_defuzz = time.time()

    salida = fuzz.defuzz(x_salida, agregada, "centroid")
    res_defuzz = round(salida, 2)

    t_defuzz = time.time() - t_defuzz

    sent_calculado = ""
    # Escala : Neg Neu Pos. Escala [0; 10]
    if 0 < (res_defuzz) < 3.33:  # R
        sent_calculado = "Negativa"

    elif 3.34 < (res_defuzz) < 6.66:
        sent_calculado = "Neutra"

    elif 6.67 < (res_defuzz) < 10:
        sent_calculado = "Positiva"

    sentimientos_calculados.append(sent_calculado)

    # ----------------------------------------------
    # impresion datos del tweet
    t_total = t_fuzz + t_defuzz

# --------------------------------------------------------------------
# Tiempo de ejecución

fin_tiempo = time.time()
tiempo_ejecucion = fin_tiempo - inicio_tiempo
print(f"\nTiempo de ejecución: {round(tiempo_ejecucion, 3)} segundos")
print(f"Tiempo promedio por tweet: {tiempo_ejecucion/len(textos_tweets)} segundos")

# --------------------------------------------------------------------
# Evaluación de la precision del modelo

# Informe de clasificación detallado
print("\nInforme de clasificación del modelo:")

# Calcular la precisión
plt.title("Puntuaciones de clasificacion")
plt.xlabel("Tweet")
plt.ylabel("Puntuacion")
plt.legend()
plt.grid(True)
plt.show()
