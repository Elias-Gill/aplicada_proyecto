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

import time

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import (accuracy_score, classification_report,
                             precision_score)

# NOTE: descomentar estas lineas si es la primera vez que se corre el programa
import nltk
nltk.download('vader_lexicon')

# --------------------------------------------------------------------
# Carga de datos y configuración inicial

inicio_tiempo = time.time()

# Cargar conjunto de datos de entrenamiento
conjunto_entrenamiento = pd.read_csv("./dataset/test_data.csv", encoding="ISO-8859-1")

# Extraer texto de los tweets y etiquetas de sentimiento
textos_tweets = conjunto_entrenamiento.Sentence
etiquetas_sentimiento = conjunto_entrenamiento.Sentiment

numero_a_sentimiento = {1: "Positiva", 0: "Negativa"}

print(f"Número total de tweets: {len(textos_tweets)}")

# --------------------------------------------------------------------
# Listas para guardar los resultados requeridos

sentimientos_esperados = []
sentimientos_calculados = []

analizador_sentimiento = SentimentIntensityAnalyzer()
open("output.csv", "w").write(
    "tweet, sentimiento, negativa, neutral, positiva, defuzz, sent_defuzz, t_fuzz, t_defuzz, t_total\n"
)

output_file = open("output.csv", "a")

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
    puntuaciones = analizador_sentimiento.polarity_scores(tweet)

    puntuacion_positiva = puntuaciones["pos"]
    puntuacion_negativa = puntuaciones["neg"]
    puntuacion_neutral = puntuaciones["neu"]

    # Redondeo y ajuste de resultados (NOTE: sin el redondeo el programa crashea)
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

for j in range(len(textos_tweets)):
    # Limpiar el tweet y guardarlo entre los tweets procesados
    tweet_original = conjunto_entrenamiento.Sentence[j]

    # El dataset ya viene con la interpretacion esperada de los datos
    sentimiento = conjunto_entrenamiento.Sentiment[j]
    sentimientos_esperados.append(numero_a_sentimiento[sentimiento])

    t_fuzz = time.time()

    # Generar puntuaciones con el analizador (retorna una tupla con las puntuaciones)
    puntuacion_negativa, puntuacion_neutral, puntuacion_positiva = generar_puntuaciones(
        tweet_original
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
    output_file.write(
        f"{tweet_original}, {numero_a_sentimiento[sentimiento]}, {puntuacion_negativa}, {puntuacion_neutral}, {puntuacion_positiva}, {res_defuzz}, {sent_calculado}, {t_fuzz}, {t_defuzz}, {t_total}\n"
    )

# --------------------------------------------------------------------
# Evaluación de la precision del modelo

# Informe de clasificación detallado
print("\nInforme de clasificación del modelo:")

# NOTE: el parametro zero division es necesario porque nuestro dataset no contiene
# tweets neutros, por tanto, pese a que el modelo predice los tweets nuetros, las metricas
# no podran ser correctamente mostradas.
print(
    classification_report(
        sentimientos_esperados, sentimientos_calculados, zero_division=1
    )
)

# Precisión global
precision_global = accuracy_score(sentimientos_esperados, sentimientos_calculados)
print(f"Precisión global: {round(precision_global * 100, 2)}%")

# Métricas macro
precision_macro = precision_score(
    sentimientos_esperados, sentimientos_calculados, average="macro", zero_division=1
)

print(f"Puntuación de precisión (MACRO): {round(precision_macro * 100, 2)}%")

# --------------------------------------------------------------------
# Tiempo de ejecución

fin_tiempo = time.time()
tiempo_ejecucion = fin_tiempo - inicio_tiempo
print(f"\nTiempo de ejecución: {round(tiempo_ejecucion, 3)} segundos")
print(f"Tiempo promedio por tweet: {tiempo_ejecucion/len(textos_tweets)} segundos")
