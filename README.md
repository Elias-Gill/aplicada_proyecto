# Requerimientos

Para correr el proyecto se debe instalar el dataset desde
[aqui](https://www.kaggle.com/datasets/krishbaisoya/tweets-sentiment-analysis).

Luego ponerlo en una carpeta que se llame "dataset" (modificar la linea de codigo que hacer
referencia al dataset de ser necesario).

Luego instale las dependencias necesarias utilizando `pip`:
```bash
pip install -r requirements.txt
```

Ahora ya puede correr el programa:

```bash
python main.py
```

El programa generara un archivo output.csv, el cual contiene el siguiente formato para cada
linea:

```txt
"tweet, sentimiento, negativa, neutral, positiva, defuzz, sent_defuzz, t_fuzz, t_defuzz, t_total"
```
