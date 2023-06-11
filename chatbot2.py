#### Configurar ambiente.
import pandas as pd
import numpy as np
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jellyfish
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
import spacy
nlp = spacy.load('es_core_news_md')
from bs4 import BeautifulSoup
import requests
import torch
import random


#### Tratamiento de datos.
## Función para encontrar la raiz de la palabra.
def raiz(palabra):
# Esta función permite obtener la raiz de un verbo, ya que de esa forma es mas facil ubicra una palabra para el algoritmo,
# por ej, comiendo y comeré tiene la misma raíz comer.
# Para llevar a cabo esta tarea con verbos en castellano se utiliza scrapping.
    radio = 0
    palabra_encontrada = palabra
    for word in lista_verbos:
        confianza = jellyfish.jaro_winkler(palabra, word)
        if (confianza >= 0.92 and confianza >= radio):
            radio = confianza
            palabra_encontrada = word
    return palabra_encontrada

def tratamiento_texto(texto):
    trans = str.maketrans('áéíóú', 'aeiou')
    texto = texto.lower()
    texto = texto.translate(trans) # Saca los acentos de las palabras.
    texto = re.sub(r"[^\w\s]", '', texto) # Retira los símbolos: punto, comas. exclamación, etc.
    texto = " ".join(texto.split())
    return texto

## Función para normalizar la palabra.
def normalizar(texto):
# Esta función permite eliminar todo aquello que no es util en la frase, solo se queda con verbos, pronombres, adjvetivos, etc.
# Para ello utiliza NLP que me permite identificar el tipo de palabra dentro de la frase. Además deja los verbos en su forma raiz.
    doc = nlp(texto)
    tokens = []
    if len(doc) <= 3:
        for t in doc:
            if t.pos_ == 'VERB':
                tokens.append(raiz(t.lemma_))
            else:
                tokens.append(t.lemma_)
    else:
        for t in doc:
            if (t.pos_ in ('VERB', 'PROPN','PRON' ,'NOUN', 'AUX', 'SCONJ', 'DET', 'ADJ', 'ADV') or any(t.dep_.startswith(elemento) for elemento in ['ROOT'])):   
                if t.pos_ == 'VERB':
                    tokens.append(raiz(t.lemma_))
                else:
                    tokens.append(t.lemma_)
    tokens = list(dict.fromkeys(tokens))
    tokens = tokens[:10]
    tokens = ' '.join(tokens)
    return tratamiento_texto(tokens)
 

#### Cargar bases de conocimiento.
## Importando verbos en español.
# Se hace scrapping de un sitio web de donde se obtiene el lsiado de verbos en su base.
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'}
trans = str.maketrans('áéíóú', 'aeiou')
lista_verbos = []
url = ['https://www.ejemplos.co/verbos-mas-usados-en-espanol/',
       'https://www.ejemplos.co/tipos-de-verbos/',
       'https://www.ejemplos.co/verbos-predicativos/',
       'https://www.ejemplos.co/verbos-personales/',
       'https://www.ejemplos.co/verbos-irregulares/',
       'https://www.ejemplos.co/verbos/',
       'https://www.ejemplos.co/100-ejemplos-de-verbos-regulares/',
       'https://www.ejemplos.co/verbos-del-decir/',
       'https://www.ejemplos.co/verbos-con-a/',
       'https://www.ejemplos.co/verbos-con-b/',
       'https://www.ejemplos.co/verbos-con-c/',
       'https://www.ejemplos.co/verbos-con-d/',
       'https://www.ejemplos.co/verbos-con-e/',
       'https://www.ejemplos.co/verbos-con-f/',
       'https://www.ejemplos.co/verbos-con-g/',
       'https://www.ejemplos.co/verbos-con-h/',
       'https://www.ejemplos.co/verbos-con-i/',
       'https://www.ejemplos.co/verbos-con-j/',
       'https://www.ejemplos.co/verbos-con-k/',
       'https://www.ejemplos.co/verbos-con-l/',
       'https://www.ejemplos.co/verbos-con-ll/',
       'https://www.ejemplos.co/verbos-con-m/',
       'https://www.ejemplos.co/verbos-con-n/',
       'https://www.ejemplos.co/verbos-con-o/',
       'https://www.ejemplos.co/verbos-con-p/',
       'https://www.ejemplos.co/verbos-con-q/',
       'https://www.ejemplos.co/verbos-con-r/',
       'https://www.ejemplos.co/verbos-con-s/',
       'https://www.ejemplos.co/verbos-con-t/',
       'https://www.ejemplos.co/verbos-con-u/',
       'https://www.ejemplos.co/verbos-con-v/',
       'https://www.ejemplos.co/verbos-con-w/',
       'https://www.ejemplos.co/verbos-con-x/',
       'https://www.ejemplos.co/verbos-con-y/',
       'https://www.ejemplos.co/verbos-con-z/']

for i in range(len(url)):
    try:       
        respuesta = requests.get(url[i], headers=headers)      
        respuesta = respuesta.content.decode('utf-8')        
        bases = pd.read_html(respuesta)        
        for i, df in enumerate(bases):
            for idx, row in bases[i].iterrows():     
                _ = [lista_verbos.append(re.sub(r"(.*?)", '', x.lower()).strip().translate(trans)) for x in row[0].split('/')]
                _ = [lista_verbos.append(re.sub(r"(.*?)", '', x.lower()).strip().translate(trans)) for x in row[1].split('/')]
                _ = [lista_verbos.append(re.sub(r"(.*?)", '', x.lower()).strip().translate(trans)) for x in row[2].split('/')]
    except Exception:
        continue

lista_verbos = [elemento for elemento in lista_verbos if len(elemento) != 2]
lista_verbos = list(set(lista_verbos))

## Importando bases de dialogo fluído.
txt_folder_path = 'textos' # '/content/textos' para Google Colab.
lista_documentos = [x for x in os.listdir(txt_folder_path) if x.endswith(".txt")]
lista_dialogos, lista_dialogos_respuesta, lista_tipo_dialogo = [], [], []
# El ciclo for recorre cada uno de los archivos que se encuentran en el directorio de los documentos.
for idx in range(len(lista_documentos)):
    f = open(txt_folder_path+'/'+lista_documentos[idx], 'r', encoding='utf-8', errors='ignore')
    flag, posicion = True, 0
    for line in f.read().split('\n'):
        # El if permite ir saltando las líneas, ya que las impares son preguntas y las pares son respuestas.
        if flag:
            line = tratamiento_texto(line)
            lista_dialogos.append(line)
            lista_tipo_dialogo.append(lista_documentos[idx][:2])
        else:
            lista_dialogos_respuesta.append(line)
            posicion+=1
        flag = not flag

## Creando dataframe de diálogos.
datos = {'dialogo': lista_dialogos, 'respuesta': lista_dialogos_respuesta, 'tipo': lista_tipo_dialogo, 'interseccion':0, 'similarity':0, 'jaro_winkler':0, 'probabilidad':0}
df_dialogo = pd.DataFrame(datos)
df_dialogo = df_dialogo.drop_duplicates(keep='first')
df_dialogo.reset_index(drop=True, inplace=True)


#### Buscar respuesta del Chatbot.
## Función para verificar si el usuário inició un diálogo.
def dialogo(user_response):
    df = df_dialogo.copy()
    vectorizer = TfidfVectorizer()
    dialogos_numero = vectorizer.fit_transform(df_dialogo['dialogo'])
    respuesta_numero = vectorizer.transform([user_response])
    for idx, row in df.iterrows():
        df.at[idx, 'interseccion'] = len(set(user_response.split()) & set(row['dialogo'].split()))/len(user_response.split())
        df.at[idx, 'similarity'] = cosine_similarity(dialogos_numero[idx], respuesta_numero)[0][0]
        df.at[idx, 'jaro_winkler'] = jellyfish.jaro_winkler(user_response, row['dialogo'])
        df.at[idx, 'probabilidad'] = max(df.at[idx, 'interseccion'], df.at[idx, 'similarity'], df.at[idx, 'jaro_winkler'])
    df.sort_values(by=['probabilidad', 'jaro_winkler'], inplace=True, ascending=False)
    return df.head(3)

## Cargar el modelo entrenado.
ruta_modelo = '/content/modelo'
Modelo_TF = BertForSequenceClassification.from_pretrained(ruta_modelo)
tokenizer_TF = BertTokenizer.from_pretrained(ruta_modelo)

def clasificacion_modelo(pregunta):
    frase = normalizar(pregunta)
    tokens = tokenizer_TF.encode_plus(
        frase,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']

    with torch.no_grad():
        outputs = Modelo_TF(input_ids, attention_mask)

    etiquetas_predichas = torch.argmax(outputs.logits, dim=1)
    etiquetas_decodificadas = etiquetas_predichas.tolist()

    diccionario = {3: 'Continuacion', 10: 'Nombre', 2: 'Contacto', 13: 'Saludos', 14: 'Sentimiento', 9: 'Identidad', 15: 'Usuario', 6: 'ElProfeAlejo', 1: 'Aprendizaje', 0: 'Agradecimiento', 5: 'Edad', 4: 'Despedida', 11: 'Origen', 12: 'Otros', 7: 'Error', 8: 'Funcion'}
    llave_buscada = etiquetas_decodificadas[0]
    clase_encontrada = diccionario[llave_buscada]

    ## Buscar respuesta más parecida en la clase encontrada.
    df = df_dialogo[df_dialogo['tipo'] == clase_encontrada]
    df.reset_index(inplace=True)
    vectorizer = TfidfVectorizer()
    dialogos_num = vectorizer.fit_transform(df['dialogo'])
    pregunta_num = vectorizer.transform([tratamiento_texto(pregunta)])
    similarity_scores = cosine_similarity(dialogos_num, pregunta_num)
    indice_pregunta_proxima = similarity_scores.argmax()
    return clase_encontrada, df['respuesta'][indice_pregunta_proxima]

#### Ejecutar Chatbot.
pregunta = 'que es machine learning?'
user_response = tratamiento_texto(pregunta)
respuesta = dialogo(user_response)
print(respuesta) # Respuesta en Google Colab.

clase = clasificacion_modelo(pregunta)
print(clase[0])
print(clase[1])
     