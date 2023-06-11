#### Configurar Ambiente.
## Instalando bibliotecas necesarias.
import pandas as pd
import numpy as np
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
nlp = spacy.load('es_core_news_md')
import jellyfish
import requests
import csv
from docx import Document
import nltk
nltk.download('punkt')
import warnings, os
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from bs4 import BeautifulSoup
import torch
import random

warnings.filterwarnings('ignore')
global diccionario_irregulares, documento, lista_frases, lista_frases_normalizadas


#### Tratamiento de los datos.
## Función para encontrar la raiz de las palabras.
def raiz(palabra):
    radio = 0
    palabra_encontrada = palabra
    for word in lista_verbos:
        confianza = jellyfish.jaro_winkler(palabra, word)
        if (confianza >= 0.93 and confianza >= radio):
            radio = confianza
            palabra_encontrada = word
    return palabra_encontrada


def tratamiento_texto(texto):
    trans = str.maketrans('áéíóú', 'aeiou')
    texto = texto.lower()
    texto = texto.translate(trans)
    texto = " ".join(texto.split())
    return texto

## Función para reemplazar el final de una palabra por 'r'.
def reemplazar_terminacion(palabra):
    # La función tiene por objetivo dejar a la palabra en si verbo base.  
    patron = r"(es|me|as|te|ste)$"
    nueva_palabra = re.sub(patron, "r", palabra)
    return nueva_palabra.split()[0]

## Función para adicionar o eliminar tokens.
def revisar_tokens(texto, tokens):
    # La función tiene por objetivo poder destacar un token(palabras llaves) por sobre el resto, pero considerando su 
    # posibles variaciones, teniendo en cuenta la entrada del usuario.
    # Esto lo logra por en la salida por verdadero del if, en else elimina todas las palabras que no son utiles y que 
    # forman parte de la pregunta.
    if len(tokens) == 0:
        if [x for x in ['elprofealejo', 'el profe alejo', 'profe alejo', 'profealejo'] if x in tratamiento_texto(texto)]: tokens.append('elprofealejo')
        elif [x for x in ['cientifico de datos', 'data scientist'] if x in tratamiento_texto(texto)]: tokens.append('datascientist')
        elif [x for x in ['ciencia de datos', 'data science'] if x in tratamiento_texto(texto)]: tokens.append('datascience')
        elif [x for x in ['big data', 'bigdata'] if x in tratamiento_texto(texto)]: tokens.append('bigdata')
    else:
        elementos_a_eliminar = ["profe", "alejo", "profealejo", "cual", "que", "quien", "cuanto", "donde", "cuando", "como"]
        if 'hablame' in texto and 'hablar' in tokens: tokens.remove('hablar')
        elif 'cuentame' in texto and 'contar' in tokens: tokens.remove('contar') 
        elif 'hago' in texto and 'hacer' in tokens: tokens.remove('hacer') 
        elif 'entiendes' in texto and 'entender' in tokens: tokens.remove('entender') 
        elif 'sabes' in texto and 'saber' in tokens: tokens.remove('saber') 
        tokens = [x.replace('datar', 'data').replace('datos', 'dato') for x in tokens if x not in elementos_a_eliminar]
    return tokens

## Función para devolver los tokens normalizados del texto.
def normalizar(texto):
    # La función devuelve una lista con las palabras de la frase.
    tokens = []
    tokens = revisar_tokens(texto, tokens)
    if 'elprofealejo' in tokens:
        texto = ' '.join(texto.split()[:15])
    else:
        texto = ' '.join(texto.split()[:25])

    doc = nlp(texto)
    for t in doc:
        lemma = diccionario_irregulares.get(t.text, t.lemma_.split()[0])
        lemma = re.sub(r'[^\w\s+\-*/]', '', lemma)
        if t.pos_ in ('VERB','PROPN','PRON','NOUN','AUX','SCONJ','ADJ','ADV','NUM') or lemma in lista_verbos:
            if t.pos_ == 'VERB':
                lemma = reemplazar_terminacion(lemma)
                tokens.append(raiz(tratamiento_texto(lemma)))
            else:
                tokens.append(tratamiento_texto(lemma))

    tokens = list(dict.fromkeys(tokens))
    tokens = list(filter(None, tokens))
    tokens = revisar_tokens(texto, tokens)
    return tokens

## Función normalizar que se utilizó para entrenar el modelo.
def normalizar_modelo(texto):
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
            if (t.pos_ in ('VERB','PROPN','PRON','NOUN','AUX','SCONJ','DET','ADJ','ADV') or any(t.dep_.startswith(elemento) for elemento in ['ROOT'])):
                if t.pos_ == 'VERB':
                    tokens.append(raiz(t.lemma_))
                else:
                    tokens.append(t.lemma_)
    tokens = list(dict.fromkeys(tokens))
    tokens = tokens[:10]
    tokens = ' '.join(tokens)
    return tratamiento_texto(tokens)


#### Cargar bases de verbos.
## Importando verbos en español.
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'}
trans = str.maketrans('áéíóú','aeiou')
lista_verbos = []
url = ['https://www.ejemplos.co/verbos-mas-usados-en-espanol/',
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
            for idx,row in bases[i].iterrows():     
                _ = [lista_verbos.append(re.sub(r"(.*?)", '', x.lower()).strip().translate(trans)) for x in row[0].split('/')]
                _ = [lista_verbos.append(re.sub(r"(.*?)", '', x.lower()).strip().translate(trans)) for x in row[1].split('/')]
                _ = [lista_verbos.append(re.sub(r"(.*?)", '', x.lower()).strip().translate(trans)) for x in row[2].split('/')]
    except Exception:
        continue
lista_verbos = [elemento for elemento in lista_verbos if (elemento == 'ir' or len(elemento) != 2)]
nuevos_verbos = ['costar', 'referir', 'datar']
lista_verbos.extend(nuevos_verbos)
lista_verbos = list(set(lista_verbos))

## Definir una lista de verbos irregulares y sus conjugaciones en pasado, presente, futuro, imperfecto, pretérito y condicional.
# El primer elemento es la raiz.
verbos_irregulares = [
    ('ser', 'soy', 'eres', 'seras', 'eras', 'es', 'serias'),
    ('estar', 'estuviste', 'estas', 'estaras', 'estabas', 'estuviste', 'estarias'),
    ('ir', 'fuiste', 'vas', 'iras', 'ibas', 'fuiste', 'irias'),
    ('ir', 'fuiste', 'vaya', 'iras', 'ibas', 'fuiste', 'irias'),
    ('tener', 'tuviste', 'tienes', 'tendras', 'tenias', 'tuviste', 'tendrias'),
    ('hacer', 'hiciste', 'haces', 'haras', 'hacias', 'hiciste', 'harias'),
    ('decir', 'dijiste', 'dices', 'diras', 'decias', 'dijiste', 'dirias'),
    ('decir', 'dimar', 'dime', 'digame', 'dimir', 'dimo', 'dimiria'),
    ('poder', 'pudiste', 'puedes', 'podras', 'podias', 'pudiste', 'podrias'),
    ('saber', 'supiste', 'sabes', 'sabras', 'sabias', 'supiste', 'sabrias'),
    ('poner', 'pusiste', 'pones', 'pondras', 'ponias', 'pusiste', 'pondrias'),
    ('ver', 'viste', 'ves', 'veras', 'veias', 'viste', 'verias'),
    ('dar', 'diste', 'das', 'daras', 'dabas', 'diste', 'darias'),
    ('dar', 'damar', 'dame', 'daras', 'dabas', 'darme', 'darias'),
    ('venir', 'viniste', 'vienes', 'vendras', 'venias', 'viniste', 'vendrias'),
    ('haber', 'haya', 'has', 'habras', 'habias', 'hubiste', 'habrias'),
    ('caber', 'cupiste', 'cabes', 'cabras', 'cabias', 'cupiste', 'cabrias'),
    ('valer', 'valiste', 'vales', 'valdras', 'valias', 'valiste', 'valdrias'),
    ('querer', 'quisiste', 'quieres', 'querras', 'querias', 'quisiste', 'querrias'),
    ('llegar', 'llegaste', 'llegares', 'llegaras', 'llegarias', 'llegaste', 'llegarrias'),
    ('hacer', 'hiciste', 'haces', 'haras', 'hacias', 'hiciste', 'harias'),
    ('decir', 'dijiste', 'dices', 'diras', 'decias', 'dijiste', 'dirias'),
    ('poder', 'pudiste', 'puedes', 'podras', 'podias', 'pudiste', 'podria'),
    ('contar', 'contaste', 'cuentas', 'contaras', 'contabas', 'cuentame', 'contarias'),
    ('saber', 'supiste', 'sabes', 'sabras', 'sabias', 'supiste', 'sabrias'),
    ('costar', 'cuesta', 'cuestan', 'costo', 'costaria', 'costarian', 'cuestas'),
    ('durar', 'duraste', 'duro', 'duraras', 'durabas', 'duraste', 'durarias')
]

# Crear el DataFrame
diccionario_irregulares = {}
df = pd.DataFrame(verbos_irregulares, columns=['Verbo', 'Pasado', 'Presente', 'Futuro', 'Imperfecto', 'Pretérito', 'Condicional'])
for columna in df.columns:
    if columna != 'Verbo':
        for valor in df[columna]:
            diccionario_irregulares[valor] = df.loc[df[columna] == valor, 'Verbo'].values[0]


#### Cargar base de documentos.
## Importando bases de dialogo fluído.
txt_folder_path = 'documentos' # '/content/dialogos' en Google Colab.
lista_documentos = [x for x in os.listdir(txt_folder_path) if x.endswith(".txt")]
lista_dialogos, lista_dialogos_respuesta, lista_tipo_dialogo = [], [], []
for idx in range(len(lista_documentos)):
    f = open(txt_folder_path+'/'+lista_documentos[idx], 'r', encoding='utf-8', errors='ignore')
    flag, posicion = True, 0
    for line in f.read().split('\n'):
        if flag:
            line = tratamiento_texto(line)
            line = re.sub(r"[^\w\s]", '', line)
            lista_dialogos.append(line)
            lista_tipo_dialogo.append(lista_documentos[idx].replace('.txt', ''))
        else:
            lista_dialogos_respuesta.append(line)
            posicion += 1
        flag = not flag

## Creando Dataframe de diálogos.
datos = {'dialogo':lista_dialogos, 'respuesta':lista_dialogos_respuesta, 'tipo':lista_tipo_dialogo, 'interseccion':0, 'similarity':0, 'jaro_winkler':0, 'probabilidad':0}
df_dialogo = pd.DataFrame(datos)
df_dialogo = df_dialogo.drop_duplicates(keep='first')
df_dialogo.reset_index(drop=True, inplace=True)

## Importando bases csv.
txt_folder_path = 'documentos' # '/content/documentos' en Google Colab.
lista_documentos = [x for x in os.listdir(txt_folder_path) if x.endswith(".csv")]
documento_csv = ''
for idx in range(len(lista_documentos)):
    with open(txt_folder_path+'/'+lista_documentos[idx], "r", encoding="utf-8") as archivo_csv:
        lector_csv = csv.reader(archivo_csv)
        for fila in lector_csv:
            if fila[-1] != 'frase':
                documento_csv += fila[-1]

## Importando bases docx.
txt_folder_path = 'documentos' # '/content/documentos' en Google Colab.
lista_documentos = [x for x in os.listdir(txt_folder_path) if x.endswith(".docx")]
documento_docx = ''
for idx in range(len(lista_documentos)):
    for paragraph in Document(txt_folder_path+'/'+lista_documentos[idx]).paragraphs:
        documento_docx += paragraph.text.replace('*','\n\n*').replace('-','\n-')

## Importando bases txt.
txt_folder_path = 'documentos' # '/content/documentos' en Google Colab.
lista_documentos = [x for x in os.listdir(txt_folder_path) if x.endswith(".txt")]
documento_txt = ''
for idx in range(len(lista_documentos)):
    with open(txt_folder_path+'/'+lista_documentos[idx], "r", encoding="utf-8") as archivo_txt:
        lector_txt = archivo_txt.read()
        for fila in lector_txt:
            documento_txt += fila

documento = documento_csv + documento_txt + documento_docx
# La línea de abajo separa, la unión del contenido de todos los documentos, en frases.
lista_frases = nltk.sent_tokenize(documento, 'spanish')
lista_frases_normalizadas = [' '.join(normalizar(x)) for x in lista_frases]


#### Buscar respuesta del Chatbot.
## Función para verificar si el usuário inició un diálogo.
def dialogo(user_response):
    user_response = tratamiento_texto(user_response)
    user_response = re.sub(r"[^\w\s]", '', user_response)
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
    probabilidad = df['probabilidad'].head(1).values[0]
    tipo = df['tipo'].head(1).values[0]
    if probabilidad >= 0.93 and tipo not in ['ElProfeAlejo']:
        print('Respuesta encontrada por el método de comparación de textos - Probabilidad: ', probabilidad)
        respuesta = df['respuesta'].head(1).values[0]
    else:
        respuesta = ''
    return respuesta

## Cargar el modelo entrenado.
ruta_modelo = 'modelos' # '/content/modelo' en Google Colab.
Modelo_TF = BertForSequenceClassification.from_pretrained(ruta_modelo)
tokenizer_TF = BertTokenizer.from_pretrained(ruta_modelo)

## Función para dialogar utilizando el modelo Transformers.
def clasificacion_modelo(pregunta):
    pregunta = re.sub(r"[^\w\s]", '', pregunta)
    frase = normalizar_modelo(pregunta)
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
    if clase_encontrada not in ['Otros','ElProfeAlejo']:
        print('Respuesta encontrada por el modelo Transformers - tipo:',clase_encontrada)
        respuesta = df['respuesta'][indice_pregunta_proxima]
    else:
        respuesta = ''
    return respuesta

## Función para devolver la respuesta de los documentos.
def respuesta_documento(pregunta):
    pregunta = normalizar(pregunta)
    def contar_coincidencias(frase):
        return sum(1 for elemento in pregunta if elemento in frase) 
    # Traigo todas las frases normalizadas.
    diccionario = {valor: posicion for posicion, valor in enumerate(lista_frases_normalizadas)}
    # De todas las frases normalizadas, trae las 6 primeras frases que coinciden con la pregunta.
    lista = sorted(list(diccionario.keys()), key=contar_coincidencias, reverse=True)[:6]
    if 'curso' not in pregunta: lista = [frase for frase in lista if 'curso' not in frase]
    # Agrega la pregunta a la lista de 6 elementos coincidentes con la pregunta.
    lista.append(' '.join(pregunta))
    # Se tokeniza la lista formada por 6 respuestas y la pregunta. Tfifd crea una matriz con cada 
    # frase como indice y cada palabra de todas las frases como columnas, y va contando por fila.
    TfidfVec = TfidfVectorizer(tokenizer=normalizar)
    tfidf = TfidfVec.fit_transform(lista)
    # Calcula la respuesta más parecida de las 6 con respecto a la pregunta, en la última posición 
    # de la lista está la pregunta por ello el -1 como primer parámetro, lo que retorna son valores.
    vals = cosine_similarity(tfidf[-1], tfidf)
    # Se ordenan esos valores de mayor a menor.
    idx = vals.argsort()[0][-2]
    # Se convierten a posiciones.
    flat = vals.flatten()
    flat.sort()
    # Devuelvo la posición -2 porque es la más parecida a la pregunta y está ordenado la lista.
    req_tfidf = round(flat[-2], 2)
    if req_tfidf >= 0.22:
        print('Respuesta encontrada por el método TfidfVectorizer - Probabilidad:', req_tfidf)
        respuesta = lista_frases[diccionario[lista[idx]]]
    else:
        respuesta = ''
    return respuesta

## Función para devolver una respuesta final buscada en todos los métodos disponibles.
def respuesta_chatbot(pregunta):
    respuesta = dialogo(pregunta)
    if respuesta != '':
        return respuesta
    else:
        respuesta = respuesta_documento(pregunta)
        if respuesta != '':
            return respuesta
        else:
            respuesta = clasificacion_modelo(pregunta)
            if respuesta != '':
                return respuesta
            else:
                return 'Respuesta no encontrada'


#### Ejecutar Chatbot.
pregunta = 'que es Machine Learning?'
respuesta = respuesta_chatbot(pregunta)
print(respuesta)