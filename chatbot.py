### Configurar ambiente
import pandas as pd
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer # Transforma texto a número.
from sklearn.metrics.pairwise import cosine_similarity # Verifica que tan parecidas son dos palabras.
import jellyfish # Compara textos.


#### Tratamiento de datos.
def tratamiento_texto(texto):
    trans = str.maketrans('áéíóú', 'aeiou')
    texto = texto.lower()
    #print('lower: ', texto)
    texto = texto.translate(trans) # Saca los acentos de las palabras.
    #print('acentuación: ', texto)
    texto = re.sub(r"[^\w\s]", '', texto) # Retira los símbolos: punto, comas. exclamación, etc.
    #print('símbolos: ', texto)
    texto = " ".join(texto.split())
    return texto


#### Cargar bases de conocimiento.
## Importando bases de dialogo fluído.
txt_folder_path = 'textos' # '/content/textos' para Google Colab.
lista_documentos = [x for x in os.listdir(txt_folder_path) if x.endswith(".txt")]
lista_dialogos, lista_dialogos_respuesta, lista_tipo_dialogo = [], [], []
# El ciclo for recorre cada uno de los archivos que se encuentran en el directorio de los documentos.
for idx in range(len(lista_documentos)):
    f=open(txt_folder_path+'/'+lista_documentos[idx], 'r', encoding='utf-8', errors='ignore')
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
vectorizer = TfidfVectorizer()
# Convierte cada dialogo en un vector de números.
dialagos_numero = vectorizer.fit_transform(df_dialogo['dialogo'])

## Función para verificar si el usuario inició un diálogo.
def dialogo(user_response):
    # La función permite comparar las frases ingresadas por el usuario y busca la mejor respuesta.
    # Para ello emplea 3 métodos(intersección, similaridad. Jaro-Winkler) y selecciona la mejor respuesta según la frase.
    df = df_dialogo.copy()
    respuesta_numero = vectorizer.transform([user_response]) # Transformo lo que ingresa el usuario a vector para poder comparar.
    for idx, row in df.iterrows():
        # Recorro cada fila del dataframe de dialogos y comparo con lo ingresado por el usuario aplicando los 3 métodos y eligiendo el mejor.
        df.at[idx, 'interseccion'] = len(set(user_response.split()) & set(row['dialogo'].split()))/len(user_response.split())
        df.at[idx, 'similarity'] = cosine_similarity(dialagos_numero[idx], respuesta_numero)[0][0]
        df.at[idx, 'jaro_winkler'] = jellyfish.jaro_winkler(user_response, row['dialogo'])
        df.at[idx, 'probabilidad'] =max(df.at[idx, 'interseccion'], df.at[idx, 'similarity'], df.at[idx, 'jaro_winkler'])
    df.sort_values(by=['probabilidad', 'jaro_winkler'], inplace=True, ascending=False)
    return df.head(5)


#### Probar Chatbot.
pregunta = 'hola buen día'
user_response = tratamiento_texto(pregunta)
print(user_response)
respuesta = dialogo(user_response)
#print(respuesta) # Respuesta en Google Colab.
print(respuesta['respuesta'].head(1).values[0])





#### Validaciones de prueba por cada paso.
# Tratamiento de texto.
#texto1 = "¡Hola! ¿Quién eres?"
#print (tratamiento_texto(texto1))

# Base de diálogos.
#print(df_dialogo) # df_dialogo en Google Colab

## Intersección.
# Cuenta la cantidad de palabras que hay en común entre las dos oraciones y calcula el porcentaje.
#texto1 = 'quien eres tu'
#texto2 = 'hola quien eres'
#print(len(set(texto1.split()) & set(texto2.split()))/len(texto1.split()))

## Similaridad de coseno.
# Transforma cada frase en un vector de números y calcula el coseno de esos vectores, el resultado indica que tan cerca están.
#texto1 = 'quien eres tu'
#texto2 = 'hola quien eres'
#texto1 = vectorizer.transform([texto1])
#texto2 = vectorizer.transform([texto2])
#print(cosine_similarity(texto1, texto2)[0][0])

## Jaro-Winkler.
#texto1 = 'quien eres tu'
#texto2 = 'hola quien eres'
#print(jellyfish.jaro_winkler(texto1, texto2))