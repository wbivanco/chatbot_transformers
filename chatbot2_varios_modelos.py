#### Configurar ambiente.
import pandas as pd
import numpy as np
import os
import re
import spacy
nlp = spacy.load('es_core_news_md')
import jellyfish
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from bs4 import BeautifulSoup
import requests
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


#### Tratameinto de datos.
## Función para encontrar la raiz de las palabras.
def raiz(palabra):
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
    texto = texto.translate(trans)
    texto = re.sub(r"[^\w\s]", '', texto)
    texto = " ".join(texto.split())
    return texto

## Función para normalizar la palabra.
def normalizar(texto):
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
            if (t.pos_ in ('VERB', 'PROPN', 'PRON', 'NOUN', 'AUX', 'SCONJ', 'DET', 'ADJ', 'ADV') or any(t.dep_.startswith(elemento) for elemento in ['ROOT'])):
                if t.pos_ == 'VERB':
                    tokens.append(raiz(t.lemma_))
                else:
                    tokens.append(t.lemma_)
    tokens = list(dict.fromkeys(tokens))
    tokens = tokens[:10]
    tokens = ' '.join(tokens)
    return tratamiento_texto(tokens)

# Pruebas
frase = '¡Hola! ¿como estás? Es un placer saludarlo'
normalizar(frase)

raiz('saludarlo')


#### Cargar bases de conocimiento.
## Importando verbos en español.
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'}
trans = str.maketrans('áéíóú', 'aeiou')
lista_verbos=[]
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
            for idx,row in bases[i].iterrows():     
                _ = [lista_verbos.append(re.sub(r"(.*?)", '', x.lower()).strip().translate(trans)) for x in row[0].split('/')]
                _ = [lista_verbos.append(re.sub(r"(.*?)", '', x.lower()).strip().translate(trans)) for x in row[1].split('/')]
                _ = [lista_verbos.append(re.sub(r"(.*?)", '', x.lower()).strip().translate(trans)) for x in row[2].split('/')]
    except Exception:
        continue
lista_verbos = [elemento for elemento in lista_verbos if len(elemento) != 2]
lista_verbos=list(set(lista_verbos))

## Importando bases de dialogo fluído.
txt_folder_path = 'textos' # '/contentx' en Google Colab.
lista_documentos = [x for x in os.listdir(txt_folder_path) if x.endswith(".txt")]
lista_dialogos, lista_dialogos_respuesta, lista_tipo_dialogo = [], [], []
for idx in range(len(lista_documentos)):
    f = open(txt_folder_path+'/'+lista_documentos[idx], 'r', encoding='utf-8', errors='ignore')
    flag,posicion = True,0
    for line in f.read().split('\n'):
        if flag:
            line = tratamiento_texto(line)
            lista_dialogos.append(line)
            lista_tipo_dialogo.append(lista_documentos[idx].replace('.txt', ''))
        else:
            lista_dialogos_respuesta.append(line)
            posicion += 1
        flag=not flag

## Creando Dataframe de diálogos.
datos = {'dialogo':lista_dialogos, 'respuesta':lista_dialogos_respuesta, 'tipo':lista_tipo_dialogo, 'interseccion':0, 'similarity':0, 'jaro_winkler':0, 'probabilidad':0}
df_dialogo = pd.DataFrame(datos)
df_dialogo = df_dialogo.drop_duplicates(keep='first')
df_dialogo.reset_index(drop=True, inplace=True)


#### Normalizando diálogos.

#Normalizando las frases
label_encoder = LabelEncoder()
df_dialogo['palabras'] = df_dialogo['dialogo'].apply(normalizar)
df_dialogo['tipo_num'] = label_encoder.fit_transform(df_dialogo['tipo'])
df_dialogo

# Pruebas.
## Ranking de frases.
word_freq = df_dialogo.explode('palabras').groupby('tipo')['palabras'].value_counts()
ranking = word_freq.reset_index(name='Frecuencia').sort_values(['tipo', 'Frecuencia'], ascending=[True, False])
ranking
     

#### Entrenando Naive Bayes.
## Separar los datos en características (X) y etiquetas (y).
X = df_dialogo['palabras']
y = df_dialogo['tipo_num']

## Dividir los datos en conjuntos de entrenamiento y prueba.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Vectorizar los datos de texto.
vectorizer = CountVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

## Entrenar el clasificador de Naive Bayes.
modelo_NB = MultinomialNB()
modelo_NB.fit(X_train_vect, y_train)

## Realizar predicciones en el conjunto de prueba.
y_pred = modelo_NB.predict(X_test_vect)

## Calcular el accuracy.
accuracy = accuracy_score(y_test, y_pred)
print("Precisión:", accuracy)
     
## Calcular la precisión por clase.
unique_classes = df_dialogo['tipo_num'].unique()
for cls in unique_classes:
    cls_indices = y_test == cls
    cls_accuracy = accuracy_score(y_test[cls_indices], y_pred[cls_indices])
    print("Accuracy para la clase", df_dialogo[df_dialogo.tipo_num == cls]['tipo'].unique()[0], ":", cls_accuracy)
     
## Procesando la nueva frase.
frase = normalizar('como haces para aprender tan rapido?')
nueva_frase_vect = vectorizer.transform([frase])

## Realizar la predicción.
prediccion = modelo_NB.predict(nueva_frase_vect)

print("La frase", frase, "se clasifica como: ", df_dialogo[df_dialogo.tipo_num == prediccion[0]]['tipo'].unique()[0])
     

#### Entrenando Random Forest.
## Separar los datos en características (X) y etiquetas (y).
X = df_dialogo['palabras']
y = df_dialogo['tipo_num']

## Dividir los datos en conjuntos de entrenamiento y prueba.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Vectorizar los datos de texto.
vectorizer = CountVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

## Entrenar el clasificador Random Forest.
Modelo_RF = RandomForestClassifier()
Modelo_RF.fit(X_train_vect, y_train)

## Realizar predicciones en el conjunto de prueba.
y_pred = Modelo_RF.predict(X_test_vect)

## Calcular el accuracy.
accuracy = accuracy_score(y_test, y_pred)
print("Precisión:", accuracy)

## Calcular la precisión por clase.
unique_classes = df_dialogo['tipo_num'].unique()
for cls in unique_classes:
    cls_indices = y_test == cls
    cls_accuracy = accuracy_score(y_test[cls_indices], y_pred[cls_indices])
    print("Accuracy para la clase", df_dialogo[df_dialogo.tipo_num == cls]['tipo'].unique()[0], ":", cls_accuracy)
     
## Procesando la nueva frase.
frase = normalizar('como haces para aprender tan rapido?')
nueva_frase_vect = vectorizer.transform([frase])

## Realizar la predicción.
prediccion = Modelo_RF.predict(nueva_frase_vect)

print("La frase", frase, "se clasifica como: ", df_dialogo[df_dialogo.tipo_num == prediccion[0]]['tipo'].unique()[0])
     

#### Entrenando con Transformers.
## Dividir los datos en conjunto de entrenamiento y conjunto de prueba.
df_train, df_test = train_test_split(df_dialogo, test_size=0.2, random_state=42)

## Cargar el modelo preentrenado de BERT para clasificación en español.
model_name = 'dccuchile/bert-base-spanish-wwm-uncased'
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=df_dialogo['tipo_num'].nunique())
tokenizer = BertTokenizer.from_pretrained(model_name)

## Tokenizar y codificar las frases de entrenamiento.
train_inputs = tokenizer.batch_encode_plus(
    df_train['palabras'].tolist(),
    max_length=128,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

## Tokenizar y codificar las frases de prueba.
test_inputs = tokenizer.batch_encode_plus(
    df_test['palabras'].tolist(),
    max_length=128,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

## Preparar los datos de entrenamiento y prueba.
train_data = torch.utils.data.TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], torch.tensor(df_train['tipo_num'].tolist()))
test_data = torch.utils.data.TensorDataset(test_inputs['input_ids'], test_inputs['attention_mask'], torch.tensor(df_test['tipo_num'].tolist()))

## Definir el optimizador y la función de pérdida.
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

## Entrenamiento del modelo.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.train()

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)

for epoch in range(5):  # Número de épocas de entrenamiento.
    total_loss = 0

    for batch in train_dataloader:
        input_ids, attention_mask, labels = tuple(t.to(device) for t in batch)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    print("Epoch:", epoch + 1, "Loss:", total_loss)

## Evaluación del modelo.
model.eval()
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

with torch.no_grad():
    predictions = []
    true_labels = []

    for batch in test_dataloader:
        input_ids, attention_mask, labels = tuple(t.to(device) for t in batch)

        outputs = model(input_ids, attention_mask=attention_mask)

        _, predicted_labels = torch.max(outputs.logits, dim=1)

        predictions.extend(predicted_labels.tolist())
        true_labels.extend(labels.tolist())

accuracy = accuracy_score(true_labels, predictions)
print("Precisión:", accuracy)

## Guardar el modelo entrenado.
ruta_modelo = 'modelos' # '/content/modelos' ne Google Colab.
model.save_pretrained(ruta_modelo)
tokenizer.save_pretrained(ruta_modelo)

## Cargar el modelo entrenado.
ruta_modelo = 'modelos' # /content/modelo' en Google Colab.
Modelo_TF = BertForSequenceClassification.from_pretrained(ruta_modelo)
tokenizer_TF = BertTokenizer.from_pretrained(ruta_modelo)

## Calcular la precisión por clase.
unique_classes = df_dialogo['tipo_num'].unique()

for class_label in unique_classes:
    ## Filtrar los datos por clase.
    class_data = df_dialogo[df_dialogo['tipo_num'] == class_label]

    ## Preparar los datos de la clase para evaluar.
    tokens = tokenizer_TF.batch_encode_plus(
        class_data['palabras'].tolist(),
        truncation=True,
        padding=True,
        return_tensors='pt'
    )

    inputs = tokens['input_ids']
    attention_mask = tokens['attention_mask']
    labels = class_data['tipo_num'].tolist()

    ## Pasar los datos de la clase por el modelo.
    with torch.no_grad():
        outputs = Modelo_TF(inputs, attention_mask=attention_mask)

    predicted_labels = outputs.logits.argmax(dim=1).tolist()

    ## Calcular la precisión para la clase.
    accuracy = accuracy_score(labels, predicted_labels)
    print(f"Precisión por clase {df_dialogo[df_dialogo.tipo_num == class_label]['tipo'].unique()[0]}: {accuracy}")
     
## Procesar nueva frase.
frase = normalizar('como haces para aprender tan rapido?')

## Tokenizar la frase de entrada.
tokens = tokenizer_TF.encode_plus(
    frase,
    add_special_tokens=True,
    max_length=128,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

## Obtener los input_ids y attention_mask.
input_ids = tokens['input_ids']
attention_mask = tokens['attention_mask']

## Realizar la predicción.
with torch.no_grad():
    outputs = Modelo_TF(input_ids, attention_mask)

## Obtener las etiquetas predichas.
etiquetas_predichas = torch.argmax(outputs.logits, dim=1)

## Decodificar las etiquetas predichas.
etiquetas_decodificadas = etiquetas_predichas.tolist()
print("La frase", frase, "se clasifica como: ", df_dialogo[df_dialogo.tipo_num == etiquetas_decodificadas[0]]['tipo'].unique()[0])
     