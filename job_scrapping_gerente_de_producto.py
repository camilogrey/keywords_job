import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import re
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import spacy
import spacy.displacy
import re
from datetime import datetime, timedelta
from PIL import Image

#Index(['Title', 'Description', 'Primary Description', 'Detail URL', 'Location','Skill']

#Lectura del df_original
file_path = "C:/File.xlsx"
df = pd.read_excel(file_path)
print(df.tail())

print(df.columns)
#eliminar las columnas del 

df = df.drop(columns=['Insight', 'Job State', 'Poster Id', 'Company Name',
'Company Logo', 'Created At', 'Scraped At'])
print(df.columns)

print(df.describe())

duplicated_records = df[df["Description"].duplicated(keep=False)]
print("heyyyyyyyyyyyyyyyyyy")
print(duplicated_records.head())

# Convert the text to lowercase
df["Description"]= df["Description"].str.lower()



def remove_special_characters(text):
    # Utiliza una expresión regular para eliminar caracteres especiales y espacios en blanco (se excluyen las vocales con acentos)
    return re.sub(r'[^a-zA-ZüÜáéíóúÁÉÍÓÚñÑ$€0-9\s]', ' ', text)

# Aplica la función a toda la columna
df_opinion= df["Description"].apply(remove_special_characters)
print(df["Description"].head(15))
print(df["Description"].tail(15))

# Importar las clases necesarias
one_hot_encoderA = CountVectorizer(binary=True)  # Inicializar un codificador One-Hot utilizando CountVectorizer con codificación binaria
TF_encoderA = CountVectorizer()                   # Inicializar un codificador de términos de frecuencia (TF) utilizando CountVectorizer

# Aplicar los encoders a los datos de entrada
A_1hot = one_hot_encoderA.fit_transform(df_opinion).toarray()  # Codificar los datos de entrada en formato One-Hot
A_TF = TF_encoderA.fit_transform(df_opinion).toarray()         # Codificar los datos de entrada en formato de frecuencia de términos (TF)

# Aplicar transformación TF-IDF a los datos codificados en formato TF
TFIDF_encoderA = TfidfTransformer()                           # Inicializar un transformador TF-IDF utilizando TfidfTransformer
A_TFIDF = TFIDF_encoderA.fit_transform(A_TF).toarray()         # Transformar los datos codificados en formato TF a formato TF-IDF

vocab_A = one_hot_encoderA.get_feature_names_out()             # Obtener el vocabulario generado por el codificador One-Hot

# Imprimir información relevante
print(A_1hot.shape)  # Imprimir la forma de la matriz codificada en formato One-Hot
print(df_opinion[0])  # Imprimir la primera opinión del dataframe
print(len(vocab_A))   # Imprimir la longitud del vocabulario generado

# Encontrar las 5 palabras más frecuentes en el corpus A
print('')
print('Total number of words in corpus A:', A_TF.sum())  # Imprimir el número total de palabras en el corpus A
word_count = A_TF.sum(axis=0)                            # Calcular la frecuencia de cada palabra en el corpus A
a = word_count.max()                                     # Encontrar la palabra más frecuente y su conteo
print('1st :', vocab_A[word_count.argmax()], '-- total count:', a)
word_count[word_count.argmax()] = 0                      # Excluir la palabra más frecuente para encontrar la siguiente
b = word_count.max()                                     # Encontrar la segunda palabra más frecuente y su conteo
print('2nd :', vocab_A[word_count.argmax()], '-- total count:', b)
word_count[word_count.argmax()] = 0                      # Excluir la segunda palabra más frecuente para encontrar la siguiente
c = word_count.max()                                     # Encontrar la tercera palabra más frecuente y su conteo
print('3rd :', vocab_A[word_count.argmax()], '-- total count:', c)
word_count[word_count.argmax()] = 0                      # Excluir la tercera palabra más frecuente para encontrar la siguiente
d = word_count.max()                                     # Encontrar la cuarta palabra más frecuente y su conteo
print('4th :', vocab_A[word_count.argmax()], '-- total count:', d)
word_count[word_count.argmax()] = 0                      # Excluir la cuarta palabra más frecuente para encontrar la siguiente
e = word_count.max()                                     # Encontrar la quinta palabra más frecuente y su conteo
print('5th :', vocab_A[word_count.argmax()], '-- total count:', e)
print('Corpus % for top 5 words:', (a + b + c + d + e) / A_TF.sum())  # Imprimir el porcentaje del corpus representado por las 5 palabras más frecuentes


# Descargar y poner dentro de una variable las stop words
nltk.download('stopwords')
nltk.download('punkt')
stop_words_sp = set(stopwords.words('spanish'))

nltk.download('stopwords')
nltk.download('punkt')
stop_words_en = set(stopwords.words('english'))

# Combinar las stopwords en un solo conjunto
combined_stopwords = stop_words_sp.union(stop_words_en)

# Generar la nube de palabras con el conjunto combinado de stopwords
wordcloud = WordCloud(
    width=800,
    height=800,
    background_color='white',
    stopwords=combined_stopwords,
    min_font_size=10
).generate(df_opinion.to_string())  # Asegúrate de usar la columna correcta

# Mostrar la nube de palabras
plt.figure(figsize=(14, 6), facecolor=None)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Descripciones", fontdict={'fontsize': 10})
plt.show()


###### TOKENIZACION #######
def tokenize_text(text):
    return word_tokenize(text)

# Aplicar tokenización a toda la columna 'Texto'
df['Tokens'] = df_opinion.apply(tokenize_text)

# Mostrar el DataFrame con la nueva columna 'Tokens'
print(df['Tokens'].head(15))

# Función para limpiar la lista de tokens
def clean_tokens(tokens):
    if isinstance(tokens, list): # Verificar si tokens es una lista
        # Eliminar espacios en blanco adicionales y stopwords
        clean_tokens = [re.sub(r'\s+', ' ', word.strip()) for word in tokens if word.strip() not in combined_stopwords]
        return clean_tokens
    else: # Si tokens es NaN, devuelve una lista vacía
        return []

# Aplicar la limpieza de tokens a toda la columna 'Tokens'
df['Cleaned_Tokens'] = df['Tokens'].apply(clean_tokens)

# Mostrar el DataFrame con la nueva columna 'Cleaned_Tokens'

print(df['Cleaned_Tokens'].head(10))
print(df['Cleaned_Tokens'].tail(10))



# Identificar los valores que son listas vacías en una columna de un DataFrame
valores_vacios = df['Cleaned_Tokens'].apply(lambda x: isinstance(x, list) and len(x) == 0)

# Obtener los índices de las filas con valores vacíos
indices_valores_vacios = valores_vacios[valores_vacios].index

# Eliminar las filas con valores vacíos de forma permanente
df.drop(indices_valores_vacios, inplace=True)

print(df['Cleaned_Tokens'].head(50))
print(df['Cleaned_Tokens'].tail(50))


# Convertir los tokens a textos planos para TF-IDF
texts = [' '.join(tokens) for tokens in df['Cleaned_Tokens']]



# Crear bigramas
bigram_vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=50)
bigram_matrix = bigram_vectorizer.fit_transform(texts)

# Obtener los nombres de los bigramas y sus frecuencias
bigram_feature_names = bigram_vectorizer.get_feature_names_out()
bigram_scores = bigram_matrix.toarray().sum(axis=0)

# Convertir a DataFrame para facilitar la visualización
bigram_df = pd.DataFrame({'Bigram': bigram_feature_names, 'Frequency': bigram_scores}).sort_values(by='Frequency', ascending=False)

# Mostrar los bigramas más frecuentes
print(bigram_df.head(20))

# Graficar los bigramas más frecuentes
plt.figure(figsize=(12, 8))
sns.barplot(x='Frequency', y='Bigram', data=bigram_df.head(50), palette='viridis')
plt.title('Top 20 Bigramas Más Frecuentes en las Reseñas')
plt.xlabel('Frecuencia')
plt.ylabel('Bigram')
plt.show()


# Crear trigramas
trigram_vectorizer = CountVectorizer(ngram_range=(3, 3), max_features=50)
trigram_matrix = trigram_vectorizer.fit_transform(texts)

# Obtener los nombres de los trigramas y sus frecuencias
trigram_feature_names = trigram_vectorizer.get_feature_names_out()
trigram_scores = trigram_matrix.toarray().sum(axis=0)

# Convertir a DataFrame para facilitar la visualización
trigram_df = pd.DataFrame({'Trigram': trigram_feature_names, 'Frequency': trigram_scores}).sort_values(by='Frequency', ascending=False)

# Mostrar los trigramas más frecuentes
print(trigram_df.head(20))

# Graficar los trigramas más frecuentes
plt.figure(figsize=(12, 8))
sns.barplot(x='Frequency', y='Trigram', data=trigram_df.head(50), palette='viridis')
plt.title('Top 20 Trigramas Más Frecuentes en las Reseñas')
plt.xlabel('Frecuencia')
plt.ylabel('Trigram')
plt.show()



# Crear el vectorizador TF-IDF para bigramas y trigramas
#tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words=stopwords.words('spanish'), max_features=100)

tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words=stopwords.words('spanish'), max_features=100)


# Ajustar y transformar los textos
tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

# Obtener los nombres de las características y los puntajes TF-IDF
feature_names = tfidf_vectorizer.get_feature_names_out()
tfidf_scores = tfidf_matrix.toarray().sum(axis=0)

# Convertir a DataFrame para facilitar la visualización
tfidf_df = pd.DataFrame({'Phrase': feature_names, 'TF-IDF Score': tfidf_scores}).sort_values(by='TF-IDF Score', ascending=False)

# Mostrar las frases con los puntajes TF-IDF más altos
print(tfidf_df.head(20))

# Graficar las frases con los puntajes TF-IDF más altos
plt.figure(figsize=(12, 8))
sns.barplot(x='TF-IDF Score', y='Phrase', data=tfidf_df.head(20), palette='viridis')
plt.title('Top 20 Frases Más Importantes según TF-IDF Español')
plt.xlabel('Puntaje TF-IDF')
plt.ylabel('Frase')
plt.show()

