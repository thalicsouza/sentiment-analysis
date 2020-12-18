#!/usr/bin/env python
# coding: utf-8

# ### NLP - Análise de Sentimentos em Pesquisas de Satisfação
# ### MBA em Ciência de Dados (Cemeai/USP)
# 
# #### Autora: Thalita Cristina de Souza
# #### Orientador: Prof. Dr. Julio Cezar Estrella
# ##### Última atualização: Dezembro/2020

# ### Configurações iniciais

# In[1]:


# Configuração do valor da semente para garantir a reprodução dos mesmos resultados
seed_value = 1313

import os
os.environ['PYTHONHASHSEED']=str(seed_value)

import random
random.seed(seed_value)

import numpy as np
np.random.seed(seed_value)

import tensorflow as tf
tf.random.set_seed(seed_value)

from keras import backend as K
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)


# In[2]:


# Importando as bibliotecas

# Bibliotecas de manipulação e visualização dos dados
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import timeit

# Pré-processamento dos elementos textuais
import texthero as hero
from texthero import preprocessing, stopwords, nlp
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import PortugueseStemmer

# Modelagem NLP
#from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD, PCA
import spacy
from scipy.sparse import random as sparse_random
from sklearn.random_projection import sparse_random_matrix
import pt_core_news_sm
nlp = pt_core_news_sm.load()

# Algoritmos clássicos de classificação
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.pipeline import make_pipeline 
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Modelo ensemble de classificação
import lightgbm as lgb

# Tensorflow e Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import preprocessing as keras_prep
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_hub as hub
from tensorflow.keras import Model, Sequential
from tensorflow.keras.utils import Sequence
from tensorflow.keras import initializers
from tensorflow.keras.layers import GRU, LSTM, Dense, RNN, SimpleRNN, Flatten, Embedding, Dropout, GlobalAveragePooling1D, LeakyReLU
from tensorflow.random import set_seed

# Métricas
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef, confusion_matrix, accuracy_score, classification_report
from sklearn import metrics
from sklearn.pipeline import make_pipeline 
from scipy.stats import ttest_ind, shapiro, levene
import baycomp

# Validação Cruzada
from sklearn.model_selection import cross_validate, cross_val_score, StratifiedKFold

# modelo treinado da NLTK para segmentação de sentença
stok = nltk.data.load('tokenizers/punkt/portuguese.pickle')


# In[3]:


print("Versões das principais bibliotecas utilizadas")
print("Pandas: ", pd.__version__)
print("Numpy: ", np.__version__)
print("Tensorflow: ", tf.__version__)
print("Keras: ", keras.__version__)
print("TensorflowHub version: ", hub.__version__)
print("LightGBM: ", lgb.__version__)
print("NLTK: ", nltk.__version__)
print("Spacy: ", spacy.__version__)
print("Seaborn: ", sns.__version__)


# In[4]:


# Lista completa dos pacotes e biliotecas utilizados no ambiente
get_ipython().system('pip list')


# In[5]:


# Exibindo a versão do Python nesse ambiente
from platform import python_version
print (python_version())


# ### Upload do dataset

# In[6]:


# Upload do dataset de treino
df = pd.read_csv('dataset_nlp.csv', encoding='latin-1')
print("Tamanho do dataset:", df.shape)
df.head()


# In[7]:


# Elimininando valores nulos e exibindo infos do dataset
df = df.dropna()
df.info()


# In[8]:


# Criando o dataset que será processado e utilizado no treino
df_prep = df.copy()
df_prep.head()


# ### Análise exploratória

# In[9]:


# Plot da distribuição das classes
plt.figure(figsize=(8,6))
sns.countplot(x='class',data=df, palette='RdBu_r')
plt.title('Distribuição de classes', fontsize=14)
plt.xlabel("Classificação", fontsize=12)
plt.ylabel("Quantidade", fontsize=12);


# In[10]:


# Percentual das classes
positive = df['class'][df['class']=='positiva'].count()/len(df['class'])
print('Percentual da classe positiva: %0.3f' % positive)

negative = df['class'][df['class']=='negativa'].count()/len(df['class'])
print('Percentual da classe negativa: %0.3f' % negative)


# In[11]:


# Contagem de palavras
df['count_words'] = df['sentence'].apply(lambda x:len(x.split(" ")))

# Contagem de letras
df['count_letters'] = df['sentence'].apply(lambda x:len(x))

# Exibindo a soma de palavras e letras
print('Quantidade total de palavras:', df['count_words'].sum())
print('Quantidade total de letras:', df['count_letters'].sum())


# In[12]:


# Exibindo as estatísticas da coluna de contagem de palavras
df['count_words'].describe()


# In[13]:


# Exibindo a mediana da coluna de contagem de palavras
df['count_words'].median()


# In[14]:


# Exibindo o tamanho da maior e menor sentença
longest = df_prep.sentence.str.len().max()
shortest = df_prep.sentence.str.len().min()
print("Tamanho da maior sentença:", longest)
print("Tamanho da menor sentença:", shortest)


# In[15]:


# Histograma de frequência de quantidade de palavras
plt.subplots(figsize=(10,6))
sns.distplot(df.count_words, color="dodgerblue", label="Contagem de Palavras", kde=False)
sns.distplot(df.count_letters, color="steelblue", label="Contagem de Letras", kde=False)

plt.title("Distribuição de Contagem de Palavras e Letras", fontsize=16)
plt.xlabel("     ", fontsize=12)
plt.ylabel("Quantidade", fontsize=12)
plt.legend(fontsize=11);


# In[16]:


# Fistograma de frequência de quantidade de letras
sns.displot(df.count_letters, color="steelblue", label="Contagem de Letras", kde=True)
plt.title("Distribuição de Contagem de Letras", fontsize=14)
plt.xlabel("Contagem de Letras")
plt.ylabel("Quantidade");


# In[17]:


# Histograma de frequência de quantidade de palavras
sns.displot(df.count_words, color="dodgerblue", label="Contagem de Palavras", kde=True)
plt.title("Distribuição de Contagem de Palavras", fontsize=14)
plt.xlabel("Contagem de Palavras")
plt.ylabel("Quantidade");


# In[18]:


# Exibindo o boxplot de palavras por classe
plt.figure(figsize=(10,5))
plt.title("Boxplot palavras por classe", fontsize=14)
sns.boxplot(y="class", x="count_words", data = df, orient="h", palette = 'ch:6.5,-.4,dark=.5')
plt.xlabel("Quantidade de Palavras", fontsize=12)
plt.ylabel("Classe", fontsize=12);


# In[19]:


# Aplicando stop words e exibindo as palavras mais frequentes
stop_words = nltk.corpus.stopwords.words('portuguese')
df['clean'] = hero.remove_stopwords(df_prep['sentence'], stop_words)

freq_words = hero.top_words(df['clean'], normalize=False)[:20]

plt.figure(figsize=(10,5))
plt.title("Palavras mais frequentes", fontsize=14)
freq_words.plot(kind='barh', color='steelblue')
plt.xlabel("Quantidade", fontsize=12);
plt.show()


# In[20]:


# Exibindo principais palavras relacionadas à classe positiva
hero.visualization.wordcloud(df[df['class'] == 'positiva']['clean'], background_color='white')


# In[21]:


# Exibindo principais palavras relacionadas à classe negativa
hero.visualization.wordcloud(df[df['class'] == 'negativa']['clean'], background_color='lightblue');


# In[22]:


# Diferença de stemização entre palavras com e sem acento
stemmer = PortugueseStemmer()
print(stemmer.stem('informacao'))
print(stemmer.stem('informação'))


# ### Pré-processamento dos dados

# In[23]:


stop_words = nltk.corpus.stopwords.words('portuguese')
# removing stop_words
stop_words = set(stop_words)
stop_words.update(['dia','q','pra','pro','vc','vcs'])
stop_words


# In[24]:


# Pipeline de pré-processamento
prep_pipeline = [preprocessing.fillna,
                 preprocessing.lowercase,
                 preprocessing.remove_whitespace,
                 preprocessing.remove_digits,
                 preprocessing.remove_punctuation,
                 preprocessing.remove_diacritics]


# In[25]:


# Aplicando o pipeline de pré-processamento dos textos
df_prep['sentence'] = hero.preprocessing.remove_stopwords(df_prep['sentence'], stop_words)
df_prep['sentence'] = hero.preprocessing.clean(df_prep['sentence'], prep_pipeline)
df_prep


# In[26]:


# Criação da coluna com a variável resposta em formato numérico
target = {'positiva': 0, 'negativa': 1}
df_prep['target'] = df_prep['class'].map(target)
df_prep


# In[27]:


# Excluindo a coluna 'class' agora que já existe a informação do target em formato int
df_prep = df_prep.drop('class', axis=1)
df_prep


# In[28]:


# Salvando o dataframe com as tranformações do pré-rpocessamento
print('Salvando o arquivo com os elementos textuais pré-processsados...')
df_prep.to_csv('dataset_prep.csv', index=False)
print('Ok!')


# In[29]:


# carregando a base de dados já pré-processada
df_prep = pd.read_csv('dataset_prep.csv')
df_prep.head()


# ### Modelagem NLP

# In[30]:


# Vetorização dos elementos textuais
# Unigrama + CountVectorizer
vectorizer_cv = CountVectorizer(ngram_range=(1,1),stop_words=stop_words)
vectorizer_cv.fit(df_prep.sentence)
cv_unigram = vectorizer_cv.transform(df_prep.sentence)
cv_unigram


# In[31]:


# Exibindo Bag of Words
# Unigrama + CountVectorizer
cv_unigram = pd.DataFrame(cv_unigram.toarray(), columns=vectorizer_cv.get_feature_names())
print('BoW: Unigrama + CountVectorizer')
print(cv_unigram.head(), cv_unigram.shape)


# In[32]:


# Bigrama + CountVectorizer
vectorizer_cv2 = CountVectorizer(ngram_range=(2,2), stop_words=stop_words)
vectorizer_cv2.fit(df_prep.sentence)
cv_bigram = vectorizer_cv2.transform(df_prep.sentence)
cv_bigram


# In[33]:


# Exibindo Bag of Words
# Bigrama + CountVectorizer
cv_bigram = pd.DataFrame(cv_bigram.toarray(), columns=vectorizer_cv2.get_feature_names())
print('BoW: Bigrama + CountVectorizer')
print(cv_bigram.head(), cv_bigram.shape)


# In[34]:


# Unigrama + TF-IDF Vectorizer
vectorizer_tf = TfidfVectorizer(analyzer='word', use_idf=True, ngram_range=(1, 1), stop_words=stop_words)
vectorizer_tf.fit(df_prep.sentence)
tf_unigram = vectorizer_tf.transform(df_prep.sentence)
tf_unigram


# In[35]:


# Exibindo Bag of Words
# Unigrama + TF-IDF
tf_unigram = pd.DataFrame(tf_unigram.toarray(), columns=vectorizer_tf.get_feature_names())
print('BoW: Unigrama + TF-IDF')
print(tf_unigram.head(), tf_unigram.shape)


# In[36]:


# Bigrama + TF-IDF Vectorizer
vectorizer_tf2 = TfidfVectorizer(analyzer='word', use_idf=True, ngram_range=(2, 2), stop_words=stop_words)
vectorizer_tf2.fit(df_prep.sentence)
tf_bigram = vectorizer_tf2.transform(df_prep.sentence)
tf_bigram


# In[37]:


# Exibindo Bag of Words
# Bigrama + TF-IDF
tf_bigram = pd.DataFrame(tf_bigram.toarray(), columns=vectorizer_tf2.get_feature_names())
print('Bigrama + TF-IDF')
print(tf_bigram.head(), tf_bigram.shape)


# ### Técnicas de redução de dimensionalidade

# In[38]:


# Aplicando SVD para verificar a necessidade de redução de dimensionalidade 
# Unigrama + CountVectorizer
svd = TruncatedSVD(n_components=50, n_iter=20, random_state=seed_value)
svd.fit(cv_unigram)
cv_unigram_svd = svd.transform(cv_unigram)

print('Explained Variance Ratio:')
print(svd.explained_variance_ratio_)
print('\n')
print('Explained Variance Ratio Acumulado:')
print(svd.explained_variance_ratio_.sum())


# In[39]:


# Exibindo plot da maior variância percentual do número de componentes do SVD
# Unigrama + BoW
def svd_plot(svd):
    """Plota a variância dos componentes do svd com seaborn;
    A variância é exibida em forma percentual
    """
    sns.scatterplot(y=svd.explained_variance_ratio_, 
                    x=np.arange(len(svd.explained_variance_ratio_)))
    plt.title("Análise SVD -  BoW + Unigrama", fontsize=14)
    plt.xlabel("Componentes", fontsize=12)
    plt.ylabel("% Variância explicada", fontsize=12)
    plt.show()
svd_plot(svd)


# In[40]:


# Plotando os 2 principais componentes do SVD
svd = svd.transform(cv_unigram)
plt.figure(figsize=(12,8))
plt.title('SVD 2 componentes principais', fontsize=15)
plt.xlabel('Componente 1', fontsize=12)
plt.ylabel('Componente 2', fontsize=12)
sns.scatterplot(x=svd[:, 0], y=svd[:, 1], hue=df_prep['target'], palette=sns.color_palette("hls", 2), alpha=0.9);


# In[41]:


# Bigrama + CountVectorizer
svd_cv_bigram = TruncatedSVD(n_components=50, n_iter=40, random_state=seed_value)
svd_cv_bigram.fit(cv_bigram)
cv_bigram_svd = svd_cv_bigram.transform(cv_bigram)

print('Explained Variance Ratio:')
print(svd_cv_bigram.explained_variance_ratio_)
print('\n')
print('Explained Variance Ratio Acumulado:')
print(svd_cv_bigram.explained_variance_ratio_.sum())


# In[42]:


# Unigrama + TF-IDF
svd_tf_unigram = TruncatedSVD(n_components=50, n_iter=20, random_state=seed_value)
svd_tf_unigram.fit(tf_unigram)
tf_unigram_svd = svd_tf_unigram.transform(tf_unigram)

print('Explained Variance Ratio:')
print(svd_tf_unigram.explained_variance_ratio_)
print('\n')
print('Explained Variance Ratio Acumulado:')
print(svd_tf_unigram.explained_variance_ratio_.sum())


# In[43]:


# Bigrama + CountVectorizer
svd_tf_bigram = TruncatedSVD(n_components=50, n_iter=20, random_state=seed_value)
svd_tf_bigram.fit(tf_bigram)
tf_bigram_svd = svd_tf_bigram.transform(tf_bigram)

print('Explained Variance Ratio:')
print(svd_tf_bigram.explained_variance_ratio_)
print('\n')
print('Explained Variance Ratio Acumulado:')
print(svd_tf_bigram.explained_variance_ratio_.sum())


# In[44]:


cv_unigram_svd.shape


# ### Treinando os classificadores

# In[45]:


# Aplicando hold-out para separar as bases de treino e teste para cada método de vetorização

# Unigrama + CountVectorizer
X_trainUCV, X_testUCV, y_trainUCV, y_testUCV = train_test_split(cv_unigram, df_prep.target, test_size=0.2, random_state=seed_value)

# Unigrama + TF-IDF Vectorizer
X_trainUIDF, X_testUIDF, y_trainUIDF, y_testUIDF = train_test_split(cv_bigram, df_prep.target, test_size=0.2, random_state=seed_value)

# Bigrama + CountVectorizer
X_trainBCV, X_testBCV, y_trainBCV, y_testBCV = train_test_split(tf_unigram, df_prep.target, test_size=0.2, random_state=seed_value)

# Bigrama + TF-IDF Vectorizer
X_trainBIDF, X_testBIDF, y_trainBIDF, y_testBIDF = train_test_split(tf_bigram, df_prep.target, test_size=0.2, random_state=seed_value)


# In[46]:


# Validação cruzada estratificada com Stratifield Kfold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed_value)
skf


# In[47]:


# Decision Tree Classifier

starttime = timeit.default_timer()
print('Treinando Decision Tree Classifier...')
tree_model = DecisionTreeClassifier(random_state=seed_value)

print('Exibindo acurácia na base de teste com Stratified Kfold = 10...')

# Unigrama + CountVectorizer
tree_model.fit(X_trainUCV, y_trainUCV)
y_predictionUCV = tree_model.predict(X_testUCV)
resultsUCV_tree = cross_validate(tree_model, X=X_testUCV, y=y_testUCV, cv=skf)
print('Acurária Unigrama + CountVectorizer: %f (%f)' %(resultsUCV_tree['test_score'].mean(), 
                                                       resultsUCV_tree['test_score'].std()))

# Unigrama + TF-IDF Vectorizer
tree_model.fit(X_trainUIDF, y_trainUIDF)
y_predictionUIDF = tree_model.predict(X_testUIDF)
resultsUIDF_tree = cross_validate(tree_model, X=X_testUIDF, y=y_testUIDF, cv=skf)
print('Acurária Unigrama + TF-IDF Vectorizer: %f (%f)' %(resultsUIDF_tree['test_score'].mean(), 
                                                         resultsUIDF_tree['test_score'].std()))

# Bigrama + CountVectorizer
tree_model.fit(X_trainBCV, y_trainBCV)
y_predictionBCV = tree_model.predict(X_testBCV)
resultsBCV_tree  = cross_validate(tree_model, X=X_testBCV, y=y_testBCV, cv=skf)
print('Acurária Bigrama + CountVectorizer: %f (%f)' %(resultsBCV_tree['test_score'].mean(), 
                                                      resultsBCV_tree['test_score'].std()))

# Bigrama + TF-IDF Vectorizer
tree_model.fit(X_trainBIDF, y_trainBIDF)
y_predictionBIDF = tree_model.predict(X_testBIDF)
resultsBIDF_tree  = cross_validate(tree_model, X=X_testBIDF, y=y_testBIDF, cv=skf)
print('Acurária Bigrama + TF-IDF Vectorizer: %f (%f)' %(resultsBIDF_tree['test_score'].mean(), 
                                                        resultsBIDF_tree['test_score'].std()))

print('Ok!')
print("Total time:", timeit.default_timer() - starttime)


# In[48]:


# Naive Bayes Multinomial

starttime = timeit.default_timer()
print('Treinando Multinomial Naive Bayes...')
nb_model = MultinomialNB()

print('Exibindo acurácia na base de teste com Stratified Kfold = 10...')

# Unigrama + CountVectorizer
nb_model.fit(X_trainUCV, y_trainUCV)
y_predictionUCV = nb_model.predict(X_testUCV)
resultsUCV_nb = cross_validate(nb_model, X=X_testUCV, y=y_testUCV, cv=skf)
print('Acurária Unigrama + CountVectorizer: %f (%f)' %(resultsUCV_nb['test_score'].mean(), 
                                                       resultsUCV_nb['test_score'].std()))

# Unigrama + TF-IDF Vectorizer
nb_model.fit(X_trainUIDF, y_trainUIDF)
y_predictionUIDF = nb_model.predict(X_testUIDF)
resultsUIDF_nb  = cross_validate(nb_model, X=X_testUIDF, y=y_testUIDF, cv=skf)
print('Acurária Unigrama + TF-IDF Vectorizer: %f (%f)' %(resultsUIDF_nb['test_score'].mean(), 
                                                         resultsUIDF_nb['test_score'].std()))

# Bigrama + CountVectorizer
nb_model.fit(X_trainBCV, y_trainBCV)
y_predictionBCV = nb_model.predict(X_testBCV)
resultsBCV_nb = cross_validate(nb_model, X=X_testBCV, y=y_testBCV, cv=skf)
print('Acurária Bigrama + CountVectorizer: %f (%f)' %(resultsBCV_nb['test_score'].mean(), 
                                                      resultsBCV_nb['test_score'].std()))

# Bigrama + TF-IDF Vectorizer
nb_model.fit(X_trainBIDF, y_trainBIDF)
y_predictionBIDF = nb_model.predict(X_testBIDF)
resultsBIDF_nb = cross_validate(nb_model, X=X_testBIDF, y=y_testBIDF, cv=skf)
print('Acurária Bigrama + TF-IDF Vectorizer: %f (%f)' %(resultsBIDF_nb['test_score'].mean(), 
                                                        resultsBIDF_nb['test_score'].std()))

print('Ok!')
print("Total time:", timeit.default_timer() - starttime)


# In[49]:


# Light Gradient Boost

starttime = timeit.default_timer()
print('Treinando Light Gradient Boost...')
lgbm_model = lgb.LGBMClassifier(num_leaves=25, 
                                n_estimators=150, 
                                max_depth=8,
                                n_jobs=-1, 
                                random_state=seed_value)

print('Exibindo acurácia na base de teste com Stratified Kfold = 10...')

# Unigrama + CountVectorizer
lgbm_model.fit(X_trainUCV.astype(float), y_trainUCV.astype(float))
y_predictionUCV = lgbm_model.predict(X_testUCV.astype(float))
resultsUCV_lgbm = cross_validate(lgbm_model, X=X_testUCV, y=y_testUCV, cv=skf)
print('Acurária Unigrama + CountVectorizer: %f (%f)' %(resultsUCV_lgbm['test_score'].mean(), 
                                                       resultsUCV_lgbm['test_score'].std()))

# Unigrama + TF-IDF Vectorizer
lgbm_model.fit(X_trainUIDF.astype(float), y_trainUIDF.astype(float))
y_predictionUIDF = lgbm_model.predict(X_testUIDF.astype(float))
resultsUIDF_lgbm  = cross_validate(lgbm_model, X=X_testUIDF, y=y_testUIDF, cv=skf)
print('Acurária Unigrama + TF-IDF Vectorizer: %f (%f)' %(resultsUIDF_lgbm['test_score'].mean(), 
                                                         resultsUIDF_lgbm['test_score'].std()))

# Bigrama + CountVectorizer
lgbm_model.fit(X_trainBCV.astype(float), y_trainBCV.astype(float))
y_predictionBCV = lgbm_model.predict(X_testBCV.astype(float))
resultsBCV_lgbm  = cross_validate(lgbm_model, X=X_testBCV, y=y_testBCV, cv=skf)
print('Acurária Bigrama + CountVectorizer: %f (%f)' %(resultsBCV_lgbm['test_score'].mean(), 
                                                      resultsBCV_lgbm['test_score'].std()))

# Bigrama + TF-IDF Vectorizer
lgbm_model.fit(X_trainBIDF.astype(float), y_trainBIDF.astype(float))
y_predictionBIDF = lgbm_model.predict(X_testBIDF.astype(float))
resultsBIDF_lgbm  = cross_validate(lgbm_model, X=X_testBIDF, y=y_testBIDF, cv=skf)
print('Acurária Bigrama + TF-IDF Vectorizer: %f (%f)' %(resultsBIDF_lgbm['test_score'].mean(), 
                                                        resultsBIDF_lgbm['test_score'].std()))

print('Ok!')
print("Total time:", timeit.default_timer() - starttime)


# In[50]:


dict_results = {'UCV_tree':resultsUCV_tree['test_score'].mean(),
               'UIDF_tree':resultsUIDF_tree['test_score'].mean(),
               'BCV_tree':resultsBCV_tree['test_score'].mean(),
               'BIDF_tree':resultsBIDF_tree['test_score'].mean(),
               'UCV_nb':resultsUCV_nb['test_score'].mean(),
               'UIDF_nb':resultsUIDF_nb['test_score'].mean(),
               'BCV_nb':resultsBCV_nb['test_score'].mean(),
               'BIDF_nb':resultsBIDF_nb['test_score'].mean(),
               'UCV_lgbm':resultsUCV_lgbm['test_score'].mean(),
               'UIDF_lgbm':resultsUIDF_lgbm['test_score'].mean(),
               'BCV_lgbm':resultsBCV_lgbm['test_score'].mean(),
               'BIDF_lgbm':resultsBIDF_lgbm['test_score'].mean()}

dict_results


# In[51]:


# criando um data frame para armazenar os resultados do dicionário
df_results = pd.DataFrame.from_dict(dict_results, orient='index')
df_results.reset_index(inplace=True)
df_results.columns=['Modelo', 'Acurácia Média']
df_results = df_results.sort_values(by='Acurácia Média', ascending=False)


# In[52]:


# exibindo o gráfico com a comparação entre acurácia média entre modelos c/ validação cruzada
ax = sns.barplot(x="Acurácia Média", y="Modelo", data=df_results,
                 palette="Blues_d")
ax.set_title('Ranking acurácia com validação cruzada', fontsize=18);


# In[53]:


# teste de levene para igualdade de variância entre as amostras
modelo1 = resultsUCV_nb['test_score']
modelo2 = resultsBCV_tree['test_score']
stat, p = levene(modelo1, modelo2)
print('Comparação de igualdade de variância os 2 modelos mais acurados...')
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Provavelmente as variâncias são iguais')
else:
    print('Provavelmente as variâncias são diferentes')


# In[54]:


# Example of the Shapiro-Wilk Normality Test
from scipy.stats import shapiro

stat, p = shapiro(modelo1)
print('Verificando a normalidade das amostras: modelo1...')
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Provavelmente é normal')
else:
    print('Provavelmente não é normal')
    
print('\n')
    
stat, p = shapiro(modelo2)
print('Verificando a normalidade das amostras: modelo2...')
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Provavelmente é normal')
else:
    print('Provavelmente não é normal')


# In[55]:


# teste de levene para igualdade de variância entre as amostras
stat, p = ttest_ind(modelo1, modelo2)
print('Comparação de igualdade das médias entre os 2 modelos mais acurados...')
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Provavelmente as médias são iguais')
else:
    print('Provavelmente as médias são diferentes')


# In[56]:


# teste Bayesiano hierárquico para comparação de modelos
print('Teste Bayesiano Hierárquico')
print('Probabilidade do modelo1 ser melhor do que o modelo2...')
print(baycomp.two_on_single(modelo1, modelo2))


# In[57]:


# avaliando a possibilidade de igualdade dos modelos 
names = ("UCV_nb", "BCV_tree")
probs, fig = baycomp.two_on_single(modelo1, modelo2, rope=0.1, plot=True, names=names)
print(probs)
plt.title('Distribuição da probabilidade de igualdade dos modelos', fontsize=13)
fig.show();


# ### Redes Neurais

# In[58]:


df_prep


# In[59]:


# Tokenização de palavras
frases = df_prep['sentence']
tokenizer = Tokenizer(oov_token='<OOV>') #definição de token para palavras não existentes no vocabulário de treino
tokens = tokenizer.fit_on_texts(frases)
vocab = tokenizer.word_index
print(vocab)


# In[60]:


# Exibindo os vetores pós-tokenização de palavras
tokens_frases = tokenizer.texts_to_sequences(frases)
tokens_frases


# In[61]:


# Aplicando pad_sequences na sequência recém transformada via tokenização
frases_tokens = pad_sequences(tokens_frases, 
                              padding='post', 
                              truncating='post',
                              maxlen=30)
print(frases_tokens)


# In[62]:


# Separando as bases de treino, teste e validação
X_train_pad, X_test_pad, y_train_pad, y_test_pad = train_test_split(frases_tokens, df_prep['target'], test_size=0.2, random_state=42)

# Bases de treino parcial + validação
# Para garantir a consistência dos resultados antes de validar nas bases teste
X_val_pad = X_train_pad[:250]
partial_X_train = X_train_pad[250:]

y_val_pad = y_train_pad[:250]
partial_y_train = y_train_pad[250:]


# In[63]:


# Transformação do formato dos dataframes em arrays
partial_X_train = np.array(partial_X_train)
partial_y_train = np.array(partial_y_train)

X_val_pad = np.array(X_val_pad)
y_val_pad = np.array(y_val_pad)

X_test_pad = np.array(X_test_pad)
y_test_pad = np.array(y_test_pad)


# In[64]:


# Definição do número de neurônios da camada embedding e do vetor de palavras vocabulário
embedding_dim = 256
vocab_size = 6500 #selecionando as 6000 palavras mais frequentes do vocabulário


# In[65]:


# Definindo a função do scheduler com decaimento exponencial a partir da época 5
def scheduler(epoch, lr):
    print("Learning rate atual: ", lr)
    if epoch <= 4:
        return lr
    else:
        return np.clip(lr * tf.math.exp(-0.4), 0.0001, 0.001)
    
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)


# In[66]:


# Primeira arquitetura
# Camada Embedding + GAP 1D + Dense

model1 = Sequential([
  layers.Embedding(vocab_size, embedding_dim),
  layers.GlobalAveragePooling1D(),
  layers.Dense(32, activation='relu'),
  layers.Dense(1, activation='sigmoid')
], name='model_emb1')

model1.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy',
                       'AUC',
                       'binary_accuracy'])

model1.summary()


# In[67]:


# Fit do modelo
starttime = timeit.default_timer()
history_pad1 = model1.fit(partial_X_train, 
                        partial_y_train,
                        batch_size=64,
                        epochs=50,
                        validation_data=(X_val_pad, y_val_pad), 
                        verbose=1,
                        callbacks=[callback])
print('Ok!')
print("Total time:", timeit.default_timer() - starttime)


# In[68]:


# Validando na base de teste
results1 = model1.evaluate(X_test_pad,  y_test_pad, verbose=2)


# In[69]:


history_dict = history_pad1.history
history_dict.keys()

# Exibindo gráfico de Loss
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Model1 - Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[70]:


# Exibindo gráfico de Accurácia
plt.clf() 

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# In[71]:


# Segunda arquitetura
# Camada Embedding + GRU + Dense

model2 = Sequential([
  layers.Embedding(vocab_size, embedding_dim, mask_zero=True),
  layers.GRU(32),
  layers.Dense(32, activation='relu'),
  layers.Dense(16, activation='softmax'),
  layers.Dense(16, activation='relu'),
  layers.Dense(1, activation='sigmoid')
], name='model_emb2')

model2.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy',
                       'AUC',
                       'binary_accuracy'])

model2.summary()


# In[73]:


# Fit do modelo
history_pad2 = model2.fit(partial_X_train, 
                          partial_y_train,
                          batch_size=64,
                          epochs=50,
                          validation_data=(X_val_pad, y_val_pad), 
                          verbose=1,
                          callbacks=[callback])
print('Ok!')
print("Total time:", timeit.default_timer() - starttime)


# In[74]:


# Validando na base de teste
results2 = model2.evaluate(X_test_pad,  y_test_pad, verbose=2)


# In[75]:


history_dict = history_pad2.history
history_dict.keys()

# Exibindo gráfico de Loss
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Model2 - Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[76]:


# Exibindo gráfico de Accurácia
plt.clf() 

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Model2 - Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# In[77]:


# Terceira arquitetura
# Camada Embedding + SimpleRNN + Dense
model3 = Sequential([
  layers.Embedding(vocab_size, embedding_dim, mask_zero=True),
  layers.SimpleRNN(units=64),
  layers.Dense(32, activation='relu'),
  layers.Dense(16, activation='softmax'),
  layers.Dense(16, activation='relu'),
  layers.Dense(1, activation='sigmoid')
], name='model_emb3')

model3.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy',
                       'AUC',
                       'binary_accuracy'])

model3.summary()


# In[78]:


# Fit do Modelo
starttime = timeit.default_timer()
history_pad3 = model3.fit(partial_X_train, 
                        partial_y_train,
                        batch_size=64,
                        epochs=50,
                        validation_data=(X_val_pad, y_val_pad), 
                        verbose=1,
                        callbacks=[callback])
print('Ok!')
print("Total time:", timeit.default_timer() - starttime)


# In[79]:


# Validando na base de teste
results3 = model3.evaluate(X_test_pad,  y_test_pad, verbose=2)


# In[80]:


history_dict = history_pad3.history
history_dict.keys()

# Exibindo gráfico de Loss
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Model3 - Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[81]:


# Exibindo gráfico de Accurácia
plt.clf() 

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Model3 - Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# In[82]:


# Quarta arquitetura
# Camada Embedding + Bidirectional Layer LSTM + Dense
model4 = Sequential([
  layers.Embedding(vocab_size, embedding_dim, mask_zero=True),
  layers.Bidirectional(layers.LSTM(32)),
  layers.Dense(32, activation='relu'),
  layers.Dense(16, activation='softmax'),
  layers.Dense(16, activation='relu'),
  layers.Dense(1, activation='sigmoid')
], name='model_emb4')

model4.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy',
                       'AUC',
                       'binary_accuracy'])

model4.summary()


# In[83]:


# Fit do modelo
starttime = timeit.default_timer()
history_pad4 = model4.fit(partial_X_train, 
                        partial_y_train,
                        batch_size=64,
                        epochs=50,
                        validation_data=(X_val_pad, y_val_pad), 
                        verbose=1,
                        callbacks=[callback])
print('Ok!')
print("Total time:", timeit.default_timer() - starttime)


# In[84]:


# Validando na base de teste
results4 = model4.evaluate(X_test_pad, y_test_pad, verbose=2)


# In[85]:


history_dict = history_pad4.history
history_dict.keys()

# Exibindo gráfico de Loss
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Model4 - Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[86]:


# Exibindo gráfico de Accurácia
plt.clf() 

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Model4 - Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.legend()

plt.show()


# In[87]:


# Quinta arquitetura
# Embedding + 2 camadas LSTM + Dense
model5 = Sequential([
  layers.Embedding(vocab_size, embedding_dim, mask_zero=True),
  layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
  layers.Bidirectional(layers.LSTM(16)),
  layers.Dense(32, activation='relu'),
  layers.Dense(16, activation='softmax'),
  layers.Dense(16, activation='relu'),
  layers.Dense(1, activation='sigmoid')
], name='model_emb5')

model5.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy',
                       'AUC',
                       'binary_accuracy'])

model5.summary()


# In[88]:


# Fit do modelo
starttime = timeit.default_timer()
history_pad5 = model5.fit(partial_X_train, 
                        partial_y_train,
                        batch_size=32,
                        epochs=50,
                        validation_data=(X_val_pad, y_val_pad), 
                        verbose=1,
                        callbacks=[callback])
print('Ok!')
print("Total time:", timeit.default_timer() - starttime)


# In[97]:


# Validando na base de teste
results5 = model5.evaluate(X_test_pad, y_test_pad, verbose=2)


# In[100]:


history_dict = history_pad5.history
history_dict.keys()

# Exibindo gráfico de Loss
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Model5 - Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[101]:


# Exibindo gráfico de Accurácia
plt.clf() 

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Model5 - Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# ### Avaliação de resultados: Teste Bayesiano Hierárquico 

# In[102]:


results1
#loss: 0.4462 - accuracy: 0.8204 - auc: 0.8192 - binary_accuracy: 0.8204


# In[103]:


dict_results = {'modelo1':results1,
               'modelo2':results2,
               'modelo3':results3,
               'modelo4':results4,
               'modelo5':results5}

dict_results


# In[104]:


# criando um data frame para armazenar os resultados do dicionário
df_results = pd.DataFrame.from_dict(dict_results, orient='index')
df_results.reset_index(inplace=True)
df_results.columns=['Modelo', 'Loss', 'Acurácia', 'AUC', 'Acurácia Binária']
df_results = df_results.sort_values(by='Acurácia', ascending=False)
df_results


# In[105]:


# exibindo o gráfico com a comparação entre acurácia média entre modelos
ax = sns.barplot(x=df_results["Acurácia"], y="Modelo", data=df_results,
                 palette="Blues_d")
ax.set_title('Ranking acurácia', fontsize=18);


# In[106]:


# exibindo o gráfico com a comparação entre acurácia média entre modelos
df_results = df_results.sort_values(by='Loss')
ax = sns.barplot(x=df_results["Loss"], y="Modelo", data=df_results,
                 palette="Blues")
ax.set_title('Ranking Loss', fontsize=18);


# In[ ]:




