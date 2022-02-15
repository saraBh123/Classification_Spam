# Classification de spam SMS (Naive Bayes, DT, RF)
### Table des matières ###
1. Aperçu
2. Importation de bibliothèques
3. Importation de données
4. Nettoyage
5. Visualisation
6. Application de modèles
7. Précision finale

## Aperçu ##
La collecte de spam SMS est un ensemble de messages SMS étiquetés qui ont été collectés pour la recherche de spam SMS. Il contient un ensemble de messages SMS en anglais de 5 574 messages, étiquetés selon qu'ils sont du jambon (légitime) ou du spam.
## Importation de bibliothèques ##

```
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import chardet
import re

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
```
## Importation de données ##
```
with open('spam.csv', 'rb') as f:
    result = chardet.detect(f.read())
df=pd.read_csv('spam.csv', encoding=result['encoding'])
df.head()
```
## Nettoyage ##
#### Préparation des données ####
afficher les numéros de lignes et de colonnes appartenant au dataset 
```
df.shape
```
les informations nécessaires sur l'ensemble de données
```
df.info()
```
somme de nombre nulls dans chaque colonne
```
df.isnull().sum()
```
suppression de colonnes contenant plusieurs valeurs nulles
```
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
```
renommez les deux colonnes et ajoutez une nouvelle colonne qui contient le nombre de caractères dans chaque ligne
```
df=df.rename(columns = {'v1': 'type', 'v2': 'message'})
df['length'] = df['message'].apply(len)
```
### Appliquer une expression régulière ###
* Remplacer les adresses e-mail par 'emailaddr'
* Remplacer les URL par 'httpaddr'
* Remplacez les symboles d'argent par 'moneysymb'
* Remplacer les numéros de téléphone par 'phonenumbr'
* Remplacer les nombres par 'numbr'
```
for i in range(0, 5572):
    df['message'][i] = re.sub('\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'emailaddr', df['message'][i])
    df['message'][i] = re.sub('(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'httpaddr', df['message'][i])
    df['message'][i] = re.sub('£|\$', 'moneysymb', df['message'][i])
    df['message'][i] = re.sub('\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b', 'phonenumbr', df['message'][i])
    df['message'][i] = re.sub('\d+(\.\d+)?', 'numbr', df['message'][i])
```
* Supprimer toutes les ponctuations
* Chaque mot en minuscule
```
for i in range(0, 5572):
    df['message'][i] = re.sub('[^\w\d\s]',' ', df['message'][i])
    df['message'][i]=df['message'][i].lower()
```
### Fractionner des mots à tokeniser ###
```
for i in range(0, 5572):
    df["message"][i] = df["message"][i].split() 
print(df["message"][2])  
```
### Stemming ###
```
ps = PorterStemmer()
for i in range(0, 5572):
    df["message"][i] = [ps.stem(word) for word in df["message"][i] if not word in set(stopwords.words('english'))]
```
### Préparer des messages avec des jetons restants ###
```
for i in range(0, 5572):
    df["message"][i] = ' '.join(df["message"][i])
```
### Préparation du corpus WordVector ###
```
corpus = []
for i in range(0, 5572):
    corpus.append(df["message"][i])
```
## Visualisation ##
...
## Application de modèles ##
```
# Remplacer ham par 0 et spam par 1
df = df.replace(['ham','spam'],[0, 1]) 
#Transformer les données en valeur numérique 
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
le = LabelEncoder()
y = df['type']
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test= train_test_split(X,y, test_size= 0.3, random_state=30)
```
### Naif Bayes Classifier ###
```
NBmodel=MultinomialNB()
NBmodel.fit(X_train, y_train)
predict=NBmodel.predict(X_test)
matrix = confusion_matrix(y_test, predict)
NBaccuracy = accuracy_score(y_test, predict)

print(matrix)
print("Accuracy Score:", NBaccuracy)
print(classification_report(y_test, predict))
```
### Random Forest ###
```
RFmodel= RandomForestClassifier()
RFmodel.fit(x_train, y_train)
RFmodel_predict=RFmodel.predict(x_test)
RFmatrix = confusion_matrix(y_test, RFmodel_predict)
RFaccuracy = accuracy_score(y_test, RFmodel_predict)

print(RFmatrix)
print("Accuracy Score:", RFaccuracy)
print(classification_report(y_test,RFmodel_predict))
```
### Decision Tree Classifier ###
```
DTmodel= DecisionTreeClassifier()
DTmodel.fit(x_train, y_train)
DTmodel_predict=DTmodel.predict(x_test)
DTmatrix = confusion_matrix(y_test, DTmodel_predict)
DTaccuracy = accuracy_score(y_test, DTmodel_predict)

print(DTmatrix)
print("Accuracy Score:", DTaccuracy)
print(classification_report(y_test,DTmodel_predict))
```
## Précision finale ##
```
print("---    Naif Bayes Classifier :     {:.2f}%".format(100 * NBaccuracy),"   ---")
print("---    Random Forest :             {:.2f}%".format(100 * RFaccuracy),"   ---")
print("---    Decision Tree Classifier :  {:.2f}%".format(100 * DTaccuracy),"   ---")
```
