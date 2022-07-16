# Cargo las librerias

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px 
from dotenv import load_dotenv
from sqlalchemy import create_engine
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import cross_val_score


#load the .env file variables
load_dotenv()
connection_string = os.getenv('DATABASE_URL')
#print(connection_string)

# Cargo la data
df = pd.read_csv(connection_string, sep =',')

# Hago una copia
df_raw = df.copy()

# Veo la correlacion entre las variables
df_raw.corr()

#Defino las variables 
X = df_raw.drop('Outcome',axis=1)
y = df_raw['Outcome']

# Divido mis datos de entrenamiento y validacion
X_train,X_test, y_train,y_test = train_test_split(X,y,stratify=y, random_state=43)

#Creo el modelo
clf = DecisionTreeClassifier(criterion='entropy', random_state=0)

clf.fit(X_train,y_train)
print('Accuracy:',clf.score(X_test,y_test))

# Creo la prediccion
clf_pred = clf.predict(X_test)

# Traigo la matriz de confusion
cm = confusion_matrix(y_test, clf_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
display_labels=clf.classes_)
disp.plot()

# Hago print de los estimadores
print(classification_report(y_test, clf_pred))



