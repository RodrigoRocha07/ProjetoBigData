#importando bibliotecas 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

#lendo arquivo excel
df = pd.read_excel("/content/tabelas de vendas 4.xlsx")

#transformando os elementos dinheiro, débito e crédito
df["Transações"] = df["Transações"].map({"Dinheiro": 0,"Débito": 1, "Crédito": 2})

#definindo as variáveis x e y
y = df["Transações"]
x = df.drop(columns = ["Data","Cliente","Transações","Produto","ID"])

#dividindo as variáveis x e y em variáveis de treino e teste 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

#utilizando código de normalização para melhorar a acurácia
sca = preprocessing.MinMaxScaler()
cols = x.columns

x_train = sca.fit_transform(x_train)
x_train = pd.DataFrame(x_train, columns=cols)
x_test = sca.fit_transform(x_test)
x_test = pd.DataFrame(x_test, columns=cols)

#treinando código
decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)

#printando a média de acertos
resultado = decision_tree.predict(x_test)
print(metrics.classification_report(y_test, resultado))