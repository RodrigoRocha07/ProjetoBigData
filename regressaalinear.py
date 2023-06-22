#importando bibliotecas
import pandas as pd
import sklearn.metrics as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#lendo arquivo excel
df = pd.read_excel("/content/tabelas de vendas 4.xlsx")

#transformando os elementos dinheiro, débito e crédito
df["Transações"] = df["Transações"].map({"Débito":0, "Dinheiro":1, "Crédito":2,})

#definindo as variáveis x e y
x = df[["Transações","Quantidade"]]
y = df[["Preço Total"]]

#dividindo as variáveis x e y em variáveis de treino e teste 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3,random_state = 42)

#treinando código
model = LinearRegression()
model.fit(x_train,y_train)

#printando a média de acertos
y_pred = model.predict(x_test)
mse = sm.mean_squared_error(y_test,y_pred)
r2 = sm.r2_score(y_test,y_pred)
print("mean squared error:" ,mse)
print("r2 score:" ,r2)