import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt 

dataframe = pd.read_csv('challenge_dataset.txt', sep = ',', names=['X', 'Y'])
x = dataframe[["X"]]
y = dataframe[["Y"]]

body_reg = linear_model.LinearRegression()
body_reg.fit(x, y)


print(dataframe)
print(body_reg.predict(x))


plt.scatter(x, y)
plt.plot(x, body_reg.predict(x))
plt.show()

print("Coefficient of determination:")
print(body_reg.score(x, y))
