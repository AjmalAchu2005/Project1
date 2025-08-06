import pandas as pd
import matplotlib.pyplot as pt
import sklearn.linear_model as lm
mydata=pd.read_csv("steps_calories.csv")
x=mydata[["Steps"]]
y=mydata[["Calories"]]

pt.scatter(x,y)
pt.show()
model=lm.LinearRegression()
model.fit(x,y)
print(model.predict([[7000]]))