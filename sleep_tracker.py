import pandas as pd
import matplotlib.pyplot as pt
import sklearn.linear_model as lm
mydata=pd.read_csv("coffee_sleep.csv")
x=mydata[["Coffee_Cups"]]
y=mydata[["Sleep_Hours"]]

pt.scatter(x,y)
pt.show()
model=lm.LinearRegression()
model.fit(x,y)
print(model.predict([[3.5]]))