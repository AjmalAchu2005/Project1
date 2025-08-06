import pandas as pd
import matplotlib.pyplot as pt
import sklearn.linear_model as lm
mydata=pd.read_csv("internet_buffer.csv")
x=mydata[["Speed"]]
y=mydata[["Buffer_Time"]]

pt.scatter(x,y)
pt.show()
model=lm.LinearRegression()
model.fit(x,y)
print(model.predict([[9]]))