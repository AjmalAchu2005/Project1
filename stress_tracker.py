import pandas as pd
import matplotlib.pyplot as pt
import sklearn.linear_model as lm
mydata=pd.read_csv("work_stress.csv")
x=mydata[["Work_Hours"]]
y=mydata[["Stress_Level"]]

pt.scatter(x,y)
pt.show()
model=lm.LinearRegression()
model.fit(x,y)
print(model.predict([[9]]))