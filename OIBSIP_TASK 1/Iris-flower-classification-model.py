import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.svm import SVC
import pickle

iris=datasets.load_iris()
print(iris)

X=iris.data
y=iris.target

sns.boxplot(x=iris.target ,y=iris.data[:,0])
plt.show()

x_train,x_test,y_train,y_test=train_test_split(X,y)

#Iris Classification using Linear Regression
lin_reg=LinearRegression()
lin_reg=lin_reg.fit(x_train,y_train)
print(lin_reg.score(x_test,y_test))

#Iris Classification using Logistic Regression
log_reg=LogisticRegression()
log_reg=log_reg.fit(x_train,y_train)
print(log_reg.score(x_test,y_test))

#Iris Classification using Support Vector Classifier
svc_model=SVC()
svc_model=svc_model.fit(x_train,y_train)
print(svc_model.score(x_test,y_test))

pickle.dump(lin_reg,open('lin_model.pkl','wb'))
pickle.dump(log_reg,open('log_model.pkl','wb'))
pickle.dump(svc_model,open('svc_model.pkl','wb'))
