from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#Load dataset
path = "C:/Users/Lisa/Desktop/ML/basic/cwurData.csv"
#names = ['quality_of_education', 'alumni_employment', 'quality_of_faculty']
#'quality_of_faculty', 'publications', 'influence', 'patents']
data = read_csv(path)
 
#Seperate data
array = data.values
X = array[:,3:9]
y = array[:,3]
y = y.astype('int')
#Train data by spliting data 70-30

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state = 1)

#Test options with different types of eval methods
#Model!

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVC', SVC(gamma='auto')))
models.append(('CART', DecisionTreeClassifier()))
#evaluate model
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits = 5, random_state=1)
	cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')	
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

#Warnings minumum number of members in any class cannot be less than n_splits =3
#LR: 0.140192 (0.012774)
#LDA: 0.048495 (0.006069)
#KNN: 0.085939 (0.014791)
#NB: 0.868175 (0.088136)
#SVC: 0.065765 (0.010075)
#CART: 0.878298 (0.090832)
#Cart wins!

#Make prediction on validation dataset
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test,predictions))
