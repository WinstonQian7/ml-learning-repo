#12/20/2019 Top50 Spotify Dataset (Top50.csv)
#Created by Winston Qian
from pandas import read_csv
from pandas.plotting import scatter_matrix
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
from sklearn.utils.multiclass import type_of_target 
#load file
path = "C:/Users/Lisa/Desktop/ML/basic/top50.csv"
names = []
data = read_csv(path, encoding='latin-1')
#separate data
array = data.values
X = array[:,4:]
y = array[:,3]
#type_of_target(y) is multiclass

#Train model
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state = 1)

models = []
models.append(('SVC', SVC(gamma='auto')))
models.append(('NB', GaussianNB()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))

results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1)
	cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring = 'accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s %f (%f)' % (names, cv_results.mean(), cv_results.std()))

