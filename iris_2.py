from numpy import argmax
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
file = read_csv(path, header=None)
x,y = file.values[:,:-1], file.values[:,-1]
x = x.astype('float32')
y = LabelEncoder().fit_transform(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33)
n_features = x_train.shape[1]

model = Sequential()
model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train,y_train,epochs=150,batch_size=32,verbose=0)

loss,acc = model.evaluate(x_test,y_test,verbose=1)
print('Test Acc: %.3f' % acc)

