from numpy import sqrt
from pandas import read_csv
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
file = read_csv(path,header=None)
x,y = file.values[:,:-1],file.values[:,-1]
x = x.astype('float32')
y = y.astype('float32')
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33)
n_features = x_train.shape[1]

model = Sequential()
model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')
model.fit(x_train,y_train,epochs=150,batch_size=32,verbose=1)
error = model.evaluate(x_test,y_test,verbose=0)
print('MSE: %.3f, RMSE: %.3f' % (error, sqrt(error)))

row = [0.00632,18.00,2.310,0,0.5380,6.5750,65.20,4.0900,1,296.0,15.30,396.90,4.98]
yhat = model.predict([row])
print('Predicted: %.3f' % yhat)