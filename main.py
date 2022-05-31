import pandas as pd

data = pd.read_excel('crop_csv_file.xlsx')

data.head()

data.head()

data.head()

data.info()

data = data.dropna()
data.info()

data.describe()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

State_Name = le.fit_transform(data.State_Name)
District_Name = le.fit_transform(data.District_Name)
Crop_Year = le.fit_transform(data.Crop_Year)
crop = le.fit_transform(data.Crop)
Season = le.fit_transform(data.Season)
data['State_Name'] = State_Name
data['District_Name'] = District_Name
data['Crop_Year'] = Crop_Year
data['Crop'] = crop
data['Season']  = Season

data.head()



data

print(data)

from sklearn.model_selection import train_test_split

X = data.iloc[:,:-1]
y = data.iloc[:,-1]

X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2,random_state=100)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score , classification_report, mean_squared_error, r2_score
forest = RandomForestRegressor(n_estimators=1000, 
                               criterion='mse', 
                               random_state=1, 
                               n_jobs=-1)
forest.fit(X_train, Y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(Y_train, y_train_pred),
        mean_squared_error(Y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(Y_train, y_train_pred),
        r2_score(Y_test, y_test_pred)))

print(forest.score(X_test,Y_test))

forest.predict(X_test)

X_test

forest.predict([[1,5,5,3,40,37,40,46,1359.0]])

state = input('enter state:')
district = input('enter district:')
year = input('enter year:')
season = input('enter season:')
crop = input('enter crop:')
Temperature = input('enter Temperature')
humidity= input('enter humidity')
soilmoisture= input('enter soilmoisture')
area = input('enter area')

out_1 = forest.predict([[float(state),
       float(district),
       float(year),
       float(season),
       float(crop),
       float(Temperature),
       float(humidity),
       float(soilmoisture),
       float(area)]])
print(out_1)
print('crop yield Production:',out_1)

import pickle

pickle.dump(forest, open('crop.pkl', 'wb'))

model = pickle.load(open('crop.pkl', 'rb'))
