import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# Load data
data = pd.read_csv("MarvellousInfosystems_PlayPredictor.csv")

# print(data.head())

# data['Play'].unique()

# Clean, Prepare and manipulate data
feature_nm=['Whether','Temperature']

print("Feture name",feature_nm)

# Creating labelEncoder
label_encoder = preprocessing.LabelEncoder()

data['Play'] = label_encoder.fit_transform(data['Play'])
data['Wether'] = label_encoder.fit_transform(data['Wether'])
data['Temperature'] = label_encoder.fit_transform(data['Temperature'])

#print(data['Play'].unique())

#print(data.head())

# Encode labels in column 'species'.
# df['species'] = label_encoder.fit_transform(df['species'])

# df['species'].unique()

# Combining weather and temp into single listof tuples
features=list(zip(data['Wether'],data['Temperature']))

#data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.5)

# train data
classifier = KNeighborsClassifier(n_neighbors=3)

classifier.fit(features, data['Play'])

# Test data data_test
predictions = classifier.predict([[0,2]]) # 0:Overcast, 2: Mild
print(predictions)

#Accuracy = accuracy_score(target_test, predictions)

#print("Accuracy", Accuracy)
