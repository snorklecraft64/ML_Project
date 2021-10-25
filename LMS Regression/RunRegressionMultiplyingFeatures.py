import sys
import numpy
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
numpy.set_printoptions(threshold=sys.maxsize)

X = []
y = []
with open('../data/train_final.csv', 'r') as f:
  for line in f:
    terms = line.strip().split(',')
    
    if terms[0] == 'age':
      continue
    
    list = []
    for i in range(len(terms)-1):
      try:
        list.append(int(terms[i]))
      except ValueError:
        list.append(terms[i])
    
    X.append(list)
    y.append(int(terms[len(terms)-1]))

categories = [['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked', '?'],
              ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool', '?'],
              ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse', '?'],
              ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces', '?'],
              ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried', '?'],
              ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black', '?'],
              ['Female', 'Male', '?'],
              ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands', '?']]
enc = make_column_transformer((OrdinalEncoder(categories = categories), [1, 3, 5, 6, 7, 8, 9, 13]), remainder = 'passthrough')
X = enc.fit_transform(X).tolist()

#add multiplying features
numAttrs = len(X[0])
for i in range(len(X)):
  for j in range(numAttrs):
    for r in range(j+1, numAttrs):
      X[i].append(X[i][j] * X[i][r])

clf = LinearRegression()

clf.fit(X,y)

#print train score
print(clf.score(X,y))

X_test = []
with open('../data/test_final.csv', 'r') as f:
  for line in f:
    terms = line.strip().split(',')
    
    if terms[0] == 'ID':
      continue
    
    list = []
    for i in range(1, len(terms)):
      try:
        list.append(int(terms[i]))
      except ValueError:
        list.append(terms[i])
    
    X_test.append(list)

X_test = enc.fit_transform(X_test).tolist()

#add multiplying features
numAttrs = len(X_test[0])
for i in range(len(X_test)):
  for j in range(numAttrs):
    for r in range(j+1, numAttrs):
      X_test[i].append(X_test[i][j] * X_test[i][r])

#print predictions
y_test = clf.predict(X_test)
#print('ID,Prediction')
#for i in range(len(y_test)):
#  print(str(i+1) + ',' + str(y_test[i]))