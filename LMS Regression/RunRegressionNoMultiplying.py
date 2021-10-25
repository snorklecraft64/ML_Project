import sys
import numpy
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score

X_in = []
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
    
    X_in.append(list)
    y.append(int(terms[len(terms)-1]))

pipe = None

if int(sys.argv[1]) == 1:
  categories = [['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'],
                ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'],
                ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'],
                ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'],
                ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
                ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'],
                ['Female', 'Male'],
                ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']]
  pipe = Pipeline([('impute', SimpleImputer(missing_values = '?', strategy = 'most_frequent')),
                   ('encode', make_column_transformer((OrdinalEncoder(categories = categories), [1, 3, 5, 6, 7, 8, 9, 13]), remainder = 'passthrough'))])

if int(sys.argv[1]) == 2:
  categories = [['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked', '?'],
                ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool', '?'],
                ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse', '?'],
                ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces', '?'],
                ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried', '?'],
                ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black', '?'],
                ['Female', 'Male', '?'],
                ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands', '?']]
  pipe = Pipeline([('encode', make_column_transformer((OrdinalEncoder(categories = categories), [1, 3, 5, 6, 7, 8, 9, 13]), remainder = 'passthrough'))])

if pipe == None:
  print('ERROR: second argument invalid, use 1 or 2')
  quit()

pipe.fit(X_in)
X_mod = pipe.transform(X_in)

#add bias term
X_list = X_mod.tolist()
for i in range(len(X_mod)):
  X_list[i].insert(0, 1)
X_mod = numpy.array(X_list)

#change X_mod to correct representation
X_train = []
for j in range(len(X_mod[0])):
  list = []
  for i in range(len(X_mod)):
    ignore = X_mod[i]
    list.append(X_mod[i][j])
  
  X_train.append(list)

#calculate optimal weight vector
X = numpy.array(X_train)
Y = numpy.array(y)
inv = numpy.linalg.inv(X.dot(X.transpose()))
XY = X.dot(Y)
weight = inv.dot(XY)

#print train score
#predict labels using weight vector
Y_train = []
for i in range(len(X_mod)):
  Y_train.append(weight.dot(X_mod[i]))
#print(roc_auc_score(Y, Y_train))

#print csv of test labels
X_in = []
i = 0
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
    
    X_in.append(list)

X_mod = pipe.transform(X_in)

#add bias term
X_list = X_mod.tolist()
for i in range(len(X_mod)):
  X_list[i].insert(0, 1)
X_mod = numpy.array(X_list)

#predict labels using weight vector
Y_test = []
for i in range(len(X_mod)):
  Y_test.append(weight.dot(X_mod[i]))
print('ID,Prediction')
for i in range(len(Y_test)):
  print(str(i+1) + ',' + str(Y_test[i]))