import sys
import numpy
import statistics
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import Binarizer
from sklearn.ensemble import AdaBoostClassifier

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
    y.append(terms[len(terms)-1])

pipe = None
iterations = int(sys.argv[1])

#BOTH impute and binarize
if int(sys.argv[2]) == 1:
  #find the threshold for each column
  columns = [0, 2, 4, 10, 11, 12]
  thresholds = {}
  for i in columns:
    #generate list of values
    values = []
    for j in range(len(X)):
      values.append(int(X[j][i]))
    
    thresholds[i] = statistics.median(values)

  categories = [['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'],
                ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'],
                ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'],
                ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'],
                ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
                ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'],
                ['Female', 'Male'],
                ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']]
  pipe = Pipeline([('impute', SimpleImputer(missing_values = '?', strategy = 'most_frequent')),
                   ('encode', make_column_transformer((OrdinalEncoder(categories = categories), [1, 3, 5, 6, 7, 8, 9, 13]), remainder = 'passthrough')),
                   ('binarize', make_column_transformer((Binarizer(threshold = thresholds[0]), [8]),
                                                        (Binarizer(threshold = thresholds[2]), [9]),
                                                        (Binarizer(threshold = thresholds[4]), [10]),
                                                        (Binarizer(threshold = thresholds[10]), [11]),
                                                        (Binarizer(threshold = thresholds[11]), [12]),
                                                        (Binarizer(threshold = thresholds[12]), [13]), remainder = 'passthrough')),
                   ('clf', AdaBoostClassifier(n_estimators = iterations, random_state = 0))])

#NO impute, YES binarize
if int(sys.argv[2]) == 2:
  #find the threshold for each column
  columns = [0, 2, 4, 10, 11, 12]
  thresholds = {}
  for i in columns:
    #generate list of values
    values = []
    for j in range(len(X)):
      values.append(int(X[j][i]))
    
    thresholds[i] = statistics.median(values)
  
  categories = [['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked', '?'],
                ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool', '?'],
                ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse', '?'],
                ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces', '?'],
                ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried', '?'],
                ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black', '?'],
                ['Female', 'Male', '?'],
                ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands', '?']]
  pipe = Pipeline([('encode', make_column_transformer((OrdinalEncoder(categories = categories), [1, 3, 5, 6, 7, 8, 9, 13]), remainder = 'passthrough')),
                   ('binarize', make_column_transformer((Binarizer(threshold = thresholds[0]), [8]),
                                                        (Binarizer(threshold = thresholds[2]), [9]),
                                                        (Binarizer(threshold = thresholds[4]), [10]),
                                                        (Binarizer(threshold = thresholds[10]), [11]),
                                                        (Binarizer(threshold = thresholds[11]), [12]),
                                                        (Binarizer(threshold = thresholds[12]), [13]), remainder = 'passthrough')),
                   ('clf', AdaBoostClassifier(n_estimators = iterations, random_state = 0))])

#YES impute, NO binarize
if int(sys.argv[2]) == 3:
  categories = [['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'],
                ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'],
                ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'],
                ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'],
                ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
                ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'],
                ['Female', 'Male'],
                ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']]
  pipe = Pipeline([('impute', SimpleImputer(missing_values = '?', strategy = 'most_frequent')),
                   ('encode', make_column_transformer((OrdinalEncoder(categories = categories), [1, 3, 5, 6, 7, 8, 9, 13]), remainder = 'passthrough')),
                   ('clf', AdaBoostClassifier(n_estimators = iterations, random_state = 0))])

#NO impute, NO binarize
if int(sys.argv[2]) == 4:
  categories = [['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked', '?'],
                ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool', '?'],
                ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse', '?'],
                ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces', '?'],
                ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried', '?'],
                ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black', '?'],
                ['Female', 'Male', '?'],
                ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands', '?']]
  pipe = Pipeline([('encode', make_column_transformer((OrdinalEncoder(categories = categories), [1, 3, 5, 6, 7, 8, 9, 13]), remainder = 'passthrough')),
                   ('clf', AdaBoostClassifier(n_estimators = iterations, random_state = 0))])
  
pipe.fit(X, y)
print(pipe.score(X, y))

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

y_test = pipe.predict(X_test)
print('ID,Prediction')
for i in range(len(y_test)):
  print(str(i+1) + ',' + str(y_test[i]))