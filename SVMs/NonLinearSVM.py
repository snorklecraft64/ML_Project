import sys
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

##1:  unknown is majority value, hinge loss
##2:  unknown is majority value, sqaured hinge loss
##3:  unknown is its own value, hinge loss
##4:  unknown is its own value, sqaured hinge loss

##import training data
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

##import testing data
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

#make unknowns majority value
if sys.argv[1] == '1':
  categories = [['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'],
                ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'],
                ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'],
                ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'],
                ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
                ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'],
                ['Female', 'Male'],
                ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']]
  imputer = SimpleImputer(missing_values = '?', strategy = 'most_frequent')
  X = imputer.fit_transform(X)
  X_test = imputer.fit_transform(X_test)
  ordinal = make_column_transformer((OrdinalEncoder(categories = categories), [1, 3, 5, 6, 7, 8, 9, 13]), remainder = 'passthrough')
  X = ordinal.fit_transform(X)
  X_test = ordinal.fit_transform(X_test)

#make unknown its own value
else:
  categories = [['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked', '?'],
                ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool', '?'],
                ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse', '?'],
                ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces', '?'],
                ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried', '?'],
                ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black', '?'],
                ['Female', 'Male', '?'],
                ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands', '?']]
  ordinal = make_column_transformer((OrdinalEncoder(categories = categories), [1, 3, 5, 6, 7, 8, 9, 13]), remainder = 'passthrough')
  X = ordinal.fit_transform(X)
  X_test = ordinal.fit_transform(X_test)

#scale data to 0 mean, makes run faster
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.fit_transform(X_test)

model = None

#train model
if sys.argv[1] == '1':
  model = SVC(random_state=1, C=1) #best model chosen by below cross validation
else:
  model = SVC(random_state=1, C=1) #best model chosen by below cross validation
model.fit(X,y)

#print train error
y_train = model.predict(X)
print(roc_auc_score(y, y_train))

##test and print predictions
#y_test = model.predict(X_test)
#print('ID,Prediction')
#for i in range(len(y_test)):
#  print(str(i+1) + ',' + str(y_test[i]))

exit()

##run cross validation and print best accuracy and configuration of best model (currently dead code)
cv = KFold(n_splits=3, shuffle=True, random_state=1)

model = SVC(random_state=1)

space = {
        'C': [1e-4,1e-3,1e-2,1e-1,1,10,100,1000]
        }

search = GridSearchCV(model, space, scoring='accuracy', cv=cv, refit=True, verbose=2)

result = search.fit(X, y)

best_model = result.best_estimator_

y_predicted = best_model.predict(X)

acc = accuracy_score(y, y_predicted)

print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))