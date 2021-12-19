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
from sklearn.linear_model import Perceptron

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
if sys.argv[1] == '1':
  model = Perceptron(eta0=0.5)
if sys.argv[1] == '2':
  model = Perceptron(penalty='l1', alpha=0.001, eta0=0.0625)
if sys.argv[1] == '3':
  model = Perceptron(penalty='l2', alpha=1e-8, eta0=4)
else:
  model = Perceptron(penalty='elasticnet', alpha=0.001, eta0=2, l1_ratio=0.7)

model.fit(X,y)

##print train error
#y_train = model.predict(X)
#print(roc_auc_score(y, y_train))

#test and print predictions
y_test = model.predict(X_test)
print('ID,Prediction')
for i in range(len(y_test)):
  print(str(i+1) + ',' + str(y_test[i]))

exit()

##run cross validation and print best accuracy and configuration of best model (currently dead code)
cv = KFold(n_splits=3, shuffle=True, random_state=1)

model = None
if sys.argv[1] == '1':
  model = Perceptron()
if sys.argv[1] == '2':
  model = Perceptron(penalty='l1')
if sys.argv[1] == '3':
  model = Perceptron(penalty='l2')
else:
  model = Perceptron(penalty='elasticnet')

space = {
        'eta0': [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
        }

if sys.argv[1] != '1':
  space['alpha'] = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
if sys.argv[1] == '4':
  space['l1_ratio'] = [0.15, 0.3, 0.5, 0.7, 0.85]

search = GridSearchCV(model, space, scoring='accuracy', cv=cv, refit=True, verbose=2, n_jobs = 6)

result = search.fit(X, y)

best_model = result.best_estimator_

y_predicted = best_model.predict(X)

acc = accuracy_score(y, y_predicted)

print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))