

#import pandas to read the dataset
import pandas as pd
#import sklear to split the train an test data
from sklearn.model_selection import train_test_split

#reading the dataset and printing it 
cars=pd.read_csv("cars.csv")
cars=pd.DataFrame(data=cars)
print (cars)

# I am dividing the inputs variables and the output variable
X=cars.iloc[:,:7]
Y=cars.iloc[:,7]
print(X)
print(Y)

# creating the random split to train and to test the model 
X_train, X_test, y_train, y_test = train_test_split(X,Y, random_state=0,test_size=0.25)


# I imported this metrics to quialify the model 
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, recall_score)


def evaluacion_modelos(model, X_train, X_test, y_test):
  y_test_pred = model.predict(X_test)
  y_train_pred = model.predict(X_train)

  print("*****Evaluation results****")

  clf_report = pd.DataFrame(classification_report(y_test, y_test_pred, output_dict=True))
  conf = confusion_matrix(y_test, y_test_pred)

  print(conf)

  TP = conf[1,1]
  TN = conf[0,0]
  FP = conf[0,1]
  FN = conf[1,0]

  print("Accuracy %s \n" % accuracy_score(y_test, y_test_pred))
  print("Sensibility %s \n" % recall_score(y_test, y_test_pred,average='macro'))
  print("Specificity %s \n" % (TN/(TN+FP)))
  print("Rate of false positive results  %s \n" % (FP/(TN+FP)))

  print(clf_report)


  

  # I am creating the model, I use the treeclassifier
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=0)

# trainig the model
clf.fit(X_train, y_train)


# I am sending the results of this model to the quialifier
evaluacion_modelos(clf, X_train, X_test, y_train, y_test)