from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

class Classification_Models():

    def logistic(X_train,Y_train,X_test):       
        param_grid = {'solver':[ 'lbfgs','newton-cg', 'liblinear', 'saga'],'penalty':['l2'],'C':[0.01,0.1,1,10]}
        grid = GridSearchCV(LogisticRegression(max_iter=1000,random_state=42),param_grid,refit=True,verbose=3,scoring='f1_weighted',n_jobs=-1)
        grid.fit(X_train,Y_train)
        y_pred = grid.predict(X_test)
        return y_pred, grid

    def SVM(X_train,Y_train,X_test):
        param_grid = {'C':[0.1,1,10],
              'gamma':['scale','auto'],
              'kernel':['rbf','linear']}
        
        grid = GridSearchCV(SVC(probability=True,random_state=42),param_grid,refit=True,verbose=3,scoring='f1_weighted',n_jobs=-1)
        grid.fit(X_train,Y_train)
        y_pred = grid.predict(X_test)
        return y_pred, grid

    def KNN(X_train,Y_train,X_test):
        param_grid = {'n_neighbors':[3,5,7],
              'weights':['uniform','distance'],
              'algorithm':['auto','kd_tree','ball_tree','brute']}

        grid = GridSearchCV(KNeighborsClassifier(),param_grid,refit=True,verbose=3,scoring='f1_weighted',n_jobs=-1)
        grid.fit(X_train,Y_train)
        y_pred = grid.predict(X_test)
        return y_pred, grid

    def NaiveBayes(X_train,Y_train,X_test):
        param_grid = {'var_smoothing': np.logspace(0, -9, num=20)}

        grid = GridSearchCV(GaussianNB(),param_grid,refit=True,verbose=3,scoring='f1_weighted',n_jobs=-1)
        grid.fit(X_train,Y_train)
        y_pred = grid.predict(X_test)
        return y_pred, grid

    def DecisionTree(X_train,Y_train,X_test):
        param_grid = {'criterion':['gini','entropy'],
              'splitter':['best','random'],
              'max_features':['sqrt','log2']}

        grid = GridSearchCV(DecisionTreeClassifier(random_state=42),param_grid,refit=True,verbose=3,scoring='f1_weighted',n_jobs=-1)
        grid.fit(X_train,Y_train)
        y_pred = grid.predict(X_test)
        return y_pred, grid

    def RandomForest(X_train,Y_train,X_test):
        param_grid = {'criterion':['gini','entropy'],
              'n_estimators':[10,50,100],
              'class_weight':['balanced','balanced_subsample']}

        grid = GridSearchCV(RandomForestClassifier(random_state=42),param_grid,refit=True,verbose=3,scoring='f1_weighted',n_jobs=-1)
        grid.fit(X_train,Y_train)   
        y_pred = grid.predict(X_test)
        return y_pred, grid    

    