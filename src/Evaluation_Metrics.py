import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

class ModelEvaluator():

    def evaluate(Y_pred, grid_model, X_test, Y_test):
        best = str(grid_model.best_params_)
        cm = confusion_matrix(Y_test,Y_pred)
        report = classification_report(Y_test,Y_pred)
        roc = roc_auc_score(Y_test,grid_model.predict_proba(X_test)[:,1])
        return best, cm, report, roc
