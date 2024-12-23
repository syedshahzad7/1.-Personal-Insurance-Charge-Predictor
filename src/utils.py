 #common functionalities that entire project can use

import os
import sys
from src.exception import CustomException
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):        #function to save an 'obj' as a pickled file in the mentioned 'file_path'
    try:
        dir_path = os.path.dirname(file_path)   #extract the directory name from the passed file_path
        os.makedirs(dir_path, exist_ok = True)  #Create a directory with that extracted directory's name. If already exists, ignore

        with open(file_path, "wb") as file_obj:     #opening the file specified by the file_path and writing to it in binary mode(wb). You use the file_obj object to write into the file in file_path. 
            dill.dump(obj, file_obj)                #serializes the "obj" into a binary format and write it to the file represented by file_obj (which is basically the file given in the file_path)
    except Exception as e:
        raise CustomException(e, sys)     


def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

             
                    