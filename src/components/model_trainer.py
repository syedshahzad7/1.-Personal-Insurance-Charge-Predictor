import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

#As usual, we'll start by defining the config class for this module which represents the input given to the module
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")         #Input to this module is the path where the final module need to be stored in the form of a pickle file.


#Now, we define the class that is actually responsible for training the model
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()                   #This variable holds the path of where the final model needs to be saved

    #Now, the function that initiates and does the model training
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing arrays into input features and target feature.")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],              #all the columns in the train_array except the last one      
                train_array[:,-1],               #only the last column in the train_array(the target feature)
                test_array[:,:-1],               #similar for test array
                test_array[:,-1]
            )

            #creating a dictionary of models
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            ## Hyperparameter Tuning

            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }                     

            #Now we call the evaluate model defined in the utils module
            model_report:dict = evaluate_models(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, models = models, param = params)

            #To get the best model score from model_report dictionary
            best_model_score = max(sorted(model_report.values()))

            #Now, get the name of that model
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            #Finally we can get our best model as follows
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found.")
            
            logging.info("Best model found!")

            #Now, lets save this model as a pickle file using the save_object
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            #(Optional) Predictions made by the best_model
            predicted = best_model.predict(X_test)

            #best_model R2 score
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)



