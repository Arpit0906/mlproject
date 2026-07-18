import os 
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from src.logger import logging
from src.exception import CustomException

from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerconfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerconfig()

    def initiate_train_model(self,train_arr,test_arr):
        try:
            logging.info("Splitting train and test data")
            X_train, X_test, y_train, y_test = (
            train_arr[:, :-1],
            test_arr[:, :-1],
            train_arr[:, -1],
            test_arr[:, -1]
                    )

            models = {
            "Linear Regression": LinearRegression(),
            "K-Neighbors Regressor": KNeighborsRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor(),
            "XGBRegressor": XGBRegressor(), 
            "CatBoosting Regressor": CatBoostRegressor(verbose=False),
            "AdaBoost Regressor": AdaBoostRegressor()
            }
            params={
                "Decision Tree":{
                    "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                    "splitter": ["best", "random"],
                    "max_features": ["sqrt", "log2", None]
                },
                "Random Forest Regressor":{
                    "n_estimators": [100, 200, 300],
                    "max_features": ["sqrt", "log2", None],
                    "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"]
                },
                "Gradient Boosting":{    
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    "n_estimators": [100, 200, 300],
                    "max_depth": [3, 4, 5, 6]
                },
                "Linear Regression":{},
                "K-Neighbors Regressor":{
                    #"weight": ["uniform", "distance"],
                    "n_neighbors": [3, 5, 7, 9]
                },
                "XGBRegressor":{
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [100, 200, 300],
                    "max_depth": [3, 4, 5, 6]        
                     },
                "CatBoosting Regressor":{
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [50, 100, 200]
                }
            }
            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)

            ## To get best model score from dict
            best_model_score = max (sorted (model_report.values()))

            ## To get best model name from dictionary
            best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)]
  ## model_report.values().index)(best_model_score)this line returns the index of the model with best model score
## and inside the [] we r putting the index ie of the best model score
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found",sys)
            logging.info("Best model found")

            best_model.fit(X_train, y_train)
            predicted=best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return r2_square
        except Exception as e:
            raise CustomException(e,sys)
