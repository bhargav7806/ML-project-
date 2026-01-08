import os 
import sys 
from dataclasses import dataclass 

from sklearn.linear_model import LinearRegression , Ridge , Lasso 
from sklearn.ensemble import RandomForestRegressor , AdaBoostRegressor , GradientBoostingRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging


from src.utils import save_object , evalute_model


@dataclass
class ModelTrainingconfig:
    training_model_file_path = os.path.join('artifacts' , 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingconfig()


    def initiate_model_trainer(self , train_array , test_array ):
        

        try:
            logging.info("Split training and test input data")
            x_train  , y_train , x_test , y_test = (
                train_array[: , :-1],
                train_array[: , -1],
                test_array[: , :-1],
                test_array[: , -1]
            )

            models = {
                "Decision Tree" : DecisionTreeRegressor(),
                "Random_Forest" : RandomForestRegressor(),
                "GradientBoost Regressor" : GradientBoostingRegressor(),
                "Linear Regression"  : LinearRegression(),
                "K-Neighbour Regressor" : KNeighborsRegressor(),
                "XGBRgressor" : XGBRegressor(),
                # "CatBoosting Regressor" : CatBoostRegressor(),
                "AdaBoost Regressor" : AdaBoostRegressor() 
               
            }

            params = {
                "Decision Tree" :{
                    'criterion' : ['squared_error' , 'friedman_mse' , 'absolute_error' , 'poisson'],
                    'splitter' :['best' , 'random'],
                    'max_features' :['log2' , 'sqrt']
                },
                'Random_Forest':{
                    'criterion':['squared_error' , 'friedman_mse' , 'absolute_error' , 'poisson'],
                    'max_features' : ['log2' ,'sqrt'],
                    'n_estimator' : [8 , 16 , 32 , 64 , 128 , 256]
                },
                'GradientBoost Regressor' :{
                    'loss':['squared_error' , 'huber' , 'absolute_error' , 'quantile'],
                    'learning_rate' :[.1 , .01 , .05 , .001],
                    'subsample' : [.6 , .7 ,.75 , .8 , .85 , .9],
                    'criterion' : ['squared_error' , 'friedman_mse'],
                    'max_feature' : ['auto' , 'sqrt' , 'log2'],
                    'n_estimator' :[8 , 16 , 32 , 64 , 128 , 256]
                },
                'Linear Regression' : {},
                'K-Neighbour Regressor' : {
                    'n_neighbour' : [5 , 7 , 9 , 11],
                    'weights' : ['uniform' , 'distance'],
                    'algorithm' : ['ball_tree' , "kd_tree" , "brute"]
                },
                "XGBRgressor" : {
                    'learning_rate' : [.1 , .01 , .05 , .001],
                    'n_estimator' : [8 , 16 , 32 , 64 , 128 , 256]
                },
                'AdaBoost Regressor' : {
                    'learning_rate'  : [.1 , .01 , .05 , .001],
                    'loss' : ['linear' , 'square' , 'exponential'],
                    'n_estimators' : [8 , 16 , 32 , 64 , 128 , 256] 
                }




                
            }

            model_report:dict=evalute_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models , param = params)

            best_model_score = max(sorted(model_report.values()))


            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best model found and both training and testing dataset : {best_model_name}")

            save_object(
                file_path = self.model_trainer_config.training_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(x_test)

            r2score = r2_score(y_test , predicted)

            return r2score


        except Exception as e:
            raise CustomException (e , sys)