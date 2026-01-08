import sys 
from dataclasses import dataclass
import os 

import numpy as np 
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder , StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class Datatransformationconfig():
    preprocessor_obj_file_path :str = os.path.join('artifact' , 'preprocessor.pkl')
    logging.info('Data Transformation')

class datatransformation():
    def __init__(self):
        self.data_transformation_config=Datatransformationconfig()
    
    def get_data_transformer(self):

        '''
        This function is responsible for data transformation.
        
        '''
        try:
            num_columns = ['reading_score', 'writing_score']
            cat_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course'] 

            num_pipeline = Pipeline(
                steps = [
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler' , StandardScaler()),
                ]
            )
            cat_pipeline = Pipeline(
                steps = [
                    ('Imputer' , SimpleImputer(strategy='most_frequent')),
                    ("OneHotEncoder" , OneHotEncoder())
                ]
            )
            
            logging.info('numerical columns scaling is completed')
            logging.info('categotical columns encoding completed')


            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline' , num_pipeline , num_columns),
                    ('cat_pipeline' , cat_pipeline , cat_columns)
                ]
            )

            return preprocessor


            
        except Exception as e:

            raise CustomException(e , sys)
         
    def initiate_transformation(self , train_path , test_path):  
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Read Train and Test data completed')

            logging.info('obtaining preprocessing object')

            preprocessor_obj=self.get_data_transformer()



            target_column_name = 'math_score'
            num_columns = ['writing_score' , 'reading_score']

            input_feature_train_df = train_df.drop(columns=[target_column_name] , axis = 1)
            target_feature_train_df = train_df[target_column_name]


            input_feature_test_df = test_df.drop(columns = [target_column_name] , axis = 1)
            target_feature_test_df = test_df[target_column_name]


            logging.info('aaplying preprocessor on training dataframe and test dataframe')

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr , np.array(target_feature_train_df)]


            test_arr = np.c_[input_feature_test_arr , np.array(target_feature_test_df)]
            
            logging.info('saved processing object')

            
            save_object(
            obj = preprocessor_obj ,
            
            file_path = self.data_transformation_config.preprocessor_obj_file_path)

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path           
            )
        except Exception as e:

            raise CustomException(e ,  sys)




