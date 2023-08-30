"""
Unit test of churn_library.py module with pytest
Author: Abdelkarim Eljandoubi
Date: August, 2023
"""

import os
import logging
import pytest
import numpy as np
import churn_library as cl


test_logger = logging.getLogger("churn_script_logging_and_tests")

test_logger.setLevel(logging.INFO)

test_formatter = logging.Formatter('%(filename)s - %(levelname)s - %(message)s')

test_file_handler = logging.FileHandler(
    './logs/churn_script_logging_and_tests.log', mode="w")
test_file_handler.setLevel(logging.INFO)
test_file_handler.setFormatter(test_formatter)

test_stream_handler = logging.StreamHandler()
test_stream_handler.setFormatter(test_formatter)

test_logger.addHandler(test_file_handler)
test_logger.addHandler(test_stream_handler)

@pytest.mark.parametrize("import_data", [cl.import_data])
def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        test_logger.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        test_logger.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        test_logger.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err
        
    pytest.df = df


@pytest.mark.parametrize("perform_eda", [cl.perform_eda])
def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    
    dataframe = pytest.df
    
    try:
        perform_eda(dataframe)
        test_logger.info("Testing perform_eda: SUCCESS")
    except KeyError as err:
        test_logger.error('Column "%s" not found', err.args[0])
        raise err
        
    for pic in ['Churn_dist.png', 'Customer_Age_dist.png', ]:
        try:
            assert os.path.isfile(f"./images/eda/{pic}")
            test_logger.info('File %s was found', pic)
            
        except AssertionError as err:
            test_logger.error('File %s was not found', pic)
            raise err
        
        

@pytest.mark.parametrize("encoder_helper", [cl.encoder_helper])
def test_encoder_helper(encoder_helper):
    '''
	test encoder helper
	'''
    
    dataframe = pytest.df
    
    try:
        
        cat_columns = dataframe.select_dtypes(include='object').columns.tolist()
        x_data = encoder_helper(dataframe, cat_columns, "Churn")
        test_logger.info("Testing encoder_helper: SUCCESS")
        
    except Exception as err:
        test_logger.error("executing encoder_helper produces the error : %s",err)
        raise err
    
    
    try:
        assert x_data.shape[1] == x_data.select_dtypes(
            include=np.number).shape[1]
    except AssertionError as err:
        test_logger.error('ERROR: Training data contains non numerical numbers')
        raise err

@pytest.mark.parametrize("perform_feature_engineering",
                         [cl.perform_feature_engineering])
def test_perform_feature_engineering(perform_feature_engineering):
    '''
	test perform_feature_engineering
	'''
    pass

@pytest.mark.parametrize("train_models", [cl.train_models])
def test_train_models(train_models):
    '''
	test train_models
	'''
    pass


if __name__ == "__main__":
    pytest.main(args=['-s', os.path.abspath(__file__)])








