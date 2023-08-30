"""
configuration test of churn_library.py module with pytest
Author: Abdelkarim Eljandoubi
Date: August, 2023
"""

import pytest
def df_plugin():
    return None
# Creating a Dataframe object 'pytest.df' in Namespace
def pytest_configure():
    pytest.df = df_plugin()