"""
configuration test of churn_library.py module with pytest
Author: Abdelkarim Eljandoubi
Date: August, 2023
"""

import pytest


def df_plugin():
    "df_plugin"
    return []


def path_plugin():
    "path_plugin"
    return "./data/bank_data.csv"


def response_plugin():
    "response_plugin"
    return "Churn"


def split_plugin():
    "split_plugin"
    return ()


def pytest_configure():
    "pytest_configure"
    pytest.df = df_plugin()
    pytest.pth = path_plugin()
    pytest.reps = response_plugin()
    pytest.split = split_plugin()
