# library doc string
"""
This module implements the main script for the customer churn project with clean code
Author: Abdelkarim Eljandoubi
Date: August, 2023
"""


# import libraries
import os
import logging

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sns.set()


main_logger = logging.getLogger("churn_library")

main_logger.setLevel(logging.INFO)

main_formatter = logging.Formatter(
    '%(filename)s - %(levelname)s - %(message)s')

main_file_handler = logging.FileHandler('./logs/churn_library.log', mode="w")
main_file_handler.setLevel(logging.INFO)
main_file_handler.setFormatter(main_formatter)

main_stream_handler = logging.StreamHandler()
main_stream_handler.setFormatter(main_formatter)

main_logger.addHandler(main_file_handler)
main_logger.addHandler(main_stream_handler)

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            data_frame: pandas dataframe
    '''

    main_logger.info("importing data from %s", pth)

    data_frame = pd.read_csv(pth)
    data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    data_frame.drop('Attrition_Flag', axis=1, inplace=True)
    data_frame.drop('CLIENTNUM', axis=1, inplace=True)

    return data_frame


def perform_eda(data_frame):
    '''
    perform eda on df and save figures to images folder
    input:
            data_frame: pandas dataframe

    output:
            None
    '''

    main_logger.info("performing EDA")

    for col in ['Churn', 'Customer_Age']:
        plt.figure(figsize=(20, 10))
        data_frame[col].hist()
        plt.savefig(f"./images/eda/{col}_dist.png")
        plt.close()

    plt.figure(figsize=(20, 10))
    data_frame.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig("./images/eda/Marital_Status_value_counts_dist.png")
    plt.close()

    plt.figure(figsize=(20, 10))
    sns.histplot(data_frame['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig("./images/eda/Total_Trans_Ct_density.png")
    plt.close()

    plt.figure(figsize=(20, 10))
    sns.heatmap(data_frame.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig("./images/eda/heatmap_corr.png")
    plt.close()

    main_logger.info("Figures are saved")


def encoder_helper(data_frame, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could
            be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''

    main_logger.info("encoding categorical columns : %s", category_lst)

    for category in category_lst:
        category_groups = data_frame.groupby(category).mean()[response]
        new_feature = f"{category}_{response}"
        data_frame[new_feature] = data_frame[category].apply(
            lambda x: category_groups.loc[x])

    data_frame.drop(category_lst, axis=1, inplace=True)

    return data_frame


def perform_feature_engineering(data_frame, response):
    '''
    input:
              data_frame: pandas dataframe
              response: string of response name [optional argument that could
              be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    main_logger.info("performing feature engineering")

    # Collect categorical features to be encoded
    cat_columns = data_frame.select_dtypes(include='object').columns.tolist()

    # Encode categorical features using mean of response variable on category
    data_frame = encoder_helper(data_frame, cat_columns, response)

    y = data_frame[response]
    X = data_frame.drop(response, axis=1)

    main_logger.info("splitting data")

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def plot_classification_report(model_name,
                               y_train,
                               y_test,
                               y_train_preds,
                               y_test_preds):
    '''
    produces classification report for training and testing results and stores
    report as image in images folder

    input:
                    model_name: (str) name of the model, ie 'Random Forest'
                    y_train: training response values
                    y_test:  test response values
                    y_train_preds: training predictions from model_name
                    y_test_preds: test predictions from model_name

    output:
                     None
    '''

    main_logger.info("plot classification report of %s", model_name)

    plt.rc('figure', figsize=(5, 5))

    # Plot Classification report on Train dataset
    plt.text(0.01, 1.25,
             str(f'{model_name} Train'),
             {'fontsize': 10},
             fontproperties='monospace'
             )
    plt.text(0.01, 0.05,
             str(classification_report(y_train, y_train_preds)),
             {'fontsize': 10},
             fontproperties='monospace'
             )

    # Plot Classification report on Test dataset
    plt.text(0.01, 0.6,
             str(f'{model_name} Test'),
             {'fontsize': 10},
             fontproperties='monospace'
             )
    plt.text(0.01, 0.7,
             str(classification_report(y_test, y_test_preds)),
             {'fontsize': 10},
             fontproperties='monospace'
             )

    plt.axis('off')

    # Save figure to ./images folder
    fig_name = f'Classification_report_{model_name}.png'
    plt.savefig(
        os.path.join(
            "./images/results",
            fig_name)
    )

    plt.close()


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    plot_classification_report('Logistic_Regression',
                               y_train,
                               y_test,
                               y_train_preds_lr,
                               y_test_preds_lr)
    plt.close()

    plot_classification_report('Random_Forest',
                               y_train,
                               y_test,
                               y_train_preds_rf,
                               y_test_preds_rf)
    plt.close()


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''

    main_logger.info("plot featre importance")

    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # Save figure to output_pth
    plt.savefig(os.path.join(output_pth, "Feature_Importance.png"))

    plt.close()


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''

    main_logger.info("train models...")

    # grid search
    rfc = RandomForestClassifier(random_state=42, n_jobs=-1)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000, n_jobs=-1)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid,
                          cv=5, n_jobs=-1)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    main_logger.info("predict with models")

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    main_logger.info("plot roc curves")

    # plot ROC-curves
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(
        cv_rfc.best_estimator_,
        X_test,
        y_test,
        ax=ax,
        alpha=0.8
    )

    plot_roc_curve(lrc, X_test, y_test, ax=ax, alpha=0.8)

    # save ROC-curves to images directory
    plt.savefig(
        os.path.join(
            "./images/results",
            'ROC_curves.png'),
        bbox_inches='tight')
    plt.close()

    main_logger.info("save models")

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # Display feature importance on train data
    feature_importance_plot(cv_rfc.best_estimator_,
                            X_train,
                            "./images/results")


def main(pth, response):
    """
    the main function that execute the hole process
    input:
            pth: a path to the csv
            response: string of response name
    output:
              None
    """
    dataframe = import_data(pth)

    perform_eda(dataframe)

    train_models(*perform_feature_engineering(dataframe, response))

    main_logger.info("execution terminated successfully")


if __name__ == "__main__":
    main("./data/bank_data.csv", "Churn")
