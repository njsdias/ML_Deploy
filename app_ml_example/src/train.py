import pickle
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.preprocessing import OneHotEncoder


def train():
    # Load data and sabe indices of columns
    df = pd.read_csv("../input/HR_comma_sep.csv")

    # select the features to the model
    features = df.drop('left', 1).columns

    # save features to a pickle file
    pickle.dump(features, open('../output/features.pickle', 'wb'))

    # One-hot-Encoding will be used in categorical features
    columns_to_fit = ['sales', 'salary']

    # encode categorical variable
    enc = OneHotEncoder(sparse=False).fit(df.loc[:, columns_to_fit])

    # save encode categorical features results in a pickle file
    pickle.dump(enc, open('../output/encoder.pickle', 'wb'))

    # transform variables, merge with existing df and keep column names
    column_names = enc.get_feature_names(columns_to_fit)
    encoded_variables = pd.DataFrame(enc.transform(df.loc[:, columns_to_fit]), columns=column_names)

    # delete categorical columns from original data-frame
    df = df.drop(columns_to_fit, 1)

    # add encoded columns to the original data-frame
    df = pd.concat([df, encoded_variables], axis=1)

    # Fit and save model
    X, y = df.drop('left', 1), df.loc[:, 'left']

    # fit model
    clf = LGBMClassifier().fit(X, y)

    # save model into a pickle file
    pickle.dump(clf, open('../output/model.pickle', 'wb'))


if __name__ == "__main__":
    train()
