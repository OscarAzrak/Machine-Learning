import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

encoderY = preprocessing.LabelEncoder()
encoderX = preprocessing.LabelEncoder()

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', 1500)



# clean data set
def clean_data(df):
    # print shape df
    print("Shape of data set: ", df.shape)

    # drop rows with missing values
    df = df.dropna()
    # encode string features to numeric

    df = df[(df != '?').all(axis=1)]

    #convert x4 column to float
    df['x4'] = df['x4'].astype(float)

    # remove rows that are not True or False in x12
    df = df[(df['x12'] == 'True') | (df['x12'] == 'False')]

    # drop rows containing olka
    df = df[df.x7 != 'olka']

    #drop rows containing chottis
    df = df[df.x7 != 'chottis']


    for col in df.columns:
        if df[col].dtype == "object":
            if col == 'y':
                df[col] = encoderY.fit_transform(df[col])
            else:
                df[col] = encoderX.fit_transform(df[col])

    #remove outliers
    df = df[(np.abs(stats.zscore(df)) < 2.85).all(axis=1)]

    print("Removing low variance features...")


    """
    This resulted in to remove column x5 and x12
    """

    var_thr = VarianceThreshold(threshold = 0.22) #Removing both constant and quasi-constant
    var_thr.fit(df)

    var_thr.get_support()

    concol = [column for column in df.columns
              if column not in df.columns[var_thr.get_support()]]

    df = df.drop(['x5', 'x12'],axis=1)


    print("Shape of data set after cleaning: ", df.shape)

    return df

# clean test
def clean_test_dataset(df):

    # print shape df
    print("Shape of data set: ", df.shape)

    # drop rows with missing/unwanted values
    df = df.dropna()
    df = df[(df != '?').all(axis=1)]
    df = df[(df['x12'] == True) | (df['x12'] == False)]

    #convert x4 column to float
    df['x4'] = df['x4'].astype(float)


    # encode string features to numeric


    for col in df.columns:
        #print name of col
        if (df[col].dtype == "object") or (df[col].dtype == bool):
            if col == 'y':
                df[col] = encoderY.fit_transform(df[col])
            else:
                df[col] = encoderX.fit_transform(df[col])


    # remove columns

    df = df.drop(['x5', 'x12'],axis=1)


    print("Shape of data set after cleaning test: ", df.shape)

    return df



def readCSV(path):
    return pd.read_csv(path, delimiter=",", index_col=0)


#decode y labels

def decode_labels(y):
    return encoderY.inverse_transform(y)

def generate_predictions(model, X):
    y_pred = model.predict(X)
    return y_pred

def save_predictions(y_pred):
    with open("results.txt", "w") as file:
        for pred in y_pred:
            file.write(pred + "\n")
    file.close()


def main():
    train_path = os.path.abspath("TrainOnMe-4.csv")
    test_path = os.path.abspath("EvaluateOnMe-4.csv")

    train = readCSV(train_path)
    test = readCSV(test_path)

    train = clean_data(train)
    test = clean_test_dataset(test)

    X_train = train.drop("y", axis=1)
    y_train = train["y"]

    XGB = XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=3, gamma=0.9, subsample=0.6, random_state=0)
    XGB.fit(X_train, y_train)

    y_pred = generate_predictions(XGB, test)
    #predict test
    y_pred = decode_labels(y_pred)
    save_predictions(y_pred)




if __name__ == '__main__':
    main()

