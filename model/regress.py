from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from connector.connector import get_data
from conf.conf import logging
from util.util import save_model, load_model

def split(df):
    df = get_data('https://raw.githubusercontent.com/5x12/ml-cookbook/master/supplements/data/heart.csv')
    logging.info('defining X and y')
    X = df.iloc[:, :-1]
    y = df['target']

    logging.info('X and y defined, start splitting')
    x_train, x_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        random_state = 3)

    logging.info('dataset is splitted')
    return x_train, x_test, y_train, y_test


def training_lr(x_train, y_train):
    lr = LogisticRegression()
    logging.info('training the Random Forest model')

    lr.fit(x_train, y_train)

    save_model(dir='conf/regression.pkl', model=lr)
    return lr


def predict(values, path_to_model):
    lr = load_model(dir = path_to_model)
    return lr.predict(values)
