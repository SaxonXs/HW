from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from connector.connector import get_data
from conf.conf import logging
from util.util import save_model, load_model


def split(df):
    logging.info('defining X and y')
    X = df.iloc[:, :-1]
    y = df['target']

    logging.info('X and y defined, start splitting')

    x_train, x_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        random_state = 3)
    logging.info('dataset is splitted')

    return x_train, x_test, y_train, y_test



def training(x_train, y_train):
    rtf = RandomForestClassifier()
    logging.info('training the Random Forest model')

    rtf.fit(x_train, y_train)

    save_model(dir='conf/random_forest.pkl', model=rtf)

    return rtf

def predict(values, path_to_model):
    rtf = load_model(dir = path_to_model)
    return rtf.predict(values)
