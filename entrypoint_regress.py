from model.regress import predict
from conf.conf import logging
#insert your data below
logging.info(f"prediction: {predict([[59,1,0,101,234,0,1,143,0,3.4,0,0,0]], path_to_model='model/conf/regression.pkl')}")