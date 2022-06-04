import numpy as np
import sys
import os
import shutil
import pandas as pd
import json

from train import train
from test import test


TRAIN_DATA_DIR = os.path.join("./Data", "Train", "Under_10_min_training")
VALID_DATA_DIR = os.path.join("./Data", "Validation", "Validation_10_percent")

x_price_train = np.load(os.path.join(TRAIN_DATA_DIR, "x_price_train.npy"))
x_news_train = list(np.load(os.path.join(TRAIN_DATA_DIR, "x_news_train.npy")))
y_train = np.load(os.path.join(TRAIN_DATA_DIR, "y_train.npy"))

x_price_valid = np.load(os.path.join(VALID_DATA_DIR, "x_price_valid.npy"))
x_news_valid = list(np.load(os.path.join(VALID_DATA_DIR, "x_news_valid.npy")))
y_valid = np.load(os.path.join(VALID_DATA_DIR, "y_valid.npy"))


optims = ['adam', 'rmsprop']
epochss = [20, 40]
lossfns = ['mse', 'mae']
lstm_units = [128, 256]

min_error = np.inf
best_hyp = {'optim':None, 'epochs':None, 'lossfn':None, 'lstm_unit':None}


cols =  ['optimizer', 'epochs', 'loss_function', 'lstm_unit', 'train_score', 'test_score']
df  = pd.DataFrame(columns = cols)

for optim in optims:
	for epochs in epochss:
		for lossfn in lossfns:
			for lstm_unit in lstm_units:
				print('-------------------------------------------------')
				print(optim, epochs, lossfn, lstm_unit)
				model = train(x_news_train, x_price_train, y_train, optim, lossfn, epochs, lstm_unit)
				error = test(x_news_valid, x_price_valid, y_valid, model)
				if error < min_error:
					best_hyp['optim'] = optim
					best_hyp['epochs'] = epochs
					best_hyp['lossfn'] = lossfn
					best_hyp['lstm_unit'] = lstm_unit
					min_error = error
				train_error = test(x_news_train, x_price_train, y_train, model)
				df.loc[len(df)] = [optim, epochs, lossfn, lstm_unit, train_error, error]

print(df)
df.to_csv(os.path.join("./Results", "tuning_results.txt"), index=False)
json.dump(best_hyp, open(os.path.join("./Results", "hyperparameter.txt"), 'w'))

