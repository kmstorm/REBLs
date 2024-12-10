from pathlib import Path
import dill
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, accuracy_score, log_loss, brier_score_loss, f1_score, roc_auc_score, roc_curve

import numpy as np
import scipy
import yaml

def load_data(data_path: str, target_column: str):
	"""
	Loads the data into pandas DataFrame, and separates them into input features and target labels
	
	Parameters
	----------
	data_path : str
		Path to the csv file containing the data
		
	target_column : str
		The name of the column in the dataset that is to be predicted by the model
		
	Returns
	-------
	X : 2d array-like
			Array of features

	y : 1d array-like
		Array of labels
	"""
	data = pd.read_csv(data_path)
	y = data[target_column]
	X = data.drop(columns=[target_column])
	return X, y

def load_model(model_path: str):
	"""
	Loads a fitted machine learning model
	
	Parameters
	----------
	model_path : str
		Path to the model
	
	Returns
	-------
	model
		The loaded model instance
	"""
	return dill.load(open(model_path, 'rb'))


def mean_confidence_interval(data, confidence=0.95):
	"""
	Calculates the mean and confidence interval of a data
	
	Parameters
	----------
	data : array-like
		The data that is used for the calculation
	
	confidence : float
		The confidence value that is used for the confidenc interval calculation
		
	Returns
	-------
	mean : float
		The mean of the data
	
	lower : float
		The lower value of the confidence interval
		
	upper : float
		The upper value of the confidence interval
	"""
	a = 1.0 * np.array(data)
	n = len(a)
	m, se = np.mean(a, axis=0), scipy.stats.sem(a, axis=0)
	h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
	return m, m-h, m+h
	
def scores_with_optimal_cutoff(tpr, fpr, thresholds, y_true, y_proba):
	"""
	Calculates multiple metrics based on the optimal cutoff of the predicted probabilities
	
	Parameters
	----------
	tpr : 1d array-like
		True positive rates
	
	fpr : 1d array-like
		False positive rates
		
	thresholds : 1d array-like
		Threshold values
	
	y_true : 1d array-like
		True labels
	
	y_proba : 1d array-like
		Predicted probabilities
		
	Returns
	-------
	log_loss_score : float
		Logloss score
		
	brier_score : float
		Brier score
		
	acc : float
		Accuracy score
		
	f1 : float
		F1 score
	
	ba : float
		Balanced accuracy score
	"""
	optimal_idx = np.argmax(tpr - fpr)
	optimal_threshold = thresholds[optimal_idx]
	th_mask = y_proba >= optimal_threshold
	y_pred = np.zeros(y_true.shape, np.int64)
	y_pred[th_mask] = 1
	acc = accuracy_score(y_true,y_pred)
	f1 = f1_score(y_true, y_pred)
	ba = balanced_accuracy_score(y_true, y_pred)
	if np.max(y_proba > 1):
		y_proba = y_pred
	log_loss_score = log_loss(y_true,y_proba)
	brier_score = brier_score_loss(y_true,y_proba)
	se = tpr[optimal_idx]
	sp = 1 - fpr[optimal_idx]

	return log_loss_score, brier_score, acc, f1, ba, se, sp

def get_param_grid_from_config(param_grid: dict, model_name: str):
	"""
	Creates a hyperparameter grid that can be passed to the grid search algorithm for hyperparameter optimization
	
	Parameters
	----------
	param_grid_path : str
		Path to the yaml file that defines the possible hyperparameter for the model. If None a default one is defined
		
	Returns
	-------
	param_grid: dict
		Dictionary that can be passed to GridSearchCV
	"""
	if param_grid is not None:
		param_grid = {f'classifier__{k}': v for k, v in param_grid.items()}
	else:
		with open('utils/default_hyperparams.yaml' , 'r') as f:
			default_params = yaml.safe_load(f)
			param_grid = {f'classifier__{k}': v for k, v in default_params[model_name].items()}
		
	return param_grid

def simplify_cross_val_result(df: pd.DataFrame):
	'''
	This function simplifies the DataFrame that is created by the CrossValidatedModel's cross_validate function
	to make it more humanly readable.

	Parameters
	----------
	df: DataFrame
		the resulting data frame after the cross-validated training

	Returns
	-------
	res_df: DataFrame
		the simplified data frame
	'''
	df = df.T
	metrics = ['auc', 'logloss', 'brier_loss', 'accuracy', 'balanced_accuracy', 'f1_score', 'average_precision']
	res_df = pd.DataFrame(columns=['mean', '+/-'], index=metrics)
	model = df.columns[0]
	for m in metrics:
		mean = df.loc[f'{m}_mean', model]
		pm = df.loc[f'{m}_upper', model] - mean
		res_df.loc[m, 'mean'] = mean
		res_df.loc[m, '+/-'] = pm

	return res_df

def plot_summarize_curve(ax, y_probas, y_tests, group=['original', 'log', 'wavelet'], offset=0, confidence=0.95):
    n = len(y_probas)  # number of repetitions or folds
    grp_str = '+'.join(group)
    mean_tpr = np.zeros(100)  # Mean true positive rate at each false positive rate (FPR)
    mean_fpr = np.linspace(0, 1, 100)  # FPR values
    
    tpr_values = []  # Store TPR values for each FPR to calculate standard deviation later
    
    for i in range(n):
        fpr, tpr, _ = roc_curve(y_tests[i], y_probas[i])
        mean_tpr += np.interp(mean_fpr, fpr, tpr)  # Interpolate to get TPR at all FPR points
        tpr_values.append(np.interp(mean_fpr, fpr, tpr))  # Store TPR values for error calculation
    
    mean_tpr /= n  # Calculate the mean TPR over all folds
    mean_tpr[-1] = 1.0  # Set the last point (fpr=1) to 1 for TPR

    # Compute the standard deviation of the TPR at each FPR
    tpr_values = np.array(tpr_values)
    tpr_std = np.std(tpr_values, axis=0)  # Standard deviation of TPR across folds

    # Compute standard error (SE) for each FPR
    tpr_se = tpr_std / np.sqrt(n)  # Standard error
    
    # Calculate the mean AUC and its standard deviation (for display purposes)

    ax.plot(mean_fpr, mean_tpr, label=f'{grp_str}')

    # Set the error bars with the computed SE
    ax.errorbar(
        mean_fpr, mean_tpr, 
        yerr=tpr_se,  # Use SE as the error bar (upper and lower bound)
        fmt='none',  # No markers on the error bars
        capsize=5,
        capthick=1,
        errorevery=18 + 2*offset,  # You can adjust this to control how often the error bars appear
        alpha=0.8,
        color=ax.get_lines()[-1].get_color()  # Use the same color as the ROC curve
    )

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8)
    
    # Set plot limits and labels
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right')
