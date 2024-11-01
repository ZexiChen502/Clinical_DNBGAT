import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def convert_timepoint_to_label(time_point):
    if time_point in [1, 2, 3]:
        return 0  # Reversible
    elif time_point in [4, 5]:
        return 1  # Irreversible
    else:
        return 2  # Disease


def calculate_correlation(clinical):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(clinical)
    scaled_clinical = pd.DataFrame(scaled_data, columns=clinical.columns)
    correlation_matrix = scaled_clinical.T.corr()
    np.fill_diagonal(correlation_matrix.values, 0)

    return correlation_matrix

