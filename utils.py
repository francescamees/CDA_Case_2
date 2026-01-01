
from sklearn.metrics import silhouette_score
import numpy as np
FEATURES = ['HR_TD_Mean','HR_TD_Median','HR_TD_std','HR_TD_Min',
            'HR_TD_Max','HR_TD_AUC','HR_TD_Kurtosis','HR_TD_Skew',
            'HR_TD_Slope_min','HR_TD_Slope_max','HR_TD_Slope_mean','HR_TD_Slope',
            'TEMP_TD_Mean','TEMP_TD_Median','TEMP_TD_std','TEMP_TD_Min',
            'TEMP_TD_Max','TEMP_TD_AUC','TEMP_TD_Kurtosis','TEMP_TD_Skew',
            'TEMP_TD_Slope_min','TEMP_TD_Slope_max','TEMP_TD_Slope_mean',
            'TEMP_TD_Slope','EDA_TD_P_Mean','EDA_TD_P_Median','EDA_TD_P_std',
            'EDA_TD_P_Min','EDA_TD_P_Max', 'EDA_TD_P_AUC','EDA_TD_P_Kurtosis',
            'EDA_TD_P_Skew','EDA_TD_P_Slope_min','EDA_TD_P_Slope_max',
            'EDA_TD_P_Slope_mean','EDA_TD_P_Slope','EDA_TD_T_Mean','EDA_TD_T_Median',
            'EDA_TD_T_std','EDA_TD_T_Min','EDA_TD_T_Max','EDA_TD_T_AUC',
            'EDA_TD_T_Kurtosis','EDA_TD_T_Skew','EDA_TD_T_Slope_min','EDA_TD_T_Slope_max',
            'EDA_TD_T_Slope_mean','EDA_TD_T_Slope','EDA_TD_P_Peaks','EDA_TD_P_RT','EDA_TD_P_ReT']

MODEL_KWARGS = {
        'kmeans': {'n_clusters': 4, 'random_state': 0},
        'hierarchical': {'n_clusters': 4, 'linkage': 'ward'},
        'gmm': {'n_components': 4, 'random_state': 0}}