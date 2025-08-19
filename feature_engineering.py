import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin



# ============================
# Custom Feature Engineer
# ============================
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Save q80 threshold for consistency across train/test
        self.q80 = X['MonthlyCharges'].quantile(0.80)
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # tenure bins
        bins = [-float('inf'), 3, 6, 12, 24, 48, float('inf')]
        labels = ['0-3','4-6','7-12','13-24','25-48','49+']
        X['tenure_bin'] = pd.cut(X['tenure'], bins=bins, labels=labels)
        X['is_new_customer'] = (X['tenure'] <= 3).astype(int)

        # ordered categorical codes
        order = ['0-3','4-6','7-12','13-24','25-48','49+']
        X['tenure_bin'] = pd.Categorical(X['tenure_bin'], categories=order, ordered=True)
        X['tenure_bin_code'] = X['tenure_bin'].cat.codes.astype('int8')

        # engineered numeric features
        X['total_per_month'] = (X['TotalCharges'] / X['tenure']).replace([np.inf], 0).fillna(0)
        X['high_monthly_flag'] = (X['MonthlyCharges'] >= self.q80).astype(int)

        return X