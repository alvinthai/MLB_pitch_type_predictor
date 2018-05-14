import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from BaseClassifier import extended_classification_report


class UniformClassifier(BaseEstimator, ClassifierMixin):
    '''
    Dumb classifier that predicts the same value for all predictions.

    Parameters
    ----------
    val: str or numeric
        Value to return in predictions.
    '''
    def __init__(self, val):
        self.val = val

    def fit(self, X=None, y=None):
        return self

    def predict(self, X):
        return np.full(len(X), self.val)

    def predict_proba(self, X):
        return np.ones(len(X)).reshape(-1, 1)

    def multiclassification_report(self, X):
        y_true = X.loc[:, 'pitch_type']
        y_pred = self.predict(X)

        return extended_classification_report(y_true, y_pred)


class BaselineClassifier(BaseEstimator, ClassifierMixin):
    '''
    Classifier that predicts without machine learning.

    Uses pitch probability data from training set to make predictions, returns
     classification for the specific pitcher's most probable pitch type.
    '''
    def __init__(self):
        self.cols = ['p(pitch_type_Changeup | pitcher_id)',
                     'p(pitch_type_Curveball | pitcher_id)',
                     'p(pitch_type_Cutter | pitcher_id)',
                     'p(pitch_type_Fastball | pitcher_id)',
                     'p(pitch_type_Off-Speed | pitcher_id)',
                     'p(pitch_type_Purpose_Pitch | pitcher_id)',
                     'p(pitch_type_Sinker | pitcher_id)',
                     'p(pitch_type_Slider | pitcher_id)']
        self.recode = {0: 'Changeup', 1: 'Curveball', 2: 'Cutter',
                       3: 'Fastball', 4: 'Off-Speed', 5: 'Purpose_Pitch',
                       6: 'Sinker', 7: 'Slider'}

    def predict(self, X):
        X = X.loc[:, self.cols].values
        y = np.argmax(X, axis=1).astype(object)

        for k, v in self.recode.iteritems():
            y[y == k] = v

        return y

    def multiclassification_report(self, X):
        y_true = X['pitch_type']
        y_pred = self.predict(X)

        return extended_classification_report(y_true, y_pred)
