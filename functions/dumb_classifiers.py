from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import OrderedOVRClassifier.oovr_utils as utils


class UniformClassifier(utils.UniformClassifier):
    '''
    Classifier that always predicts the same value.

    Class inherited from UniformClassifier.
     source: https://github.com/alvinthai/OrderedOVRClassifier/blob/master/OrderedOVRClassifier/oovr_utils.py
    '''
    def __init__(self, val):
        self.val = val

    def multiclassification_report(self, X):
        y_true = X.loc[:, 'pitch_type']
        y_pred = self.predict(X)

        return utils.extended_classification_report(y_true, y_pred)


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

        return utils.extended_classification_report(y_true, y_pred)
