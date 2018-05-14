from __future__ import division, print_function
import datetime
import logging
import numpy as np
import pandas as pd
import sklearn.metrics as m

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder


def extended_classification_report(y_true, y_pred):
    '''
    Extention of sklearn.metrics.classification_report. Builds a text report
    showing the main classification metrics and the total count of multiclass
    predictions per class.

    Parameters
    ----------
    y_true: array-like, shape = [n_samples, ]
        Ground truth (correct) target values.

    y_pred: array-like, shape = [n_samples, ]
        Estimated targets as returned by a classifier.
    '''
    acc = m.accuracy_score(y_true, y_pred)

    output = m.classification_report(y_true, y_pred, digits=3)
    class_labels = sorted(pd.Series(y_true).unique())

    n_pred = pd.Series(y_pred).value_counts()

    if class_labels == [False, True] and type(class_labels[0]) != np.int64:
        n_pred = np.array([n_pred[False], n_pred[True]])
    else:
        n_pred = n_pred[class_labels].values

    padding = max([15, np.ceil(max(np.log10(n_pred))) + 2])
    n_pred = np.char.array(n_pred)

    output = output.split('\n')
    output[0] += 'n_predictions'.rjust(padding)

    for i, x in enumerate(n_pred):
        output[i+2] += x.rjust(padding)

    output.extend(['', 'accuracy: {}'.format(acc), ''])

    print('\n'.join(output))


class BaseClassifier(BaseEstimator, ClassifierMixin):
    '''
    A custom scikit-learn module that provides general utility functions that:
    (1) Provides Numpy and Pandas input compatibility
    (2) Allows extraction of target variable from X DataFrame inputs
    (3) LabelEncodes categorical target variable inputs
    (4) Support easy handling for early stopping on the sklearn wrapper for
        XGBoost and LightGBM

    Parameters
    ----------
    target: str
        Label for target variable in pandas DataFrame. If provided, all future
        future inputs with an X DataFrame do not require an accompanying y
        input, as y will be extracted from the X DataFrame; however, the target
        column must be included in the X DataFrame for all fitting steps if the
        target parameter is provided.

    model: sklearn compatible model
        Model to train classifier on.

        i.e. model = XGBClassifier()

    model_fit_params: dict
        Additional parameters (inputted as a dict) to pass to the fit step of
        self.model.

        i.e. model_fit_params = {'verbose': False}
    '''
    def __init__(self, target=None, model=None, model_fit_params=None):
        self.target = target
        self.model = model
        self.model_fit_params = model_fit_params

        self._default_attributes()

        self._logger = logging.Logger('')
        self._logger.addHandler(logging.StreamHandler())

    def _check_eval_metric(self, clf, fit_params, eval_X, eval_y):
        '''
        A function for ensuring default metrics for early stopping
        evaluation are compatible with xgboost and lightgbm. Also adds
        eval_set into fit parameters for xgboost and lightgbm.

        The default evaluation metric for binary classification is accuracy
        and the default evaluation metric for multiclass classification is
        multiclass logloss.

        Parameters
        ----------
        clf: model
            Unfitted model. Used to check the type of model being fitted.

        fit_params: dict
            Fit parameters to pass into clf.

        eval_X: array-like, shape = [n_samples, n_features]
            X input evaluation data for early stopping.

        eval_y: array-like, shape = [n_samples, ]
            True classification values to score early stopping.

        Returns
        -------
        fit_params: dict
            Early stopping compatible fit parameters to pass into clf.
        '''
        early_stop_berror = {'xgboost.sklearn': 'error',
                             'lightgbm.sklearn': 'binary_error'}
        early_stop_mlog = {'xgboost.sklearn': 'mlogloss',
                           'lightgbm.sklearn': 'multi_logloss'}
        berror = early_stop_berror.values()
        mlog = early_stop_mlog.values()
        module = clf.__module__

        if module in early_stop_berror:
            fit_params['eval_set'] = [(eval_X, eval_y)]
            binary = len(np.unique(eval_y)) == 2

            if 'early_stopping_rounds' not in fit_params:
                fit_params['early_stopping_rounds'] = 10

            if 'eval_metric' not in fit_params:
                if binary:
                    fit_params['eval_metric'] = early_stop_berror[module]
                else:
                    fit_params['eval_metric'] = early_stop_mlog[module]
            elif binary:
                if fit_params['eval_metric'] in mlog:
                    fit_params['eval_metric'] = early_stop_berror[module]
            elif fit_params['eval_metric'] in berror:
                fit_params['eval_metric'] = early_stop_mlog[module]

        return fit_params

    def _default_attributes(self):
        '''
        Sets/resets defualt attributes.
        '''
        self._le = LabelEncoder()

        if self.model is None:
            for _ in xrange(1):
                try:  # default model: xgboost if installed
                    from xgboost import XGBClassifier
                    self.model = XGBClassifier()
                    break

                except ImportError:
                    pass

                try:  # if xgboost not found, next default: lightgbm
                    from lightgbm import LGBMClassifier
                    self.model = LGBMClassifier(n_estimators=100)
                    break

                except ImportError:  # gradientboosting in absense of above
                    from sklearn.ensemble import GradientBoostingClassifier
                    self.model = GradientBoostingClassifier()

    def _encode_y(self, y, eval_y=None):
        '''
        Encodes y values with LabelEncoder to allow classification on string
        or numerical values.

        Parameters
        ----------
        y: array-like, shape = [n_samples, ]
            y values from the training set to encode.

        eval_y: array-like, shape = [n_samples, ], optional
            y values from the evaluation set to encode.

        Returns
        ------
        y: array-like, shape = [n_samples, ]
            y values from the training set after LabelEncoder transform.

        eval_y: array-like, shape = [n_samples, ] or None
            y values from the evaluation set after LabelEncoder transform.
            Returns None if no eval_y is provided as input.

        enc: LabelEncoder
            LabelEncoder object for transforming the y values.
        '''
        y = self._le.transform(y)

        if eval_y is not None:
            eval_y = self._le.transform(eval_y)

        return y, eval_y

    def _eval_set(self, eval_set):
        '''
        Cleans up eval_set into proper format for xgboost and lightgbm. If
        eval_set is a DataFrame, unpacks it into a list containing an (X, y)
        tuple. Aside from xgboost/lightgbm, eval_set is also used to evaluate
        trained models on unseen data and validate grid searches.

        Parameters
        ----------
        eval_set: DataFrame or list of (X, y) tuple
            Dataset to use as validation set for early-stopping and/or scoring
            trained models.

        Returns
        -------
        eval_set: list of (X, y) tuple
            Cleaned up eval_set for further usage in OrderedOVRClassifier.
        '''
        if eval_set is None:
            return None

        if eval_set.__class__ == pd.DataFrame:  # eval_set is a DataFrame
            eval_X = eval_set.drop(self.target, axis=1)
            eval_y = eval_set[self.target]
            eval_set = [(eval_X, eval_y)]

        elif eval_set[0].__class__ == tuple:  # eval_set is a (X, y) tuple
            if len(eval_set[0]) != 2:
                raise AssertionError('Invalid shape for eval_set')

            if eval_set[0][0].__class__ == pd.DataFrame:
                eval_set = [(eval_set[0][0], pd.Series(eval_set[0][1]))]

        else:  # bad input for eval_set
            raise TypeError('Invalid input for eval_set')

        assert len(eval_set[0][0]) == len(eval_set[0][1])

        return eval_set

    def _fit_model_with_params(self, X, y, fit_params):
        '''
        Wrapper for training a general model with additional fit parameters.

        Parameters
        ----------
        X: array-like, shape = [n_samples, n_features]
            Training input data.

        y: array-like, shape = [n_samples, ]
            True classification values for training data.

        fit_params: dict
            Key-value pairs of optional arguments to pass into model fit
            function.
        '''
        if type(fit_params) == dict and len(fit_params) > 0:
            try:
                self.model.fit(X, y, **fit_params)
            except TypeError:

                msg = str("Warning: incompatible fit_params found for model. "
                          "Fit method will ignore parameters:\n\n{0}\n\n"
                          "".format(fit_params.keys()))

                self._logger.warn(msg)

                self.model.fit(X, y)
        else:
            self.model.fit(X, y)

    @staticmethod
    def _xg_cleanup(clf):
        '''
        Utility function to delete the Booster.feature_names attributes in
        XGBClassifier. Deleting this attribute allows XGBClassifier to make
        predictions from either a numpy array or DataFrame input.

        Parameters
        ----------
        clf: model
            Model used to make predictions.
        '''
        if clf.__module__ == 'xgboost.sklearn':
            if clf._Booster.feature_names:
                del clf._Booster.feature_names

    def _xy_transform(self, X, y):
        '''
        Utility function for extracting y from a DataFrame when an empty y
        input is provided.

        Parameters
        ----------
        X: array-like, shape = [n_samples, n_features]
            Input data. Can be DataFrame (with or without target column
            included) or numpy array.

        y: array-like, shape = [n_samples, ] or None
            True labels for X. If not provided and X is a DataFrame, column
            specified by self.target is extracted as the y output.

        Returns
        -------
        X: array-like, shape = [n_samples, <= n_features]
            Data for future modeling/predicting steps, with (if applicable)
            target column removed.

        y: array-like, shape = [n_samples, ]
            True labels for X.
        '''
        if y is not None:
            y = pd.Series(y)
        elif self.target is not None:
            # set y from pandas DataFrame if self.target is set
            y = X[self.target].copy()
            X = X.drop(self.target, axis=1)
        else:
            raise RuntimeError('Please initiate class with a label for target '
                               'variable OR input a y parameter into fit')

        assert len(X) == len(y)
        return X, y

    def fit(self, X, y=None, eval_set=None, fit_params=None):
        '''
        Fits the classifier.

        Parameters
        ----------
        X: array-like, shape = [n_samples, n_features]
            Input data for model training.

        y: array-like, shape = [n_samples, ], optional
            True labels for X. If not provided and X is a DataFrame, will
            extract y column from X with the provided self.target value.

        eval_set: DataFrame or list of (X, y) tuple, optional
            Dataset to use as validation set for early-stopping and/or scoring
            trained models.

        fit_params: dict
            Additional parameters (inputted as a dict) to pass to the fit step
            of self.model.

            i.e. fit_params = {'verbose': False}

        Returns
        -------
        self
        '''
        start = datetime.datetime.now()
        self._default_attributes()

        X, y = self._xy_transform(X, y)
        eval_set = self._eval_set(eval_set)
        model = self.model

        if fit_params is None:
            fit_params = self.model_fit_params

        if eval_set is not None:
            eval_X = eval_set[0][0]
            eval_y = eval_set[0][1]
        else:
            eval_X, eval_y = None, None

        self._le.fit(y)  # fit LabelEncoder
        y, eval_y = self._encode_y(y, eval_y)

        # Fit the model
        fit_params = self._check_eval_metric(model, fit_params, eval_X, eval_y)
        self._fit_model_with_params(X, y, fit_params)

        if eval_set is not None:
            y_true = eval_set[0][1]
            y_pred = self.predict(eval_set[0][0])
        else:
            y_true = y
            y_pred = self.predict(X)

        print('-'*80, 'OVERALL REPORT', '', sep='\n')

        extended_classification_report(y_true, y_pred)
        stop = datetime.datetime.now()
        duration = (stop-start).total_seconds()/60

        prt = 'total training time: {0:.1f} minutes'.format(duration)
        print(prt, '', '-'*80, sep='\n')

        return self

    def multiclassification_report(self, X, y=None):
        '''
        Wrapper function for extended_classification_report, which is an
        extension of sklearn.metrics.classification_report. Builds a text
        report showing the main classification metrics and the total count of
        multiclass predictions per class.

        Parameters
        ----------
        X: array-like, shape = [n_samples, n_features]
            Data used for predictions.

        y: array-like, shape = [n_samples, ], optional
            True labels for X. If not provided and X is a DataFrame, will
            extract y column from X with the provided self.target value.
        '''
        X, y = self._xy_transform(X, y)
        y_pred = self.predict(X)

        return extended_classification_report(y, y_pred)

    def predict(self, X):
        '''
        Predict multi-class targets using underlying estimators.

        Parameters
        ----------
        X: array-like, shape = [n_samples, n_features]
            Data used for predictions.

        Returns
        -------
        pred: array-like, shape = [n_samples, ]
            Predicted multi-class targets.
        '''
        self._xg_cleanup(self.model)
        pred = self.model.predict(X)
        pred = self._le.inverse_transform(pred)

        return pred

    def predict_proba(self, X):
        '''
        Predict probabilities for multi-class targets using underlying
        estimators.

        Parameters
        ----------
        X: array-like, shape = [n_samples, n_features]
            Data used for predictions.

        Returns
        -------
        pred: array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in self._le.classes_.
        '''
        self._xg_cleanup(self.model)
        pred = self.model.predict_proba(X)

        return pred

    def score(self, X, y=None, sample_weight=None):
        '''
        Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X: array-like, shape = [n_samples, n_features]
            Test samples.

        y: array-like, shape = [n_samples, ], optional
            True labels for X.

        sample_weight: array-like, shape = [n_samples], optional
            Sample weights.

        Returns
        -------
        scr: float
            Mean accuracy of self.predict(X) wrt y.
        '''
        X, y = self._xy_transform(X, y)
        scr = m.accuracy_score(y, self.predict(X), sample_weight=sample_weight)

        return scr
