from collections import defaultdict
from encoders.OneHotLabelEncoder import OneHotLabelEncoder
from iml.links import IdentityLink
from IPython.display import display
from BaseClassifier import BaseClassifier

import json
import numpy as np
import pandas as pd  # pandas 0.22 required

import shap
shap.initjs()


class OneHotLabelGrouper(OneHotLabelEncoder):
    '''
    Performs One Hot Encoding on categorical features.

    Also stores information on column names, indicies for one hot encoded
     transformed features, and baseline values (mode for categorical features,
     median for numerical features).

    Class inherited from OneHotLabelEncoder from the follwing repo:
     https://github.com/alvinthai/custom_preprocessors/
    '''
    def __init__(self, labels=None):
        super(OneHotLabelGrouper, self).__init__(labels=labels)
        self.fitted = False

    def fit(self, X, y=None):
        if len(self.labels) == 0:
            self._get_categorical(X)

        dict1 = dict([(y, x) for x, y in enumerate(X.columns)])
        medians = X.median()

        self.baseline = X.mode().iloc[0]
        self.baseline[medians.index] = medians
        self.baseline_df = pd.DataFrame(self.baseline).T

        self.colnames = X.columns
        self.feature_groups = []

        super(OneHotLabelGrouper, self).fit(X, y)
        X = super(OneHotLabelGrouper, self).transform(self.baseline_df)

        dict2 = dict([(y, x) for x, y in enumerate(X.columns)])
        dict3, dict4 = defaultdict(list), dict()

        for k1 in dict1:
            if k1 not in self.labels:
                dict3[dict1[k1]].append(dict2[k1])
                dict4[dict2[k1]] = dict1[k1]
            else:
                prefix = str(k1) + '_'

                for k2 in dict2:
                    if k2.startswith(prefix):
                        dict3[dict1[k1]].append(dict2[k2])
                        dict4[dict2[k2]] = dict1[k1]

                dict3[dict1[k1]] = sorted(dict3[dict1[k1]])

        for key in dict3:
            self.feature_groups.append(dict3[key])

        self.baseline_df = X.convert_objects(convert_numeric=True)
        self.fitted = True
        self.fitted_colnames = X.columns

        # self.dictmap = {}
        # self.dictmap['pre_col2idx'] = dict1
        # self.dictmap['post_col2idx'] = dict2
        # self.dictmap['preidx2postidx'] = dict3
        # self.dictmap['postidx2preidx'] = dict4

        return self


class BaseballPitchClassifier(BaseClassifier):
    '''
    Custom API that extends sklearn classifications with:
     - Automatic one hot encoding of categorical features
     - Convenient handling of evaluation datasets for early stopping in LightGBM and XGBoost
     - Convenient integration with shap library to explain features contributing to prediction
     - Multiclassification accuracy/precision/recall/f1 reports

    More info on shap library:
     source: https://github.com/slundberg/shap
     research paper: https://arxiv.org/pdf/1705.07874.pdf
    '''
    def __init__(self, target=None, model=None, model_fit_params=None,
                 categorical_cols=None):
        self._ohlg = OneHotLabelGrouper(categorical_cols)

        init = super(BaseballPitchClassifier, self).__init__
        init(target, model=model, model_fit_params=model_fit_params)

    def _create_shap_kernal(self, class_idx, link):
        data = self._ohlg.baseline_df.values.reshape(1, -1)
        group_names = self._ohlg.colnames.tolist()
        feature_groups = self._ohlg.feature_groups
        data_median = shap.DenseData(data, group_names, feature_groups)
        f = lambda x: self.model.predict_proba(x)[:, class_idx]
        explainer = shap.KernelExplainer(f, data_median, link, nsamples=1000)

        return explainer

    def _eval_set(self, eval_set):
        es = super(BaseballPitchClassifier, self)._eval_set

        eval_set = es(eval_set)
        eval_set = [(self._ohlg.transform(eval_set[0][0]), eval_set[0][1])]

        return eval_set

    def _input_cleanup(self, X):
        if X.__class__ == pd.DataFrame:
            return X.copy()
        elif X.__class__ == pd.Series:
            return pd.DataFrame(X).T.convert_objects(convert_numeric=True)
        elif X.__class__ == str or X.__class__ == unicode:
            row = pd.Series(json.loads(X), index=self.columns)
            return pd.DataFrame(row).T.convert_objects(convert_numeric=True)

    def _pred_cleanup(self, X, return_raw=False):
        if self.target in X.columns:
            X = X.drop(self.target, axis=1)

        if len(X.columns) == len(self._ohlg.colnames):
            if all(X.columns == self._ohlg.colnames):
                Xt = self._ohlg.transform(X)
        elif len(X.columns) == len(self._ohlg.fitted_colnames):
            assert all(X.columns == self._ohlg.fitted_colnames)
            Xt = X
        else:
            raise RuntimeError('Column names mismatch')

        if return_raw:
            return Xt, X
        else:
            return Xt

    def _xy_transform(self, X, y):
        xy_tf = super(BaseballPitchClassifier, self)._xy_transform
        X, y = xy_tf(X, y)

        if not self._ohlg.fitted:
            X = self._ohlg.fit_transform(X)
        else:
            X = self._ohlg.transform(X)

        return X, y

    def fit(self, X, y=None, eval_set=None):
        self._ohlg.fitted = False
        return super(BaseballPitchClassifier, self).fit(X, y, eval_set)

    def predict(self, X):
        X = self._input_cleanup(X)
        X = self._pred_cleanup(X)
        return super(BaseballPitchClassifier, self).predict(X)

    def predict_proba(self, X):
        X = self._input_cleanup(X)
        X = self._pred_cleanup(X)
        return super(BaseballPitchClassifier, self).predict_proba(X)

    def predict_explain_one(self, X, explain_class=None, link=IdentityLink(),
                            explainer=None, detail=True):
        X = self._input_cleanup(X)
        Xt, X = self._pred_cleanup(X, return_raw=True)

        if explainer is None:
            if explain_class is not None:
                # explain features contributing to user-specified class
                class_idx = self._le.transform([explain_class])[0]
            else:
                # explain features contributing to predicted class
                probas = self.predict_proba(Xt)
                class_idx = np.argmax(probas)
                explain_class = self._le.inverse_transform([class_idx])[0]

            explainer = self._create_shap_kernal(class_idx, link)

        instance = shap.Instance(Xt.values.reshape(1, -1), X.values.reshape(-1))
        explaination = explainer.explain(instance)

        if detail:  # shows comparison of input values to baseline values
            detail_cols = ['baseline_val', 'input_val', 'effect', 'abs_effect']
            pred_label = ['prediction for {}'.format(explain_class)]

            predictions = [[explaination.base_value, explaination.out_value]]
            pred_df = pd.DataFrame(predictions, pred_label, detail_cols[:2])

            col1 = self._ohlg.baseline.values.reshape(-1, 1)
            col2 = X.values.reshape(-1, 1)
            col3 = np.round(explaination.effects, 3).reshape(-1, 1)
            col4 = np.abs(col3)

            effects = np.hstack([col1, col2, col3, col4])
            effects = pd.DataFrame(effects, self._ohlg.colnames, detail_cols)
            effects = effects.sort_values('abs_effect', 0, False).iloc[:5, :-1]

            display(pred_df)
            display(effects)

        return explaination

    def predict_explain(self, X, explain_class, link=IdentityLink()):
        class_idx = self._le.transform([explain_class])[0]
        explainer = self._create_shap_kernal(class_idx, link)
        exp_arr = []

        # explain features contributing to user-specified class for entire dataset
        for i in xrange(X.shape[0]):
            explaination = self.predict_explain_one(X.iloc[i], explain_class,
                                                    link, explainer, False)
            exp_arr.append(explaination)

        return shap.visualize(exp_arr)
