from encoders.OneHotLabelEncoder import OneHotLabelEncoder
from featuretools.primitives import Count, Mean, Min
from sklearn.base import BaseEstimator, TransformerMixin

import featuretools as ft
import json
import numpy as np
import pandas as pd


def make_agg_features(X):
    '''
    Creates aggregated features for every pitcher_id in the training set.

    Engineered aggregation features:
        COUNT(pitcher)
        MEAN(pitcher.inning)
        MIN(pitcher.inning)

    Target-encoded engineered aggregation features:
        MEAN(pitcher.pitch_type_Changeup)
        MEAN(pitcher.pitch_type_Curveball)
        MEAN(pitcher.pitch_type_Cutter)
        MEAN(pitcher.pitch_type_Fastball)
        MEAN(pitcher.pitch_type_Off-Speed
        MEAN(pitcher.pitch_type_Purpose_Pitch)
        MEAN(pitcher.pitch_type_Sinker)
        MEAN(pitcher.pitch_type_Slider)
    '''
    X = X.copy()
    X['all'] = -1  # aggregation for all data, -1 used as index for pitcher_id

    subset_cols = ['pitch_type', 'pitcher_id', 'all', 'inning']
    X = OneHotLabelEncoder(labels=['pitch_type']).fit_transform(X[subset_cols])

    es = ft.EntitySet(id='baseball_pitches')

    es = es.entity_from_dataframe(entity_id='pitcher', dataframe=X,
                                  make_index=True, index='index')

    es = es.normalize_entity(base_entity_id='pitcher',
                             new_entity_id='pitcher_ids', index='pitcher_id')
    es = es.normalize_entity(base_entity_id='pitcher',
                             new_entity_id='pitchers_all', index='all')

    p_pitcher, _ = ft.dfs(target_entity='pitcher_ids',
                          entityset=es,
                          agg_primitives=[Count, Mean, Min],
                          drop_contains=['MIN(pitcher.pitch_type'],
                          ignore_variables={'pitcher': ['all']})

    p_pitcher_all, _ = ft.dfs(target_entity='pitchers_all',
                              entityset=es,
                              agg_primitives=[Count, Mean, Min],
                              drop_contains=['MIN(pitcher.pitch_type'])

    p_pitcher = p_pitcher[sorted(p_pitcher.columns)]
    p_pitcher_all = p_pitcher_all[sorted(p_pitcher_all.columns)]
    p_pitcher = pd.concat([p_pitcher, p_pitcher_all], axis=0)

    return p_pitcher


def make_pitcher_dict(p_pitcher):
    '''
    Renames columns from the DataFrame returned from make_agg_features.
    Also converts COUNT(pitcher) column to dictionary with pitcher_id mapping.

    Returns results as a dictionary.
    '''
    cols_p_pitcher_id = ['MEAN(pitcher.inning) | pitcher_id',
                         'p(pitch_type_Changeup | pitcher_id)',
                         'p(pitch_type_Curveball | pitcher_id)',
                         'p(pitch_type_Cutter | pitcher_id)',
                         'p(pitch_type_Fastball | pitcher_id)',
                         'p(pitch_type_Off-Speed | pitcher_id)',
                         'p(pitch_type_Purpose_Pitch | pitcher_id)',
                         'p(pitch_type_Sinker | pitcher_id)',
                         'p(pitch_type_Slider | pitcher_id)',
                         'MIN(pitcher.inning) | pitcher_id']
    p_pitcher_id = pd.DataFrame(p_pitcher.iloc[:, 1:].values,
                                columns=cols_p_pitcher_id)
    p_pitcher_id = p_pitcher_id.set_index(p_pitcher.index)
    p_pitcher_id = p_pitcher_id.iloc[:, [0, 9, 1, 2, 3, 4, 5, 6, 7, 8]]

    dict_pitch = {'pitcher_cnt_dict': p_pitcher['COUNT(pitcher)'].to_dict(),
                  'p_pitcher_id': p_pitcher_id}

    return dict_pitch


class FeatureEngineering(BaseEstimator, TransformerMixin):
    '''
    Class for adding engineered features to a DataFrame, Series, or JSON input.

    Engineered features:
        runs_diff
        lead
        on_base_1
        on_base_2
        on_base_3
        on_base_any
        batterHand_pitcherHand
        slg_2010
        park_factor_H

    Engineered near-realtime features:
        last_count_type
        last_pitch
        Changeup_L10
        Curveball_L10
        Cutter_L10
        Fastball_L10
        Off-Speed_L10
        Purpose_Pitch_L10
        Sinker_L10
        Slider_L10

    Engineered aggregation features:
        MEAN(pitcher.inning) | pitcher_id
        MIN(pitcher.inning) | pitcher_id

    Target-encoded engineered aggregation features:
        p(pitch_type_Changeup | pitcher_id)
        p(pitch_type_Curveball | pitcher_id)
        p(pitch_type_Cutter | pitcher_id)
        p(pitch_type_Fastball | pitcher_id)
        p(pitch_type_Off-Speed | pitcher_id)
        p(pitch_type_Purpose_Pitch | pitcher_id)
        p(pitch_type_Sinker | pitcher_id)
        p(pitch_type_Slider | pitcher_id)
    '''
    def __init__(self, dict_slg, dict_park_H, epp_df,
                 min_pitches_from_pitcher=150):
        self.dict_slg = dict_slg  # dict of 2010 SLG data with batter_id mapping
        self.dict_park_H = dict_park_H  # dict of park factors with team_id mapping
        self.epp_df = epp_df  # DataFrame with engineered near-realtime features
        self.min_pitches_from_pitcher = min_pitches_from_pitcher

    @staticmethod
    def _height(val):
        feet, inches = map(int, val.split("-"))
        return feet*12 + inches

    @staticmethod
    def _lead(val):
        if val > 0:
            return 'Y'
        elif val == 0:
            return 'T'
        else:
            return 'N'

    def _features(self, X):
        X.loc[:, 'runs_diff'] = (X['home_team_runs'] - X['away_team_runs']) * (1 - 2*X['top'])
        X.loc[:, 'lead'] = X['runs_diff'].map(self._lead)

        X.loc[:, 'on_base_1'] = pd.notnull(X['on_1b'])
        X.loc[:, 'on_base_2'] = pd.notnull(X['on_2b'])
        X.loc[:, 'on_base_3'] = pd.notnull(X['on_3b'])
        X.loc[:, 'on_base_any'] = X['on_base_1'] | X['on_base_2'] | X['on_base_3']

        X.loc[:, 'batterHand_pitcherHand'] = X['stand'] + '_' + X['p_throws']

        X.loc[:, 'b_height'] = X['b_height'].map(lambda x: self._height(x))
        X.loc[:, 'slg_2010'] = X['batter_id'].map(self.dict_slg)

        h1 = X['team_id_p'].map(self.dict_park_H)  # pitching team is at home when on top
        h2 = X['team_id_b'].map(self.dict_park_H)  # batting team is at home when on bottom
        X.loc[:, 'park_factor_H'] = np.where(X['top'] == 1, h1, h2)

    def _input_cleanup(self, X):
        if X.__class__ == pd.DataFrame:
            return X.copy()
        elif X.__class__ == pd.Series:
            return pd.DataFrame(X).T.convert_objects(convert_numeric=True)
        elif X.__class__ == str or X.__class__ == unicode:
            row = pd.Series(json.loads(X), index=self.columns)
            return pd.DataFrame(row).T.convert_objects(convert_numeric=True)

    def _make_pitch_freq(self, X, y=None):
        if 'pitch_type' not in X.columns:
            X = pd.concat([X, pd.Series(y, name='pitch_type')], axis=1)

        p_pitcher = make_agg_features(X)
        dict_pfreq = make_pitcher_dict(p_pitcher)

        self.pfreq = dict_pfreq

    def fit(self, X, y=None):
        self.columns = X.columns
        self._make_pitch_freq(X, y)
        return self

    def transform(self, X):
        X = self._input_cleanup(X)

        self._features(X)
        X = X.join(self.epp_df, on='uid')

        # overall pitch distribution (key == -1) will be used for pitcher_id
        #   with insufficient pitches thrown (< self.min_pitches_from_pitcher)
        #   or unknown pitcher_id
        pitches_thrown = X['pitcher_id'].map(self.pfreq['pitcher_cnt_dict'])
        pid_recode = X['pitcher_id'].where(pitches_thrown >= self.min_pitches_from_pitcher, -1)

        freq_data1 = self.pfreq['p_pitcher_id'].loc[pid_recode]
        freq_data1 = freq_data1.reset_index(drop=True).set_index(X.index)

        drops = ['home_team_runs', 'away_team_runs', 'on_1b', 'on_2b', 'on_3b',
                 'stand', 'p_throws', 'batter_id', 'team_id_p', 'team_id_b',
                 'top', 'uid', 'game_pk', 'year', 'date', 'at_bat_num',
                 'start_tfs', 'start_tfs_zulu', 'pitch_id', 'pitcher_id']
        X = X.drop(drops, axis=1)

        output = pd.concat([X, freq_data1], axis=1)

        return output
