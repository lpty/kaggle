# -*- coding: utf-8 -*
import os, sys
sys.path.append(os.path.abspath('../..'))
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from quora.feature.feature import ClassicalFeature
from quora import config


class Model(object):

    def __init__(self):
        self.params = config.model_params[self.__class__.__name__]
        self.clf = None
        self.model = None

    def _init_train_corpus(self):
        self.x = ClassicalFeature.load()
        self.y = pd.read_csv(config.origin_train_file)['is_duplicate']

    def _init_predict_corpus(self):
        self.x_submission = ClassicalFeature.load(mode='test')

    def _cv(self, clf, params):
        """
        Cross validate as see.
        """
        cv = GridSearchCV(estimator=clf(), param_grid=params, scoring='neg_log_loss', cv=5, verbose=2, n_jobs=-1)
        cv.fit(self.x, self.y)
        return cv.best_params_, cv.best_score_

    def _fit(self, clf, params):
        """
        Fit the model .
        """
        model = clf(**params)
        model.fit(self.x, self.y)
        return model

    def _predict(self, clf, filename):
        """
        Generate submission file use model.
        """
        submission = pd.read_csv(config.origin_submission_file)
        submission['is_duplicate'] = 1 - clf.predict_proba(self.x_submission)
        submission = submission[['test_id', 'is_duplicate']].set_index('test_id')
        submission.to_csv(config.root_path.format(filename))
        return submission

    def cv(self):
        assert self.clf, 'Classifier must exit.'
        if 'x' not in self.__dict__:
            self._init_train_corpus()
        best_params, best_score = self._cv(self.clf, self.params['cv'])
        print('{}: Best_params: {}, Best_score: {}'.format(self.__class__.__name__,
                                                           best_params, best_score))
        return best_params, best_score

    def fit(self):
        assert self.clf, 'Classifier must exit.'
        if 'x' not in self.__dict__:
            self._init_train_corpus()
        self.model = self._fit(self.clf, self.params['fit'])
        importance = self.model.feature_importances_
        for feature, imp in sorted(zip(self.x.columns.tolist(), importance), key=lambda x: -x[1]):
            print('{}, {}'.format(feature, str(imp)))
        return self.model

    def predict(self):
        assert self.model, 'Model must exit.'
        if 'x_submission' not in self.__dict__:
            self._init_predict_corpus()
        submission = self._predict(self.model, '{}.csv'.format(self.__class__.__name__))
        return submission


class XGB(Model):

    def __init__(self):
        Model.__init__(self)
        self.clf = XGBClassifier


class LGB(Model):

    def __init__(self):
        Model.__init__(self)


class RF(Model):

    def __init__(self):
        Model.__init__(self)
        self.clf = RandomForestClassifier


if __name__ == '__main__':
    m = XGB()
    m.cv()
    m.fit()
    m.predict()
