# -*- coding: utf-8 -*-
"""
MODEL
------
Some model of kaggle titanic.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from titanic.corpus import Corpus
from titanic import config


class Survived(object):

    def __init__(self):
        self.initialize()

    def initialize(self):
        """
        Init corpus.
        """
        corpus = Corpus()
        self.x = corpus.train_data_x_std
        self.y = corpus.train_data_y
        self.x_submission = corpus.submission_data_x_std

    def _cv(self, clf, params, x=None, y=None):
        """
        Cross validate as see.
        """
        cv = GridSearchCV(estimator=clf(), param_grid=params, scoring='roc_auc', cv=5, verbose=0, n_jobs=-1)
        cv.fit(self.x if x is None else x, self.y if y is None else y)
        return cv.best_params_, cv.best_score_

    def _fit(self, clf, params):
        """
        Fit the model .
        """
        model = clf(**params)
        model.fit(self.x, self.y)
        return model

    def _predict(self, clf, filename, x_submission=None):
        """
        Generate submission file use model.
        """
        submission = pd.read_csv(config.submission_path)
        submission['Survived'] = clf.predict(self.x_submission if x_submission is None else x_submission)
        submission = submission[['PassengerId', 'Survived']].set_index('PassengerId')
        submission.to_csv(config._root_path.format(filename))

    def cv(self):
        """
        CV all single model.
        """
        res = {}
        for cls_name, cls_content in config.model_params.items():
            best_params, best_score = self._cv(cls_content[0], cls_content[1])
            res[cls_name] = [best_params, best_score]
            print('{}: Best_params: {}, Best_score: {}'.format(cls_name, best_params, best_score))
        return res

    def fit(self):
        """
        Fit all single model.
        """
        res = {}
        for cls_name, cls_content in config.model_params.items():
            model = self._fit(cls_content[0], cls_content[2])
            res[cls_name] = model
        return res

    def predict(self, res):
        """
        Genetate for all single model.
        """
        for name, clf in res.items():
            self._predict(clf=clf, filename='{}.csv'.format(name))

    def voting(self):
        """
        Ensemble model -- voting.
        """
        estimators = [(cls_name, cls_content[0](**cls_content[2]))
                      for cls_name, cls_content in config.model_params.items()]
        vot = VotingClassifier(estimators=estimators, voting='hard')
        vot.fit(self.x, self.y)
        self._predict(vot, 'voting.csv')

    def stacking(self):
        """
        Ensemble model -- stacking.
        """
        clfs = [cls_content[0](**cls_content[2]) for cls_name, cls_content in config.model_params.items()]
        train_blend = np.zeros((self.x.shape[0], len(clfs)))
        submission_blend = np.zeros((self.x_submission.shape[0], len(clfs)))
        skf = list(StratifiedKFold(self.y, 5))
        for i, clf in enumerate(clfs):
            submission_blend_i = np.zeros((submission_blend.shape[0], len(skf)))
            for j, (train, test) in enumerate(skf):
                clf.fit(self.x[train], self.y[train])
                train_blend[test, i] = clf.predict_proba(self.x[test])[:, 1]
                submission_blend_i[:, j] = clf.predict_proba(self.x_submission)[:, 1]
            submission_blend[:, i] = submission_blend_i.mean(1)
        clf = config.model_params['lr'][0]
        best_params, best_score = self._cv(clf, config.lr_params, train_blend)
        lr = clf(**best_params)
        lr.fit(train_blend, self.y)
        self._predict(lr, 'stacking.csv', submission_blend)

if __name__ == '__main__':
    survived = Survived()
    # survived.cv()
    res = survived.fit()
    survived.predict(res)
    # survived.stacking()
    # survived.voting()
#