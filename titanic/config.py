from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Data params
_root_path = 'data/{}'
__train_file = 'train.csv'
__test_file = 'test.csv'
__submission_file = 'gender_submission.csv'

train_path = _root_path.format(__train_file)
test_path = _root_path.format(__test_file)
submission_path = _root_path.format(__submission_file)

# Model params
# RandomForestClassifier
rf_params = {
             'n_estimators': range(150, 250, 5),
             # 'n_estimators': [200],
             'min_samples_leaf': range(1, 5, 1),
             # 'min_samples_leaf': [1],
             'max_depth': range(5, 20, 1),
             # 'max_depth': [14],
             'min_samples_split': range(4, 12, 1),
             # 'min_samples_split': [8],
             'oob_score': [True]}
# 0.8853737536043419
rf_best_params = {
                  'n_estimators': 150,
                  'min_samples_leaf': 2,
                  'max_depth': 18,
                  'min_samples_split': 6,
                  'oob_score': True
                  }

# LogisticRegression
lr_params = {'C': [0.001, 0.01, 0.1, 1, 10],
             "max_iter": range(100, 500, 10),
             'class_weight': ['balanced']}
# 0.8678689733984286
lr_best_params = {'class_weight': 'balanced',
                  'C': 0.01,
                  'max_iter': 100}

# SVC
svc_params = {'C': [0.001, 0.01, 0.1, 1, 10],
              "max_iter": range(100, 500, 10)}
# 0.8633673405277086
svc_best_params = {'C': 1,
                   'max_iter': 300,
                   'probability': True}

# GradientBoostingClassifier
gbdt_params = {'learning_rate': [0.5, 0.6, 0.7, 0.8],
               'n_estimators': range(100, 300, 10)}
# 0.8602577255896211
gbdt_best_params = {'learning_rate': 0.6,
                    'n_estimators': 100}

# XGBClassifier
xgb_params = {'learning_rate': [0.5, 0.6, 0.7, 0.8],
              'n_estimators': range(100, 300, 10)}
# 0.8715292287137468
xgb_best_params = {'learning_rate': 0.5,
                   'n_estimators': 100}

# ExtraTreesClassifier
et_params = {'n_estimators': range(100, 300, 10)}
# 0.8298925142516563
et_best_params = {'n_estimators': 100}

# model
model_params = {'rf': [RandomForestClassifier, rf_params, rf_best_params],
                'lr': [LogisticRegression, lr_params, lr_best_params],
                'svc': [SVC, svc_params, svc_best_params],
                'gbdt': [GradientBoostingClassifier, gbdt_params, gbdt_best_params],
                'xgb': [XGBClassifier, xgb_params, xgb_best_params],
                'et': [ExtraTreesClassifier, et_params, et_best_params]}
