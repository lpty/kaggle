# abs
abs_path = '{Your Path}/kaggle/quora/{}'
# corpus
root_path = abs_path.format('data/{}')
origin_train_file = root_path.format('train.csv')
origin_test_file = root_path.format('test.csv')
origin_submission_file = root_path.format('sample_submission.csv')

clean_train_file = root_path.format('clean.csv')
clean_test_file = root_path.format('clean_test.csv')

pos_train_file = root_path.format('pos_train.csv')
pos_test_file = root_path.format('pos_test.csv')

zh_train_file_txt = root_path.format('zh_train.txt')
zh_test_file_txt = root_path.format('zh_test.txt')
zh_train_file = root_path.format('zh_train.csv')
zh_test_file = root_path.format('zh_test.csv')

# feature
feature_path = abs_path.format('data/feature/{}')
feature_columns = [
    'CharLengthDiff',
    'CharLengthDiffRatio',
    'Concurrence',
    'ConcurrenceTFIDF',
    'Distance',
    'Doc2VectorDistance',
    'InterrogativeWord',
    'KeyWordDistance',
    'LevenshteinDistance',
    'LongestCommonSeq',
    'NGramW2VDistance',
    'SpecialConcurrence',
    'Statistics',
    'StrikeAMatch',
    'Structure',
    'TFIDFDistance',
    'W2VWeightDistance',
    'Word2VecDistance',
    'WordLengthDiff',
    'WordLengthDiffRatio',
    'WordMoverDistance'
]

# model
model_path = abs_path.format('data/model/{}')
model_params = {
    'XGB': {
        'fit': {
            'learning_rate': 0.1,
            'n_estimators': 100,
            'max_depth': 8,
            'min_child_weight': 50,
            'scale_pos_weight': 0.333536
        },
        'cv': {
            # 'learning_rate': [0.5, 0.6, 0.7, 0.8],
            'learning_rate': [0.1],
            # 'n_estimators': range(100, 300, 10)
            'n_estimators': [100],
            'max_depth': [8],
            'min_child_weight': [50],
            # 'scale_pos_weight': [0.333536]
        }
    },
    'LGB': {
        'fit': {

        },
        'cv': {

        }
    },
    'RF': {
        'fit': {
            'n_estimators': 150,
            'min_samples_leaf': 2,
            'max_depth': 18,
            'min_samples_split': 6,
            'oob_score': True
        },
        'cv': {
            # 'n_estimators': range(150, 250, 5),
            'n_estimators': [150],
            # 'min_samples_leaf': range(1, 5, 1),
            'min_samples_leaf': [2],
            # 'max_depth': range(5, 20, 1),
            'max_depth': [18],
            # 'min_samples_split': range(4, 12, 1),
            'min_samples_split': [6],
            'oob_score': [True]
        }
    },
    'MatchPyramid': {
        'conv1_1': (1, 20, 3),
        'conv1_2': (20, 20, 3),
        'pool1': (10,),
        'conv2': (20, 36, 3),
        'pool2': (2, 2),
        'mlp3': (36 * 4 * 4, 200),
        'mlp4': (200, 2),
        'lr': (0.1, 0.001),
        'lr_gamma': 0.5,
        'momentum': 0.9,
        'init_weight': True,

        'batch_size': 200,
        'epoch': 100,
        'auto_lr_epoch': 10,
        'summary_count': 50,
        'save_epoch': 20
    },
    'LSTM_VEC': {
        'input_size': 300,
        'hidden_size': 300,
        'dropout': 0.5,
        'init_weight': True,
        'num_layers': 2,
        'lr': (0.1, 0.001),
        'lr_gamma': 0.5,
        'momentum': 0.9,

        'epoch': 100,
        'batch_size': 200,
        'auto_lr_epoch': 5,
        'summary_count': 50,
        'save_epoch': 20
    },
}
