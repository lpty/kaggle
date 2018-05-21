# -*- coding: utf-8 -*-
"""
CORPUS
-------
A corpus Pre-processing, for kaggle titanic.
"""
import pandas as pd
from titanic import config
from sklearn.preprocessing import StandardScaler


class Corpus(object):

    def __init__(self):
        self.initialize_corpus()
        self.feature_engineer()
        self.standardlize()

    def initialize_corpus(self):
        """
        Read corpus.
        """
        train = pd.read_csv(config.train_path)
        test = pd.read_csv(config.test_path)
        test['Survived'] = 0
        self.__dict__['database'] = train.append(test)

    def feature_engineer(self):
        """
        extract feature from original data.
        """
        feature = Feature(self.database)
        feature.extract()
        self.database = feature.database

    def standardlize(self):
        """
        multi dimension one-hot vector should be normalized.
        """
        train_data = self.database[:891]
        submission_data = self.database[891:]
        train_data_x = train_data.drop(['Survived'], axis=1)
        self.train_data_y = train_data['Survived']
        submission_data_x = submission_data.drop(['Survived'], axis=1)
        std = StandardScaler()
        std.fit(train_data_x)
        self.train_data_x_std = std.transform(train_data_x)
        self.submission_data_x_std = std.transform(submission_data_x)


class Feature(object):

    def __init__(self, database):
        self.database = database

    def extract_pclass(self):
        """
        Pclass may means social identity or wealth
        """
        self.database = pd.get_dummies(self.database, columns=['Pclass'], prefix='P')

    def extract_name(self):
        """
        Prefix of Name, as call Name1, can figure out women or man, master and the other rare prefix may means some
        kind of social identity such as royalty.
        Family name, as call Name2, it suppose that have the same Family Name may be one family.And they would have
        some relation about alive.
        """
        self.database['Name1'] = self.database['Name'].str.extract('.*?,(.*?)\.', expand=False).str.strip()
        self.database['Name1'].replace(['Master'], 'Master', inplace=True)
        self.database['Name1'].replace(['Jonkheer', 'Don', 'Sir', 'the Countess', 'Dona', 'Lady',
                                        'Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Royalty', inplace=True)
        self.database['Name1'].replace(['Mme', 'Ms', 'Mrs', 'Mlle', 'Miss'], 'Women', inplace=True)
        self.database['Name1'].replace(['Mr'], 'Man', inplace=True)
        self.database = pd.get_dummies(self.database, columns=['Name1'], prefix='N')

        self.database['Name2_'] = self.database['Name'].apply(lambda x: x.split('.')[1].strip())
        names = self.database['Name2_'].value_counts().reset_index()
        names.columns = ['Name2_', 'Name2_sum']
        self.database = pd.merge(self.database, names, how='left', on='Name2_')
        self.database.loc[self.database['Name2_sum'] <= 2, 'Name2'] = 'small'
        self.database.loc[self.database['Name2_sum'] > 2, 'Name2'] = self.database['Name2_']
        self.database = pd.get_dummies(self.database, columns=['Name2'], prefix='N')

    def extract_sex(self):
        """
        Female have more possibility to live.
        """
        self.database = pd.get_dummies(self.database, columns=['Sex'], prefix='S')

    def extract_family(self):
        """
        we figure out that the family number it a reason for P(alive).
        """
        self.database['F_size'] = self.database['SibSp'] + self.database['Parch'] + 1
        self.database['F_Single'] = self.database['F_size'].map(lambda x: 1 if x == 1 else 0)
        self.database['F_Small'] = self.database['F_size'].map(lambda x: 1 if 2 <= x <= 3 else 0)
        self.database['F_Med'] = self.database['F_size'].map(lambda x: 1 if x == 4 else 0)
        self.database['F_Large'] = self.database['F_size'].map(lambda x: 1 if x >= 5 else 0)

    def extract_ticket(self):
        """
        who share the same ticket may be stay together.
        """
        tpc = self.database['Ticket'].value_counts().reset_index()
        tpc.columns = ['Ticket', 'Ticket_sum']
        self.database = pd.merge(self.database, tpc, how='left', on='Ticket')
        self.database.loc[self.database['Ticket_sum'] == 1, 'T_share'] = 0
        self.database.loc[self.database['Ticket_sum'] != 1, 'T_share'] = 1

    def extract_fare(self):
        """
        The more expansive Ticket mean more P(alive).
        """
        self.database['Fare'].fillna(14.644083, inplace=True)
        self.database['Fare_bin'] = pd.cut(self.database['Fare'], 3, labels=[3, 2, 1])
        self.database = pd.get_dummies(self.database, columns=['Fare_bin'], prefix='F')

    def extract_cabin(self):
        """
        Cabin present of the position of person.
        """
        self.database['Cabin'] = self.database['Cabin'].apply(lambda x: str(x)[0] if pd.notnull(x) else x)
        self.database.loc[self.database['Cabin'].isnull(), 'C_nan'] = 1
        self.database.loc[self.database['Cabin'].notnull(), 'C_nan'] = 0
        self.database = pd.get_dummies(self.database, columns=['Cabin'], prefix='C')

    def extract_embarked(self):
        """
        Embarked have some kind of relation of P(alive).
        """
        self.database['Embarked'].fillna('S')
        self.database = pd.get_dummies(self.database, columns=['Embarked'], prefix='E')

    def extract_age(self):
        """
        Children/adult/elder
        """
        self.database.loc[self.database['Age'].isnull(), 'A_nan'] = 1
        self.database.loc[self.database['Age'].notnull(), 'A_nan'] = 0
        miss_age = self.database.drop(['PassengerId', 'Name', 'Name2_sum', 'Name2_', 'Ticket', 'Fare', 'Survived'], axis=1)
        miss_age_train = miss_age[miss_age['Age'].notnull()]
        miss_age_test = miss_age[miss_age['Age'].isnull()]
        miss_age_train_x = miss_age_train.drop(['Age'], axis=1)
        miss_age_train_y = miss_age_train['Age']
        miss_age_test_x = miss_age_test.drop(['Age'], axis=1)

        std = StandardScaler()
        std.fit(miss_age_train_x)
        miss_age_train_x_std = std.transform(miss_age_train_x)
        miss_age_test_x_std = std.transform(miss_age_test_x)
        from sklearn import linear_model
        model = linear_model.BayesianRidge()
        model.fit(miss_age_train_x_std, miss_age_train_y)
        self.database.loc[self.database['Age'].isnull(), 'Age'] = model.predict(miss_age_test_x_std)
        self.database['Age'] = pd.cut(self.database['Age'], bins=[0, 18, 30, 45, 100], labels=[1, 2, 3, 4])
        self.database = pd.get_dummies(self.database, columns=['Age'], prefix='A')

    def drop_feature(self):
        """
        Drop some feature we don not need.
        """
        feature_columns = ['PassengerId', 'Name', 'Name2_sum', 'Name2_', 'Ticket', 'Fare', 'SibSp', 'Parch', 'F_size']
        self.database = self.database.drop(feature_columns, axis=1)

    def extract(self):
        self.extract_pclass()
        self.extract_name()
        self.extract_sex()
        self.extract_family()
        self.extract_ticket()
        self.extract_fare()
        self.extract_cabin()
        self.extract_embarked()
        self.extract_age()
        self.drop_feature()

if __name__ == '__main__':
    corpus = Corpus()
    x = corpus.train_data_x_std
    y = corpus.train_data_y
    x_submission = corpus.submission_data_x_std
