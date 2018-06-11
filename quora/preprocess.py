# -*- coding: utf-8 -*-
import os, sys
sys.path.append(os.path.abspath('..'))
import nltk
import pandas as pd
from quora import config
from quora.util.translate.baidu import BaiDu
from nltk.stem import SnowballStemmer


class Channel(object):

    def __init__(self):
        self.file = ''
        self.channel = ''
        self.process_stage = []

    def init_data(self):
        assert self.file, 'File Name must exit.'
        self.data = pd.read_csv(self.file).fillna('')

    def process(self):
        self.init_data()
        for stage in self.process_stage:
            stage()
        self.save()

    def save(self):
        assert self.channel, 'Channel Name must exit.'
        self.data.to_csv(self.channel)


class OriginChannel(Channel):

    def __init__(self):
        Channel.__init__(self)


class TextCleanChannel(Channel):

    def __init__(self):
        Channel.__init__(self)
        # unit
        unit_paris = [
            (r"(\d+)kgs ", lambda m: m.group(1) + ' kg '),  # e.g. 4kgs => 4 kg
            (r"(\d+)kg ", lambda m: m.group(1) + ' kg '),  # e.g. 4kg => 4 kg
            (r"(\d+)k ", lambda m: m.group(1) + '000 '),  # e.g. 4k => 4000
            (r"\$(\d+)", lambda m: m.group(1) + ' dollar '),
            (r"(\d+)\$", lambda m: m.group(1) + ' dollar ')]
        # acronym
        acronym_pairs = [
            (r"can\'t", "can not"),
            (r"cannot", "can not "),
            (r"what\'s", "what is"),
            (r"What\'s", "what is"),
            (r"\'ve ", " have "),
            (r"n\'t", " not "),
            (r"i\'m", "i am "),
            (r"I\'m", "i am "),
            (r"\'re", " are "),
            (r"\'d", " would "),
            (r"\'ll", " will "),
            (r"c\+\+", "cplusplus"),
            (r"c \+\+", "cplusplus"),
            (r"c \+ \+", "cplusplus"),
            (r"c#", "csharp"),
            (r"f#", "fsharp"),
            (r"g#", "gsharp"),
            (r" e mail ", " email "),
            (r" e \- mail ", " email "),
            (r" e\-mail ", " email "),
            (r",000", '000'),
            (r"\'s", " ")]
        # spelling correction
        spelling_pairs = [
            (r"ph\.d", "phd"),
            (r"PhD", "phd"),
            (r"pokemons", "pokemon"),
            (r"pokémon", "pokemon"),
            (r"pokemon go ", "pokemon-go "),
            (r" e g ", " eg "),
            (r" b g ", " bg "),
            (r" 9 11 ", " 911 "),
            (r" j k ", " jk "),
            (r" fb ", " facebook "),
            (r"facebooks", " facebook "),
            (r"facebooking", " facebook "),
            (r"insidefacebook", "inside facebook"),
            (r"donald trump", "trump"),
            (r"the big bang", "big-bang"),
            (r"the european union", "eu"),
            (r" usa ", " america "),
            (r" us ", " america "),
            (r" u s ", " america "),
            (r" U\.S\. ", " america "),
            (r" US ", " america "),
            (r" American ", " america "),
            (r" America ", " america "),
            (r" quaro ", " quora "),
            (r" mbp ", " macbook-pro "),
            (r" mac ", " macbook "),
            (r"macbook pro", "macbook-pro"),
            (r"macbook-pros", "macbook-pro"),
            (r" 1 ", " one "),
            (r" 2 ", " two "),
            (r" 3 ", " three "),
            (r" 4 ", " four "),
            (r" 5 ", " five "),
            (r" 6 ", " six "),
            (r" 7 ", " seven "),
            (r" 8 ", " eight "),
            (r" 9 ", " nine "),
            (r"googling", " google "),
            (r"googled", " google "),
            (r"googleable", " google "),
            (r"googles", " google "),
            (r" rs(\d+)", lambda m: ' rs ' + m.group(1)),
            (r"(\d+)rs", lambda m: ' rs ' + m.group(1)),
            (r"the european union", " eu "),
            (r"dollars", " dollar ")]
        # punctuation
        punctuation_pairs = [
            (r"\+", " + "),
            (r"'", " "),
            (r"-", " - "),
            (r"/", " / "),
            (r"\\", " \ "),
            (r"=", " = "),
            (r"\^", " ^ "),
            (r":", " : "),
            (r"\.", " . "),
            (r",", " , "),
            (r"\?", " ? "),
            (r"!", " ! "),
            (r"\"", " \" "),
            (r"&", " & "),
            (r"\|", " | "),
            (r";", " ; "),
            (r"\(", " ( "),
            (r"\)", " ( "),
        ]
        # symbol replacement
        symbol_pairs = [
            (r"&", " and "),
            (r"\|", " or "),
            (r"=", " equal "),
            (r"\+", " plus "),
            (r"₹", " rs "),
            (r"\$", " dollar ")]
        self.clean_pairs = acronym_pairs + spelling_pairs + punctuation_pairs +\
                           symbol_pairs + unit_paris
        self.process_stage = [self.clean,
                              self.steam]
        self.file = config.origin_train_file
        self.channel = config.clean_test_file

    def clean(self):
        self.data['question1'] = self.data['question1'].str.lower()
        self.data['question2'] = self.data['question2'].str.lower()
        for (old_rex, new_rex) in self.clean_pairs:
            self.data['question1'] = self.data['question1'].str.replace(old_rex, new_rex)
            self.data['question2'] = self.data['question2'].str.replace(old_rex, new_rex)

    def steam(self):
        _stem = SnowballStemmer('english')
        self.data['question1'] = self.data.question1.map(lambda x: ' '.join(
            [_stem.stem(word) for word in nltk.word_tokenize(x)]))
        self.data['question2'] = self.data.question2.map(lambda x: ' '.join(
            [_stem.stem(word) for word in nltk.word_tokenize(x)]))


class PosChannel(Channel):

    def __init__(self):
        Channel.__init__(self)


class TranslateChannel(Channel):

    def __init__(self):
        Channel.__init__(self)
        self.engine = BaiDu()
        self.process_stage = [self.translate]
        self.file = config.origin_train_file
        self.channel = config.zh_train_file

    def translate(self):
        f = open(config.zh_train_file_txt, 'a')
        for _, row in self.data.iterrows():
            print(_)
            seq1 = str(row['question1']).strip()
            seq2 = str(row['question2']).strip()
            zh1 = self.engine.translate(seq1)
            f.write(str(_) + ' ' + zh1 + '\n')
            zh2 = self.engine.translate(seq2)
            f.write(str(_) + ' ' + zh2 + '\n')


class ConcurrenceDelChannel(Channel):

    def __init__(self):
        Channel.__init__(self)


if __name__ == '__main__':
    TranslateChannel().process()
