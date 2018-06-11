# -*- coding: utf-8 -*-
import string
from zhon import hanzi
from nltk.corpus import stopwords


class Punctuations(object):
    AllPuncUnicode = string.punctuation + hanzi.punctuation
    ZHPuncUnicode = hanzi.punctuation
    ENPuncUnicode = string.punctuation
    AllPunc = AllPuncUnicode
    ZHPunc = ZHPuncUnicode
    ENPunc = ENPuncUnicode
    PUNCTUATIONSUnicode = "".join(set(ENPuncUnicode + ZHPuncUnicode + u''':!),.:;?]}¢'"、。〉》」』】〕〗〞︰︱︳﹐､﹒﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻︽︿﹁﹃﹙﹛﹝（｛“‘-—_…'''))
    PUNCTUATIONS = PUNCTUATIONSUnicode


class StopWords(object):
    StopWordsEN = set(stopwords.words('english'))
