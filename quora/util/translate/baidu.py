# -*- coding: utf-8 -*-
import execjs
import demjson, requests
from quora import config


def error(func):
    def wrapper(self, *args, **kwargs):
        try:
            result = func(self, *args, **kwargs)
        except Exception as e:
            print(str(e))
            result = func(self, *args, **kwargs)
        return result
    return wrapper


class BaiDu(object):
    def __init__(self):
        self.headers = {
            "Cookie": "BAIDUID=1F6A837B1730B3CCB9E65C1757365239:FG=1; BIDUPSID=1F6A837B1730B3CCB9E65C1757365239; "
                      "PSTM=1525403957; REALTIME_TRANS_SWITCH=1; FANYI_WORD_SWITCH=1; HISTORY_SWITCH=1; "
                      "SOUND_SPD_SWITCH=1; SOUND_PREFER_SWITCH=1;H_PS_PSSID=1426_21092_26350_20929; "
                      "BDSFRCVID=C80sJeC62CrXfR6ApGfjUONONf4hm05TH6aIAcJBFBFLTumcX4keEG0PDf8g0KubMVkPogKK0eOTHk6P; "
                      "H_BDCLCKID_SF=tJu8oIt-JIK3fnKkb-nfhPt_qxby26POyeceaJ5nJDoAsj6SKPTNK-LeMbo9WfRwXbvBXxj4QpP-HJ7YKh"
                      "J_3J3W-xTDKnIj5IcrKl0MLpbtbb0xynoDX-D0MfnMBMPj5mOnankKLIFKMKtwe5-KjTPW5ptX--Q0aCOasJOOaCv1HInRy4"
                      "oTj6DDeJ8Ob5JC-ITyof5vBq7Fs-bjDP6-3MvB-fndaboKaHnNXUc-a45RqlbHQft20M0AeMtjBbQaW26r5R7jWhk2eq72y"
                      "-4-05TXja-qt6DOtJ3fL-08KbnEDRjwMR-_h4L3qxbXq5O-W57Z0l8KttKaJl3wyxQ2y4Fq3tnwWM6JtKoZWJcmWILh8hPmj"
                      "qr85UAk3-RetxvdBHR4KKJx-4PWeIJo5t5s5n8ehUJiB5O-Ban7BKQxfD_MhKI6e5-aen-W5gTQa4Qj2C_X3b7Ef-nfh-O"
                      "_bf--D6-gyUvk5fojWD5Q-l7kBhrqsC5bKpQxy550X-b4-lcLHm7-oUbH0RrHeUbHQT3mQhQbbN3i-4ji04DOWb3cWKJq8Ub"
                      "SMTjme6j3ja-DJ5-DfK7QBROo-b5KfRKkb-QK5bt8-q52aI6X56Pssln1-hcqEIL4Ln5ibljBLq3xBtnPB2o92JL22nC2Mfb"
                      "Sj4QzLttpL4jG0f7nbCj2hIbo5l5nhMJeb67JDMP0-4cpahOy523ion5vQpnOMxtuD68MDToyjNLs5-70KK3e04oK56rfHt"
                      "omKPoHK4tJqxT--TnmfnReaJ5nJDoTqxnI06jNKM-EMRo9WfRw5Jn3WCtaQpP-HqTq-U6IQxDzXbOt24JvJI3-Kl0MLpbtbb"
                      "0xynoDXqKf0MnMBMPj5mOnankKLIcjqR8ZDT8bDTjP; "
                      "BDORZ=B490B5EBF6F3CD402E515D22BCDA1598; "
                      "from_lang_often=%5B%7B%22value%22%3A%22en%22%2C%22text%22%3A%22%u82F1%u8BED%22%7D%2C%7B%22value"
                      "%22%3A%22zh%22%2C%22text%22%3A%22%u4E2D%u6587%22%7D%5D; "
                      "PSINO=5; locale=zh; Hm_lvt_64ecd82404c51e03dc91cb9e8c025574=1526953787,1526954667,1526955278,"
                      "1526984876; "
                      "Hm_lpvt_64ecd82404c51e03dc91cb9e8c025574=1526984876; to_lang_often=%5B%7B%22value%22%3A%22zh%22%"
                      "2C%22text%22%3A%22%u4E2D%u6587%22%7D%2C%7B%22value%22%3A%22en%22%2C%22text%22%3A%22%u82F1%u8BED%"
                      "22%7D%5D",
        }
        self.url = 'http://fanyi.baidu.com/v2transapi'
        self.init_ctx()

    def init_ctx(self):
        f = open(config.abs_path.format("util/translate/des_rsa.js"), 'r', encoding='UTF-8')
        content = f.read()
        f.close()
        self.ctx = execjs.compile(content)

    @error
    def translate(self, query):
        data = {
            'from': 'en',
            'to': 'zh',
            'query': query,
            'transtype': 'translang',
            'simple_means_flag': '3',
            'sign': self.ctx.call('detect', query),
            'token': '1dc19b5d66fa4d0e13d2ec22449e268a'
        }
        response = requests.post(self.url, data=data, headers=self.headers, timeout=10)
        string = response.text
        dic = demjson.decode(string)
        result = str(dic['trans_result']['data'][0]['dst'])
        return result


if __name__ == '__main__':
    res = BaiDu().translate('What is the step by step guide to invest in share market in india?')
    print(res)
