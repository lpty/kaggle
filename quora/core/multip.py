# -*- coding: utf-8 -*-
import sys
import time
import os
from multiprocessing import Pool, Queue

queue = Queue()


def _worker(calculate, data_split, columns):
    data = {}
    count = 0
    for i, row in data_split.iterrows():
        res = calculate(row)
        for column in columns:
            data[column] = data.get(column, []) + [res[columns.index(column)]]
        count += 1
        if count % 100 == 0: print('PID: {}, All: {}, COUNT: {}'.format(
            str(os.getpid()), str(data_split.shape[0]), str(count)))
    queue.put((i, data))
    print('PID: {}, All: {}, COUNT: {}, END.'.format(str(os.getpid()), str(data_split.shape[0]), str(count)))
    sys.exit(0)


def _map(data, calculate, columns, n):
    print('Map...')
    split_count = round(data.shape[0] / n)
    datas = []
    for i in range(0, n):
        if i == n-1:
            split = data.loc[i * split_count:, :]
        else:
            split = data.loc[i * split_count: ((i + 1) * split_count) - 1, :]
        datas.append(split)
    pool = Pool(n)
    for data_split in datas:
        pool.apply_async(_worker, args=(calculate, data_split, columns))
    pool.close()
    # pool.join()
    print('Map End.')


def _reduce(n):
    while queue.qsize() != n:
        print('Check...')
        time.sleep(1*60)
    print('Reduce...')
    que = [queue.get() for _ in range(queue.qsize())]
    que = sorted(que, key=lambda x: x[0])
    data = que[0][1]
    for q in que[1:]:
        for column in data:
            data[column] = data[column] + q[1][column]
    print('Reduce End.')
    return data


def map_reduce(data, calculate, columns, n):
    _map(data, calculate, columns, n)
    return _reduce(n)
