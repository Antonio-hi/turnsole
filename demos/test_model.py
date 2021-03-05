# -*- coding: utf-8 -*-
# @Author        : Lyu Kui
# @Email         : 9428.al@gmail.com
# @Created Date  : 2021-03-05 16:51:22
# @Last Modified : 2021-03-05 16:51:57
# @Description   :

from turnsole import model

if __name__ == '__main__':
    model = model.dbnet(phi=0)
    model.summary()

    import time
    import numpy as np

    x = np.random.random_sample((1, 640, 640, 3))
    # warm up
    output = model.predict(x)

    print('\n[INFO] Test start')
    time_start = time.time()
    for i in range(1000):
        output = model.predict(x)
        
        time_end = time.time()
        print('[INFO] Time used: {:.2f} ms'.format((time_end - time_start)*1000/(i+1)))