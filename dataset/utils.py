import numpy as np

import time

def string2timestamp(strings, offset_frame):
    ts = []
    for t in strings:
        dtstr = '-'.join([t[:4].decode(), t[4:6].decode(), t[6:8].decode()])
        slot = int(t[8:])-1
        ts.append(np.datetime64(dtstr, 'm') + slot * offset_frame)

    return ts


def timestamp2array(timestamps, halfhour_flag=False):
    vec_wday = [time.strptime(str(t)[:10], '%Y-%m-%d').tm_wday for t in timestamps]
    vec_hour = [time.strptime(str(t)[11:13], '%H').tm_hour for t in timestamps]
    vec_halfhour = [time.strptime(str(t)[14:16], '%M').tm_min for t in timestamps]
    ret = []
    for idx, wday in enumerate(vec_wday):
        #day
        v = [0 for _ in range(7)]
        v[wday] = 1
        if wday >= 6 or wday <= 0: 
            v.append(0)  # weekend
        else:
            v.append(1)  # weekday

        #hour
        if halfhour_flag:
            sign_of_day = 48
            sign_of_hour = 2
            plus_one_flag = 1 if vec_halfhour[idx] >= 30 else 0
        else:
            sign_of_day = 24
            sign_of_hour = 1
            plus_one_flag = 0

        v += [0 for _ in range(sign_of_day)]
        hour = vec_hour[idx]
        v[hour * sign_of_hour + 8 + plus_one_flag] = 1
        if hour >= 18 or hour < 6:
            v.append(0) # night
        else:
            v.append(1) # day
        ret.append(v)
        
    return np.asarray(ret)

def timestamp2vec_origin(timestamps):
    vec = [time.strptime(str(t)[:10], '%Y-%m-%d').tm_wday for t in timestamps]  # python3
    #vec = [time.strptime(t[:8], '%Y%m%d').tm_wday for t in timestamps]  # python2
    ret = []
    for i in vec:
        v = [0 for _ in range(7)]
        v[i] = 1
        if i >= 5:
            v.append(0)  # weekend
        else:
            v.append(1)  # weekday
        ret.append(v)
    return np.asarray(ret)

def transtr(ts):
    tstr = str(ts)
    return tstr[:4] + tstr[5:7] + tstr[8:10]

def transtrlong(ts):
    tstr = str(ts)
    return tstr[:4] + tstr[5:7] + tstr[8:10] + str((int(tstr[11:13]) * 2 + 0 if tstr[14:16] == '00' else 1) + 1).zfill(2)

def transtr24(ts):
    tstr = str(ts)
    return tstr[:4] + tstr[5:7] + tstr[8:10] + str(int(tstr[11:13]) + 1).zfill(2)
