import os
import numpy as np
import h5py

from .utils import timestamp2array, timestamp2vec_origin, transtr, transtrlong, transtr24


def external_taxibj(datapath, fourty_eight, previous_meteorol):
    def f(tsx, tsy, ext_time):
        exd = ExtDat(datapath)
        tsx = np.asarray([exd.get_bjextarray(N, ext_time, fourty_eight=fourty_eight) for N in tsx])  # N * len_seq
        tsy = exd.get_bjextarray(tsy, ext_time, fourty_eight=fourty_eight, previous_meteorol=previous_meteorol)  # N

        print('there are totally', exd.tot_holiday, 'holidays in constructed data')

        return tsx, tsy

    return f


def external_bikenyc():
    def f(tsx, tsy, ext_time):
        timestampfunc = timestamp2array if ext_time else timestamp2vec_origin
        tsx = np.asarray([timestampfunc(N) for N in tsx])
        tsy = timestampfunc(tsy)
        return tsx, tsy
    return f

def external_taxinyc(datapath, fourty_eight, previous_meteorol):
    def f(tsx, tsy, ext_time):
        exd = ExtDat(datapath, dataset='TaxiNYC')
        tsx = np.asarray([exd.get_bjextarray(N, ext_time, fourty_eight=fourty_eight) for N in tsx])  # N * len_seq
        tsy = exd.get_bjextarray(tsy, ext_time, fourty_eight=fourty_eight, previous_meteorol=previous_meteorol)  # N

        print('there are totally', exd.tot_holiday, 'holidays in constructed data')

        return tsx, tsy

    return f

class ExtDat:
    def __init__(self, datapath, dataset='TaxiBJ'):
        self.tot_holiday = 0

        self.holidayfname = os.path.join(datapath, dataset, 'Holiday.txt')
        f = open(self.holidayfname, 'r')
        holidays = f.readlines()
        self.holidays = set([h.strip() for h in holidays])
        '''
        timeslots: the predicted timeslots
        In real-world, we dont have the meteorol data in the predicted timeslot, instead, we use the meteoral at
         previous timeslots, i.e., slot = predicted_slot - timeslot (you can use predicted meteorol data as well)
        '''
        fname = os.path.join(datapath, dataset, 'Meteorology.h5')
        f = h5py.File(fname, 'r')
        timeslots = f['date'].value
        wind_speed = f['WindSpeed'].value
        weather = f['Weather'].value
        temperature = f['Temperature'].value
        f.close()

        self.M = dict()  # map timeslot to index
        for i, slot in enumerate(timeslots):
            self.M[slot.decode()] = i
        ws = []  # WindSpeed
        wr = []  # Weather
        te = []  # Temperature
        for slot in timeslots:
            cur_id = self.M[slot.decode()]
            ws.append(wind_speed[cur_id])
            wr.append(weather[cur_id])
            te.append(temperature[cur_id])

        ws = np.asarray(ws)
        wr = np.asarray(wr)
        te = np.asarray(te)

        # 0-1 scale
        ws = 1. * (ws - ws.min()) / (ws.max() - ws.min())
        te = 1. * (te - te.min()) / (te.max() - te.min())

        print("meteor shape: ", ws.shape, wr.shape, te.shape)
        # concatenate all these attributes
        self.meteor_data = np.hstack([wr, ws[:, None], te[:, None]])

    def get_bjextarray(self, timestamp_list, ext_time, fourty_eight=False, previous_meteorol=False):
        vecs_timestamp = timestamp2array(timestamp_list, fourty_eight) if ext_time else timestamp2vec_origin(timestamp_list)
        bits_holiday = self.get_holidayarray(timestamp_list)
        vecs_meteorol = self.get_meteorolarray(timestamp_list, previous_meteorol, fourty_eight)

        return np.hstack([vecs_timestamp, bits_holiday, vecs_meteorol])

    def get_holidayarray(self, timeslots):
        h = [0 for _ in range(len(timeslots))]
        for i, slot in enumerate(timeslots):
            transformat = transtr(slot)
            if transformat in self.holidays:
                h[i] = 1
                self.tot_holiday += 1
        return np.vstack(h)

    def get_meteorolarray(self, timestamp_list, previous_meteorol, fourty_eight):
        if fourty_eight:
            return np.array(
                [
                    self.meteor_data[
                        self.M[transtrlong(ts-np.timedelta64(30, 'm') if previous_meteorol else ts)]
                    ] for ts in timestamp_list
                ]
            )
        else:
            return np.array(
                [
                    self.meteor_data[
                        self.M[transtr24(ts-np.timedelta64(60, 'm') if previous_meteorol else ts)]
                    ] for ts in timestamp_list
                ]
            )
