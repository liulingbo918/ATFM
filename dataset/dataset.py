import numpy as np
import h5py
import os
import math
import torch
import torch.utils.data as data

from .external import external_taxibj, external_bikenyc, external_taxinyc
from .minmax_normalization import MinMaxNormalization
from .data_fetcher import DataFetcher

class Dataset:
    datapath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

    def __init__(self, dconf, test_days=-1, datapath=datapath):
        self.dconf = dconf
        self.dataset = dconf.name
        self.datapath = datapath
        if self.dataset == 'TaxiBJ':
            self.datafolder = 'TaxiBJ'
            self.dataname = [
                'BJ13_M32x32_T30_InOut.h5',
                'BJ14_M32x32_T30_InOut.h5',
                'BJ15_M32x32_T30_InOut.h5',
                'BJ16_M32x32_T30_InOut.h5'
            ]
            self.nb_flow = 2
            self.dim_h = 32
            self.dim_w = 32
            self.T = 48
            test_days = 28 if test_days == -1 else test_days

            self.external_builder = external_taxibj(
                self.datapath,
                fourty_eight=dconf.fourty_eight,
                previous_meteorol=dconf.previous_meteorol
            )
            self.m_factor = 1.
            ext_dim = 28
            if dconf.ext_time_flag:
                ext_dim += 25
                if dconf.fourty_eight:
                    ext_dim += 24
            self.ext_dim = ext_dim

        elif self.dataset == 'BikeNYC':
            self.datafolder = 'BikeNYC'
            self.dataname = ['NYC14_M16x8_T60_NewEnd.h5']
            self.nb_flow = 2
            self.dim_h = 16
            self.dim_w = 8
            self.T = 24
            test_days = 10 if test_days == -1 else test_days

            self.external_builder = external_bikenyc()
            self.m_factor = math.sqrt(1. * 16 * 8 / 81)
            self.ext_dim = 33 if dconf.ext_time_flag else 8

        elif self.dataset == 'TaxiNYC':
            self.datafolder = 'TaxiNYC'
            self.dataname = ['NYC2014.h5']
            self.nb_flow = 2
            self.dim_h = 15
            self.dim_w = 5
            self.T = 48
            test_days = 28 if test_days == -1 else test_days

            self.external_builder = external_taxinyc(
                self.datapath,
                fourty_eight=dconf.fourty_eight,
                previous_meteorol=dconf.previous_meteorol
            )
            self.m_factor = math.sqrt(1. * 15 * 5 / 64)
            self.ext_dim = 77

        else:
            raise ValueError('Invalid dataset')

        self.len_test = test_days * self.T
        self.portion = dconf.portion

    def get_raw_data(self):
        """
         data:
         np.array(n_sample * n_flow * height * width)
         ts:
         np.array(n_sample * length of timestamp string)
        """
        raw_data_list = list()
        raw_ts_list = list()
        print("  Dataset: ", self.datafolder)
        for filename in self.dataname:
            f = h5py.File(os.path.join(self.datapath, self.datafolder, filename), 'r')
            _raw_data = f['data'].value
            _raw_ts = f['date'].value
            f.close()

            raw_data_list.append(_raw_data)
            raw_ts_list.append(_raw_ts)
        # delete data over 2channels

        return raw_data_list, raw_ts_list

    @staticmethod
    def remove_incomplete_days(data, timestamps, t=48):
        print("before removing", len(data))
        # remove a certain day which has not 48 timestamps
        days = []  # available days: some day only contain some seqs
        days_incomplete = []
        i = 0
        while i < len(timestamps):
            if int(timestamps[i][8:]) != 1:
                i += 1
            elif i + t - 1 < len(timestamps) and int(timestamps[i + t - 1][8:]) == t:
                days.append(timestamps[i][:8])
                i += t
            else:
                days_incomplete.append(timestamps[i][:8])
                i += 1
        print("incomplete days: ", days_incomplete)
        days = set(days)
        idx = []
        for i, t in enumerate(timestamps):
            if t[:8] in days:
                idx.append(i)

        data = data[idx]
        timestamps = [timestamps[i] for i in idx]
        print("after removing", len(data))
        return data, timestamps

    def trainset_of(self, vec):
        return vec[:math.floor((len(vec)-self.len_test) * self.portion)]

    def testset_of(self, vec):
        return vec[-math.floor(self.len_test * self.portion):]

    def split(self, x, y):
        x_train = self.trainset_of(x)
        x_test = self.testset_of(x)
        y_train = self.trainset_of(y)
        y_test = self.testset_of(y)

        return x_train, y_train, x_test, y_test

    def load_data(self):
        """
        return value:
            X_train & X_test: [XC, XP, XT, Xext]
            Y_train & Y_test: vector
        """
        # read file and place all of the raw data in np.array. 'ts' means timestamp
        # without removing incomplete days
        print('Preprocessing: Reading HDF5 file(s)')
        raw_data_list, ts_list = self.get_raw_data()

        data_list, ts_new_list = [], []
        for idx in range(len(ts_list)):
            raw_data = raw_data_list[idx]
            ts = ts_list[idx]

            if self.dconf.rm_incomplete_flag:
                raw_data, ts = self.remove_incomplete_days(raw_data, ts, self.T)

            data_list.append(raw_data)
            ts_new_list.append(ts)

        raw_data = np.concatenate(data_list)

        print('Preprocessing: Min max normalizing')
        mmn = MinMaxNormalization()
        train_dat = self.trainset_of(raw_data)
        mmn.fit(train_dat)
        new_data_list = [
            mmn.transform(data).astype('float32', copy=False)
            for data in data_list
        ]

        x_list, y_list, ts_x_list, ts_y_list = [], [], [], []
        for idx in range(len(ts_new_list)):
            x, y, ts_x, ts_y = \
                DataFetcher(new_data_list[idx], ts_new_list[idx], self.T).fetch_data(self.dconf)
            x_list.append(x)
            y_list.append(y)
            ts_x_list.append(ts_x)
            ts_y_list.append(ts_y)
        x = np.concatenate(x_list)
        y = np.concatenate(y_list)
        ts_x = np.concatenate(ts_x_list)
        ts_y = np.concatenate(ts_y_list)

        x_train, y_train, x_test, y_test = self.split(x, y)

        if self.dconf.ext_flag:
            ext_x, ext_y = self.external_builder(ts_x, ts_y, ext_time=self.dconf.ext_time_flag)
            ext_x_train, ext_y_train, ext_x_test, ext_y_test = self.split(ext_x, ext_y)
        else:
            ext_x_train, ext_y_train, ext_x_test, ext_y_test = None, None, None, None

        class TempClass:
            def __init__(self_2):
                self_2.X_train = x_train
                self_2.Y_train = y_train
                self_2.X_test = x_test
                self_2.Y_test = y_test
                self_2.ext_X_train = ext_x_train
                self_2.ext_Y_train = ext_y_train
                self_2.ext_X_test = ext_x_test
                self_2.ext_Y_test = ext_y_test
                self_2.mmn = mmn
                self_2.ts_Y_train = self.trainset_of(ts_y)
                self_2.ts_Y_test = self.testset_of(ts_y)
            
            def show(self_2):
                print(
                    "Run: X inputs shape: ", self_2.X_train.shape, self_2.X_test.shape,
                    "Y inputs shape: ", self_2.Y_train.shape, self_2.Y_test.shape
                )
                print("Run: min~max: ", self_2.mmn.min, '~', self_2.mmn.max)
                if self.dconf.ext_flag:
                    print(
                        "Run: X-ext inputs' shapes:", self_2.ext_X_train.shape, self_2.ext_X_test.shape,
                        "Y-ext inputs' shapes:", self_2.ext_Y_train.shape, self_2.ext_Y_test.shape
                    )
        return TempClass()

class TorchDataset(data.Dataset):
    def __init__(self, ds, mode='train'):
        super(TorchDataset, self).__init__()
        self.ds = ds
        self.mode = mode
    
    def __getitem__(self, index):
        if self.mode == 'train':
            X = torch.from_numpy(self.ds.X_train[index])
            X_ext = torch.from_numpy(self.ds.ext_X_train[index])
            Y = torch.from_numpy(self.ds.Y_train[index])
            Y_ext = torch.from_numpy(self.ds.ext_Y_train[index])
        else:
            X = torch.from_numpy(self.ds.X_test[index])
            X_ext = torch.from_numpy(self.ds.ext_X_test[index])
            Y = torch.from_numpy(self.ds.Y_test[index])
            Y_ext = torch.from_numpy(self.ds.ext_Y_test[index])

        return X.float(), X_ext.float(), Y.float(), Y_ext.float()

    def __len__(self):
        if self.mode == 'train':
            return self.ds.X_train.shape[0]
        else:
            return self.ds.X_test.shape[0]

class DatasetFactory(object):
    def __init__(self, dconf):
        self.dataset = Dataset(dconf)
        self.ds = self.dataset.load_data()

    def get_train_dataset(self):
        return TorchDataset(self.ds, 'train')
    
    def get_test_dataset(self):
        return TorchDataset(self.ds, 'test')

if __name__ == '__main__':
    class DataConfiguration:
        # Data
        name = 'TaxiBJ'
        portion = 1.  # portion of data

        len_close = 3
        len_period = 2
        len_trend = 0
        pad_forward_period = 0
        pad_back_period = 0
        pad_forward_trend = 0
        pad_back_trend = 0

        len_all_close = len_close * 1
        len_all_period = len_period * (1 + pad_back_period + pad_forward_period)
        len_all_trend = len_trend * (1 + pad_back_trend + pad_forward_trend)

        len_seq = len_all_close + len_all_period + len_all_trend
        cpt = [len_all_close, len_all_period, len_all_trend]

        interval_period = 1
        interval_trend = 7

        ext_flag = True
        ext_time_flag = True
        rm_incomplete_flag = True
        fourty_eight = True
        previous_meteorol = True

    df = DatasetFactory(DataConfiguration())
    ds = df.get_train_dataset()
    X, X_ext, Y, Y_ext = next(iter(ds))
    print('train:')
    print(X.size())
    print(X_ext.size())
    print(Y.size())
    print(Y_ext.size())

    ds = df.get_train_dataset()
    X, X_ext, Y, Y_ext = next(iter(ds))
    print('test:')
    print(X.size())
    print(X_ext.size())
    print(Y.size())
    print(Y_ext.size())