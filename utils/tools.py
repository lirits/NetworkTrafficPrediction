import pandas as pd


def sliding_windows(
        time_values,
        data_values,
        windows_size: int,
        target_size: int,
        start_index=0) -> list:
    X = []
    y = []
    for i in range(len(time_values)):
        X.append(data_values[start_index:start_index + windows_size])
        y.append(data_values[start_index +
                             windows_size:start_index +
                             windows_size +
                             target_size])
        start_index += 1
    return X, y


def preprocessing(data_frame, CellName):
    data_frame = data_frame[data_frame['CellName'] == CellName].copy()
    data_frame.loc[:, 'Date'] = pd.to_datetime(data_frame.Date.astype(str))
    data_frame.loc[:, 'Hour'] = pd.to_timedelta(data_frame.Hour, unit='h')
    data_frame.loc[:, 'Date_time'] = pd.to_datetime(
        data_frame.Date + data_frame.Hour)
    data_frame = data_frame.drop(['Date', 'Hour', 'CellName'], axis=1)
    data_frame = data_frame.set_index('Date_time')
    data_frame = data_frame.sort_values(by='Date_time')
    return data_frame


if __name__ == '__main__':
    import numpy as np
    x = np.arange(100)
    X, y = sliding_windows(x, x, 12, 1)
    print('Test-sliding-windows-function')
    print(len(X), '\n', X[0], y[0], '\n', X[1], y[1])
