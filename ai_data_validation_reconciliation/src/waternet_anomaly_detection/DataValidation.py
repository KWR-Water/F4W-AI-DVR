import json
import pandas as pd
import numpy as np
import joblib
import math
import matplotlib.pyplot as plt
from adtk.data import validate_series
from adtk.detector import ThresholdAD, CustomizedDetector1D
import datetime as dt
import pathlib
from os.path import join
import pathlib
from os.path import join

path_metadata = pathlib.Path(__file__).parent.absolute()

file_metadata_limits = 'sensor_metadata.json'

fpath_metadata_limits = join(path_metadata, file_metadata_limits)
# from data.metadata import fpath_metadata_limits


def init_dataframe(data) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    else:
        try:
            return pd.DaraFrame(data.copy())
        except:
            raise Exception("Input data is not convertible to pandas DataFrame")


class DataValidation (object):
    """
    This is the master class for the Data Validation Framework. All common properties are defined here.
    Subsequent children classes inherit from this class.
    """

    # constructor
    def __init__(self, data, tag, correct_format_tseries=True):
        """
        Initialises an instance of the DataValidation class.
        :param data:
        :param tag:
        :param correct_format_tseries: Variable to describe whether the tseries has been already validated using ADTK
        """
        self.raw_data = init_dataframe(data)
        self.data = self.raw_data.copy()
        self.tag = tag
        if correct_format_tseries:
            self._validate_series()
        self._corrected_format_tseries = correct_format_tseries

    def _validate_series(self):
        """
        Using the in-built ADTK method for validation of series to identify and remove duplicates and timestamps with
        no values present.
        :return:
        Replaces the data instance with the validated data.
        """
        self.data = validate_series(self.data)

    @property
    def corrected_format_tseries(self):
        return self._corrected_format_tseries

    @corrected_format_tseries.setter
    def corrected_format_tseries(self, value):
        """
        Method to ensure that the data is validated.
        :param value:
        :return:
        """
        currently_corrected = self._corrected_format_tseries
        if value is False and currently_corrected is False:
            self._corrected_format_tseries = value
        elif value is False and currently_corrected is True:
            raise ValueError("Cannot retrieve the raw data. Set the correct_format_tseries to False when initialising the object.")
        else:
            self._corrected_format_tseries = value
            self._validate_series()

    def find_value_anomaly_flag(self, anomalies):
        """
        A method to get the data values of the timestamps that have been flagged as anomalies
        :param anomalies: dataframe containing the anomaly flags from the different anomaly detection methods.
        :return: pd.Dataframe with the flagged timestamps containing the data values and the rest of the points being
        np.nans
        """

        df_anomalies_values = pd.DataFrame(data=np.nan, index=self.data.index, columns=anomalies.columns)

        for columns in anomalies:
            if anomalies[columns].sum() == 0:
                continue
            else:
                index = anomalies[columns][anomalies[columns] == 1].index
                df_anomalies_values.loc[index, columns] = self.data.loc[index, self.tag]

        return df_anomalies_values

    def lag_finder(self, y2):
        """
        Method to find the lag between two signals.
        :param y2: Second signal used to find the lag with the data in question
        :return: The lab between the signals
        """
        n = len(self.data)
        y1_auto_corr = np.dot(self.data, self.data) / n
        y2_auto_corr = np.dot(y2, y2) / n
        corr = np.correlate(self.data, y2, mode='same') / np.sqrt(y1_auto_corr * y2_auto_corr)
        shift = n // 2
        max_corr = np.max(corr)
        argmax = np.argmax(corr)
        lag = argmax - shift

        return lag

    def calc_residuals(self, y2):
        """
        Find the residuals between two signals while incorporating the lag between the two signals
        (using the lag_finder method).
        :param y2: Second signal used for calculating residuals.
        :return: a pd.Dataframe containing the residual in a column
        """

        df = pd.DataFrame()
        df['y1'] = self.data
        df['y2'] = y2
        lag = self.lag_finder(y2)
        df['Residual'] = df.diff(periods=lag, axis=1)

        return df['Residual']


class UnivariateAD(DataValidation):

    def __init__(self, data, tag=None, param=None, correct_format_tseries=True, sensor_threshold_low=None,
                 sensor_threshold_high=None, process_threshold_low=None, process_threshold_high=None,
                 flatline_min_length=None, offline_min_length=None, get_metadata=True):
        self.tag = tag
        self._sensor_threshold_low = sensor_threshold_low
        self._sensor_threshold_high = sensor_threshold_high
        self._process_threshold_low = process_threshold_low
        self._process_threshold_high = process_threshold_high
        self._flatline_min_length = flatline_min_length
        self._offline_min_length = offline_min_length
        self.param = param

        DataValidation.__init__(self, data=data, tag=tag,
                                correct_format_tseries=correct_format_tseries)
        if get_metadata:
            self.get_metadata_wnt()

    def get_metadata_wnt(self, sensor_lower_threshold_label='sensor_threshold_low',
                         sensor_upper_threshold_label='sensor_threshold_high',
                         process_lower_threshold_label='process_threshold_low',
                         process_upper_threshold_label='process_threshold_high',
                         flatline_min_length_label='flatline_min_length',
                         sensor_offline_min_length_label='offline_min_length',
                         param_label='param'):

        if self.tag is None:
            raise Exception("Please enter valid sensor tag")
        with open(fpath_metadata_limits) as f:
            metadata_dict = json.load(f)

        self._sensor_threshold_low = metadata_dict[self.tag][sensor_lower_threshold_label]
        self._sensor_threshold_high = metadata_dict[self.tag][sensor_upper_threshold_label]
        self._process_threshold_low = metadata_dict[self.tag][process_lower_threshold_label]
        self._process_threshold_high = metadata_dict[self.tag][process_upper_threshold_label]
        self._flatline_min_length = metadata_dict[self.tag][flatline_min_length_label]
        self._offline_min_length = metadata_dict[self.tag][sensor_offline_min_length_label]
        self.param = metadata_dict[self.tag][param_label]

    def threshold_detection(self, threshold_type='sensor'):
        """
        This method is the detection of anomalies that are beyond the boundaries
        of the thresholds provided for a given signal. The thresholds were provided
        by Waternet and are stored as metadata. Once a specific parameter is passed
        into this method, the corresponding thresholds are pulled and the analysis
        is conducted.
        :param threshold_type: The type of theshold to use for anomaly detection. Choices are 'sensor' and 'process'.
        The 'sensor' threshold_type represents thresholds provided by a sensor manufacturer/calibration reports.
        The 'process' threshold_type represents thresholds decided by a process engineer to flag unphysical values
        above or below a the provided limits.
        :return: a pd.Series containing the flags of the timestamps where the values are beyond the thresholds.
        """

        if threshold_type == 'sensor':
            lower_threshold = self._sensor_threshold_low
            upper_threshold = self._sensor_threshold_high
        else:
            lower_threshold = self._process_threshold_low
            upper_threshold = self._process_threshold_high

        threshold_ad = ThresholdAD(low=lower_threshold, high=upper_threshold)
        anomalies_thres = threshold_ad.detect(self.data)

        return anomalies_thres

    def spike_drop_detection(self, lower_threshold=0.001, upper_threshold=99.999):
        """
        A method to detect a sudden spike or drop. The analysis flags timesteps where the values represent an unphysical
        jump or drop in consecutive timesteps. A differencing is conducted and then a threshold based detection
        is done based on user defined low and high thresholds.
        :param lower_threshold: Lower threshold based on user-provided confidence intervals.
        :param upper_threshold: Higher threshold based on user-provided confidence intervals.
        :return: A pd.Series containing the flags of the timestamps where the values are beyond the thresholds.
        """

        df_data = pd.DataFrame()
        df_data['data'] = self.data[self.tag]
        df_data['data_difference'] = df_data.diff()
        diff = df_data['data_difference'].to_numpy()
        diff = diff[1:]
        perc_low = np.percentile(diff, lower_threshold)
        perc_high = np.percentile(diff, upper_threshold)

        threshold_ls = ThresholdAD(low=perc_low, high=perc_high)
        anomalies_ls = threshold_ls.detect(df_data['data_difference'])

        return anomalies_ls, perc_low, perc_high

    @property
    def nanvalues_detection(self):
        """
        A method to detect any existing nan values in the datasets.
        :return: A pd.Series contraining the flags of the timestamps where nan values are present.
        """

        def custom_nan(df):
            return (df.isna())

        nan_values = CustomizedDetector1D(detect_func=custom_nan)
        anomalies_nan = nan_values.detect(self.data)
        return anomalies_nan

    @property
    def negativevalues_detection(self):
        """
        A method to detect any existing negatives values in the datasets.
        :return: A pd.Series containing the flags of the timestamps where 0 values are present.
        """
        def custom_negative(df):
            return (df < 0)

        negative_values = CustomizedDetector1D(detect_func=custom_negative)
        anomalies_negative = negative_values.detect(self.data)
        return anomalies_negative

    @property
    def zerovalues_detection(self):
        """
        A method to detect any existing 0 values in the datasets. To be used for signals where 0 values are not
        physically feasible.
        :return: A pd.Series containing the flags of the timestamps where 0 values are present.
        """
        def custom_negative(df):
            return (df == 0)

        zero_values = CustomizedDetector1D(detect_func=custom_negative)
        anomalies_zeros = zero_values.detect(self.data)
        return anomalies_zeros

    def flatline_detection(self, min_length=None, in_flatline=False):
        """
        A method to detect flat lines in datasets. The minimal length of data of the flat line is user defined.
        :param min_length: User-defined minimum length of data of the flat line. Default is 5 minutes. The input can be
        an integer as a time interval or a integer to represent number of data points.
        :return: A pd.Series containing the flags of the timestamps of the flat line identified.
        """
        if min_length is None:
            length = self._flatline_min_length
        elif isinstance(min_length, int):
            length = min_length
        else:
            try:
                length = int(min_length)
            except:
                raise Exception('min_length parameter is not in acceptable format. Please provide an integer value or'
                                'remove the provided input to use the default value of '+str(self._flatline_min_length)+
                                'stored in metadata for the given sensor tag')
        df_data = pd.DataFrame()
        df_data['data'] = self.data[self.tag]
        df_data['diff'] = df_data['data'].diff()
        count = 0
        if in_flatline is False:
            df_data.dropna(inplace=True)
            if df_data.iloc[0, 1] != 0 and math.isclose(df_data.iloc[1, 1], 0, abs_tol=0.0001):
                df_data.iloc[0, 1] = 0
            anomalies_flatline = pd.DataFrame(data=np.nan,
                                              index=df_data.index, columns=['anomalies_flatline'])
            for i, (index, rows) in enumerate(df_data.iterrows()):
                if math.isclose(rows['diff'], 0, abs_tol=0.0001):
                    if rows['data'] == 0:
                        continue
                    count = count + 1
                    if count >= length:
                        for j in range(0, count+1):
                            anomalies_flatline.iloc[i - j, 0] = 1
                else:
                    count = 0
            anomalies_flatline.fillna(value=0, inplace=True)
        else:
            anomalies_flatline = pd.DataFrame(data=np.nan,
                                              index=df_data.index, columns=['anomalies_flatline'])
            for i, (index, rows) in enumerate(df_data.iterrows()):
                if math.isclose(rows['diff'], 0, abs_tol=0.0001):
                    if rows['data'] == 0:
                        continue
                    anomalies_flatline.iloc[i] = 1
            anomalies_flatline.fillna(value=0, inplace=True)

        return anomalies_flatline.tail(1)

    def plot_anomalies(self, anomalies):

        df_anomalies_visual = self.find_value_anomaly_flag(anomalies)

        sensor_threshold_low = np.ones(len(anomalies)) * self._sensor_threshold_low
        sensor_threshold_high = np.ones(len(anomalies)) * self._sensor_threshold_high
        process_threshold_low = np.ones(len(anomalies)) * self._process_threshold_low
        process_threshold_high = np.ones(len(anomalies)) * self._process_threshold_high
        fig = plt.figure(figsize=(20, 10))
        plt.plot(self.data, label='Sensor Signal', alpha=0.8)
        # plt.plot(self.data, label=self.param+' ('+self.tag+')', alpha=0.8)
        # plt.plot(df_anomalies_visual.index, sensor_threshold_low, 'y-',
        #          label='Sensor Threshold Low (' + str(self._sensor_threshold_low) + ')', alpha=0.8)
        # plt.plot(df_anomalies_visual.index, sensor_threshold_high, 'g-',
        #          label='Sensor Threshold High (' + str(self._sensor_threshold_high) + ')', alpha=0.8)
        # plt.plot(df_anomalies_visual.index, process_threshold_low, 'b-',
        #          label='Process Threshold Low (' + str(self._process_threshold_low) + ')', alpha=0.8)
        # plt.plot(df_anomalies_visual.index, process_threshold_high, 'r-',
        #          label='Process Threshold High (' + str(self._process_threshold_high) + ')', alpha=0.8)

        for columns in df_anomalies_visual:
            plt.scatter(x=df_anomalies_visual.index, y=df_anomalies_visual[columns], s=50, label=columns)
        plt.legend(loc='upper right')
        plt.xlabel('Time', fontweight='bold', fontsize=12)
        plt.ylabel('Concentration (mg/l)', fontweight='bold', fontsize=12)
        plt.grid(True)
        plt.title('Detection of Anomalies for Sensor: '+self.param, fontweight='bold', fontsize=14)
        plt.show()
