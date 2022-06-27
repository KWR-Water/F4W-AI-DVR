import pandas as pd
import numpy as np

from waternet_anomaly_detection.DataValidation import UnivariateAD
from autoencoderPreprocessor.dataPreprocessor import DataGenerator


def single_anomaly_detection(raw_data=None, tag=None):
    """
    A function to call the single anomaly detection methods for a given inputted raw data
    :param raw_data: raw data point(s) assessed for anomalies
    :param tag: tag of the signal the data is from. It must be a tag that is in the metadata json file.
    :return: A pd.DataFrame containing the flags of the different anomaly detection methods
    """

    # Checking for single value based anomalies, such as negative values or zero values by creating
    # a class instance of the data validation univariate AD.
    ad = UnivariateAD(data=raw_data, tag=tag, correct_format_tseries=False)

    # Detection of threshold breaches
    single_anomalies = ad.threshold_detection(threshold_type='process')
    single_anomalies.rename(columns={tag: 'threshold'}, inplace=True)
    # Detection of negative values
    single_anomalies['negative_value'] = ad.negativevalues_detection
    # Detection of zero values
    single_anomalies['zero_value'] = ad.zerovalues_detection
    # Detection of nan values
    single_anomalies['nan_value'] = ad.nanvalues_detection

    # If there are single anomalies detected, the flagging will be returned for the value,
    # including, what anomaly detection technique caused the flag (by the name of the column).

    return single_anomalies


def flatline_anomaly_detection(raw_data=None, tag=None, in_flatline=False):
    """
    A function to call the flatline anomaly detection method
    :param raw_data: sequential raw data points assessed to be the flatline
    :param tag: tag of the signal the data is from. It must be a tag that is in the metadata json file.
    :param in_flatline: A variable to check whether the current point is part of a flatline. Mostly unused.
    :return: pd.DataFrame that includes the flag of the last point of the input (the point assessed for an anomaly).
    """
    ad_flatline = UnivariateAD(data=raw_data, tag=tag, correct_format_tseries=False)
    flatline_anomalies = ad_flatline.flatline_detection(min_length=3, in_flatline=in_flatline)

    return flatline_anomalies


def autoencoder_predictor(history=None, model=None, power_transformer=None, resolution='5min'):
    """
    A function to preprocess the raw data (resample and normalise), conduct a prediction and
    denormalise back.
    :param history: Historical data inputted to make a one step ahead prediction.
    :param model: The AI autoencoder model used to make a prediction.
    :param power_transformer: The saved powertransformer fitted based on the training data used
    :param resolution: Resolution of the data that determines which model is used (5min or 30 min)
    :return: The one step ahead predicted value.
    """

    # Acquiring the n_steps needed to convert the history into a sequence
    if resolution == '5min':
        n_steps = 36
    else:
        n_steps = 48

    # Creating a DataGenerator instance to preprocess data for predictions
    dg_norm = DataGenerator(history)
    # Processing data to be resampled, normalised and converted to a numpy array
    normalised_data = dg_norm.prep_data_to_array(to_resample=True, resample=resolution,
                                                 to_normalise=True, normaliser=power_transformer)
    # Creating a sequence of the normalised data
    sequence = dg_norm.split_sequence(sequence=normalised_data, n_steps=n_steps)

    # Preforming a prediction with the model.
    current_pred = model.predict(sequence)[0][0]

    # Denormalising the prediction
    pred_denorm = dg_norm.denormalise_data(predictions=np.array([current_pred]),
                                           normaliser=power_transformer)[0][0]

    # Returning the current autoencoder prediction
    return pred_denorm


def reconcile_anomalies(raw_data=None, anomalies=None, prediction=None):
    """
    A function to conduct the reconcilation of a data point, based on the anomaly flag determined.
    It will assess whether it should return the raw data itself or the predicted value from the
    autoencoder
    :param raw_data: Raw data point that was assessed to be an anomaly.
    :param anomalies: Anomaly flags for the given raw data point.
    :param prediction: Predicted value as calculated from an autoencoder prediction.
    :return: Returns the raw data point or predicted value based on the reconciliation.
    """

    if anomalies.any(axis=None).sum() >= 1:
        columns = list(raw_data.columns.values.tolist())
        return pd.DataFrame(data={columns[0]: prediction}, index=raw_data.index)
    else:
        return raw_data


def autoencoder_aggregation(processed_5_min=None, processed_30_min=None, tag=None, ratio_of_importance=0.1,
                            resample_target='15min'):
    """
    A function to conduct the aggregation of the intermediate data streams as computed from the
    individual reconciliations of the 5 min and 30 min raw data streams.
    :param processed_5_min: Intermediate 5 min processed data
    :param processed_30_min: Intermediate 30 min processed data
    :param tag: tag of the signal the data is from. It must be a tag that is in the metadata json file.
    :param ratio_of_importance: Importance value given to the higher resolution data stream.
    :param resample_target: Final resample target desired. Default is 15 mins
    :return: Returns the aggregated data at the desired resolution
    """

    processed_30_min_resampled_5_min = processed_30_min.resample('5min').interpolate(method='linear')
    beta = np.log(ratio_of_importance) / 6

    w = pd.DataFrame(data={tag: np.exp(6 * beta * pd.Series(range(len(processed_30_min_resampled_5_min))))})
    w.index = processed_5_min.index
    processed_data = pd.DataFrame(data=w[tag] * processed_5_min[tag] + (1 - w[tag]) * processed_30_min_resampled_5_min[tag])

    # Resampling data to the resample target
    reconciled_signal = processed_data.resample(resample_target).mean()

    return reconciled_signal
