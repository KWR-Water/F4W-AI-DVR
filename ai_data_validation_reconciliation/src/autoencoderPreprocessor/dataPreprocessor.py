""""
A helper class containing preprocessing methods to get the data into the right format for inputting
into an autoencoder model and to make predictions. Main functionalities include:
- Converting data in pandas dataframe into numpy array as the accepted format
  to input into a Tensorflow model.
- Normalising of the data
- Denormalising of normalised data
- Preparing data into sequences
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing


def init_dataframe(data) -> pd.DataFrame:
    """
    Ensuring the propvided data is a Dataframe (might want to just use numpy) and make a copy.
    :param data:  Feature data
    :return: data copied into a dataframe
    """
    if isinstance(data, pd.DataFrame):
        return data.copy()
    else:
        try:
            return pd.DataFrame(data.copy())
        except:
            raise Exception("Input data is not convertible to a pandas "
                            "DataFrame")


class DataGenerator:
    """
    Master class for Data Generation.
    """

    # constructor
    def __init__(self, data):
        self.raw_data = init_dataframe(data)
        self.data = self.raw_data.copy()

    def prep_data_to_array(self, to_resample=False, resample='5min',
                           to_normalise=True, normaliser=None):
        """
        Preparing input raw data into a numpy array ready to be inputted into the autoencoder model.
        The raw data is resampled into the frequency provided by user, and normalised (if user
        defines this choice), using a sklearn.preprocessing method.
        :param to_resample: A variable to define whether the data should be resampled or not.
        Default is False.
        :param resample: Resampling frequency to conduct. Should be of the form '#min' or using the
        same format as defined in the pandas resampling method for the 'rule' variable.
        :param to_normalise: A variable to define whether the data should be normalised or not.
        Default is True
        :param normaliser: The normalisation rule to use. Using the sklearn.preprocessing methods.
        Currently, only two options provided- PowerTransformer or StandardScaler.
        Default is PowerTransformer.
        :return:
        A numpy array containing the processed data to be inputted into an autoencoder model.
        """

        if to_resample is True:
            data_processed = self.data.copy().resample(resample).mean()
            data_processed.dropna(inplace=True)
        else:
            data_processed = self.data.copy()

        if to_normalise is True:
            data_standardised = pd.DataFrame(normaliser.transform(data_processed),
                                             columns=data_processed.columns,
                                             index=data_processed.index)
            data_array = data_standardised.to_numpy()
        else:
            data_array = data_processed.to_numpy()

        return data_array

    def denormalise_data(self, predictions, normaliser=None):
        """
        A method to conduct the denormalisation or descaling of normalised predictions, as provided
        as outputs from an autoencoder model.
        :param predictions: Predictions from the autonecoder as a numpy array.
        :param normaliser: The normalisation rule to use. Using the sklearn.preprocessing methods.
        Currently, only two options provided- PowerTransformer or StandardScaler.
        Default is PowerTransformer.
        :return:
        A numpy array containing the predictions of the same scale as the raw data of a given
        variable.
        """
        if predictions.shape[0] > 1:
            pred_unscaled = normaliser.inverse_transform(predictions[:,0].reshape(-1, 1))
        else:
            pred_unscaled = normaliser.inverse_transform(predictions.reshape(1, -1))

        return pred_unscaled

    def split_sequence(self, sequence, n_steps=36):
        """
        Splitting the dataset into sequences based on an inputted number of timesteps. As this is an
        autoencoder, there is no target (y) values.
        :param sequence: the data in np.array with the time series data
        :param n_steps: The number of timesteps to be considered for the sequencing. Default taken
        as 36 timesteps
        :return: a np.array of the sequenced data.
        """
        X = list()
        for i in range(sequence.shape[0]):

            end = i + n_steps
            if end > sequence.shape[0] - 1:
                break
            seq_x = sequence[i:end, :].reshape(n_steps, sequence.shape[1])
            X.append(seq_x)

        return np.array(X)
