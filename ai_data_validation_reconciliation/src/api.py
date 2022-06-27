from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import pandas.io.sql as sqlio
from absl import logging
import re
import requests
import traceback
import time
from datetime import datetime, timedelta
import psycopg2
from pickle import load
from tensorflow.keras.models import load_model
from AI_data_validation import single_anomaly_detection, flatline_anomaly_detection, autoencoder_predictor, reconcile_anomalies, autoencoder_aggregation
import pytz

import pathlib
from os.path import join


def load_autoencoder(variable='nh4', resolution=36):
    path_src = pathlib.Path(__file__).parent.absolute()
    if variable == 'nh4' and resolution == 36:
        model_folder = 'nh4_autoencoder_36'

    elif variable == 'nh4' and resolution == 48:
        model_folder = 'nh4_autoencoder_48'

    elif variable == 'no3' and resolution == 36:
        model_folder = 'no3_autoencoder_36'

    else:
        model_folder = 'no3_autoencoder_48'

    fpath_model = join(path_src, model_folder)

    return load_model(filepath=fpath_model)


def load_powerTransformer(variable='nh4', resolution=36):
    path_src = pathlib.Path(__file__).parent.absolute()
    if variable == 'nh4' and resolution == 36:
        file = 'power_transformer_5mins_NH4.pkl'

    elif variable == 'nh4' and resolution == 48:
        file = 'power_transformer_30mins_NH4.pkl'

    elif variable == 'no3' and resolution == 36:
        file = 'power_transformer_5mins_NO3.pkl'

    else:
        file = 'power_transformer_30mins_NO3.pkl'

    fpath_transformer = join(path_src, 'PowerTransformers', file)

    return load(open(fpath_transformer, 'rb'))



logging.set_verbosity(logging.DEBUG)

postgres_dbname="postgres"
postgres_user="postgres"
postgres_password="postgresql"
postgres_host="postgresql-db"
postgres_port=5432

app = Flask(__name__)


@app.route('/test')
def test():
    return jsonify('API call successful')

# Loading the autoencoder models

nh4_autoencoder_36 = load_autoencoder("nh4", resolution=36)
nh4_autoencoder_48 = load_autoencoder("nh4", resolution=48)
no3_autoencoder_36 = load_autoencoder("no3", resolution=36)
no3_autoencoder_48 = load_autoencoder("no3", resolution=48)

# Load PowerTransformers for normalisation

power_transformer_5mins_NH4 = load_powerTransformer(variable="nh4", resolution=36)
power_transformer_30mins_NH4 = load_powerTransformer(variable="nh4", resolution=48)
power_transformer_5mins_NO3 = load_powerTransformer(variable="no3", resolution=36)
power_transformer_30mins_NO3 = load_powerTransformer(variable="no3", resolution=48)

type_to_device_id_mapping = {
    "WasteWaterTank": ["urn:ngsi-ld:WasteWaterTank:WasteWaterTank02"],
    "WasteWaterSimulationResults":
        ["urn:ngsi-ld:WasteWaterSimulationResults:NitrateReconciled",
         "urn:ngsi-ld:WasteWaterSimulationResults:NitrateProcessedFiveMinutes",
         "urn:ngsi-ld:WasteWaterSimulationResults:NitrateProcessedThirtyMinutes",
         "urn:ngsi-ld:WasteWaterSimulationResults:NitratePredictionsFiveMinutes",
         "urn:ngsi-ld:WasteWaterSimulationResults:NitratePredictionsThirtyMinutes",
         "urn:ngsi-ld:WasteWaterSimulationResults:AmmoniaReconciled",
         "urn:ngsi-ld:WasteWaterSimulationResults:AmmoniaProcessedFiveMinutes",
         "urn:ngsi-ld:WasteWaterSimulationResults:AmmoniaProcessedThirtyMinutes",
         "urn:ngsi-ld:WasteWaterSimulationResults:AmmoniaPredictionsFiveMinutes",
         "urn:ngsi-ld:WasteWaterSimulationResults:AmmoniaPredictionsThirtyMinutes"],
    "Anomaly": ["urn:ngsi-ld:Anomaly:NitrateAnomaly", "urn:ngsi-ld:Anomaly:AmmoniaAnomaly"]
}

def device_ids():
    return [item for sublist in type_to_device_id_mapping.values() for item in sublist]


def device_id_to_property(device_id):
    found_type = None
    for device_type, device_ids in type_to_device_id_mapping.items():
        for full_device_id in device_ids:
            if full_device_id.endswith(device_id):
                found_type = device_type
                break
    return device_type_to_property(found_type)


def tag_to_device_id(tag):

    mapping = {
        "NitrateReconciled": {
            "id": "urn:ngsi-ld:WasteWaterSimulationResults:NitrateReconciled",
            "type": "WasteWaterSimulationResults",
            "concentration": {
                "type": "Property",
                "value": None
            },
            "targetURI": {
                "type": "Relationship",
                "value": "urn:ngsi-ld:WasteWaterTank:WasteWaterTank02"
            }
        },
        "AmmoniaReconciled": {
            "id": "urn:ngsi-ld:WasteWaterSimulationResults:AmmoniaReconciled",
            "type": "WasteWaterSimulationResults",
            "concentration": {
                "type": "Property",
                "value": None
            },
            "targetURI": {
                "type": "Relationship",
                "value": "urn:ngsi-ld:WasteWaterTank:WasteWaterTank02"
            }
        },
        "NitrateProcessedFiveMinutes": {
            "id": "urn:ngsi-ld:WasteWaterSimulationResults:NitrateProcessedFiveMinutes",
            "type": "WasteWaterSimulationResults",
            "concentration": {
                "type": "Property",
                "value": None
            },
            "targetURI": {
                "type": "Relationship",
                "value": "urn:ngsi-ld:WasteWaterTank:WasteWaterTank02"
            }
        },
        "AmmoniaProcessedFiveMinutes": {
            "id": "urn:ngsi-ld:WasteWaterSimulationResults:AmmoniaProcessedFiveMinutes",
            "type": "WasteWaterSimulationResults",
            "concentration": {
                "type": "Property",
                "value": None
            },
            "targetURI": {
                "type": "Relationship",
                "value": "urn:ngsi-ld:WasteWaterTank:WasteWaterTank02"
            }
        },
        "NitrateProcessedThirtyMinutes": {
            "id": "urn:ngsi-ld:WasteWaterSimulationResults:NitrateProcessedThirtyMinutes",
            "type": "WasteWaterSimulationResults",
            "concentration": {
                "type": "Property",
                "value": None
            },
            "targetURI": {
                "type": "Relationship",
                "value": "urn:ngsi-ld:WasteWaterTank:WasteWaterTank02"
            }
        },
        "AmmoniaProcessedThirtyMinutes": {
            "id": "urn:ngsi-ld:WasteWaterSimulationResults:AmmoniaProcessedThirtyMinutes",
            "type": "WasteWaterSimulationResults",
            "concentration": {
                "type": "Property",
                "value": None
            },
            "targetURI": {
                "type": "Relationship",
                "value": "urn:ngsi-ld:WasteWaterTank:WasteWaterTank02"
            }
        },
        "NitratePredictionsFiveMinutes": {
            "id": "urn:ngsi-ld:WasteWaterSimulationResults:NitratePredictionsFiveMinutes",
            "type": "WasteWaterSimulationResults",
            "concentration": {
                "type": "Property",
                "value": None
            },
            "targetURI": {
                "type": "Relationship",
                "value": "urn:ngsi-ld:WasteWaterTank:WasteWaterTank02"
            }
        },
        "AmmoniaPredictionsFiveMinutes": {
            "id": "urn:ngsi-ld:WasteWaterSimulationResults:AmmoniaPredictionsFiveMinutes",
            "type": "WasteWaterSimulationResults",
            "concentration": {
                "type": "Property",
                "value": None
            },
            "targetURI": {
                "type": "Relationship",
                "value": "urn:ngsi-ld:WasteWaterTank:WasteWaterTank02"
            }
        },
        "NitratePredictionsThirtyMinutes": {
            "id": "urn:ngsi-ld:WasteWaterSimulationResults:NitratePredictionsThirtyMinutes",
            "type": "WasteWaterSimulationResults",
            "concentration": {
                "type": "Property",
                "value": None
            },
            "targetURI": {
                "type": "Relationship",
                "value": "urn:ngsi-ld:WasteWaterTank:WasteWaterTank02"
            }
        },
        "AmmoniaPredictionsThirtyMinutes": {
            "id": "urn:ngsi-ld:WasteWaterSimulationResults:AmmoniaPredictionsThirtyMinutes",
            "type": "WasteWaterSimulationResults",
            "concentration": {
                "type": "Property",
                "value": None
            },
            "targetURI": {
                "type": "Relationship",
                "value": "urn:ngsi-ld:WasteWaterTank:WasteWaterTank02"
            }
        },
        "NitrateAnomaly": {
            "id": "urn:ngsi-ld:Anomaly:NitrateAnomaly",
            "type": "Anomaly",
            "thresholdBreach": {
                "type": "Property",
                "value": None
            },
            "negativeValue": {
                "type": "Property",
                "value": None
            },
            "nanValue": {
                "type": "Property",
                "value": None
            },
            "zeroValue": {
                "type": "Property",
                "value": None
            },
            "flatline": {
                "type": "Property",
                "value": None
            },
            "dateDetected": {
                "type": "Property",
                "value": None,
            },
            "measuredValue": {
                "type": "Property",
                "value": None
            },
            "detectedIn": {
                "type": "Relationship",
                "value": "urn:ngsi-ld:WasteWaterTank:WasteWaterTank02"
            }
        },
        "AmmoniaAnomaly": {
            "id": "urn:ngsi-ld:Anomaly:AmmoniaAnomaly",
            "type": "Anomaly",
            "thresholdBreach": {
                "type": "Property",
                "value": None
            },
            "negativeValue": {
                "type": "Property",
                "value": None
            },
            "nanValue": {
                "type": "Property",
                "value": None
            },
            "zeroValue": {
                "type": "Property",
                "value": None
            },
            "flatline": {
                "type": "Property",
                "value": None
            },
            "dateDetected": {
                "type": "Property",
                "value": None,
            },
            "measuredValue": {
                "type": "Property",
                "value": None
            },
            "detectedIn": {
                "type": "Relationship",
                "value": "urn:ngsi-ld:WasteWaterTank:WasteWaterTank02"
            }
        }
    }
    device_id_mapping = mapping.get(tag, None)
    device_id = device_id_mapping['id'].split(':')[-1]

    if device_id:
        return device_id_mapping, device_id
    logging.warning(f"Received tag \"{tag}\", for which we do not know the Fiware device id.")
    return tag


device_property_tag_mapping = {
        "urn:ngsi-ld:WasteWaterTank:WasteWaterTank02": {
            "no3": "6254_SPRS_873",
            "nh4": "6218_SPLM_873"
        }
    }


def device_id_and_property_to_tag(device_id, property):
    # tag = device_property_tag_mapping.get(device_id, dict()).get(property, None)
    tag = None
    for full_device_id, property_tag_mapping in device_property_tag_mapping.items():
        if full_device_id.endswith(device_id):
            tag = property_tag_mapping.get(property, None)
            break
    if tag:
        return tag
    logging.warning(f"Received device id \"{device_id}\" and \"{property}\", for which we do not know the PIMS tag.")
    return device_id


def pims_tag_to_device_id(tag):
    device_id = None
    for device_id, property_to_tag_mapping in device_property_tag_mapping.items():
        for _, mapped_tag in property_to_tag_mapping.items():
            if mapped_tag == tag:
                return device_id
    logging.warning(f"Received tag \"{tag}\" for which we do not know the device id.")
    return tag


def tag_to_type(tag):
    """Translate PIMS tag to the Fiware NGSI-LD type of the device."""
    return "Thing"
    if re.match("6254_SPRS_873", tag):
        return "WasteWaterTank" # "urn:ngsi-ld:WasteWaterTank"
    elif re.match("6218_SPLM_873", tag):
        return "WasteWaterTank"
    logging.warning(f"Received tag \"{tag}\", for which we do not know the Fiware data model type.")
    return tag


# Additions of the PIMS tags that represent the output of the airflow soft sensor to the tag_to_property function
def tag_to_property(tag):
    """Translate PIMS tag to the property of the device."""
    # if re.match("WasteWaterTank02", tag):
    #     return "no3"
    if tag == "6254_SPRS_873":
        return "no3"
    elif tag == "6218_SPLM_873":
        return "nh4"
    elif re.match("(Ammonia|Nitrate)Reconciled", tag) or re.match("(Ammonia|Nitrate)(Predictions|Processed)(Five|Thirty)Minutes", tag):
        return "concentration"
    logging.warning(f"Received tag \"{tag}\", for which we do not know the Fiware data model property.")
    return tag


device_type_to_property_mapping = {
    "WasteWaterTank02": ["no3", "nh4"],
    "WasteWaterSimulationResults": ["concentration"],
    "Anomaly": ["NitrateAnomaly", "AmmoniaAnomaly"]
}


def device_type_to_property(device_type):
    """Translate Fiware device type to the property that is watched."""
    value = device_type_to_property_mapping.get(device_type, None)
    if not value:
        logging.warning(f"Received device type \"{device_type}\", for which we do not know the Fiware data model property to watch.")
    return value


def run_data_reconciliation_procedures(raw_data_no3=None, raw_data_nh4=None, processed_data_no3=None, processed_data_nh4=None, resolution='5min', conduct_flatline=False, timestamp_from: datetime = None, timestamp_to: datetime = None):
    try:
        parameter_to_data_stream_mapping = {
            "no3": {
                "tag": "6254_SPRS_873",
                "raw_data": raw_data_no3,
                "processed_data": processed_data_no3,
                "tag_processed_5_min": "NitrateProcessedFiveMinutes",
                "tag_predictions_5_min": "NitratePredictionsFiveMinutes",
                "tag_processed_30_min": "NitrateProcessedThirtyMinutes",
                "tag_predictions_30_min": "NitratePredictionsThirtyMinutes",
                "tag_anomaly": "NitrateAnomaly"
            },
            "nh4": {
                "tag": "6218_SPLM_873",
                "raw_data": raw_data_nh4,
                "processed_data": processed_data_nh4,
                "tag_processed_5_min": "AmmoniaProcessedFiveMinutes",
                "tag_predictions_5_min": "AmmoniaPredictionsFiveMinutes",
                "tag_processed_30_min": "AmmoniaProcessedThirtyMinutes",
                "tag_predictions_30_min": "AmmoniaPredictionsThirtyMinutes",
                "tag_anomaly": "AmmoniaAnomaly"
                }
        }
        for parameter, data_mapping in parameter_to_data_stream_mapping.items():
            try:
                # Getting the raw data and setting the index in the dataframe. Finally, resampling to 5 mins.
                raw_data = data_mapping["raw_data"]
                if not isinstance(raw_data.index, pd.DatetimeIndex):
                    raw_data.set_index(f'{parameter}_observedat', inplace=True)
                    logging.info(f"The isinstance check index dtype gave a {isinstance(raw_data.index, pd.DatetimeIndex)}")
                    raw_data.rename(columns={parameter: data_mapping['tag']}, inplace=True)
                    raw_data[data_mapping['tag']] = raw_data[data_mapping['tag']].astype('float')
                logging.info(f"data is: {raw_data.head()}; {raw_data.tail()}; {raw_data.index.dtype}; {raw_data[data_mapping['tag']].dtype}")
                
                # Sometimes, there may be missing values, especially when there was a long disruption or the component starts for the first time or because of events being dropped by orion and/or cygnus.
                # Do a linear interpolation in these situations. That makes sure there is enough data to process it trough the model.
                # Eventually, the 5 minutes and 30 minutes intermediate signals will have a high enough quality to work accurately.
                if timestamp_from and timestamp_to:
                    raw_data = interpolate_data(raw_data, timestamp_from=timestamp_from, timestamp_to=timestamp_to, resolution="1min")

                raw_data_resampled = raw_data.resample(resolution).mean()
                logging.info(f"The resampled raw data of resolution {resolution} for {parameter} is: {raw_data_resampled}")

                # Taking out the new point to be considered for anomaly detection
                data_point = raw_data_resampled.tail(1)
                logging.info(f"Current data point is: {data_point}")
                # Check for single anomalies
                try:
                    anomalies_new = single_anomaly_detection(raw_data=data_point, tag=data_mapping["tag"])
                    logging.debug(f"Successfully conducted a single anomaly detection and it looks like {anomalies_new}")
                except Exception as e:
                    logging.warning(f"Error attempting to perform a single anomaly detection {e}")
                    continue

                # Checking for flatlines if conduct_flatline is true
                if conduct_flatline:
                    try:
                        logging.info(f'The data that will be inputted into the flat line is {raw_data_resampled.tail(4)}')
                        flatline_anomalies = flatline_anomaly_detection(raw_data=raw_data_resampled.tail(4), tag=data_mapping["tag"])
                        logging.debug(
                            f"Successfully conducted a flatline anomaly detection for {resolution} data "
                            f"and it looks like {flatline_anomalies}")
                    except Exception as e:
                        logging.warning(f"Error attempting to perform a flatline anomaly detection for "
                                        f"{resolution} data {e}")
                    # Consolidating the anomaly flags for the new point
                    anomalies_new = pd.concat((anomalies_new, flatline_anomalies), axis=1)

                # Making an autoencoder prediction by inputting the necessary history using the intermediate reconciled stream
                if resolution == '5min':
                    if parameter == 'no3':
                        model = no3_autoencoder_36
                        transformer = power_transformer_5mins_NO3
                    else:
                        model = nh4_autoencoder_36
                        transformer = power_transformer_5mins_NH4
                else:
                    if parameter == 'no3':
                        model = no3_autoencoder_48
                        transformer = power_transformer_30mins_NO3
                    else:
                        model = nh4_autoencoder_48
                        transformer = power_transformer_30mins_NH4
                try:
                    processed_data = data_mapping["processed_data"]
                    processed_data.set_index('concentration_observedat', inplace=True)
                    processed_data['concentration'] = processed_data['concentration'].astype('float')
                    processed_data = processed_data.sort_index()
                    logging.info(f'The processed data looks like: {processed_data} and is of size {len(processed_data)}')
                    try:
                        prediction = autoencoder_predictor(history=processed_data, model=model,
                                                           power_transformer=transformer,
                                                           resolution=resolution)
                    except:
                        prediction = None
                    # If something went wrong with the prediction, take the raw value instead, the reconciliation step would take the raw value as well.
                    if prediction is None or np.isnan(prediction) or np.isposinf(prediction) or np.isneginf(prediction):
                        prediction = data_point.iloc[0, 0]
                    logging.debug(f"The {resolution} autoencoder prediction for {parameter} and the predicted value is {prediction} (before casting) and of type {prediction.dtype}")
                    prediction = np.float64(prediction)
                    logging.debug(f"Successfully performed a {resolution} autoencoder prediction for {parameter} and the predicted value is {prediction} and of type {prediction.dtype}")
                except Exception as e:
                    logging.warning(f"Error performing a {resolution} autoencoder prediction for {parameter} {e}")
                    logging.error(traceback.format_exc())
                    continue

                # Conduct the intermediate resolution
                try:
                    processed_point = reconcile_anomalies(raw_data=data_point, anomalies=anomalies_new,
                                                          prediction=prediction)
                    logging.debug(
                        f"Successfully performed intermediate reconciling of anomalies for the "
                        f"{resolution} resolution and the processed point is {processed_point}")
                except Exception as e:
                    logging.warning(f"Error performing intermediate reconciling of anomalies for the "
                                    f"{resolution} resolution {e}")
                    continue

                if resolution == '5min':
                    tag_processed = data_mapping["tag_processed_5_min"]
                    tag_predictions = data_mapping["tag_predictions_5_min"]
                else:
                    tag_processed = data_mapping["tag_processed_30_min"]
                    tag_predictions = data_mapping["tag_predictions_30_min"]

                # Sending back the processed point to the relevant device
                request_id_mapping, request_id = tag_to_device_id(tag_processed)
                timestamp_resampled = processed_point.index[0].tz_convert(None).strftime("%Y%m%dT%H%M%S")
                request_url = f"http://iot-agent-json:7896/iot/json?k=1234&i={request_id}&t={timestamp_resampled}"
                request_id_mapping['concentration']['value'] = processed_point.loc[processed_point.index[0], data_mapping['tag']]
                try:
                    # requests.post(url=request_url, json=request_id_mapping)
                    requests.post(url=request_url,
                                  json={tag_to_property(tag_processed):
                                        processed_point.loc[processed_point.index[0], data_mapping['tag']]})
                except Exception as e:
                    logging.error(f"Error sending request to Fiware. URL=\"{request_url}\","
                                f"json data= \"{str(request_id_mapping)}\", "
                                f"Exception: {traceback.format_exc()}")

                # Sending back the prediction point to the relevant device
                request_id_mapping, request_id = tag_to_device_id(tag_predictions)
                timestamp_resampled = processed_point.index[0].tz_convert(None).strftime("%Y%m%dT%H%M%S")
                request_url = f"http://iot-agent-json:7896/iot/json?k=1234&i={request_id}&t={timestamp_resampled}"
                request_id_mapping['concentration']['value'] = prediction
                try:
                    # requests.post(url=request_url, json=request_id_mapping)
                    requests.post(url=request_url,
                                  json={tag_to_property(tag_predictions): prediction})
                except Exception as e:
                    logging.error(f"Error sending request to Fiware. URL=\"{request_url}\", "
                                  f"json data= \"{str(request_id_mapping)}\", "
                                  f"Exception: {traceback.format_exc()}")

            except Exception as e:
                logging.error(traceback.format_exc())
                logging.error(f'Some internal error ocurred while processing {parameter} | {e}')
    except Exception as e:
        logging.error(traceback.format_exc())
        return f'Internal error | {e}', 500


def interpolate_data(df: pd.DataFrame, timestamp_from: datetime, timestamp_to: datetime, resolution: str) -> pd.DataFrame:
    timestamp_from = timestamp_from.astimezone(pytz.UTC)
    if min(df.index) > timestamp_from:
        df = df.append(pd.Series(name=timestamp_from))
    timestamp_to = timestamp_to.astimezone(pytz.UTC)
    if max(df.index) < timestamp_to:
        df = df.append(pd.Series(name=timestamp_to))
    return df.resample(resolution).apply(lambda x: x if len(x) > 0 else np.nan).interpolate(method="time", limit_direction="both")


@app.route('/subscription_notification', methods=['post'])
def subscription_notification():
    # {'id': 'urn:ngsi-ld:Notification:60d5c935dad46689d3b3c371', 'type': 'Notification', 'subscriptionId': 'urn:ngsi-ld:Subscription:60d5c908dad46689d3b3c36e', 'notifiedAt': '2021-06-25T12:16:53.309Z', 'data': [{'id': 'urn:ngsi-ld:Blower:Blower2', 'type': 'Blower', 'power': {'type': 'Property', 'value': 5, 'observedAt': '2021-06-25T11:54:00.000Z'}, 'airflow': {'type': 'Property', 'value': 3, 'observedAt': '2021-06-25T11:42:57.000Z'}}]}
    logging.debug(f"Received notification: {request.json}")
    # Get the timestamp that the data was originally measured.
    try:
        # device_type = request.json.get("data", [])[0].get("type")  # Does not work, Fiware thinks every device is of type 'Thing" instead of 'Blower', 'Valve', or something else.
        # device_type = request.json.get("data", [])[0].get("id", "").split(":")[-1][:-2]
        # Need to also see is that what we get, based on the response. To be tested.
        # Essentially we want to get either NitrateValidation for example or AmmoniaValidation, or the anomalies?
        device_type = request.json.get("data", [])[0].get("id", "").split(":")[-1]
    except Exception as e:
        logging.warning(f"Error parsing the device type from Fiware notification | {e}")
        # When you try to run it, look out for incomplete data kind of log.
    if device_type not in ["WasteWaterTank02"]:
        logging.debug(f"Received notification on a {device_type}, which will be ignored.")
        return jsonify("")
    try:
        for device_property in device_type_to_property(device_type):
            timestamp = request.json.get("data", [])[0].get(device_property, dict()).get("observedAt", None)
            logging.info(f"Received a timestamp from the request of {timestamp}")
            timestamp = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
            logging.info(f"Timestamp after strptime is {timestamp}")
            fiware_timestamp = int(timestamp.timestamp())
    except Exception as e:
        logging.warning(f"Error parsing the observedAt timestamp from Fiware notification | {e}")
        return jsonify("")
    logging.debug(f"ObservedAt = '{timestamp}' = '{fiware_timestamp}'")
    return run_for_timestamp(timestamp)

def run_for_timestamp(timestamp):
    # Query context data for all devices the model needs at that timestamp.
    # The timstamp is important, because data will come in in an arbitrary order,
    # as well as historic data (to catch up after a disruption), as well as
    # batches (instead of streams) of data with multiple timestamps.
    # Perform the query per property to get the full data we are after.

    # Ignoring any analytics for FIWARE notifications that are not divisible by 5. This is
    # necessary as FIWARE provides a notification for every new value PIMS provides. We have set
    # that to every minute. However the highest resolution we conduct any data validation is 5 mins.
    data = get_historic_data_from_database('def_serv_ld.urn_ngsi_ld_thing_WasteWaterTank02', 'no3', timestamp_from=timestamp - timedelta(days=2), timestamp_to=timestamp)
    data_length = len(data)
    logging.info(f"The raw data available in the database has a length of {data_length} from len and from shape {data.shape[0]}")
    logging.debug(f"The raw data available in the database is = {data}")
    if data_length < 180:
        logging.info(f"Not enough new data (< 180 rows) to make a prediction with the 5 minute autoencoder, the length is {data_length}")
        return jsonify("")
    elif data_length == 180:
        timestamp_from = timestamp - timedelta(hours=4)
        raw_data_no3 = get_historic_data_from_database('def_serv_ld.urn_ngsi_ld_thing_WasteWaterTank02', 'no3', timestamp_from=timestamp_from, timestamp_to=timestamp)
        raw_data_nh4 = get_historic_data_from_database('def_serv_ld.urn_ngsi_ld_thing_WasteWaterTank02', 'nh4', timestamp_from=timestamp_from, timestamp_to=timestamp)
        logging.info(f"Length of raw data of no3 queried is {len(raw_data_no3)}")
        logging.info(f"Length of raw data of nh4 queried is {len(raw_data_nh4)}")
        logging.debug(f"The no3 raw data is = {raw_data_no3}")
        logging.debug(f"The nh4 raw data is = {raw_data_nh4}")

        try:
            raw_data_mapping = {
                "no3": {
                    "data": raw_data_no3,
                    "tag": "NitrateProcessedFiveMinutes"
                },
                "nh4": {
                    "data": raw_data_nh4,
                    "tag": "AmmoniaProcessedFiveMinutes"
                }
            }
            # tags = ["NitrateProcessedFiveMinutes", "AmmoniaProcessedFiveMinutes"]

            for parameter, metadata in raw_data_mapping.items():
                # Convert datetime to UTC and format in a compressed ISO 8601 format.
                data = metadata["data"]
                data.set_index(f'{parameter}_observedat', inplace=True)
                data[parameter] = data[parameter].astype('float')
                logging.debug(f"data is: {data.head()}; {data.index.dtype}, {data[parameter].dtype}")

                data = interpolate_data(data, timestamp_from=timestamp_from, timestamp_to=timestamp, resolution="1min")

                data = data.resample('5min').mean()

                logging.info(f"Successfully resampled the data to 5 mins: {data.head()}")
                # data[f'{parameter}_observedat'] = data[f'{parameter}_observedat'].dt.tz_convert(None).dt.strftime("%Y%m%dT%H%M%S")
                for index, row in data.iterrows():
                    # send data to Fiware IoT agent
                    request_id_mapping, request_id = tag_to_device_id(metadata["tag"])
                    timestamp_resampled = index.tz_convert(None).strftime("%Y%m%dT%H%M%S")
                    logging.info(f"Index posted is {index} and the string form is {timestamp_resampled}")
                    request_url = f"http://iot-agent-json:7896/iot/json?k=1234&i={request_id}&t={timestamp_resampled}"
                    request_id_mapping['concentration']['value'] = row[parameter]
                    try:

                        # requests.post(url=request_url, json=request_id_mapping)
                        logging.info(f"row[parameter] is {row[parameter]} and of type {row[parameter].dtype}")
                        requests.post(url=request_url, json={tag_to_property(metadata['tag']): row[parameter]})
                    except Exception as e:
                        logging.error(f"Error sending request to Fiware. URL=\"{request_url}\","
                                        f"json data= \"{str(request_id_mapping)}\", "
                                        f"Exception: {traceback.format_exc()}")
        except Exception as e:
            logging.error(traceback.format_exc())
            return f'Internal error | {e}', 500
        return jsonify("")

    elif 180 < data_length < 1440:
        # Checking if there is enough data to conduct a new 5 minute resample and analysis.
        if timestamp.minute % 5 != 0:
            logging.debug(f"The timestamp is {timestamp}")
            logging.info("<5min Notif 1: Not enough new data yet to conduct 5 minute based analysis")
            return jsonify("")
        # Querying historical database to get the last 1.5 hours of data for NH4 and NO3.
        # First we get the timestamp_from variable based on the current timestamp:
        time.sleep(0.5)
        raw_timestamp_from = timestamp - timedelta(hours=2, minutes=30)
        logging.info(f"The from timestamp for the raw data is {raw_timestamp_from}")
        raw_data_no3 = get_historic_data_from_database(
            'def_serv_ld.urn_ngsi_ld_thing_WasteWaterTank02',
            'no3', raw_timestamp_from,
            timestamp)
        logging.info(f"The first index of raw no3 is {raw_data_no3['no3_observedat'].head(1)} and the last index is {raw_data_no3['no3_observedat'].tail(1)}")
        raw_data_nh4 = get_historic_data_from_database(
            'def_serv_ld.urn_ngsi_ld_thing_WasteWaterTank02',
            'nh4', raw_timestamp_from,
            timestamp)
        logging.info(f"The first index of raw nh4 is {raw_data_nh4['nh4_observedat'].head(1)} and the last index is {raw_data_nh4['nh4_observedat'].tail(1)}")
        processed_5_min_timestamp_from = timestamp - timedelta(hours=3, minutes=5)
        logging.info(f"The from timestamp for the processed data is {processed_5_min_timestamp_from}")
        processed_data_5_min_no3 = get_historic_data_from_database(
                'def_serv_ld.urn_ngsi_ld_thing_NitrateProcessedFiveMinutes', 'concentration',
                processed_5_min_timestamp_from, timestamp, query_all=False)
        processed_data_5_min_nh4 = get_historic_data_from_database(
            'def_serv_ld.urn_ngsi_ld_thing_AmmoniaProcessedFiveMinutes', 'concentration',
            processed_5_min_timestamp_from, timestamp, query_all=False)

        run_data_reconciliation_procedures(raw_data_no3=raw_data_no3, raw_data_nh4=raw_data_nh4,
                                           processed_data_no3=processed_data_5_min_no3,
                                           processed_data_nh4=processed_data_5_min_nh4,
                                           resolution='5min', conduct_flatline=False,
                                           timestamp_from=raw_timestamp_from, timestamp_to=timestamp)
        return jsonify("")

    elif data_length == 1440:

        # Conduct the five minute based analytics as per the previous conditional step.
        time.sleep(0.5)
        raw_timestamp_from = timestamp - timedelta(hours=2, minutes=30)
        logging.info(f"The from timestamp for the raw data is {raw_timestamp_from}")
        raw_data_no3 = get_historic_data_from_database(
            'def_serv_ld.urn_ngsi_ld_thing_WasteWaterTank02',
            'no3', raw_timestamp_from,
            timestamp)
        logging.info(f"The first index of raw no3 is {raw_data_no3['no3_observedat'].head(1)} and the last index is {raw_data_no3['no3_observedat'].tail(1)}")
        raw_data_nh4 = get_historic_data_from_database(
            'def_serv_ld.urn_ngsi_ld_thing_WasteWaterTank02',
            'nh4', raw_timestamp_from,
            timestamp)
        logging.info(f"The first index of raw nh4 is {raw_data_nh4['nh4_observedat'].head(1)} and the last index is {raw_data_nh4['nh4_observedat'].tail(1)}")
        processed_5_min_timestamp_from = timestamp - timedelta(hours=3, minutes=5)
        logging.info(f"The from timestamp for the processed data is {processed_5_min_timestamp_from}")
        processed_data_5_min_no3 = get_historic_data_from_database(
            'def_serv_ld.urn_ngsi_ld_thing_NitrateProcessedFiveMinutes', 'concentration',
            processed_5_min_timestamp_from, timestamp, query_all=False)
        processed_data_5_min_nh4 = get_historic_data_from_database(
            'def_serv_ld.urn_ngsi_ld_thing_AmmoniaProcessedFiveMinutes', 'concentration',
            processed_5_min_timestamp_from, timestamp, query_all=False)

        run_data_reconciliation_procedures(raw_data_no3=raw_data_no3, raw_data_nh4=raw_data_nh4,
                                           processed_data_no3=processed_data_5_min_no3,
                                           processed_data_nh4=processed_data_5_min_nh4,
                                           resolution='5min', conduct_flatline=False,
                                           timestamp_from=raw_timestamp_from, timestamp_to=timestamp)

        # Conduct the resampling to 30 mins and then send the data to the 30 minute processed data
        # streams. Like how it is done for the 5 mins.
        raw_data_no3 = get_historic_data_from_database('def_serv_ld.urn_ngsi_ld_thing_WasteWaterTank02', 'no3', timestamp_to=timestamp, query_all=True)
        raw_data_nh4 = get_historic_data_from_database('def_serv_ld.urn_ngsi_ld_thing_WasteWaterTank02', 'nh4', timestamp_to=timestamp, query_all=True)
        logging.info(f"Length of raw data of no3 queried is {len(raw_data_no3)}")
        logging.info(f"Length of raw data of no3 queried is {len(raw_data_nh4)}")
        logging.debug(f"The no3 raw data is = {raw_data_no3}")
        logging.debug(f"The nh4 raw data is = {raw_data_nh4}")

        try:
            raw_data_mapping = {
                "no3": {
                    "data": raw_data_no3,
                    "tag": "NitrateProcessedThirtyMinutes"
                },
                "nh4": {
                    "data": raw_data_nh4,
                    "tag": "AmmoniaProcessedThirtyMinutes"
                }
            }

            for parameter, metadata in raw_data_mapping.items():
                # Convert datetime to UTC and format in a compressed ISO 8601 format.
                data = metadata["data"]
                data.set_index(f'{parameter}_observedat', inplace=True)
                data[parameter] = data[parameter].astype('float')
                logging.info(f"data is: {data.head()}; {data.index.dtype}, {data[parameter].dtype}")
                data = data.resample('30min').mean()
                logging.info(f"Sucessfully resampled the data to 30 mins: {data.head()}")
                # data[f'{parameter}_observedat'] = data[f'{parameter}_observedat'].dt.tz_convert(None).dt.strftime("%Y%m%dT%H%M%S")
                for index, row in data.iterrows():
                    # send data to Fiware IoT agent
                    request_id_mapping, request_id = tag_to_device_id(metadata["tag"])
                    timestamp_resampled = index.tz_convert(None).strftime("%Y%m%dT%H%M%S")
                    logging.info(f"Index posted is {index} and the string form is {timestamp_resampled}")
                    request_url = f"http://iot-agent-json:7896/iot/json?k=1234&i={request_id}&t={timestamp_resampled}"
                    request_id_mapping['concentration']['value'] = row[parameter]
                    try:
                        # requests.post(url=request_url, json=request_id_mapping)
                        logging.info(f"row[parameter] is{row[parameter]} and of type {row[parameter].dtype}")
                        requests.post(url=request_url, json={tag_to_property(metadata['tag']): row[parameter]})
                    except Exception as e:
                        logging.error(f"Error sending request to Fiware. URL=\"{request_url}\","
                                        f"json data= \"{str(request_id_mapping)}\", "
                                        f"Exception: {traceback.format_exc()}")
        except Exception as e:
            logging.error(traceback.format_exc())
            return f'Internal error | {e}', 500
        return jsonify("")

    else:
        if timestamp.minute % 5 != 0:
            logging.info("<5min Notif 2: Not enough new data yet to conduct 5 minute based analysis")
            return jsonify("")

        # Querying historical database to get the last 1.5 hours of data for NH4 and NO3.
        # First we get the timestamp_from variable based on the current timestamp:
        time.sleep(0.5)
        raw_timestamp_from = timestamp - timedelta(hours=2, minutes=30)
        raw_data_no3 = get_historic_data_from_database('def_serv_ld.urn_ngsi_ld_thing_WasteWaterTank02',
                                                       'no3', raw_timestamp_from,
                                                       timestamp)
        logging.info(
            f"The first index of raw no3 is {raw_data_no3['no3_observedat'].head(1)} and the last "
            f"index is {raw_data_no3['no3_observedat'].tail(1)}")
        raw_data_nh4 = get_historic_data_from_database('def_serv_ld.urn_ngsi_ld_thing_WasteWaterTank02',
                                                       'nh4', raw_timestamp_from,
                                                       timestamp)
        logging.info(
            f"The first index of raw nh4 is {raw_data_nh4['nh4_observedat'].head(1)} and the last "
            f"index is {raw_data_nh4['nh4_observedat'].tail(1)}")

        # Querying historical database to get the last 3 hours of the intermediate reconciled stream
        # for the 5 minute autoencoder. It will already be of resolution 5 minutes
        processed_5_min_timestamp_from = timestamp - timedelta(hours=3, minutes=5)
        logging.info(f"The from timestamp for the 5 min processed data is {processed_5_min_timestamp_from}")
        processed_data_5_min_no3 = get_historic_data_from_database(
            'def_serv_ld.urn_ngsi_ld_thing_NitrateProcessedFiveMinutes', 'concentration',
            processed_5_min_timestamp_from, timestamp, query_all=False)
        processed_data_5_min_nh4 = get_historic_data_from_database(
            'def_serv_ld.urn_ngsi_ld_thing_AmmoniaProcessedFiveMinutes', 'concentration',
            processed_5_min_timestamp_from, timestamp, query_all=False)

        run_data_reconciliation_procedures(raw_data_no3=raw_data_no3, raw_data_nh4=raw_data_nh4,
                                           processed_data_no3=processed_data_5_min_no3,
                                           processed_data_nh4=processed_data_5_min_nh4,
                                           resolution='5min', conduct_flatline=True,
                                           timestamp_from=raw_timestamp_from, timestamp_to=timestamp)

        if timestamp.minute == 0 or timestamp.minute == 30:
            time.sleep(0.5)
            # Querying historical database to get the last 24 hours of the intermediate reconciled
            # stream for the 30 minute autoencoder. It will already be of resolution 30 minutes
            processed_30_min_timestamp_from = timestamp - timedelta(hours=24, minutes=30)
            logging.info(f"The from timestamp for the 30 min processed data is {processed_5_min_timestamp_from}")
            processed_data_30_min_no3 = get_historic_data_from_database(
                'def_serv_ld.urn_ngsi_ld_thing_NitrateProcessedThirtyMinutes', 'concentration',
                processed_30_min_timestamp_from, timestamp - timedelta(minutes=30))
            logging.debug(f"Here is what is in the raw data for no3 before 30 min analysis: {raw_data_no3}; {raw_data_no3.index.dtype}; {raw_data_no3.index.dtype} ")

            processed_data_30_min_nh4 = get_historic_data_from_database(
                'def_serv_ld.urn_ngsi_ld_thing_AmmoniaProcessedThirtyMinutes', 'concentration',
                processed_30_min_timestamp_from, timestamp - timedelta(minutes=30))
            logging.debug(f"Here is what is in the raw data for nh4 before 30 min analysis: {raw_data_nh4}; {raw_data_nh4.index.dtype}; {raw_data_nh4.index.dtype} ")

            run_data_reconciliation_procedures(raw_data_no3=raw_data_no3, raw_data_nh4=raw_data_nh4,
                                               processed_data_no3=processed_data_30_min_no3,
                                               processed_data_nh4=processed_data_30_min_nh4,
                                               resolution='30min', conduct_flatline=True,
                                               timestamp_from=raw_timestamp_from, timestamp_to=timestamp)

            # Querying the 5 minute and 30 minute processed data streams for aggregation.
            processed_aggregation_timestamp_from = timestamp - timedelta(minutes=30)
            logging.info(f"The from timestamp for the aggregation is {processed_aggregation_timestamp_from}")

            processed_data_5_min_no3 = get_historic_data_from_database(
                'def_serv_ld.urn_ngsi_ld_thing_NitrateProcessedFiveMinutes', 'concentration',
                processed_aggregation_timestamp_from, timestamp, query_all=False)
            processed_data_5_min_nh4 = get_historic_data_from_database(
                'def_serv_ld.urn_ngsi_ld_thing_AmmoniaProcessedFiveMinutes', 'concentration',
                processed_aggregation_timestamp_from, timestamp, query_all=False)

            processed_data_30_min_no3 = get_historic_data_from_database(
                'def_serv_ld.urn_ngsi_ld_thing_NitrateProcessedThirtyMinutes', 'concentration',
                processed_aggregation_timestamp_from, timestamp, query_all=False)
            processed_data_30_min_nh4 = get_historic_data_from_database(
                'def_serv_ld.urn_ngsi_ld_thing_AmmoniaProcessedThirtyMinutes', 'concentration',
                processed_aggregation_timestamp_from, timestamp, query_all=False)

            try:
                # Small mapping to conduct the aggregation and posting of results to the IoT agent
                aggregation_data_mapping = {
                    "no3": {
                        "tag": "2521 ATQT420_TRMW",
                        "processed_data_5_min": processed_data_5_min_no3,
                        "processed_data_30_min": processed_data_30_min_no3,
                        "tag_reconciled_data": "NitrateReconciled"
                    },
                    "nh4": {
                        "tag": "2521 ATQT440_TRMW",
                        "processed_data_5_min": processed_data_5_min_nh4,
                        "processed_data_30_min": processed_data_30_min_nh4,
                        "tag_reconciled_data": "AmmoniaReconciled"
                    }
                }
                for parameter, metadata in aggregation_data_mapping.items():
                    # Getting the 5 minute processed data
                    # Convert datetime to UTC and format in a compressed ISO 8601 format.
                    processed_data_5_min = metadata["processed_data_5_min"]
                    if not isinstance(processed_data_5_min.index, pd.DatetimeIndex):
                        processed_data_5_min.set_index('concentration_observedat', inplace=True)
                        logging.info(f"The isinstance check index dtype gave a {isinstance(processed_data_5_min.index, pd.DatetimeIndex)}")
                        processed_data_5_min.rename(columns={'concentration': metadata['tag']}, inplace=True)
                        processed_data_5_min[metadata['tag']] = processed_data_5_min[metadata['tag']].astype('float')
                    logging.debug(f"The 5 min processed data is: {processed_data_5_min}")

                    # Getting the 30 minute processed data
                    processed_data_30_min = metadata["processed_data_30_min"]
                    if not isinstance(processed_data_30_min.index, pd.DatetimeIndex):
                        processed_data_30_min.set_index('concentration_observedat', inplace=True)
                        logging.info(f"The isinstance check index dtype gave a {isinstance(processed_data_30_min.index, pd.DatetimeIndex)}")
                        processed_data_30_min.rename(columns={'concentration': metadata["tag"]}, inplace=True)
                        processed_data_30_min[metadata["tag"]] = processed_data_30_min[metadata["tag"]].astype("float")
                    logging.debug(f"The 30 min processed data is: {processed_data_30_min}")

                    # Calling the aggregation function which also resampled to 15 min data.
                    reconciled_signal = None
                    try:
                        reconciled_signal = autoencoder_aggregation(
                            processed_5_min=processed_data_5_min,
                            processed_30_min=processed_data_30_min, tag=metadata["tag"],
                            resample_target='15min')
                        logging.info(f"Successfully conducted the aggregation for {parameter}")
                        logging.debug(f"The aggregation values are: {reconciled_signal}")
                    except Exception as e:
                        logging.error(f"Error conducting the aggregation for {parameter}: {e}")

                    # Posting the reconciled data to the IoT Agent.
                    if reconciled_signal is not None:
                        for index, row in reconciled_signal.iterrows():
                            # send data to Fiware IoT agent
                            request_id_mapping, request_id = tag_to_device_id(metadata["tag_reconciled_data"])
                            timestamp_resampled = index.tz_convert(None).strftime("%Y%m%dT%H%M%S")
                            logging.info(f"Index posted is {index} and the string form is {timestamp_resampled}")
                            request_url = f"http://iot-agent-json:7896/iot/json?k=1234&i={request_id}&t={timestamp_resampled}"
                            request_id_mapping['concentration']['value'] = row[metadata["tag"]]
                            try:
                                # requests.post(url=request_url, json=request_id_mapping)
                                requests.post(url=request_url,
                                              json={tag_to_property(metadata['tag_reconciled_data']): row[metadata["tag"]]})
                            except Exception as e:
                                logging.error(f"Error sending request to Fiware. URL=\"{request_url}\","
                                              f"json data= \"{str(request_id_mapping)}\", "
                                              f"Exception: {traceback.format_exc()}")
            except Exception as e:
                logging.error(traceback.format_exc())
                return f'Internal error | {e}', 500
            return jsonify("")
    return jsonify("")


tags_for_model = [#"WasteWaterTank02",
    "6218_SPLM_873", "6254_SPRS_873",
    "NitrateReconciled",
    "NitrateProcessedFiveMinutes", "NitrateProcessedThirtyMinutes",
    "NitratePredictionsFiveMinutes", "NitratePredictionsThirtyMinutes",
    "AmmoniaReconciled",
    "AmmoniaProcessedFiveMinutes", "AmmoniaProcessedThirtyMinutes",
    "AmmoniaPredictionsFiveMinutes", "AmmoniaPredictionsThirtyMinutes",
    "NitrateAnomaly", "AmmoniaAnomaly"]


def ensure_devices_are_registered():
    # Get all devices registered in Fiware.
    registered_devices = get_devices()
    logging.debug(f"type(registered_devices) = \"{type(registered_devices)}\", registered_devices = \"{registered_devices}\"")
    registered_devices = registered_devices.get("devices", [])
    # Make a flat list of all device ids that we want to have registered.
    device_ids = [item for sublist in type_to_device_id_mapping.values() for item in sublist]
    # For each device, check whether it is already registered, or we have to register it.
    for device_id in set(device_ids):
        found = False
        for registered_device in registered_devices:
            if registered_device.get("service", "") == "openiot" and \
                registered_device.get("service_path", "") == "/" and \
                registered_device.get("entity_name", "") == device_id and \
                registered_device.get("entity_type", "") == device_id.split(":")[-2] and \
                registered_device.get("device_id", "") == device_id.split(":")[-1]:
                # This device is registered correctly.
                found = True
                logging.info(f"Device \"{device_id}\" already registered correctly")
                break
        if not found:
            # Register the device.
            response = requests.post(
                url="http://iot-agent-json:4041/iot/devices",
                headers={"fiware-service": "openiot",
                        "fiware-servicepath": "/",
                        "Content-Type": "application/json"},
                data=f'''{{
                    "devices": [
                        {{
                            "@context": "https://smartdatamodels.org/context.jsonld",
                            "device_id": "{device_id.split(':')[-1]}",
                            "device_name": "{device_id}",
                            "device_type": "{device_id.split(':')[-2]}",
                            "timezone": "Europe/Amsterdam"
                        }}
                    ]
                }}'''
            )
            if response.status_code == 201:
                logging.info(f"Registered device {device_id}")
            else:
                logging.warning(f"Failed to register device {device_id} Fiware orion request statuscode {response.status_code} - {response.text}")
    return


def ensure_subscriptions_exist():
    # Get all subscriptions registered in Fiware.
    subscriptions = get_subscriptions()
    # For each device type and property, we want a subscription, so check if there is one per device.
    types_properties = []
    for tag in tags_for_model:
        types_properties.append((tag_to_type(tag), tag_to_property(tag)))
    for device_type, device_property in set(types_properties):
        if device_property not in ["no3", "nh4"]:
            # No need to notify this component of any property, except the ammonia and nitrate.
            continue
        found = False
        for subscription in subscriptions:
            if {"type": device_type} in subscription.get("entities", []) and \
                device_property in subscription.get("watchedAttributes", []) and \
                subscription.get("notification", dict()).get("endpoint", dict()).get("uri", "") == \
                    "http://ai_data_validation_reconciliation:80/subscription_notification":
                # This subscription should cover these devices.
                found = True
                logging.info(f"Subscription for {device_type}.{device_property} already registered correctly")
                break
        if not found:
            # Create a subscription for this type and property.
            response = requests.post(
                url="http://orion:1026/ngsi-ld/v1/subscriptions",
                headers={"NGSILD-Tenant": "openiot",
                        "Content-Type": "application/ld+json"},
                data=f'''{{
                    "description": "Notify of all {device_type}.{device_property} changes",
                    "type": "Subscription",
                    "entities": [{{"type": "{device_type}"}}],
                    "watchedAttributes": ["{device_property}"],
                    "notification": {{
                        "format": "normalized",
                        "endpoint": {{
                            "uri": "http://ai_data_validation_reconciliation:80/subscription_notification"
                        }}
                    }},
                    "@context": "https://smartdatamodels.org/context.jsonld"
                }}'''
            )
            if response.status_code == 201:
                logging.info(f"Created subscription for {device_type}.{device_property} to this component")
            else:
                logging.warning(f"Failed to create subscription for {device_type}.{device_property} to this component, Fiware orion request statuscode {response.status_code} - {response.text}")
    # We also need a subscription to send a copy of any data to Cygnus, which in turn will store it in a database.
    # But before we create the subscription, we have to create some tables in the PostgreSQL database.
    # Due to some bug or misuse of Cygnus, only 1 property of the NitrateValidated ends up in PostgreSQL.
    # Workaround is to first create those tables, and then create the subscriptions.
    commands = [
        """CREATE SCHEMA IF NOT EXISTS def_serv_ld AUTHORIZATION postgres;""",
        """
            CREATE TABLE IF NOT EXISTS def_serv_ld.urn_ngsi_ld_thing_WasteWaterTank02 (
                recvtime text,
                entityid text,
                entitytype text,
                no3 text,
                no3_observedat text,
                nh4 text,
                nh4_observedat text);
        """,
        """
            CREATE TABLE IF NOT EXISTS def_serv_ld.urn_ngsi_ld_thing_NitrateReconciled (
                recvtime text,
                entityid text,
                entitytype text,
                concentration text,
                concentration_observedat text);
        """,
        """
            CREATE TABLE IF NOT EXISTS def_serv_ld.urn_ngsi_ld_thing_NitrateProcessedFiveMinutes (
                recvtime text,
                entityid text,
                entitytype text,
                concentration text,
                concentration_observedat text);
        """,
        """
            CREATE TABLE IF NOT EXISTS def_serv_ld.urn_ngsi_ld_thing_NitrateProcessedThirtyMinutes (
                recvtime text,
                entityid text,
                entitytype text,
                concentration text,
                concentration_observedat text);
        """,
        """
            CREATE TABLE IF NOT EXISTS def_serv_ld.urn_ngsi_ld_thing_NitratePredictionsFiveMinutes (
                recvtime text,
                entityid text,
                entitytype text,
                concentration text,
                concentration_observedat text);
        """,
        """
            CREATE TABLE IF NOT EXISTS def_serv_ld.urn_ngsi_ld_thing_NitratePredictionsThirtyMinutes (
                recvtime text,
                entityid text,
                entitytype text,
                concentration text,
                concentration_observedat text);
        """,
        """
            CREATE TABLE IF NOT EXISTS def_serv_ld.urn_ngsi_ld_thing_AmmoniaReconciled (
                recvtime text,
                entityid text,
                entitytype text,
                concentration text,
                concentration_observedat text);
        """,
        """
            CREATE TABLE IF NOT EXISTS def_serv_ld.urn_ngsi_ld_thing_AmmoniaProcessedFiveMinutes (
                recvtime text,
                entityid text,
                entitytype text,
                concentration text,
                concentration_observedat text);
        """,
        """
            CREATE TABLE IF NOT EXISTS def_serv_ld.urn_ngsi_ld_thing_AmmoniaProcessedThirtyMinutes (
                recvtime text,
                entityid text,
                entitytype text,
                concentration text,
                concentration_observedat text);
        """,
        """
            CREATE TABLE IF NOT EXISTS def_serv_ld.urn_ngsi_ld_thing_AmmoniaPredictionsFiveMinutes (
                recvtime text,
                entityid text,
                entitytype text,
                concentration text,
                concentration_observedat text);
        """,
        """
            CREATE TABLE IF NOT EXISTS def_serv_ld.urn_ngsi_ld_thing_AmmoniaPredictionsThirtyMinutes (
                recvtime text,
                entityid text,
                entitytype text,
                concentration text,
                concentration_observedat text);
        """,
        """
            CREATE TABLE IF NOT EXISTS def_serv_ld.urn_ngsi_ld_thing_NitrateAnomaly (
                recvtime text,
                entityid text,
                entitytype text,
                thresholdBreach text,
                negativeValue text,
                nanValue text,
                zeroValue text,
                flatline text,
                measuredValue text,
                flag_observedat text);
        """,
        """
            CREATE TABLE IF NOT EXISTS def_serv_ld.urn_ngsi_ld_thing_AmmoniaAnomaly (
                recvtime text,
                entityid text,
                entitytype text,
                thresholdBreach text,
                negativeValue text,
                nanValue text,
                zeroValue text,
                flatline text,
                measuredValue text,
                flag_observedat text);
        """,
    ]
    conn = None
    try:
        conn = psycopg2.connect(
            dbname=postgres_dbname,
            user=postgres_user,
            password=postgres_password,
            host=postgres_host,
            port=postgres_port
        )
        cur = conn.cursor()
        # create table one by one
        for command in commands:
            cur.execute(command)
            logging.info(f"Executed {command}")
        cur.close()
        logging.info('Cur is closed')
        conn.commit()
        logging.info('conn is commit')
        logging.info("Successfully created tables in PostgreSQL for the Cygnus-notification workaround.")
    except (Exception, psycopg2.DatabaseError) as error:
        logging.error(f"Error occurred while creating the PostgreSQL schema and tables: {error}")
    finally:
        if conn is not None:
            conn.close()
    # Now make sure the subscription to Cygnus exists.
    for device_type, device_property in set(types_properties):
        found = False
        for subscription in subscriptions:
            if {"type": device_type} in subscription.get("entities", []) and \
                device_property in subscription.get("watchedAttributes", []) and \
                subscription.get("notification", dict()).get("endpoint", dict()).get("uri", "") == \
                    "http://cygnus:5050/notify":
                # This subscription should cover these devices.
                found = True
                logging.info(f"Subscription for {device_type}.{device_property} already registered correctly in Cygnus")
                break
        if not found:
            # Create a Cygnus subscription for this type and property.
            response = requests.post(
                url="http://orion:1026/ngsi-ld/v1/subscriptions",
                headers={"NGSILD-Tenant": "openiot",
                         "Content-Type": "application/ld+json"},
                data=f'''{{
                               "description": "Notify Cygnus of all {device_type}.{device_property} changes",
                               "type": "Subscription",
                               "entities": [{{"type": "{device_type}"}}],
                               "watchedAttributes": ["{device_property}"],
                               "notification": {{
                                   "format": "normalized",
                                   "endpoint": {{
                                       "uri": "http://cygnus:5050/notify"
                                   }}
                               }},
                               "@context": "https://smartdatamodels.org/context.jsonld"
                           }}'''
            )
            if response.status_code == 201:
                logging.info(f"Created subscription for {device_type}.{device_property} to Cygnus")
            else:
                logging.warning(f"Failed to create subscription for {device_type}.{device_property} to Cygnus, Fiware orion request statuscode {response.status_code} - {response.text}")
    return


def get_historic_data_from_database(table, field, timestamp_from=None, timestamp_to=None, query_all=False):
    """Queries the database for the given table with field for the given time window (on the field his observed datetime).
    Returns a pandas dataframe.

    Example: get_historic_data_from_database('def_serv_ld.urn_ngsi_ld_thing_nitratevalidated01', 'concentration', '2021-11-30 11:30:00', '2021-11-30 14:29:00')
    """
    df = pd.DataFrame()
    if query_all:
        # query = f"SELECT {field}, {field}_observedat FROM {table} WHERE {field}_observedat::timestamp < '{timestamp_to}'::timestamp;"
        query = f"SELECT DISTINCT ON ({field}_observedat) {field}, {field}_observedat FROM {table} WHERE {field}_observedat::timestamp < '{timestamp_to}'::timestamp ORDER BY {field}_observedat, recvtime DESC;"
    else:
        # query = f"SELECT {field}, {field}_observedat FROM {table} WHERE {field}_observedat::timestamp BETWEEN '{timestamp_from}'::timestamp AND '{timestamp_to}'::timestamp;"
        query = f"SELECT DISTINCT ON ({field}_observedat) {field}, {field}_observedat FROM {table} WHERE {field}_observedat::timestamp BETWEEN '{timestamp_from}'::timestamp AND '{timestamp_to}'::timestamp ORDER BY {field}_observedat, recvtime DESC;"
    conn = None
    try:
        conn = psycopg2.connect(
            dbname=postgres_dbname,
            user=postgres_user,
            password=postgres_password,
            host=postgres_host,
            port=postgres_port
        )
        df = sqlio.read_sql_query(query, conn)
        df[f"{field}_observedat"] = pd.to_datetime(df[f"{field}_observedat"])
        logging.info(f"Successfully queried table \"{table}.{field}\" for \"{field}_observedat\" between \"{timestamp_from}\" and \"{timestamp_to}\".")
    except (Exception, psycopg2.DatabaseError) as error:
        logging.error(f"Error occurred while querying table \"{table}.{field}\" for \"{field}_observedat\" between \"{timestamp_from}\" and \"{timestamp_to}\": {error}")
    finally:
        if conn is not None:
            conn.close()
    return df


def get_subscriptions():
    return requests.get(
        url="http://orion:1026/ngsi-ld/v1/subscriptions",
        headers={"NGSILD-Tenant": "openiot"}
    ).json()


def get_devices():
    return requests.get(
        url="http://iot-agent-json:4041/iot/devices",
        headers={"fiware-service": "openiot", "fiware-servicepath": "/"}
    ).json()

# Ensure devices are registered in Fiware.
ensure_devices_are_registered()
# Ensure Fiware subscriptions exist for all tags / devices the model needs.
ensure_subscriptions_exist()


if __name__ == '__main__':
    app.run(debug=True)
