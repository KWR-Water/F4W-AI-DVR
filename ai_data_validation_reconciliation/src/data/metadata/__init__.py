import pathlib
from os.path import join

path_metadata = pathlib.Path(__file__).parent.absolute()

file_metadata_limits = 'sensor_metadata.json.json'

fpath_metadata_limits = join(path_metadata, file_metadata_limits)