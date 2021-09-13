import os
import yaml
from os.path import join, realpath, dirname


class SettingKeyError(BaseException):
    pass


class SettingDict(dict):

    def __init__(self, filename, **kwargs):
        super().__init__(**kwargs)
        self.filename = filename

        with open(filename) as f:
            super().update(yaml.load(f))

    def __getitem__(self, key):

        if super().__contains__(key):
            return super().__getitem__(key)
        else:
            raise SettingKeyError('No entry for ' + key + ' was found in file ' + self.filename)


path_filename = join(dirname(realpath(__file__)), '..', 'paths.yaml')
config_filename = join(dirname(realpath(__file__)), '..', 'config.yaml')

if not os.path.isfile(path_filename):
    # create sample paths

    with open(config_filename, 'w') as f:
        sample_content = ('AVA_ROOT: ' + realpath(join(__file__, '..')) + '\n'
                          'AVA_DATA: ' + realpath(join(__file__, '../ava/data')) + '\n'
                          'CACHE_PATH: cache\n'
                          'TRAINED_MODELS_PATH: pretrained_models')
        f.write(sample_content)

    print('paths.yaml was not found. Therefore, a new file was created. Please check its content.')

if not os.path.isfile(path_filename):
    # create sample config
    with open(join(dirname(realpath(__file__)), '..', 'config.yaml.sample')) as f:
        with open(config_filename, 'w') as f2:
            f2.write(f.read())

    print('config.yaml was not found. Therefore, a new file was created from config.yaml.sample. '
          'Please check the newly created config.yaml.')

PATHS = SettingDict(path_filename)
CONFIG = SettingDict(config_filename)


