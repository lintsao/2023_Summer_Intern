"""Copyright 2020 ETH Zurich, Yufeng Zheng, Seonwook Park
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import glob
import json
import os
import shutil
import sys

import logging
logger = logging.getLogger(__name__)


class DefaultConfig(object):

    # Batch Size-related
    batch_size = 2
    eval_batch_size = 2

    # Coefficients
    coeff_discriminator_loss = 1.0
    coeff_discriminator_r1_loss = 1.0
    coeff_disentangle_embedding_loss = 2.0
    coeff_disentangle_pseudo_label_loss = 2.0
    coeff_embedding_consistency_loss = 2.0
    coeff_gaze_head_label_loss = 5.0
    coeff_id_loss = 1.0
    coeff_l1_loss = 200.0
    coeff_l2_loss = 1.0
    coeff_perceptual_loss = 1.0
    coeff_redirection_feature_loss = 200.0
    coeff_redirection_gaze_loss = 2.0
    coeff_redirection_head_loss = 2.0
    r1 = 10
    w_discriminator_lambda = 0.0002

    # Decay-related
    decay = 0.8
    decay_steps = 3000

    # # Densenet-related
    densenet_blocks = 5

    # Delta Norm-related
    delta_norm = 2
    delta_norm_lambda = 2e-4

    # General Configuration
    is_training_discriminator = True
    compute_full_result = True
    d_reg_every = 16
    is_progressive_training = True
    load_step = 0
    num_data_loaders = 4
    print_freq_test = 5000
    print_freq_train = 200
    save_freq_images = 1000
    save_interval = 1000
    semi_supervised = False
    skip_training = False
    start_from_latent_avg = True
    store_redirect_dataset = False
    test_subsample = 0.02
    use_apex = False
    use_mixing = True
    use_tensorboard = True
    use_w_pool = True
    w_pool_size = 50

    # Learning Rates
    base_learning_rate = 0.0004
    w_discriminator_lr = 2e-5

    # Size-related
    num_2d_units = 2
    size_2d_unit = 16

    # Steps and Training
    # num_training_steps = 206865
    num_training_steps = 40000
    progressive_steps = None
    global_step = 0

    # Other
    growth_rate = 32
    num_labeled_samples = 0
    l2_reg = 1e-4
    pick_at_least_per_person = None

    # Data Paths
    mpiigaze_file = "./dataset/MPIIGaze_128.h5"
    gazecapture_file = "./dataset/GazeCapture_128.h5"
    pretrained_head_gaze_net_train = "./pretrained_models/baseline_estimator_vgg.tar"
    pretrained_head_gaze_net_eval = "./pretrained_models/baseline_estimator_resnet.tar"
    gazenet_savepath = "./output/baseline_estimator_vgg/"
    eval_gazenet_savepath = "./output/baseline_estimator_resnet/"
    save_path = "./output/ST-ED/save_2"
    log_path = "./output/ST-ED/log_2"

    # Redirect Inference
    input_path = ""
    redirect_path = "./output/redirect"
    checkpoint_path = "./output/ST-ED/save_4/checkpoints/40000.pt"
    # Rad (Degrees × (π / 180))
    head_pitch = 0.0
    head_yaw = 0.0
    gaze_pitch = 0.0
    gaze_yaw = 0.0
    preprocess = False

    @property
    def lr(self):
        lr = self.batch_size * self.base_learning_rate
        # print(lr, self.batch_size)
        return lr 
    # Available strategies:
    #     'exponential': step function with exponential decay
    #     'cyclic':      spiky down-up-downs (with exponential decay of peaks)

    # Below lie necessary methods for working configuration tracking

    __instance = None

    # Make this a singleton class
    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.__filecontents = cls.__get_config_file_contents()
            cls.__pycontents = cls.__get_python_file_contents()
            cls.__immutable = True
        return cls.__instance

    def import_json(self, json_path, strict=True):
        """Import JSON config to over-write existing config entries."""
        print(json_path)
        assert os.path.isfile(json_path)
        assert not hasattr(self.__class__, '__imported_json_path')
        logger.info('Loading ' + json_path)
        with open(json_path, 'r') as f:
            json_string = f.read()
        self.import_dict(json.loads(json_string), strict=strict)
        self.__class__.__imported_json_path = json_path
        self.__class__.__filecontents[os.path.basename(json_path)] = json_string

    def import_dict(self, dictionary, strict=True):
        """Import a set of key-value pairs from a dict to over-write existing config entries."""
        self.__class__.__immutable = False
        for key, value in dictionary.items():
            if strict is True:
                if not hasattr(self, key):
                    raise ValueError('Unknown configuration key: ' + key)
                assert type(getattr(self, key)) is type(value)
                if not isinstance(getattr(DefaultConfig, key), property):
                    setattr(self, key, value)
            else:
                if hasattr(DefaultConfig, key):
                    if not isinstance(getattr(DefaultConfig, key), property):
                        setattr(self, key, value)
                else:
                    setattr(self, key, value)
        self.__class__.__immutable = True

    def __get_config_file_contents():
        """Retrieve and cache default and user config file contents."""
        out = {}
        for relpath in ['config_default.py']:
            path = os.path.relpath(os.path.dirname(__file__) + '/' + relpath)
            assert os.path.isfile(path)
            with open(path, 'r') as f:
                out[os.path.basename(path)] = f.read()
        return out

    def __get_python_file_contents():
        """Retrieve and cache default and user config file contents."""
        out = {}
        base_path = os.path.relpath(os.path.dirname(__file__) + '/../')
        source_fpaths = [
            p for p in glob.glob(base_path + '/**/*.py')
            if not p.startswith('./3rdparty/')
        ]
        source_fpaths += [os.path.relpath(sys.argv[0])]
        for fpath in source_fpaths:
            assert os.path.isfile(fpath)
            with open(fpath, 'r') as f:
                out[fpath[2:]] = f.read()
        return out

    def get_all_key_values(self):
        return dict([
            (key, getattr(self, key))
            for key in dir(self)
            if not key.startswith('_DefaultConfig')
            and not key.startswith('__')
            and not callable(getattr(self, key))
        ])

    def get_full_json(self):
        return json.dumps(self.get_all_key_values(), indent=4)

    def write_file_contents(self, target_base_dir):
        """Write cached config file contents to target directory."""
        assert os.path.isdir(target_base_dir)

        # Write config file contents
        target_dir = target_base_dir + '/configs'
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)
        outputs = {  # Also output flattened config
            'combined.json': self.get_full_json(),
        }
        outputs.update(self.__class__.__filecontents)
        for fname, content in outputs.items():
            fpath = os.path.relpath(target_dir + '/' + fname)
            with open(fpath, 'w') as f:
                f.write(content)
                logger.info('Written %s' % fpath)

        # Write Python file contents
        # NOTE: older version copying individual files
        # target_dir = target_base_dir + '/src'
        # for fname, content in self.__pycontents.items():
        #     fpath = os.path.relpath(target_dir + '/' + fname)
        #     dpath = os.path.dirname(fpath)
        #     if not os.path.isdir(dpath):
        #         os.makedirs(dpath)
        #     with open(fpath, 'w') as f:
        #         f.write(content)
        # logger.info('Written %d source files to %s' %
        #             (len(self.__pycontents), os.path.relpath(target_dir)))

        # Copy source folder contents over
        target_path = os.path.relpath(target_base_dir + '/src')
        shutil.make_archive(target_path, "tar",
                            os.path.relpath(os.path.dirname(__file__) + '/../'))
        logger.info('Written source folder to %s' % os.path.relpath(target_path))

    def __setattr__(self, name, value):
        """Initial configs should not be overwritten!"""
        if self.__class__.__immutable:
            raise AttributeError('DefaultConfig instance attributes are immutable.')
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        """Initial configs should not be removed!"""
        if self.__class__.__immutable:
            raise AttributeError('DefaultConfig instance attributes are immutable.')
        else:
            super().__delattr__(name)
