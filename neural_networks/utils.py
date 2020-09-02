import os
import json

import numpy as np
from tensorflow import keras

class MultiInitLog():

    def __init__(self):
        super(MultiInitLog, self).__init__()
        self.inits = list()
        self.model_config = None
        self.current_init = 0

    def add_initialization(self, model, history):
        if self.current_init == 0:
            self.model_config = model.to_json()
        else:
            if self.model_config != model.to_json():
                raise ValueError(f'The multi init must be used with only one model configuration. The current model is different from the previous ones')
			
        self.inits.append({'weights': model.get_weights(), 'logs': history.history, 'params': history.params})
        self.current_init += 1

    def get_best_init(self, metric, mode, training_end):

        self._check_inits_integrity()
        best_init = 0

        if mode == 'max':
            best_metric = -np.inf
            is_best = lambda x: x>best_metric
            get_best = np.amax
        elif mode == 'min':
            best_metric = np.inf
            is_best = lambda x: x<best_metric
            get_best = np.amin
        else:
            raise ValueError(f'Only "max" and "min" are options for the mode parameter. {mode} was passed')

        if training_end:
            for init in range(len(self.inits)):
                current_metric = self.inits[init]['logs'][metric][-1]
                best_init = init if is_best(current_metric) else best_init
        else:
            for init in range(len(self.inits)):
                current_metric = get_best(self.inits[init]['logs'][metric])
                best_init = init if is_best(current_metric) else best_init
        
        return best_init

    def best_weights(self, metric, mode, training_end):
        best_init = self.get_best_init(metric, mode, training_end)
        return self.inits[best_init]['weights']  

    def _check_inits_integrity(self):
        for init in range(len(self.inits)):
            if self.inits[init]['logs'] is None or self.inits[init]['params'] is None:
                raise ValueError(f'Init {init} is missing its callback.History. You can add it using add_history method')

    def to_json(self, filepath):
        """Saves the log into a json file"""

        with open(filepath, 'w') as json_file:
            json_dict = utils.cast_to_python(self.__dict__)
            json.dump(json_dict, json_file, indent=4)

    @classmethod
    def from_json(cls, filepath):
        """Loads the log from a json file"""
         
        with open(filepath, 'r') as json_file:
            json_dict = json.load(json_file)

        for init in json_dict['inits']:
            init['weights'] = [np.array(weight) for weight in init['weights']]

        log = cls()
        log.__dict__ = json_dict

        return log

    def get_best_model(self, metric, mode, training_end, custom_objects=None):
        """ Loads the best model and returns it"""

        model = keras.models.model_from_json(self.model_config, custom_objects)
        best_init = self.get_best_init(metric, mode, training_end)
        model.set_weights(self.inits[best_init]['weights'])

        return model
    
def load_expert_committee(multi_init_path, metric, mode, training_end, custom_objects=None):
    """Loads an expert committee form a directory containing a multi init log for each expert"""
    
    experts = list()
    experts_log = np.sort(os.listdir(multi_init_path))
    custom_objects = (None for _ in range(len(experts_log))) if custom_objects is None else custom_objects
    for expert_log, custom_object in zip(experts_log, custom_objects):
        log = MultiInitLog.from_json(os.path.join(multi_init_path, expert_log))
        expert = log.get_best_model(metric, mode, training_end, custom_object)
        experts.append(expert)
    #import pdb; pdb.set_trace()
    input_layer = keras.Input(tuple(experts[0].input.shape[1:]))
    experts = [expert(input_layer) for expert in experts]
    output_layer = keras.layers.concatenate(experts)

    return keras.Model(input_layer, output_layer)

def reshape_conv_input(data):
    """Returns an numpy.ndarray reshaped as an input for a convolutional layer from keras

    Parameters:

    data: numpy.ndarray
        Data to be reshaped
    """
    
    shape = list(data.shape)
    shape.append(1)
    return data.reshape(tuple(shape))