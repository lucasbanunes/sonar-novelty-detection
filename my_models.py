import os
from copy import deepcopy

import numpy as np
from tensorflow import keras
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from data_analysis.neural_networks import losses, metrics, training
from data_analysis.utils.utils import to_sparse_tanh, DataSequence, shuffle_pair
from data_analysis.utils.lofar_operators import LofarImgSequence

NUMBER_OF_INITS = 10

class standard_model():

    @staticmethod
    def compile_and_fit(model,train_set, val_set, bot=None, chat_id=None, message_id=None):

        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        early_stop = callbacks.EarlyStopping(monitor = 'val_sparse_accuracy', min_delta=0.001, patience=30, restore_best_weights=True, mode='max')
        lr_reducer = callbacks.ReduceLROnPlateau(monitor='val_sparse_accuracy', factor=0.1, patience=7, min_lr=0.000001, mode='max')
        multi_init = training.MultiInitLog()

        for init in range(NUMBER_OF_INITS):
            if not bot is None:
                bot.sendMessage(chat_id, message_id + f'Starting init {init+1}')

            model.compile(loss=keras.losses.mean_squared_error, 
                        optimizer = deepcopy(optimizer),
                        metrics = [metrics.sparse_accuracy])
            cb = [deepcopy(early_stop), deepcopy(lr_reducer)]

            log = model.fit(x=train_set, epochs=1000, validation_data=val_set, callbacks=cb)

            multi_init.add_initialization(model, log)

            training.reinitialize_weights(model)
        
        return multi_init

class expert_commitee():
    
    @staticmethod
    def get_experts(input_shape, intermediate_neurons, classes):
        raise NotImplementedError

    @staticmethod
    def fit_committee(experts, classes, class_mapping, train_set, val_set, bot=None, chat_id=None, message_id=None):

        input_layer = keras.Input(train_set.input_shape())
        optimizer = keras.optimizers.Adam(learning_rate=0.01)
        early_stop = keras.callbacks.EarlyStopping(monitor = 'val_expert_accuracy', min_delta=0.001, patience=30, mode='max')
        lr = keras.callbacks.ReduceLROnPlateau(monitor='val_expert_accuracy', factor=0.1, patience=7, min_lr = 0.000001, mode='max')
        callbacks = [early_stop, lr]
        multi_inits = dict()
        committee_experts = list()

        for class_name in classes:

            target_label = class_mapping[class_name]
            change_func = lambda x: np.array([1 if np.all(label == target_label) else -1 for label in x])
            set_func = lambda x,y: (x, change_func(y))
            train_exp = deepcopy(train_set)
            train_exp.apply(set_func)
            val_exp = deepcopy(val_set)
            val_exp.apply(set_func)

            multi_init = training.MultiInitLog()
            expert = experts[class_name]

            for init in range(NUMBER_OF_INITS):
                message = f'expert for class {class_name}\nStarting init {init+1}'
                print(message)
                if bot:
                    bot.sendMessage(chat_id, message_id + message)

                expert.compile(loss=losses.expert_mse, 
                                optimizer=deepcopy(optimizer), 
                                metrics=[metrics.expert_accuracy])

                log = expert.fit(x=train_exp, epochs=1000, validation_data=val_exp, callbacks=deepcopy(callbacks))
                multi_init.add_initialization(expert, log)
                training.reinitialize_weights(expert)

            expert.set_weights(multi_init.best_weights(metric = 'val_expert_accuracy', mode='max', training_end = False))
            committee_experts.append(expert(input_layer))
            multi_inits[class_name] = multi_init

        concat_layer = keras.layers.concatenate(committee_experts)

        committee = keras.Model(input_layer, concat_layer)

        return committee, multi_inits

class neural_committee(standard_model):

    @staticmethod
    def get_data(model_architecture, *args, **kwargs):
        return model_architecture.get_data(*args, **kwargs)

    @staticmethod
    def get_wrapper_layers(intermediate_neurons):
        layers = [Dense(intermediate_neurons, activation='tanh'), Dense(3, activation='tanh')]
        return layers

    @staticmethod
    def get_model(multi_inits_path, wrapper_layers, input_shape):
        input_layer = keras.Input(input_shape)
        wrapper_experts = list()
        for expert_log in np.sort(os.listdir(multi_inits_path)):
            multi_init_log = training.MultiInitLog.from_json(os.path.join(multi_inits_path, expert_log))
            expert = keras.models.model_from_json(multi_init_log.model_config)
            best_weights = [np.array(x) for x in multi_init_log.best_weights('val_expert_accuracy', 'max', False)]
            expert.set_weights(best_weights)
            expert_as_layer = keras.Model(expert.input, expert.layers[-2].output, name=expert.name)(input_layer)
            wrapper_experts.append(expert_as_layer)
        x = keras.layers.concatenate(wrapper_experts)
        for layer in wrapper_layers:
            x = layer(x)
        neural_committee = keras.Model(input_layer, x, name='neural_committee')
        
        for expert in neural_committee.layers:
            if training.is_model(expert):
                expert.trainable = False

        return neural_committee

class cnn(standard_model):

    @staticmethod
    def get_model(input_shape, conv_neurons):
        model = keras.Sequential()
        model.add(Conv2D(conv_neurons, kernel_size=4, activation = 'tanh', input_shape=input_shape))
        model.add(MaxPool2D())
        model.add(Flatten())
        model.add(Dense(3, activation='tanh'))

        return model
    
    @staticmethod
    def get_data(base_path, split, to_known_value, parsed_args):
        splitpath = os.path.join(base_path, f'kfold_{parsed_args.folds}_folds', 
                        f'window_size_{parsed_args.window_size}_stride_{parsed_args.stride}', 
                        f'split_{split}.npz')
        split_set = np.load(splitpath, allow_pickle=True)

        x_train, y_train = shuffle_pair(split_set['x_train'], split_set['y_train'])
        x_val, y_val = shuffle_pair(split_set['x_val'], split_set['y_val'])
        x_test, y_test = shuffle_pair(split_set['x_test'], split_set['y_test'])

        train_set = LofarImgSequence(split_set['data'], 
                                    x_train, 
                                    to_sparse_tanh(to_known_value(y_train)))
        val_set = LofarImgSequence(split_set['data'], 
                                    x_val, 
                                    to_sparse_tanh(to_known_value(y_val)))

        x_test = np.concatenate((x_test, split_set['x_novelty']), axis=0)
        y_test = np.concatenate((y_test, split_set['y_novelty']), axis=0)
        test_set = LofarImgSequence(split_set['data'], x_test, y_test)

        return train_set, val_set, test_set

class cnn_expert(expert_commitee):

    @staticmethod
    def get_data(base_path, split, to_known_value, parsed_args):
        return cnn.get_data(base_path, split, to_known_value, parsed_args)
    @staticmethod
    def get_experts(input_shape, conv_neurons, classes):
        experts = dict()
        for class_name  in classes:
            expert_name = f'{class_name}_expert'
            input_layer = keras.Input(shape=input_shape, name=expert_name + '_input')
            x = Conv2D(conv_neurons, kernel_size=4, activation = 'tanh', input_shape=input_shape)(input_layer)
            x = MaxPool2D()(x)
            x = Flatten()(x)
            output_layer = Dense(1, activation='tanh')(x) 
            model = keras.Model(input_layer, output_layer, name=expert_name)
            experts[class_name] = model
        return experts

class cnn_mlp_expert(expert_commitee):

    @staticmethod
    def get_experts(input_shape, conv_neurons, classes):
        experts=list()
        for class_name in classes:
            expert_name = f'{class_name}_expert'
            input_layer = keras.Input(shape=input_shape, name=expert_name + '_input')
            x = Conv2D(conv_neurons, kernel_size=4, activation = 'tanh')(input_layer)
            x = MaxPool2D()(x)
            x = Flatten()(x)
            x = Dense(10, activation='tanh')(x)
            output_layer = Dense(1, activation='tanh')(x) 
            model = keras.Model(input_layer, output_layer, name=expert_name)
            experts.append(model)
        return np.array(experts)


class mlp(standard_model):
    
    @staticmethod
    def get_model(input_shape, intermediate_neurons):
        input_layer = keras.Input(input_shape)
        x = Dense(intermediate_neurons, activation='tanh', input_shape=input_shape)(input_layer)
        output_layer = Dense(3, activation='tanh')(x)
        model = keras.Model(input_layer, output_layer, name='mlp')
        return model

    @staticmethod
    def get_data(base_path, split, to_known_value, parsed_args):
        splitpath = os.path.join(base_path, f'kfold_{parsed_args.folds}_folds', 'vectors', f'split_{split}.npz')
        split_set = np.load(splitpath, allow_pickle=True)

        x_train, y_train = shuffle_pair(split_set['x_train'], split_set['y_train'])
        x_val, y_val = shuffle_pair(split_set['x_val'], split_set['y_val'])
        x_test, y_test = shuffle_pair(split_set['x_test'], split_set['y_test'])
        
        train_set = DataSequence(x_train, to_sparse_tanh(to_known_value(y_train)))
        val_set = DataSequence(x_val, to_sparse_tanh(to_known_value(y_val)))

        x_test = np.concatenate((x_test, split_set['x_novelty']), axis=0)
        y_test = np.concatenate((y_test, split_set['y_novelty']), axis=0)
        test_set = DataSequence(x_test, y_test)

        return train_set, val_set, test_set

class mlp_expert(expert_commitee):

    @staticmethod
    def get_data(base_path, split, to_known_value, parsed_args):
        return mlp.get_data(base_path, split, to_known_value, parsed_args)

    @staticmethod
    def get_experts(input_shape, intermediate_neurons, classes):
        experts = dict()
        for class_name  in classes:
            expert_name = f'{class_name}_expert'
            input_layer = keras.Input(shape=input_shape, name=expert_name + '_input')
            x = Dense(intermediate_neurons, activation='tanh', input_shape=input_shape)(input_layer)
            output_layer = Dense(1, activation='tanh')(x) 
            model = keras.Model(input_layer, output_layer, name=expert_name)
            experts[class_name] = model
        return experts