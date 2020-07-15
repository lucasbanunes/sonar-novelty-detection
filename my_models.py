import os
from copy import deepcopy
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from data_analysis.neural_networks import models, losses, metrics
import sonar_utils

class standard_model():
    is_expert_committee = False

    @staticmethod
    def compile_and_fit(model,train_set, val_set, cache, bot=None, chat_id=None):
        model.compile(loss=keras.losses.mean_squared_error, 
                    optimizer = keras.optimizers.Adam( learning_rate=0.001),
                    metrics = [metrics.sparse_accuracy])
        early_stop = callbacks.EarlyStopping(monitor = 'val_sparse_accuracy', min_delta=0.001, patience=30, restore_best_weights=True, mode='max')
        lr = callbacks.ReduceLROnPlateau(monitor='val_sparse_accuracy', factor=0.1, patience=7, min_lr=0.000001, mode='max')
        cb = [early_stop, lr]
        if bot is None:
            log = model.multi_init_fit(x=train_set, epochs=1000, n_inits=10, init_metric='val_sparse_accuracy', 
                    validation_data=val_set, save_inits=True, cache_dir=cache, callbacks=cb)
        else:
            log = model.multi_init_fit(x=train_set, epochs=1000, n_inits=10, init_metric='val_sparse_accuracy', 
                                validation_data=val_set, save_inits=True, cache_dir=cache, callbacks=cb, 
                                inits_functions=[sonar_utils.SendInitMessage(bot, chat_id)])
        
        return log

class expert_commitee():
    is_expert_committee = True
    
    @staticmethod
    def get_experts(input_shape, intermediate_neurons, classes):
        raise NotImplementedError

    @staticmethod
    def change_to_expert_data(classes, class_mapping, dataset):
        expert_sets = dict()
        for class_ in classes:
            target_label = class_mapping[class_]
            change_func = lambda x: np.array([1 if np.all(label == target_label) else -1 for label in x])
            exp_set = deepcopy(dataset)
            exp_set.apply(lambda x,y: (x, change_func(y)))
            expert_sets[class_] = exp_set
        return expert_sets

    def get_committee(self, input_shape, classes, neurons, cache_dir):
        return models.ExpertsCommittee(classes, self.get_experts(input_shape, neurons, classes), cache_dir)

    @staticmethod
    def get_wrapper(input_shape):
        model = models.MultiInitSequential()
        model.add(Dense(40, activation='tanh', input_shape=input_shape))
        model.add(Dense(3, activation='tanh'))
        return model

    @staticmethod
    def fit_wrapper(model, train_set, val_set, cache, bot=None, chat_id=None):
        model.compile(loss=keras.losses.mean_squared_error, 
            optimizer = keras.optimizers.Adam( learning_rate=0.01),
            metrics = [metrics.sparse_accuracy])
        early_stop = keras.callbacks.EarlyStopping(monitor = 'val_sparse_accuracy', min_delta=0.001, patience=30, mode='max')
        lr = callbacks.ReduceLROnPlateau(monitor='val_sparse_accuracy', factor=0.1, patience=7, min_lr = 0.000001, mode='max')
        if bot is None:
            log = model.multi_init_fit(x=train_set, epochs=1000, n_inits=10, init_metric='val_sparse_accuracy', validation_data=val_set, save_inits=True,
                        cache_dir=cache, callbacks=[early_stop, lr])
        else:
            log = model.multi_init_fit(x=train_set, epochs=1000, n_inits=10, init_metric='val_sparse_accuracy', validation_data=val_set, save_inits=True,
                            cache_dir=cache, callbacks=[early_stop, lr], 
                            inits_functions=[sonar_utils.SendInitMessage(bot, chat_id)])
        
        return log

    @staticmethod
    def fit_committee(committee, train_set, val_set, bot=None, chat_id=None):
        #Compile
        classes = list(committee.experts.keys())
        compile_mapping = dict(loss=losses.expert_mse, optimizer=keras.optimizers.Adam( learning_rate=0.01), metrics=[metrics.expert_accuracy])
        compile_params = {class_: compile_mapping for class_ in classes}
        committee.compile(compile_params)

        #Fit
        class_weights = {class_: None for class_ in classes}
        init_metrics = {class_: 'val_expert_accuracy' for class_ in classes}
        modes = {class_: 'max' for class_ in classes}
        early_stop = keras.callbacks.EarlyStopping(monitor = 'val_expert_accuracy', min_delta=0.001, patience=30, mode='max')
        lr = callbacks.ReduceLROnPlateau(monitor='val_expert_accuracy', factor=0.1, patience=7, min_lr = 0.000001, mode='max')
        if bot is None:
            model, log = committee.fit(x=train_set, epochs=1000, validation_data=val_set, class_weight=class_weights, init_metric=init_metrics, 
                            mode=modes, n_inits=10, save_inits=True, callbacks=[early_stop, lr])
        else:
            model, log = committee.fit(x=train_set, epochs=1000, validation_data=val_set, class_weight=class_weights, init_metric=init_metrics, 
                                        mode=modes, n_inits=10, save_inits=True, callbacks=[early_stop, lr], 
                                        inits_functions=[sonar_utils.SendInitMessage(bot, chat_id)])
        
        return model, log

class cnn(standard_model):
    """class with the parameters I'm using as cnn"""
    is_cnn = True
    is_mlp = False

    @staticmethod
    def get_model(input_shape, conv_neurons):
        model = models.MultiInitSequential()
        model.add(keras.Input(input_shape))
        model.add(Conv2D(conv_neurons, kernel_size=4, activation = 'tanh'))
        model.add(MaxPool2D())
        model.add(Flatten())
        model.add(Dense(3, activation='tanh'))
        return model

class cnn_expert(expert_commitee):

    is_cnn = True
    is_mlp = False

    @staticmethod
    def get_experts(input_shape, conv_neurons, classes):
        experts =list()
        for class_name  in classes:
            expert_name = f'{class_name}_expert'
            input_layer = keras.Input(shape=input_shape, name=expert_name + '_input')
            x = Conv2D(conv_neurons, kernel_size=4, activation = 'tanh', input_shape=input_shape)(input_layer)
            x = MaxPool2D()(x)
            x = Flatten()(x)
            output_layer = Dense(1, activation='tanh')(x) 
            model = keras.Model(input_layer, output_layer, name=expert_name)
            experts.append(model)
        return np.array(experts)

class cnn_mlp_expert(expert_commitee):

    is_cnn = True
    is_mlp = False

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
    """class with the parameters I'm using as mlp"""
    is_cnn = False
    is_mlp = True
    
    @staticmethod
    def get_model(input_shape, intermediate_neurons):
        model = models.MultiInitSequential()
        model.add(Dense(intermediate_neurons, activation='tanh', input_shape=input_shape))
        model.add(Dense(3, activation='tanh'))
        return model

class mlp_expert(expert_commitee):

    is_cnn = False
    is_mlp = True

    @staticmethod
    def get_experts(input_shape, intermediate_neurons, classes):
        experts =list()
        for class_name  in classes:
            expert_name = f'{class_name}_expert'
            input_layer = keras.Input(shape=input_shape, name=expert_name + '_input')
            x = Dense(intermediate_neurons, activation='tanh', input_shape=input_shape)(input_layer)
            output_layer = Dense(1, activation='tanh')(x) 
            model = keras.Model(input_layer, output_layer, name=expert_name)
            experts.append(model)
        return np.array(experts)