# coding=utf-8

from __future__ import print_function
import numpy as np
import os
import pickle as cp
import sys

def hall_of_fame (experiment, size, desc, model, score): # Escribe en un fichero los 50 mejores modelos, con su fitness y su desc (?)
    try:
        if not os.path.isfile('{}/halloffame.pkl'.format(experiment)):
            cp.dump([], open('{}/halloffame.pkl'.format(experiment), 'w+'), protocol=cp.HIGHEST_PROTOCOL)

        f = open('{}/halloffame.pkl'.format(experiment))
        hall = cp.load(f)
        f.close()

        changed = False
        if len(hall) < size:
            hall += [(score, desc, model)]
            changed = True
        elif hall[-1][0] < score:
            hall[-1] = (score, desc, model)
            changed = True

        if changed:
            f = open('{}/halloffame.pkl'.format(experiment), 'wb')
            hall = sorted(hall, key=lambda i : i[0], reverse=True)
            cp.dump(hall, f, protocol=cp.HIGHEST_PROTOCOL)
            f.close()
    except:
        pass

def cifar_tf (**kwargs):
    #print("comenzando la construccion de red")
    import numpy as np
    import pickle as cp
    import tensorflow as tf
    import keras
    from keras.utils import to_categorical
    import sklearn
    from sklearn.metrics import accuracy_score
    from keras.models import Sequential
    from keras.layers import Dense, Conv2D, Flatten, SimpleRNN, CuDNNLSTM, CuDNNGRU, Reshape
    from keras import optimizers
    
    # Parámetros de la red
    BATCH_SIZE = kwargs["batch"]
    EPOCHS = kwargs['epochs']
    T_SAMPLE = kwargs['sample']
    
    # Carga del mnist y adaptación de los datos para la red
    cifar = tf.keras.datasets.cifar10
    (X_train, y_train), (X_test, y_test) = cifar.load_data()
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    indices = np.random.choice(range(len(X_train)), size = int(len(X_train) * T_SAMPLE), replace = False)
    sample_images = X_train[indices]
    sample_labels = y_train[indices]
    
    # Inicio de la red
    try:
        ### PARTE CONVOLUCIONAL ###
        
        # La primera capa es segura así que va fuera del bucle
        keras.backend.clear_session()
        model = Sequential()
        model.add(Conv2D(kwargs["nk1"], kwargs["nr1"], kernel_initializer=keras.initializers.glorot_normal(), input_shape=X_train.shape[1:], padding='same'))
        model.add(keras.layers.Activation(kwargs["ca1"]))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D(pool_size=kwargs["pr1"], padding='same'))
        
        for nc in range(1, kwargs["nconv"]):
            nk = kwargs["nk{}".format(nc+1)] # Número de kernels
            nr = kwargs["nr{}".format(nc+1)] # Tamaño del kernel
            model.add(Conv2D(nk, nr, kernel_initializer=keras.initializers.glorot_normal(), padding='same'))
            model.add(keras.layers.Activation(kwargs["ca{}".format(nc+1)]))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.MaxPooling2D(pool_size=kwargs["pr{}".format(nc+1)], padding='same'))
        
        model.add(Flatten())
        
        ### PARTE DENSA Y RECURSIVA ###
        for nd in range(kwargs["ndense"]):
            nn = kwargs["nn{}".format(nd+1)] # Número de neuronas en esta capa
            # Según el tipo de capa
            if kwargs['dt{}'.format(nd+1)] == 'rnn':
                #Se adapta el input al output de la capa anterior en las redes recurrentes
                model.add(Reshape((1, model.layers[-1].output_shape[1])))
                if kwargs["re{}".format(nd+1)] == "l1":
                    model.add(SimpleRNN(nn, kernel_initializer=keras.initializers.glorot_normal(), kernel_regularizer=keras.regularizers.l1(0.1)))
                elif kwargs["re{}".format(nd+1)] == "l2":
                    model.add(SimpleRNN(nn, kernel_initializer=keras.initializers.glorot_normal(), kernel_regularizer=keras.regularizers.l2(0.1)))
                elif kwargs["re{}".format(nd+1)] == "l1l2":
                    model.add(SimpleRNN(nn, kernel_initializer=keras.initializers.glorot_normal(), kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)))
                elif kwargs["re{}".format(nd+1)] == "None":
                    model.add(SimpleRNN(nn, kernel_initializer=keras.initializers.glorot_normal()))
            
            elif kwargs['dt{}'.format(nd+1)] == 'lstm':
                model.add(Reshape((1, model.layers[-1].output_shape[1])))
                if kwargs["re{}".format(nd+1)] == "l1":
                    model.add(CuDNNLSTM(nn, kernel_initializer=keras.initializers.glorot_normal(), kernel_regularizer=keras.regularizers.l1(0.1)))
                elif kwargs["re{}".format(nd+1)] == "l2":
                    model.add(CuDNNLSTM(nn, kernel_initializer=keras.initializers.glorot_normal(), kernel_regularizer=keras.regularizers.l2(0.1)))
                elif kwargs["re{}".format(nd+1)] == "l1l2":
                    model.add(CuDNNLSTM(nn, kernel_initializer=keras.initializers.glorot_normal(), kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)))
                elif kwargs["re{}".format(nd+1)] == "None":
                    model.add(CuDNNLSTM(nn, kernel_initializer=keras.initializers.glorot_normal()))
            
            elif kwargs['dt{}'.format(nd+1)] == 'gru':
                model.add(Reshape((1, model.layers[-1].output_shape[1])))
                if kwargs["re{}".format(nd+1)] == "l1":
                    model.add(CuDNNGRU(nn, kernel_initializer=keras.initializers.glorot_normal(), kernel_regularizer=keras.regularizers.l1(0.1)))
                elif kwargs["re{}".format(nd+1)] == "l2":
                    model.add(CuDNNGRU(nn, kernel_initializer=keras.initializers.glorot_normal(), kernel_regularizer=keras.regularizers.l2(0.1)))
                elif kwargs["re{}".format(nd+1)] == "l1l2":
                    model.add(CuDNNGRU(nn, kernel_initializer=keras.initializers.glorot_normal(), kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)))
                elif kwargs["re{}".format(nd+1)] == "None":
                    model.add(CuDNNGRU(nn, kernel_initializer=keras.initializers.glorot_normal()))

            elif kwargs['dt{}'.format(nd+1)] == 'dense':
                if kwargs["re{}".format(nd+1)] == "l1":
                    model.add(Dense(nn, kernel_initializer=keras.initializers.glorot_normal(), kernel_regularizer=keras.regularizers.l1(0.1)))
                elif kwargs["re{}".format(nd+1)] == "l2":
                    model.add(Dense(nn, kernel_initializer=keras.initializers.glorot_normal(), kernel_regularizer=keras.regularizers.l2(0.1)))
                elif kwargs["re{}".format(nd+1)] == "l1l2":
                    model.add(Dense(nn, kernel_initializer=keras.initializers.glorot_normal(), kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)))
                elif kwargs["re{}".format(nd+1)] == "None":
                    model.add(Dense(nn, kernel_initializer=keras.initializers.glorot_normal()))

            model.add(keras.layers.Activation(kwargs["da{}".format(nd+1)]))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Dropout(kwargs["dr{}".format(nd+1)]))
        
        # Capa de salida
        model.add(Dense(len(y_train[0])))
        model.add(keras.layers.Activation('softmax'))
        
        # Ratios de aprendizaje
        LR = kwargs['lr']
        if kwargs['opt'] == 'sgd':
            opt = keras.optimizers.SGD(lr = LR)
        # No aparece momentum en Keras @TODO: Cambiar la codificación para que no aparezca 
        elif kwargs['opt'] == 'momentum':
            opt = keras.optimizers.SGD(lr = LR)
        elif kwargs['opt'] == 'nesterov':
            opt = keras.optimizers.Nadam(lr = LR) # Sustituimos Nesterov por Nadam
        elif kwargs['opt'] == 'adagrad':
            opt = keras.optimizers.Adagrad(lr = LR)
        elif kwargs['opt'] == 'rmsprop':
            opt = keras.optimizers.RMSprop(lr = LR)
        elif kwargs['opt'] == 'adadelta':
            opt = keras.optimizers.Adadelta(lr = LR)
        elif kwargs['opt'] == 'adam':
            opt = keras.optimizers.Adam(lr = LR)
        elif kwargs['opt'] == 'adamax':
            opt = keras.optimizers.Adamax(lr = LR)
        
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        historial = model.fit(x = X_train, y = y_train, batch_size = BATCH_SIZE, epochs = EPOCHS, validation_data = (X_test, y_test), verbose = 0)
        #queda incluir toda la parte del hall-of-fame y del mejor modelo
        return max(historial.history["val_acc"])
    
    except Exception as e:
        print(e, file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 0
    
def clean_args (args): # Se limpian los argumentos y se meten en un diccionario
    #print("comenzando la limpieza de argumentos")
    ret_args = dict()
    for a in args:
        #print(args)
        key, val = a.split('=')
        key = key.strip(' -')
        import re
        if re.match("^-?\d+$", val):
            val = int(val)
        elif re.match("^-?\d+?(\.\d+?)?([Ee]-?\d+)?$", val):
            val = float(val)
        ret_args[key] = val
    return ret_args

args = clean_args(sys.argv[1:])
os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(args["device"])
print(cifar_tf(**args))