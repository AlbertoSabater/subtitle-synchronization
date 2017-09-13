#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import pickle
import os
import pysrt
import audio_converter
import time
from sklearn import metrics
import collections

from keras.layers import Dense, Input, LSTM, Conv1D, Conv2D, Dropout, Flatten, Activation, MaxPooling2D
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam, RMSprop

import matplotlib.pylab as plt


DATA_DIR = 'data/'
STORE_DIR = 'datasets/'


        
# CNN architecture
# conv_layers -> [(filters, kernel_size, BatchNormaliztion, Dropout, MaxPooling)]
# dense_layers -> [(num_neurons, BatchNormaliztion, Dropout)]
def model_cnn(net_layers, input_shape):
    
    inp = Input(shape=input_shape)
    model = inp
  
    for cl in net_layers['conv_layers']:
        model = Conv2D(filters=cl[0], kernel_size=cl[1], activation='relu')(model)
        if cl[4]:
            model = MaxPooling2D()(model)
        if cl[2]:
            model = BatchNormalization()(model)
        if cl[3]:
            model = Dropout(0.2)(model)
    
    model = Flatten()(model)
    
    for dl in net_layers['dense_layers']:
        model = Dense(dl[0])(model)
        model = Activation('relu')(model)
        if dl[1]:
            model = BatchNormalization()(model)
        if dl[2]:
            model = Dropout(0.2)(model)
    
    model = Dense(1)(model)
    model = Activation('sigmoid')(model)
    
    model = Model(inp, model)
    return model

        
            
# %%
    
# LSTM architecture
# conv_layers -> [(filters, kernel_size, BatchNormaliztion, Dropout, MaxPooling)]
# dense_layers -> [(num_neurons, BatchNormaliztion, Dropout)]
def model_lstm(input_shape):
    
    inp = Input(shape=input_shape)
    model = inp
    
    if input_shape[0] > 2: model = Conv1D(filters=24, kernel_size=(3), activation='relu')(model)
#    if input_shape[0] > 0: model = TimeDistributed(Conv1D(filters=24, kernel_size=3, activation='relu'))(model)
    model = LSTM(16)(model)
    model = Activation('relu')(model)
    model = Dropout(0.2)(model)
    model = Dense(16)(model)
    model = Activation('relu')(model)
    model = BatchNormalization()(model)

    model = Dense(1)(model)
    model = Activation('sigmoid')(model)
    
    model = Model(inp, model)
    return model

# %% 
    
# Conv-1D architecture. Just one sample as input
def model_dense(input_shape):
    
    inp = Input(shape=input_shape)
    model = inp
    
    model = Conv1D(filters=12, kernel_size=(3), activation='relu')(model)
    model = Conv1D(filters=12, kernel_size=(3), activation='relu')(model)
    model = Flatten()(model)

    model = Dense(56)(model)
    model = Activation('relu')(model)
    model = BatchNormalization()(model)
    model = Dropout(0.2)(model)
    model = Dense(28)(model)
    model = Activation('relu')(model)
    model = BatchNormalization()(model)

    model = Dense(1)(model)
    model = Activation('sigmoid')(model)
    
    model = Model(inp, model)
    return model
# %%
     

     
# Better accuracy by removing mean, removing first sample and rotating 90 degrees
# Adam:
#   raw:                    0.840602374754
#   without mean:           0.841123660592
#   without first sample:   0.844946423462
#   rotated:                0.85873153781

# RMSprop:
#   raw:                    0.845004344059
#   rotated:                0.832493483944
#   without mean:           0.854039965255
#   without first sample:   0.851086012177

# CNN + LSTM:
#   raw:                    0.8609904431
#   rotated:                0.860237474729
#   without mean:           0.875354763984  <--
#   without first sample:   0.873211699978
#   without mean, no rotated, no first: 864176078841
#   without mean, no rotated, all: 858094410668

# CNN + CNN + LSTM:
#   raw:                    0.859542426882
#   rotated:                0.86950477846
#   without mean:           0.873211699971
#   without first sample:   0.868751810082
    
# Train and store LSTM NN with different parameters
def train_lstm():
    
# %%
    t = time.time()
    
#    freq = 16000.0
#    hop_len = 128.0
#    len_sample = 0.25    # Length in seconds for the input samples
#    step_sample = 0.05    # Space between the beginingof each sample
    step_sample = 0.05    # Space between the beginingof each sample
    
    train_files = ['v1', 'v2', 'v3', 'v4']
    
#    for len_sample in [0.5, 0.25, 0.125, 0.075]:
    for len_sample in [0.075]:
#        for f in [1000, 2000, 4000, 8000, 16000]:
        for f in [4000, 8000, 16000]:
            for hop_len in [128.0, 256.0, 512.0, 1024.0, 2048.0]:
                
                print 'FREQ:',  f, hop_len, len_sample
                
                t = time.time()
                
                len_mfcc = audio_converter.get_len_mfcc(len_sample, hop_len, f)     #  Num of samples to get LEN_SAMPLE
                step_mfcc = audio_converter.get_step_mfcc(step_sample, hop_len, f)     #  Num of samples to get STEP_SAMPLE
                
                X, Y = audio_converter.generateDatasets(train_files, True, len_mfcc, step_mfcc, hop_len=hop_len, freq=f)
                
                rand = np.random.permutation(np.arange(len(Y)))
                X = X[rand]
                Y = Y[rand]
                
                X = np.array([ np.rot90(val) for val in X ])
                X = X - np.mean(X, axis=0)
            #    X = X[:,1:,:]
            
                print X.shape, len(Y[Y==0]), len(Y[Y==1]), float(len(Y[Y==0]))/len(Y[Y==1])
                
                if X.shape[1] == 0:
                    print "NEXT\n"
                    continue
                
            
                input_shape = (X.shape[1], X.shape[2])   
                model = model_lstm(input_shape)
            
                earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, verbose=0, mode='min', patience=5)
                filename = 'models/v2/model_cnn_lstm_' + str(f) + '_' + str(len_mfcc) + '_' + str(step_mfcc) + '_' + str(hop_len) + '.hdf5'
                checkpoint = ModelCheckpoint(filepath=filename, 
                             monitor='val_loss', verbose=0, save_best_only=True)
                callbacks_list = [earlyStopping, checkpoint]
                model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
        #        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.01), metrics=['accuracy'])
            #    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])
                hist = model.fit(X, Y, epochs=2000, batch_size=32, shuffle=True, validation_split=0.25, verbose=0, callbacks=callbacks_list)
                
                print 'val_loss:', min(hist.history['val_loss'])
                print 'val_acc:', max(hist.history['val_acc'])
                
                print "Total training time:", (time.time()-t)/60
                print "-----------------------------"
                print "-----------------------------"
                print "-----------------------------"
                print "-----------------------------\n\n\n"
          
# %%
                
# Train and store dense NN with different parameters
def train_dense():
    
# %%
    t = time.time()
    
    v = 'v4'
    
    train_files = ['v1', 'v2', 'v3', 'v4']
    
    for f in [1000, 2000, 4000, 8000, 16000]:
        for hop_len in [256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0]:
#        for hop_len in [128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0]:
            
            window_time = hop_len/f
            if window_time < 0.05 or window_time > 0.4:
                print 'Skip:', f, hop_len, window_time
                continue
            continue
            
            print 'FREQ:',  f, hop_len,
            
            t = time.time()
            
            
            X, Y = audio_converter.generateDatasets(train_files, True, 1, 1, hop_len=hop_len, freq=f)
            
#            X = X[:,:,0]
            rand = np.random.permutation(np.arange(len(Y)))
            X = X[rand]
            Y = Y[rand]
            
            X = X - np.mean(X, axis=0)
        #    X = X[:,1:,:]
        
            print X.shape, float(len(Y[Y==0]))/len(Y), float(len(Y[Y==1]))/len(Y), float(len(Y[Y==0]))/len(Y[Y==1])
            
            if X.shape[1] == 0:
                print "NEXT\n"
                continue
            
        
            input_shape = (X.shape[1], 1)
            model = model_dense(input_shape)
        
            earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, verbose=0, mode='min', patience=5)
            filename = 'models/'+v+'/model_cnn_lstm_' + str(f) + '_' + str(1) + '_' + str(1) + '_' + str(hop_len) + '.hdf5'
            checkpoint = ModelCheckpoint(filepath=filename, 
                         monitor='val_loss', verbose=0, save_best_only=True)
            callbacks_list = [earlyStopping, checkpoint]
            model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
            hist = model.fit(X, Y, epochs=2000, batch_size=32, shuffle=True, validation_split=0.3, verbose=0, callbacks=callbacks_list)
            
            print 'val_loss:', min(hist.history['val_loss'])
            print 'val_acc:', max(hist.history['val_acc'])
            
            print "Total training time:", (time.time()-t)/60
            print "-----------------------------"
            print "-----------------------------"
            print "-----------------------------"
            print "-----------------------------\n\n\n"
          
            
# %%
    
# Subtitle Synchronization
# Calculate and plot Log Loss minimization in subtitle synchronization with 
# different hyperparameters and video files
def test():
# %%

    v = 'v3'
    nn = model_dense
    divide_and_conquer = True
    model_dir = 'models/' + v + '/'
    model_name = 'model_cnn_lstm_'
    files = os.listdir(model_dir)
    files = [ f for f in files if f.startswith(model_name)]
    print files

#    test_files = ['v1', 'v2', 'v3', 'v4']
    test_files = ['v6']

    results = []
    
    for f in files:
        
        params = f.replace(model_name, '').split('.hdf5')[0].split('_')
        params = [ float(p) for p in params ]
        
        freq, len_mfcc, step_mfcc, hop_len = params
      
        if freq != 16000: continue
        if hop_len != 512: continue
        if freq >= 24000.0 or [freq,hop_len] in [ [q['freq'], q['hop_len']] for q in results ]:
            print "Skip:", freq, hop_len
            continue

        
        d = {}
        d['freq'] = freq
        d['len_mfcc'] = len_mfcc
        d['step_mfcc'] = step_mfcc
        d['hop_len'] = hop_len
        d['window_time'] = hop_len/freq
        
        
        for tf in test_files:
        
            print '\n\n................................................'
            print '................................................'
            print f
            print '................................................'
            print '................................................\n'
        
            print params, hop_len/freq

          
            t = time.time()
            
            t_aux = time.time()
            X, Y = audio_converter.generateSingleDataset(tf, cut_data=False, 
                                                         len_mfcc=len_mfcc, step_mfcc=step_mfcc, hop_len=hop_len, freq=freq)
#            X = np.array([ np.rot90(val) for val in X ])
            X = X - np.mean(X, axis=0)
            d[tf + '_time_load_dataset'] = (time.time()-t_aux)/60
            d[tf + '_Xshape'] = X.shape
            
            # Lodad neural network
            input_shape = (X.shape[1], 1)   
            model = nn(input_shape)
            model.load_weights(model_dir + f)
        
            t_aux = time.time()
            preds = model.predict(X)
            d[tf + '_time_predictions'] = (time.time()-t_aux)/60
           
            subs = pysrt.open(DATA_DIR+tf+'.srt', encoding='iso-8859-1')
            start_time = subs[0].start            
            start = audio_converter.timeToSec(subs[0].start)
            subs.shift(seconds=-start)
            
            t_aux = time.time()
            # Create mask
            mask = np.zeros(audio_converter.timeToPos(subs[len(subs)-1].end, step_mfcc, freq, hop_len)+1)
            
            print "Synchronizing"
            for sub in subs:
                for i in np.arange(audio_converter.timeToPos(sub.start, step_mfcc, freq, hop_len), audio_converter.timeToPos(sub.end, step_mfcc, freq, hop_len)+1):
                    if i<len(mask):
                        mask[i] = 1        
                    
            if not divide_and_conquer:
                mtrs = []
                t_aux = time.time()
                for i in np.arange(0, (len(preds)-len(mask))):
                    if i % 1000 == 0: 
                        print i, (len(preds)-len(mask)), (time.time()-t_aux)/60
                        t_aux = time.time()
                    mtrs.append(metrics.log_loss(mask, preds[i:i+len(mask)]))
                    
                pos_to_delay = mtrs.index(min(mtrs))
                
            else:
                mtrs_aux = []
                step_len = 50
                second_step_len = 500
                t_aux = time.time()
                for i in np.arange(0, (len(preds)-len(mask)), step_len):
                    if i % 1000 == 0: 
#                        print i, (len(preds)-len(mask)), (time.time()-t_aux)/60
                        t_aux = time.time()
                    mtrs_aux.append(metrics.log_loss(mask, preds[i:i+len(mask)]))
                    
                min_index = mtrs_aux.index(min(mtrs_aux))*step_len
                
                plt.figure(figsize=(10,6))
                plt.plot(mtrs_aux) 
                plt.xlabel('Steps to delay', fontsize=18)
                plt.ylabel('Los Loss value', fontsize=18)
                plt.show()
                
                print 'Best_mtr_index:', min_index
                
                mtrs = []
                for i in np.arange(min_index-second_step_len, min_index+second_step_len):
                    if i<0 or i>=(len(preds)-len(mask)): continue
                    mtrs.append(metrics.log_loss(mask, preds[i:i+len(mask)]))
                            
                pos_to_delay = min_index+second_step_len-mtrs.index(min(mtrs))
            
                
                
                
            print "Synchronized"
            
            secsToDelay = audio_converter.posToTime(pos_to_delay, step_mfcc, freq, hop_len)
            subs.shift(seconds=secsToDelay)
            d[tf + '_time_sync'] = (time.time()-t_aux)/60
            
            print "\nMin loss:", min(mtrs)
            print audio_converter.timeToPos(start_time, step_mfcc, freq, hop_len), audio_converter.secToPos(start, step_mfcc, freq, hop_len), pos_to_delay
            print start, secsToDelay

            plt.figure(figsize=(10,6))
            plt.plot(mtrs) 
            plt.xlabel('Steps to delay', fontsize=18)
            plt.ylabel('Los Loss value', fontsize=18)
            plt.show()
           
            
            d[tf+'_loss'] = min(mtrs)
            d[tf+'_shift'] = [start, secsToDelay]
            total_elapsed_time = (time.time()-t)
            d[tf + '_time'] = (time.time()-t)/60
            print 'Load: {0:.2f}, Preds: {0:.2f}, Sync: {0:.2f}'.format(d[tf + '_time_load_dataset'], d[tf + '_time_predictions'], d[tf + '_time_sync'])
            print ' - Time elapsed: {0:02d}:{1:02d}'.format(int(total_elapsed_time/60), int(total_elapsed_time % 60))
        
        results.append(d)


# %%
def sotreResults(results, v):
# %%
    import pickle
    
    with open('test_results_'+v+'.pickle', 'w') as f:
        pickle.dump(results, f)


# %% 

# Plot stored training statistics. Look for the best model
def evalResults(v):

# %% 
    with open('test_results_'+v+'.pickle', 'r') as f:
        results = pickle.load(f)

    vals = [ d['v3_loss'] for d in results ]
    fig, ax = plt.subplots(figsize=(35,5))
    plt.plot(vals)
    ax.set_xticks(np.arange(0, len(vals)))
    ax.set_xticklabels([ str(d['freq'])+'_'+str(d['hop_len'])+'_'+str(d['len_mfcc']) for d in results ], rotation=20)
    plt.show()
    
    
    # Metrics to plot
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    
    width = 0.15
        
    freqs = sorted(list(set([ d['freq'] for d in results ])))
    hop_lens = sorted(list(set([ d['hop_len'] for d in results ])))
    pos = list(np.arange(len(hop_lens)))
    m = 'v1_loss'
        
    # Plot metrics
    fig, ax = plt.subplots()
    for i in np.arange(len(freqs)):
        vals = { d['hop_len']: d[m] for d in results if d['freq']==freqs[i] }
        vals.update({ hl:0.0 for hl in hop_lens if hl not in vals.keys() })
        vals = collections.OrderedDict(sorted(vals.items()))

        plt.bar([p + width*i for p in pos],
                [ v for k,v in vals.items() ],
#                [ d.values()[0][freqs[i]] for d in mtrs ],
                width,
                alpha=0.5,
                color=colors[i])

    ax.set_ylabel('Value')
    ax.set_xticks([p + 1.5 * width for p in pos])
    ax.set_xticklabels(hop_lens, rotation=20)
    plt.xlim(min(pos)-width, max(pos)+width*4)
#    plt.legend(freqs, loc=1)
    ax.legend(freqs, loc='center left', bbox_to_anchor=(1, 0.75))

    plt.show()

    
    res = [ d for d in results if d['window_time']<=0.3 ]
    
    # Metrics to plot
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    
    width = 0.12
        
    freqs = sorted(list(set([ d['freq'] for d in res ])))
    hop_lens = sorted(list(set([ d['hop_len'] for d in res ])))
    pos = list(np.arange(len(freqs)))
    vers = 'v'+str(1)
        
    # Plot metrics
    fig, ax = plt.subplots(figsize=(8,4))
    for i in np.arange(len(hop_lens)):
#        vals = { d['freq']: d[vers+'_loss'] for d in res if d['hop_len']==hop_lens[i] }
        vals = { d['freq']: d[vers+'_time'] for d in res if d['hop_len']==hop_lens[i] }
#        vals = { d['freq']: math.fabs((d[vers+'_shift'][0]-d[vers+'_shift'][1])) for d in res if d['hop_len']==hop_lens[i] and d[vers+'_time']<=max_time }
        print hop_lens[i], { hl:0.0 for hl in freqs if hl not in vals.keys() }
        vals.update({ hl:0.003 for hl in freqs if hl not in vals.keys() })
        vals = collections.OrderedDict(sorted(vals.items()))
        print hop_lens[i], vals

        plt.bar([p + width*i for p in pos],
                [ v for k,v in vals.items() ],
#                [ d.values()[0][freqs[i]] for d in mtrs ],
                width,
                alpha=0.5,
                color=colors[i])

    ax.set_ylabel('Value')
    ax.set_xticks([p + 3 * width for p in pos])
    ax.set_xticklabels(freqs, rotation=20)
    plt.xlim(min(pos)-width, max(pos)+width*len(hop_lens) + width)
    ax.legend(hop_lens, loc='center left', bbox_to_anchor=(1, 0.75))

    plt.show()


# %%  

# Subtitle Synchronization
# Calculate and plot Log Loss minimization in subtitle synchronization with 
# the specified hypermarameters and video_file
def predictTest(video_file):
    
    t = time.time()
    
#    video_file = 'v3'
    # Load test dataset
    freq = 16000.0
    hop_len = 512.0
    len_sample = 0.5    # Length in seconds for the input samples
    step_sample = 0.05    # Space between the beginingof each sample
    len_mfcc = audio_converter.get_len_mfcc(len_sample, hop_len, freq)     #  Num of samples to get LEN_SAMPLE
    step_mfcc = audio_converter.get_step_mfcc(step_sample, hop_len, freq)     #  Num of samples to get STEP_SAMPLE
    
  
    t_aux = time.time()
    X, Y = audio_converter.generateSingleDataset(video_file, cut_data=False, len_mfcc=len_mfcc, step_mfcc=step_mfcc, hop_len=hop_len)
    print "* Datased calculated: {0:02d}:{1:02d}".format(int((time.time()-t_aux)/60), int((time.time()-t_aux) % 60))
    
    
    t_aux = time.time()
    X = np.array([ np.rot90(val) for val in X ])
    X = X - np.mean(X, axis=0)
    
    # Lodad neural network
    input_shape = (X.shape[1], X.shape[2])   
    model = model_lstm(input_shape)
    model.load_weights('models/' + 'model_cnn_15_13.hdf5')
    
    preds = model.predict(X)
    print "* Output predicted: {0:02d}:{1:02d}".format(int((time.time()-t_aux)/60), int((time.time()-t_aux) % 60))

    
    t_aux = time.time()
    # Load test subtitles
    subs = pysrt.open(DATA_DIR+video_file+'.srt', encoding='iso-8859-1')

    start = audio_converter.timeToSec(subs[0].start)
    start = start if start>0 else 0
    subs.shift(seconds=-start)
    
    # Create mask
    mask = np.zeros(audio_converter.timeToPos(subs[len(subs)-1].end, step_mfcc, freq, hop_len)+1)
    for sub in subs:
        for i in np.arange(audio_converter.timeToPos(sub.start, step_mfcc, freq, hop_len), audio_converter.timeToPos(sub.end, step_mfcc, freq, hop_len)+1):
            mask[i] = 1        
            
            
    mtrs = []
    for i in np.arange(0, (len(preds)-len(mask))):
        mtrs.append(metrics.log_loss(mask, preds[i:i+len(mask)]))
        
    
    secsToDelay = audio_converter.posToTime(mtrs.index(min(mtrs)), step_mfcc, freq, hop_len)
    subs.shift(seconds=secsToDelay)
    print "* Subtitles synchronized: {0:02d}:{1:02d}".format(int((time.time()-t_aux)/60), int((time.time()-t_aux) % 60))
    

    print "Min loss:", min(mtrs), mtrs.index(min(mtrs))
    plt.plot(mtrs) 
    print start, secsToDelay
    total_elapsed_time = time.time()-t
    print "** Total time elapsed: {0:02d}:{1:02d}".format(int(total_elapsed_time/60), int(total_elapsed_time % 60))

        

