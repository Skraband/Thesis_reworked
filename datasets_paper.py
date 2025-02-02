# -*- coding: utf-8 -*-
"""
Created on Sun May  5 21:06:56 2024

@author: Fabian
"""
import json

mydict = {'M4_Yearly':
          {'numEpochs':10000,
           'batchsize':256,
           'fftWinSize':6,
           'fftComp':1,
           'pwnLR':0.004,
           'contexttime':24,
           'predictiontime':6,
           'timestep':3,
           'archLR':0.001,
           'archepochs':50,
           'EarlyStop':90},
          'M4_Quaterly':
                    {'numEpochs':10000,
                     'batchsize':256,
                     'fftWinSize':8,
                     'fftComp':1,
                     'pwnLR':0.004,
                     'contexttime':32,
                     'predictiontime':8,
                     'timestep':4,
                     'archLR':0.001,
                     'archepochs':50,
                     'EarlyStop':90},
          'M4_Monthly':
                   {'numEpochs':10000,
                    'batchsize':256,
                    'fftWinSize':18,
                    'fftComp':1,
                    'pwnLR':0.004,
                    'contexttime':108,
                    'predictiontime':18,
                    'timestep':9,
                    'archLR':0.001,
                    'archepochs':50,
                    'EarlyStop':90},
          'M4_Weekly':
                   {'numEpochs':10000,
                    'batchsize':256,
                    'fftWinSize':14,
                    'fftComp':2,
                    'pwnLR':0.004,
                    'contexttime':63,
                    'predictiontime':14,
                    'timestep':7,
                    'archLR':0.001,
                    'archepochs':50,
                    'EarlyStop':90},
           'M4_Daily':
                    {'numEpochs':10000,
                     'batchsize':256,
                     'fftWinSize':14,
                     'fftComp':2,
                     'pwnLR':0.004,
                     'contexttime':70,
                     'predictiontime':14,
                     'timestep':7,
                     'archLR':0.001,
                     'archepochs':50,
                     'EarlyStop':90},
           'M4_Hourly':
                    {'numEpochs':10000,
                     'batchsize':256,
                     'fftWinSize':24,
                     'fftComp':2,
                     'pwnLR':0.004,
                     'contexttime':480,
                     'predictiontime':48,
                     'timestep':12,
                     'archLR':0.001,
                     'archepochs':50,
                     'EarlyStop':90},
           'Power':
                     {'numEpochs':5000,
                      'batchsize':256,
                      'fftWinSize':96,
                      'fftComp':4,
                      'pwnLR':0.004,
                      'contexttime':1440,
                      'predictiontime':144,
                      'timestep':96,
                      'archLR':0.001,
                      'archepochs':50,
                      'EarlyStop':90},
           'Exchange':
                     {'numEpochs':1000,
                      'batchsize':32,
                      'fftWinSize':60,
                      'fftComp':2,
                      'pwnLR':0.003,
                      'contexttime':180,
                      'predictiontime':30,
                      'timestep':15,
                      'archLR':0.001,
                      'archepochs':50,
                      'EarlyStop':90},
            'Wiki':
                      {'numEpochs':1000,
                       'batchsize':512,
                       'fftWinSize':60,
                       'fftComp':2,
                       'pwnLR':0.004,
                       'contexttime':180,
                       'predictiontime':30,
                       'timestep':30,
                       'archLR':0.001,
                       'archepochs':50,
                       'EarlyStop':90},
            'Solar':
                      {'numEpochs':200,
                       'batchsize':64,
                       'fftWinSize':24,
                       'fftComp':2,
                       'pwnLR':0.003,
                       'contexttime':720,
                       'predictiontime':24,
                       'timestep':240,
                       'archLR':0.001,
                       'archepochs':50,
                       'EarlyStop':90}
         }

with open('params.json', 'w') as fp:
    json.dump(mydict, fp)

f = open('params.json')

data = json.load(f)
