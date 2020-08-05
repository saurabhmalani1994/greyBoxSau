'''A collection of functions for working with and visualizing Keras models.'''

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.utils import plot_model as plot
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

#import tqdm
from tqdm import tqdm as tqdm
getattr(tqdm, '_instances', {}).clear()  # ⬅ add this line
import tensorflow.keras.callbacks


from IPython.display import Image


def displayModel(model, outpath='/tmp/model.png'):
    '''Show the layer structure of a Keras model as a graph.'''
    plot(model, to_file=outpath, show_shapes=True)
    return Image(filename=outpath)


def feedForward(inputShape, nhidden, hiddenSize, outputSize,
                hiddenType='relu', outputType='linear', **layerkwargs):
    '''Construct a multi-layer perceptron.'''
    if isinstance(hiddenType, str):
        hiddenType = [hiddenType]
    layers = []
    if isinstance(hiddenSize, list):
        hiddenSizes = hiddenSize
    else:
        hiddenSizes = [hiddenSize]
    for i in range(nhidden):
        for ht in hiddenType:
            layers.append(
                Dense(hiddenSizes[i], activation=ht, name='hidden_%s_%d' % (ht, i), **layerkwargs)
            )

    # Add an output layer with the N outputs.
    layers.append(Dense(outputSize, activation=outputType, name='output', **layerkwargs))

    # Apply the stack to an input tensor to get an output tensor.
    inputTensor = Input(shape=inputShape, name='input')
    outputTensor = inputTensor
    for layer in layers:
        outputTensor = layer(outputTensor)
    return Model(inputTensor, outputTensor)


def visWeightsBiases(model, logscale=True, zeroThresh=1e-9):
    '''Show weights and biases of a Keras model as a heatmap.'''
    nc = max([
        layer.get_weights()[0].shape[1]
        for layer in model.layers
        if len(layer.get_weights()) > 0
    ])
    params = []
    for layer in model.layers:
        wb = layer.get_weights()
        if len(wb) > 0:
            wbpadded = []
            for wob in wb:
                ncactual = wob.shape[-1]
                if ncactual < nc:
                    paddingAmounts = [(0, 0), (0, nc-wob.shape[-1])][-len(wob.shape):]
                    wob = np.pad(
                        wob,
                        paddingAmounts,
                        mode='constant',
                        constant_values=-np.inf,
                    )
                wbpadded.append(wob.reshape((int(wob.size / nc), nc)))
            params.append(wbpadded)
    ticks = []
    layerLines = []
    biasLocs = []
    row = 0
    for layerNumber, wb in enumerate(params):
        addLabel = lambda row, s: ticks.append((row, s))
        labelLocation = int(np.mean([row, row+wb[0].shape[0]+wb[1].shape[0]]))
        addLabel(labelLocation, 'layer %d' % layerNumber)

        row += wb[0].shape[0]
        biasLocs.append(row)

        row += wb[1].shape[0]
        layerLines.append(row)

    image = np.vstack([np.vstack(paramRow) for paramRow in params])
    image[np.abs(image) <= zeroThresh] = -np.inf

    fig, ax = plt.subplots(figsize=(3, 12))
    def transform(value):
        '''Modify label or do transformation.'''
        if isinstance(value, str):
            return r'$\log_{10}(|$%s$|)$' % value
        else:
            return np.log10(np.abs(value))
    if not logscale:
        transform = lambda x: x
    im = ax.imshow(transform(image), interpolation='nearest')
    powerPart = int(np.log10(zeroThresh))
    basePart = zeroThresh / 10**powerPart
    if basePart == 1:
        zeroThreshEngNot = '10^{%.d}' % (powerPart,)
    else:
        zeroThreshEngNot = r'%.1f\cdot 10^{%.d}' % (basePart, powerPart)
    ax.set_title(
        transform('weights and biases')
        + '\n' + r'$|$values$|\,\leq%s$ set to $\infty$' % (zeroThreshEngNot,)
    )
    ax.set_xticks([])
    ax.set_yticks([tick[0] for tick in ticks])
    for row in biasLocs:
        ax.axhline(row-.5, color='red', linestyle='--')
    for row in layerLines:
        ax.axhline(row-.5, color='red')
    ax.set_yticklabels([tick[1] for tick in ticks])
    fig.colorbar(im)
    fig.tight_layout()
    ax.grid(False)
    return fig, ax


class Pbar(tensorflow.keras.callbacks.Callback):
    '''Keras callback for displaying a tqdm progress bar.'''
    def __init__(self, numEpochs, **kw_tqdm_notebook):
        self.numEpochs = numEpochs
        kw_tqdm_notebook.setdefault('unit', 'Epoch')
        kw_tqdm_notebook.setdefault('desc', 'Training')
        self.tqdmNotebook = tqdm(total=numEpochs, **kw_tqdm_notebook)
    def on_epoch_end(self, *ignoredArgs, **ignoredKwargs):
        self.tqdmNotebook.update(1)

def TensorboardCallback(log_dir):
    '''Keras callback for logging progress in TensorBoard.'''
    return tensorflow.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=0, write_graph=True,
        write_images=False, embeddings_freq=0,
        embeddings_layer_names=None, embeddings_metadata=None
    )

def PlateauSlowdownCallback(
        monitor='val_loss', factor=0.97, patience=50, 
        verbose=1, mode='auto', min_lr=.001,
        **kwargs
    ):
    return tensorflow.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor, factor=factor, patience=patience,
        verbose=verbose, mode=mode, min_lr=min_lr,
        **kwargs
        )

def EarlyStoppingCallback(
        monitor='val_loss', min_delta=0.001,
        patience=1000, verbose=1, mode='auto',
        **kwargs
    ):
    return tensorflow.keras.callbacks.EarlyStopping(
        monitor=monitor, min_delta=min_delta,
        patience=patience, verbose=verbose, mode=mode,
        **kwargs
        )
