from tensorflow.keras.callbacks import Callback
import numpy as np

# GetWeights() .weight_dict["layer_name"]["weights" | "bias" | "activation"] .layer_names[]
# print_node_params(weights, bias, activation, node_index)

class NodeAnalysisCallback(Callback):
    # Keras callback which collects values of weights and biases at each epoch
    def __init__(self):
        super(NodeAnalysisCallback, self).__init__()
        self.weights = {}

    def on_epoch_end(self, epoch, logs=None):
        # this function runs at the end of each epoch

        # loop over each layer and get weights and biases
        for layer_i in range(len(self.model.layers)):
            #print(self.model.layers[layer_i].name)
            try:
                w = self.model.layers[layer_i].get_weights()
            except:
                w = None

            # save all weights and biases inside a dictionary
            if epoch == 0:
                # create array to hold weights and biases
                name = self.model.layers[layer_i].name
                self.weights[name] = [w]
                
            else:
                # append new weights to previously-created weights array
                name = self.model.layers[layer_i].name
                
                self.weights[name] = np.vstack((self.weights[name], np.array([w]) ))
