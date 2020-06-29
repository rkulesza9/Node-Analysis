from tensorflow.keras.callbacks import Callback
import numpy as np

# GetWeights() .weight_dict["layer_name"]["weights" | "bias" | "activation"] .layer_names[]
# print_node_params(weights, bias, activation, node_index)

class GetWeights(Callback):
    # Keras callback which collects values of weights and biases at each epoch
    def __init__(self):
        super(GetWeights, self).__init__()
        self.weight_dict = {}
        self.layer_names = []

    def on_epoch_end(self, epoch, logs=None):
        # this function runs at the end of each epoch

        # loop over each layer and get weights and biases
        for layer_i in range(len(self.model.layers)):
            #print(self.model.layers[layer_i].name)
            if self.model.layers[layer_i].name not in self.layer_names:
                    self.layer_names.append(self.model.layers[layer_i].name)
            try:
                w = self.model.layers[layer_i].get_weights()[0]
                b = self.model.layers[layer_i].get_weights()[1]
                a = self.model.layers[layer_i].activation
                print(a)
            except:
                continue
                
            print('Layer %s has weights of shape %s and biases of shape %s' %(
                layer_i, np.shape(w), np.shape(b)))

            # save all weights and biases inside a dictionary
            if epoch == 0:
                # create array to hold weights and biases
                name = self.model.layers[layer_i].name
                self.weight_dict[name] = { "weights" : [w], "bias" : [b], "activation" : [a]}
                
            else:
                # append new weights to previously-created weights array
                name = self.model.layers[layer_i].name
                
                self.weight_dict[name] = {
                    "weights" : np.vstack((self.weight_dict[name]["weights"], np.array([w]) )),
                    "bias" :  np.vstack((self.weight_dict[name]["bias"], np.array([b]) )),
                    "activation" : np.vstack((self.weight_dict[name]["activation"], np.array([a])))
                }
    
    def getActivationFunction(self,layer_i):
        try:
            layer = self.model.layers[layer_id]
            act = layer.activation
            print(f'Layer {layer_i} activation function: {act}')
            return act
        except:
            return None
        

def print_node_params(weights, bias, activations, node_index):
    print(f'NODE index={node_index} PARAMS')
    
    n_bias = len(node_index)-1
    
    for epoch in range(0,len(weights)):
        print(f'epoch {epoch}')
        
        w = weights[epoch]
        b = bias[epoch][n_bias]
        a = activations[epoch]
        
        for n in range(0, len(node_index)):
            w_index = node_index[n]
            w = w[w_index]
        
        print(f'\tactivation {a}')
        print(f'\tweight {w}')
        print(f'\tbias {b}')