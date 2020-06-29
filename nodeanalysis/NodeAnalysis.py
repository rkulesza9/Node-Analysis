from tensorflow.keras.callbacks import Callback
import numpy as np
from nodeanalysis.Nodes import Node

# GetWeights() .weight_dict["layer_name"]["weights" | "bias" | "activation"] .layer_names[]
# print_node_params(weights, bias, activation, node_index)

class NodeAnalysisCallback(Callback):
    # Keras callback which collects values of weights and biases at each epoch
    def __init__(self):
        super(NodeAnalysisCallback, self).__init__()
        self.weight_dict = {}
        self.layer_names = []
        self.nnlayers = []
        self.nnlayer_names = []

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
                
                if self.model.layers[layer_i] not in self.nnlayers:
                    self.nnlayers.append(self.model.layers[layer_i])
                    self.nnlayer_names.append(self.model.layers[layer_i].name)
                
            except:
                continue

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
    
    def getNode(self, layer_index, node_index, epoch=-1):
        node = Node()
        node.model = self.model
        node.layer = self.nnlayers[layer_index]
        node.epoch = epoch
        
        if layer_index != 0:
            node.layer_in = self.nnlayers[layer_index - 1]
        if layer_index <= len(self.nnlayers) and layer_index != -1:
            node.layer_out = self.nnlayers[layer_index + 1]
        
        node.name = f"Node #{node_index}"
        
        node.weight = self.weight_dict[self.nnlayer_names[layer_index]]["weights"][epoch]
        for n in range(0, len(node_index)):
            w_index = node_index[n]
            node.weight = node.weight[w_index]
        
        node.bias = self.weight_dict[self.nnlayer_names[layer_index]]["bias"][epoch][node_index[-1]]
        node.activation = self.weight_dict[self.nnlayer_names[layer_index]]["activation"][epoch]
        
        return node
    
    def getNodeHistory(self, layer_index, node_index):
        nodelist = []
        epochs = len(self.weight_dict[self.nnlayer_names[layer_index]]["weights"])
        for e in range(epochs):
            nodelist.append(self.getNode(layer_index, node_index, epoch=e))
        return nodelist