import numpy as np

class EmptyNode:
    def __init__(self):
        self.name = None
        self.model = None
        self.epoch = None
        
        self.layer = None
        self.layer_in = None
        self.layer_out = None
        
        self.layer_index = None
        self.node_index = None
        self.nac = None
        
        self.epochs = None
    
    def report(self):
        print(f"NODE [name={self.name} epoch={self.epoch}]")
        print(f"\t layer_index={self.layer_index}")
        print(f"\t node_index={self.node_index}")
        print()
        print(f"\t model={self.model}")
        print(f"\t layer={self.layer}")
        print()
        print(f"\t layer_in={self.layer_in}")
        print(f"\t layer_out={self.layer_out}")
        
    
    def get(self, nac, layer_index, node_index, epoch=0):
        self.model = nac.model
        self.epoch = epoch
        
        self.layer_index = layer_index
        self.node_index = node_index
        self.nac = nac
        
        self.layer = self.model.layers[layer_index]
        
        if(layer_index != 0):
            self.layer_in = self.model.layers[layer_index - 1]
        
        if(layer_index < len(self.model.layers) - 1 or layer_index < -1):
            self.layer_out = self.model.layers[layer_index + 1]

        self.name = f"Node {self.layer.name} : { node_index }"
        
        self.epochs = len(nac.weights[self.layer.name])
        
        return self
    
    def history(self, epochs, node_type):
        h = [node_type().get(self.nac,self.layer_index,self.node_index,epoch=e) for e in range(epochs)]
        return h