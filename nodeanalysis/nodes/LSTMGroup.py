from nodeanalysis.nodes.EmptyNode import EmptyNode
from nodeanalysis.nodes.SimpleNeuron import SimpleNeuron


class LSTMGroup(EmptyNode):
    def __init__(self):
        super(LSTMGroup, self).__init__()
        self.units = None
        self.kernel = None
        self.recurrent_kernel = None
        self.bias_all = None
        
        self.kernel_dict = None
        self.recurrent_kernel_dict = None
        self.bias_dict = None
        
        # weights type
        self.KERNEL = "KERNEL"
        self.RE_KERNEL = "RECURRENT KERNEL"
        
        # weights subtype 
        self.INPUT = "input"
        self.FORGET = "forget"
        self.CELL_STATE = "cell_state"
        self.OUTPUT = "output"
    
    def report(self):
        super(LSTMGroup, self).report()
    
    def getSimpleNeuron(self, weights_type, weights_subtype, node_index, epoch=0):
        self.get(self.nac, self.layer_index, self.node_index,  epoch=epoch)
        
        node = SimpleNeuron()
        
        node.name = f"{self.name} -- {weights_type} > {weights_subtype}: {node_index}"
        node.model = self.model
        node.epoch = epoch
        
        node.layer = self.layer
        node.layer_in = self.layer_in
        node.layer_out = self.layer_out
        
        node.layer_index = self.layer_index
        node.node_index = [self.node_index, node_index]
        node.nac = self.nac
        
        node.epochs = len(self.nac.weights[self.layer.name])
        
        if weights_type == self.KERNEL:
            node.weight = self.kernel_dict

        if weights_type == self.RE_KERNEL:
            node.weight = self.recurrent_kernel_dict
        
        node.weight = node.weight[weights_subtype]
            
        for index in node_index:
            node.weight = node.weight[index]
        
        node.bias = self.bias_all[node_index[-1]]
        
        return node
        
    def getSimpleNeuronHistory(self, weights_type, weights_subtype, node_index, epochs):
        h = [self.getSimpleNeuron(weights_type, weights_subtype,  node_index, epoch=e) for e in range(epochs)]
        return h
        
    def get(self, nac, layer_index, node_index,  epoch=-1):
        super(LSTMGroup, self).get(nac, layer_index, node_index, epoch=epoch)
        
        W = nac.weights[self.layer.name][epoch][node_index*3]
        U = nac.weights[self.layer.name][epoch][node_index*3 + 1]
        b = nac.weights[self.layer.name][epoch][node_index*3 + 2]
        
        self.kernel = W
        self.recurrent_kernel = U
        self.bias_all = b

        units = int(int(self.kernel.shape[1])/4)
        self.units = units
        
        # input - forget - cell state - output
        W_i = W[:, :units]
        W_f = W[:, units: units * 2]
        W_c = W[:, units * 2: units * 3]
        W_o = W[:, units * 3:]
        
        self.kernel_dict = {
            "input" : W_i,
            "forget" : W_f,
            "cell_state" : W_c,
            "output" : W_o
        }

        U_i = U[:, :units]
        U_f = U[:, units: units * 2]
        U_c = U[:, units * 2: units * 3]
        U_o = U[:, units * 3:]
        
        self.recurrent_kernel_dict = {
            "input" : U_i,
            "forget" : U_f,
            "cell_state" : U_c,
            "output" : U_o 
        }

        b_i = b[:units]
        b_f = b[units: units * 2]
        b_c = b[units * 2: units * 3]
        b_o = b[units * 3:]
        
        self.bias_dict = {
            "input" : b_i,
            "forget" : b_f,
            "cell_state" : b_c,
            "output" : b_o 
        }
        
        return self