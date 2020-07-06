from nodeanalysis.nodes.EmptyNode import EmptyNode

class SimpleNeuron(EmptyNode):
    def __init__(self):
        super(SimpleNeuron,self).__init__()
        
        self.weight = None
        self.bias = None
        self.activation = None
    
    def report(self):
        super(SimpleNeuron, self).report()
        print()
        print(f"\t activation={self.activation}")
        print(f"\t weight={self.weight}")
        print(f"\t bias={self.bias}")
        print(f"\t epochs={self.epochs}")
    
    def get(self, nac, layer_index, node_index, epoch=0):
        super(SimpleNeuron, self).get(nac, layer_index, node_index, epoch=epoch)
        
        self.activation = self.layer.activation
 
        weights = nac.weights[self.layer.name][epoch][0]
        bias = nac.weights[self.layer.name][epoch][1]
        
        self.weight = weights
        for a in range(len(node_index)):
            self.weight = self.weight[a]
            
        self.bias = bias[node_index[-1]]
        self.epochs = len(nac.weights[self.layer.name])