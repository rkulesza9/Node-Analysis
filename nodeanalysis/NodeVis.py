import matplotlib.pyplot as plt

def showWeightsA(nodes):
    y = [n.weight for n in nodes]
    x = [n.epoch for n in nodes]
    
    plt.title(f"Weight Change Over Time {nodes[0].name}")
    plt.plot(x,y)
    plt.ylabel('weight')
    plt.xlabel("epochs")
    plt.show()

def showWeightsB(nac, layer_index, node_index):
    nodes = nac.getNodeHistory(layer_index,node_index)
    showWeightsA(nodes)

def showBiasA(nodes):
    y = [n.bias for n in nodes]
    x = [n.epoch for n in nodes]
    
    plt.title(f"Bias Change Over Time {nodes[0].name}")
    plt.plot(x,y)
    plt.ylabel('bias')
    plt.xlabel("epochs")
    plt.show()

def showBiasB(nac, layer_index, node_index):
    nodes = nac.getNodeHistory(layer_index,node_index)
    showBiasA(nodes)