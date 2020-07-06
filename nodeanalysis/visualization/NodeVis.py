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
    
def crossLists(listA, listB):
    if type(listA[0]).__name__ != "list" :
        listA = [[a] for a in listA]
    if type(listB[0]).__name__ != "list" :
        listB = [[b] for b in listB]
    return [a + b for a in listA for b in listB]

def showForLayer(nac, layer_index, show=["weights", "bias"]):
    layer_name = nac.nnlayer_names[layer_index]
    shape = nac.weight_dict[layer_name]["weights"].shape[1:]
    
    indexList = None
    for i in range(len(shape)):
        index = range(shape[i])
        if indexList == None:
            indexList = index
            continue
        indexList = crossLists(indexList,index)
    
    for node_index in indexList:
        if show != None and "weights" in show:
            showWeightsB(nac, layer_index, node_index)
        if show != None and "bias" in show:
            showBiasB(nac, layer_index, node_index)