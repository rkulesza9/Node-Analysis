import matplotlib.pyplot as plt

## HELP FUNCTIONS

def crossLists(listA, listB):
    if type(listA[0]).__name__ != "list" :
        listA = [[a] for a in listA]
    if type(listB[0]).__name__ != "list" :
        listB = [[b] for b in listB]
    return [a + b for a in listA for b in listB]

def getIndexListForShape(shape):
    indexList = None
    for i in range(len(shape)):
        index = range(shape[i])
        if indexList == None:
            indexList = index
            continue
        indexList = crossLists(indexList,index)
    return indexList

def getNodesFromLayer(nac, layer_index, nodetype):
    layer_name = nac.model.layers[layer_index].name
    weight_shape = nac.weights[layer_name][0][0].shape
    index_list = getIndexListForShape(weight_shape)
    
    node_list = []
    for index in index_list:
        node_list.append(nodetype().get(nac, layer_index, index))
    
    return node_list

## VISUALIZATION USING MATPLOT 

def showWeights(history):
    y = [n.weight for n in history]
    x = [n.epoch for n in history]
    
    plt.title(f"Weight Change Over Time {history[0].name}")
    plt.plot(x,y)
    plt.ylabel('weight')
    plt.xlabel("epochs")
    plt.show()

def showBias(history):
    y = [n.bias for n in history]
    x = [n.epoch for n in history]
    
    plt.title(f"Bias Change Over Time {history[0].name}")
    plt.plot(x,y)
    plt.ylabel('bias')
    plt.xlabel("epochs")
    plt.show()