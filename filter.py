
from pybrain.datasets.classification import ClassificationDataSet

from pybrain.optimization.populationbased.ga import GA
from pybrain.tools.shortcuts import buildNetwork
file = open('C:/Users/ronoy/OneDrive/Documents/Hepatitis/data.csv')

d = ClassificationDataSet(19)
for line in file:
    storageList = []
    classification = 100
    for i in line.split(','):
        if(i == 'live' or i == 'die'):
            if i == 'live':
                classification = 1
            else:
                classification = 0
        elif (i == 'True'):
            storageList.append(1)
        elif (i == 'False'):
            storageList.append(0)
        else:
            storageList.append(i)

    d.addSample(storageList,[classification])
    print storageList

# create dataset
'''
d.addSample([181, 80], [1])
d.addSample([177, 70], [1])
d.addSample([160, 60], [0])
d.addSample([154, 54], [0])
'''

d.setField('class', [ [0.],[1.],[1.],[0.]])

nn = buildNetwork(2, 3, 1)

# d.evaluateModuleMSE takes nn as its first and only argument
ga = GA(d.evaluateModuleMSE, nn, minimize=True)

for i in range(100):
    nn = ga.learn(0)[0]

print nn.activate([41,female,True,True,True,False,False,True,True,False,False,False,False,0.9,81,60,3.9,52,False])
