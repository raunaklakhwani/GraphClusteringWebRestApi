import numpy
from numpy import linalg
import json
import random
import math
from sets import Set
from itty import *
import copy
import operator
import Queue

finalClusters = []
numberOfNodes = 500
numberOfLinks = 1000
width = 800
height = 800
nodeSet = []
nodeSetId = numberOfNodes
nodesDict = {}
linksDict = {}
threshold = 1.5
nodeSetDict = {}

def initialize() : 
    # global finalClusters
    global nodeSet, finalClusters, numberOfNodes, numberOfLinks, width, height
    global nodeSetId, nodesDict, nodesDict, linksDict, threshold, nodeSetDict
    
    finalClusters = []
    numberOfNodes = 500
    numberOfLinks = 1000
    width = 800
    height = 800
    nodeSet = []
    nodeSetId = numberOfNodes
    nodesDict = {}
    linksDict = {}
    threshold = 1.5
    nodeSetDict = {} 

def getVolume(cluster):
    volume = 0
    for node in cluster:
        connectedVertices = linksDict.get(node, [])
        for vertex in connectedVertices :
            if vertex in cluster:
                volume = volume + 1
    return volume

def getCut(cluster1, cluster2):
    cut = 0
    if len(cluster1) > len(cluster2) :
        for node in cluster2 :
            connectedVertices = linksDict.get(node, [])
            for vertex in connectedVertices :
                if vertex in cluster1 :
                    cut = cut + 1
    else :
        for node in cluster1 :
            connectedVertices = linksDict.get(node, [])
            for vertex in connectedVertices :
                if vertex in cluster2 :
                    cut = cut + 1
    return cut

def checkQuality(cluster1, cluster2):
    volume1 = getVolume(cluster1)
    volume2 = getVolume(cluster2)
    
    if volume1 == 0 and volume2 == 0 :
        return False
    
    normalized_cut = 0
    cut = getCut(cluster1, cluster2)
    if volume1 != 0 and volume2 != 0 :
        normalized_cut = (cut / float(volume1)) + (cut / float(volume2))
    elif volume1 != 0 :
        normalized_cut = (cut / float(volume1))
    elif volume2 != 0 :
        normalized_cut = (cut / float(volume2))
    print "cluster 1 = ", len(cluster1), " cluster 2 = ", len(cluster2), " normalized_cut = ", normalized_cut
    print cluster1
    print cluster2
    if normalized_cut < threshold :
        return True
    else :
        return False
    
def generateLinksDictionary(links):
    for link in links:
        source = link['source']
        target = link['target']
        sourceDict = linksDict.get(source)
        if sourceDict :
            sourceDict.add(target)
        else :
            sourceDict = Set([target])
            linksDict[source] = sourceDict
        
        targetDict = linksDict.get(target)
        if targetDict :
            targetDict.add(source)
        else :
            targetDict = Set([source])
            linksDict[target] = targetDict    
        
            
    return linksDict


def getSecondLargestEigenVector(adjacencyMatrix) :
    eigenValues, eigenVectors = linalg.eig(adjacencyMatrix)
    # a = map(round,eigenValues.tolist(),[5] * len(eigenValues.tolist()))
    realValues = []
    for i, imag in enumerate(eigenValues.imag) :
        if imag == 0 :
            realValues.append((i, eigenValues.real[i]))
    realValues.sort(key=lambda tup : tup[1])
    
    if len(realValues) > 1 :
        index = realValues[1][0]
    
    # a = eigenValues
    # b = []
    # for i in a:
    #    b.append(i)
    # a.sort()
    # index = b.index(a[1])                    
    # eigenVectors.transpose()[index]
        eigenVector = eigenVectors.transpose()[index]
        eigenVectorList = eigenVector.tolist()
        
        for index, item in enumerate(eigenVectorList[0]) :
            eigenVectorList[0][index] = round(eigenVectorList[0][index].real, 5)
        print 'eigenVectorList'    
        print eigenVectorList
        return eigenVectorList[0]
    else : 
        return None
    

def isEdge(source, target):
    sourceDict = linksDict.get(source)
    if sourceDict :
        try :
            if target in sourceDict  :
                return True
        except ValueError:
            return False
    return False



def getClusters(nodes, links) :
    # nodeIds = []
    # for node in nodes:
    #    nodeIds.append(node['id'])
    # nodeIds = map(str,nodeIds)
    nodeIds = nodes
    size = len(nodeIds)
    
    
    degreeArray = numpy.zeros(size)

    for link in links:
        sIndex = nodeIds.index(link['source'])
        tIndex = nodeIds.index(link['target'])
        if sIndex == tIndex :
            print 'hello'
            degreeArray[tIndex] += 2
        else :
            degreeArray[tIndex] += 1
            degreeArray[sIndex] += 1
            
   
        
    lap = numpy.matrix(numpy.zeros((size, size)))
    for i in range(size):
        for j in range(size):
            value = 0;
            if i == j and degreeArray[j] != 0:
                value = 1
            elif isEdge(nodeIds[i], nodeIds[j]) :
                value = -1 / math.sqrt(degreeArray[i] * degreeArray[j])
            
            lap.itemset((i, j), value)    
                
                
                
     
        
    
        
    eigenVector = getSecondLargestEigenVector(lap)
    # eigenVector = eigenVector.real
    if eigenVector != None :
        cluster1 = []
        cluster2 = []
        cluster3 = []
        for i, s in enumerate(eigenVector):
            # s = eigenVector[0].item(i)
            if s > 0 :
                cluster1.append(nodeIds[i])
            elif s < 0 :
                cluster2.append(nodeIds[i])
            else :
                cluster3.append(nodeIds[i])
    
                
        # clusters = {}
        # clusters['cluster1'] = cluster1
        # clusters['cluster12'] = cluster2
        # clusters['adj'] = adj
        # clusters['nodeIds'] = nodeIds
        
        if len(cluster1) != 0 and len(cluster2) != 0 and len(cluster3) != 0:
            if checkQuality(cluster1, cluster2) and checkQuality(cluster1, cluster3) and checkQuality(cluster2, cluster3) :
                clusters = []
                if len(cluster1) > 1:
                    clusters.append(cluster1);
                if len(cluster2) > 1 :
                    clusters.append(cluster2);
                if len(cluster3) > 1 :
                    clusters.append(cluster3);
                clusters.append("OK")
            else :
                clusters = []
                clusters.append(nodes)
                clusters.append("DONE_POOR")
        elif len(cluster1) != 0 and len(cluster2) != 0 :
            if checkQuality(cluster1, cluster2) : 
                clusters = []
                if len(cluster1) > 1:
                    clusters.append(cluster1);
                if len(cluster2) > 1:
                    clusters.append(cluster2);
                clusters.append("OK")
            else :
                clusters = []
                clusters.append(nodes)
                clusters.append("DONE_POOR")
        elif len(cluster1) != 0 and len(cluster3) != 0 :
            if checkQuality(cluster1, cluster3) : 
                clusters = []
                if len(cluster1) > 1:
                    clusters.append(cluster1);
                if len(cluster3) > 1:
                    clusters.append(cluster3);
                clusters.append("OK")
            else :
                clusters = []
                clusters.append(nodes)
                clusters.append("DONE_POOR")
        elif len(cluster2) != 0 and len(cluster3) != 0 :
            if checkQuality(cluster2, cluster3) : 
                clusters = []
                if len(cluster2) > 1:
                    clusters.append(cluster2);
                if len(cluster3) > 1:
                    clusters.append(cluster3);
                clusters.append("OK")
            else :
                clusters = []
                clusters.append(nodes)
                clusters.append("DONE_POOR")
        else :
            clusters = []
            clusters.append(cluster1 if len(cluster1) != 0 else cluster2 if len(cluster2) != 0 else cluster3);
            clusters.append("DONE")
    else :
        clusters = []
        clusters.append(nodes)
        clusters.append("ERROR")
        
    return clusters 


def graphClustering(nodes, links, nodeSetIdInfo) :
    global nodeSetId 
    
    clusters = getClusters(nodes, links);
    result = clusters.pop()
    if result == "OK" :
        for cluster in clusters :
            clusterLinks = [i for i in links if i['source'] in cluster and i['target'] in cluster]
            dummyNodeSetIdInfo = [] 
            graphClustering(cluster, clusterLinks, dummyNodeSetIdInfo)
            if len(dummyNodeSetIdInfo) > 1 :
                aDict = {}
                aDict['id'] = nodeSetId + 1
                nodeSetId = nodeSetId + 1
                aDict['type'] = 'nodeSet'
                aDict['nodes'] = dummyNodeSetIdInfo
                aDict['x'] = random.randrange(0, width)
                aDict['y'] = random.randrange(0, height)
                nodeSet.append(aDict)
                nodeSetIdInfo.append(nodeSetId)
            elif len(dummyNodeSetIdInfo) == 1 :
                nodeSetIdInfo.append(dummyNodeSetIdInfo[0])
    elif result != "ERROR" :
        aDict = {}
        aDict['id'] = nodeSetId + 1
        nodeSetId = nodeSetId + 1
        aDict['type'] = 'nodeSet'
        aDict['nodes'] = nodes
        aDict['x'] = random.randrange(0, width)
        aDict['y'] = random.randrange(0, height)
        nodeSet.append(aDict) 
        finalClusters.append(clusters.pop())
        nodeSetIdInfo.append(nodeSetId)
        
    
    
        


def getConnectedComponents(nodes, links) : 
    print 'connected components'
    queue = Queue.Queue(maxsize=0)
    connectedComponents = []
    for node in nodes:
        if node.get('considered') is None:
            conComponent = [node['id']]
            connectedComponents.append(conComponent)
            queue.put(node)
            node['considered'] = True
            while queue.empty() == False : 
                nodesConsidered = queue.get()
                if linksDict.get(nodesConsidered['id']) is not None : 
                    for nodeId in linksDict.get(nodesConsidered['id']) : 
                        if(nodesDict[nodeId].get('considered') is None) : 
                            queue.put(nodesDict[nodeId])
                            nodesDict[nodeId]['considered'] = True
                            conComponent.append(nodesDict[nodeId]['id'])
                        
                        
    print connectedComponents

    


@post('/post')
def clustering(request):
    initialize()
    print 'Service called'
    # a =  request.POST.get('json')
    body = json.loads(request.body)
    nodes = body['nodes']
    links = body['links']
    originalNodes = copy.deepcopy(nodes)
    originalLinks = copy.deepcopy(links)
    
    
    
    
    print nodes
    print links
    nodeIds = []
    generateLinksDictionary(links)
    
    for node in nodes:
        nodeIds.append(node['id'])
        nodesDict[node['id']] = node
    # nodeIds = map(str,nodeIds)
    nodeSetIdInfo = []
    
    
    
    
    # BFS starts
    
    getConnectedComponents(nodes, links)
    # BFS end
    
    
    
    graphClustering(nodeIds, links, nodeSetIdInfo)
    
    # print finalClusters
    print len(finalClusters)
    
    # dictNodes = {}
    # for node in nodes : 
    #    dictNodes[node['id']] = node
    #===========================================================================
    # global nodeSetId
    # nodeSet = []
    # for cluster in finalClusters :
    #     aDict = {}
    #     aDict['id'] = nodeSetId + 1
    #     aDict['type'] = 'nodeSet'
    #     aDict['nodes'] = cluster
    #     aDict['x'] = random.randrange(0,width)
    #     aDict['y'] = random.randrange(0,height)
    #     print len(cluster)
    #     nodeSet.append(aDict)
    #     nodeSetId = nodeSetId + 1
    # print nodeSet
    #===========================================================================
    setCoordinateOfNodeSet()
    print nodeSet
    responseString = json.dumps({"nodes" : originalNodes, "links" : originalLinks, "nodeSet" : nodeSet})
    response = Response(responseString, content_type='application/json')
    response.add_header("Access-Control-Allow-Origin", "*")
    response.add_header("Access-Control-Expose-Headers", "Access-Control-Allow-Origin")
    # response.add_header("Access-Control-Allow-Headers: Origin, X-Requested-With, Content-Type, Accept")
    return response


def setCoordinateOfNodeSet() :
    for eachNodeSet in nodeSet:
        nodeSetDict[eachNodeSet['id']] = eachNodeSet 
        
        
    for eachNodeSet in nodeSet :
        nodes = eachNodeSet['nodes']
        numberOfNodes = len(eachNodeSet['nodes'])
        x = 0
        y = 0;
        for nodeId in nodes : 
            if nodesDict.get(nodeId) is not None :
                x = x + nodesDict[nodeId]['x']
                y = y + nodesDict[nodeId]['y']
            else :
                x = x + nodeSetDict[nodeId]['x']
                y = y + nodeSetDict[nodeId]['y']

        x = x / numberOfNodes
        y = y / numberOfNodes
        eachNodeSet['x'] = x
        eachNodeSet['y'] = y


#######################################################################
####Stacys Algorithm
#===============================================================================
# def generateLinksDictionary(links):
#     for link in links:
#         source = link['source']['id']
#         target = link['target']['id']
#         sourceDict = linksDict.get(source)
#         if sourceDict :
#             sourceDict.add(target)
#         else :
#             sourceDict = Set([target])
#             linksDict[source] = sourceDict
#         
#         targetDict = linksDict.get(target)
#         if targetDict :
#             targetDict.add(source)
#         else :
#             targetDict = Set([source])
#             linksDict[target] = targetDict    
#         
#             
#     return linksDict
#===============================================================================

def getNextLevelNodes(sourceNode, nodesConnected, excluseNodes) :
    nextLevelNodes = []
    nodesConnected = Set(nodesConnected)
    for node in nodesConnected :
        if node not in excluseNodes : 
            if nodesDict[node]['included'] == False : 
                nextLevelNodes.append(node)
    return nextLevelNodes
    
def generateNodeSet(sourceNode, nodesConnected, excluseNodes) :
    global nodeSetId 
    if nodesDict[sourceNode]['included'] == False : 
        nodesDict[sourceNode]['included'] = True
        nextLevelNodes = getNextLevelNodes(sourceNode, nodesConnected, excluseNodes)
        if len(nextLevelNodes) == 0 :
            return sourceNode
        else :
            nodes = [sourceNode]
            for nextNode in nextLevelNodes :
                updatedNodesConnected = linksDict[nextNode]
                excluseNodes = copy.deepcopy(excluseNodes)
                for nodeConnected in nodesConnected : 
                    if nodeConnected not in excluseNodes :
                        excluseNodes.add(nodeConnected)
                id = generateNodeSet(nextNode, updatedNodesConnected, excluseNodes)
                if id is not None :
                    nodes.append(id)
            
            aDict = {}
            aDict['id'] = nodeSetId + 1
            nodeSetId = nodeSetId + 1
            aDict['type'] = 'nodeSet'
            aDict['nodes'] = nodes
            aDict['x'] = random.randrange(0, width)
            aDict['y'] = random.randrange(0, height)
            nodeSet.append(aDict) 
            return aDict['id']
    return None
    
@post('/post1')    
def stacyAlgorithm(request) :
    initialize()
    print 'Calledx'
    body = json.loads(request.body)
    nodes = body['nodes']
    links = body['links']
    nodeIds = []
    
    for node in nodes:
        node['included'] = False
        nodeIds.append(node['id'])
        nodesDict[node['id']] = node
    # nodeSetIdInfo = []
    degreeDict = {}
    generateLinksDictionary(links)
    for keys, items in linksDict.items() : 
        degreeDict[keys] = len(items)
    
      
    tommy = sorted(degreeDict.items(), key=operator.itemgetter(1), reverse=True)   
    
    
    generateNodeSet(tommy[0][0], linksDict[tommy[0][0]], Set([tommy[0][0]]))  
    setCoordinateOfNodeSet()
    responseString = json.dumps({"nodes" : nodes, "links" : links, "nodeSet" : nodeSet})
    response = Response(responseString, content_type='application/json')
    response.add_header("Access-Control-Allow-Origin", "*")
    response.add_header("Access-Control-Expose-Headers", "Access-Control-Allow-Origin")
    # response.add_header("Access-Control-Allow-Headers: Origin, X-Requested-With, Content-Type, Accept")
    return response

#######################################################################
        
        
        
if __name__ == '__main__':
    run_itty()
