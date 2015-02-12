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
from grandalf.graphs import Vertex, Edge, Graph
from  grandalf.graphs  import *
from  grandalf.layouts import *

finalClusters = []
numberOfNodes = 10000
numberOfLinks = 1000
width = 800
height = 800
nodeSet = []
nodeSetId = numberOfNodes
nodesDict = {}
linksDict = {}
threshold = 1.5
nodeSetDict = {}
base = -1
hierarchy = {"cloud" : 0, "border router" : 1, "distribution" : 2, "access" : 3, "host" : 4}

def initialize() : 
    # global finalClusters
    global nodeSet, finalClusters, numberOfNodes, numberOfLinks, width, height
    global nodeSetId, nodesDict, nodesDict, linksDict, threshold, nodeSetDict
    
    finalClusters = []
    numberOfNodes = 10000
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
        
        
def graphClusteringUpdated(nodes, links) :
    global nodeSetId 
    
    if len(nodes) > 1 :
        clusters = getClusters(nodes, links);
        result = clusters.pop()
        if result == "OK" :
            dividedPartsNodeSetInfo = []
            for cluster in clusters :
                clusterLinks = [i for i in links if i['source'] in cluster and i['target'] in cluster]
                nodeSetIdGenerated = graphClusteringUpdated(cluster, clusterLinks)
                if nodeSetIdGenerated is not None:
                    dividedPartsNodeSetInfo.append(nodeSetIdGenerated)
            if len(dividedPartsNodeSetInfo) > 1 : 
                aDict = {}
                aDict['id'] = nodeSetId + 1
                nodeSetId = nodeSetId + 1
                aDict['type'] = 'nodeSet'
                aDict['nodes'] = dividedPartsNodeSetInfo
                aDict['x'] = random.randrange(0, width)
                aDict['y'] = random.randrange(0, height)
                nodeSet.append(aDict)
                return nodeSetId 
            else : 
                return None
        elif result != "ERROR" :
            aDict = {}
            aDict['id'] = nodeSetId + 1
            nodeSetId = nodeSetId + 1
            aDict['type'] = 'nodeSet'
            aDict['nodes'] = nodes
            aDict['x'] = random.randrange(0, width)
            aDict['y'] = random.randrange(0, height)
            nodeSet.append(aDict) 
            return nodeSetId
    else : 
        return None
        
    
    
        


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
    return connectedComponents

    


@post('/post')
def clustering(request):
    global nodeSetId
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
    
    connectedComponents = getConnectedComponents(nodes, links)
    # BFS end
    
    
    # # Earlier this stable version available
    # graphClustering(nodeIds, links, nodeSetIdInfo)
    # #
    # # 
    # Added may be unstable
    dividedPartsNodeSetInfo = []
    for connectedComponent in  connectedComponents : 
        updatedLinks = [i for i in links if i['source'] in connectedComponent and i['target'] in connectedComponent]
        nodeSetIdGenerated = graphClusteringUpdated(connectedComponent, updatedLinks)
        if nodeSetIdGenerated is not None : 
            dividedPartsNodeSetInfo.append(nodeSetIdGenerated)
    #===========================================================================
    # if len(dividedPartsNodeSetInfo) > 1 : 
    #     aDict = {}
    #     aDict['id'] = nodeSetId + 1
    #     nodeSetId = nodeSetId + 1
    #     aDict['type'] = 'nodeSet'
    #     aDict['nodes'] = dividedPartsNodeSetInfo
    #     aDict['x'] = random.randrange(0, width)
    #     aDict['y'] = random.randrange(0, height)
    #     nodeSet.append(aDict)
    #===========================================================================
    # # End
    
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


###########Sugiyama changes begins###############################
class defaultview(object):
    w, h = 300, 300

@post('/sugiyama') 
def sugiyama(request):
    initialize()
    body = json.loads(request.body)
    nodes = body['nodes']
    links = body['links']
    print nodes
    print links
    nodeIds = []
    for node in nodes:
        nodeIds.append(node['id'])
        nodesDict[node['id']] = node
    # V = [Vertex(data) for data in range(10)]
    V = []
    nodeIdIndexDict = {}
    index = 0;
    for node in nodes :
        V.append(Vertex(node['id']))
        nodeIdIndexDict[node['id']] = index
        index = index + 1 
        
    # X = [(link['source'], link['target']) for link in links]
    
    X = []
    hierarchy = ['cloud', 'border router', 'distribution', 'access', 'host']
    hierarchy.reverse()
    for link in links:
        sourceRole = nodesDict[link['source']]['role'].lower()
        targetRole = nodesDict[link['target']]['role'].lower()
        sourceRoleIndex = hierarchy.index(sourceRole)
        targetRoleIndex = hierarchy.index(targetRole)
        if sourceRoleIndex > targetRoleIndex :
            X.append((link['source'], link['target']))
        else :
            X.append((link['target'], link['source']))
    
    E = [Edge(V[nodeIdIndexDict[v]], V[nodeIdIndexDict[w]]) for (v, w) in X]
    
    #===========================================================================
    # X = [(0,1),(0,2),(1,3),(2,3),(4,0),(1,4),(4,5),(5,6),(3,6),(3,7),(6,8),(7,8),(8,9),(5,9)]
    # E = [Edge(V[v],V[w]) for (v,w) in X]
    #===========================================================================
    g = Graph(V, E)
    
    for v in V:
        v.view = defaultview()
        
    connectedComponents = len(g.C)
    
    for i in range(connectedComponents):
        connectedComponent = g.C[i]
        sug = SugiyamaLayout(connectedComponent)
        connectedComponentVertices = connectedComponent.V()
        rootVertex = connectedComponentVertices.next()
        sug.init_all()      
        sug.draw(10)
        for l in sug.layers:
            for n in l:
                try : 
                    nodesDict[n.data]['x'] = n.view.xy[0]
                    nodesDict[n.data]['y'] = n.view.xy[1]
                except :
                    pass
                
    
    responseString = json.dumps({"nodes" : nodes, "links" : links})
    response = Response(responseString, content_type='application/json')
    response.add_header("Access-Control-Allow-Origin", "*")
    response.add_header("Access-Control-Expose-Headers", "Access-Control-Allow-Origin")
    # response.add_header("Access-Control-Allow-Headers: Origin, X-Requested-With, Content-Type, Accept")
    return response

###########Sugiyama changes ends###############################

###########Sugiyama Hierarchy changes begins###############################


@post('/sugiyamaHierarchy') 
def sugiyamaHierarchy(request):
    initialize()
    body = json.loads(request.body)
    nodes = body['nodes']
    links = body['links']
    print nodes
    print links
    nodeIds = []
    for node in nodes :
        nodeIds.append(node['id'])
        nodesDict[node['id']] = node
    
    hierarchy = ['cloud', 'border router', 'distribution', 'access', 'host']
    
    for index in range(len(hierarchy)) : 
        if index < len(hierarchy) - 1 :
            role = hierarchy[index]
            nextRole = hierarchy[index + 1]
            workingNodes = [node for node in nodes if (node.get('role') is not None and (node.get('role').lower() == role or node.get('role').lower() == nextRole)) or node.get('role') is None]
            workingNodesIds = [node['id'] for node in workingNodes]
            workingLinks = [link for link in links if link['source'] in workingNodesIds and link['target'] in workingNodesIds]
            sugiyamaUtil(workingNodes, workingLinks)
        index = index + 1
    
    
    #===========================================================================
    # for role in hierarchy:
    #     workingNodes = [node for node in nodes if (node.get('role') is not None and node.get('role').lower() == role) or node.get('role') is None]
    #     workingNodesIds = [node['id'] for noce in workingNodes]
    #     workingLinks = [link for link in links if link['source'] in workingNodesIds and link['target'] in workingNodesIds]
    #     sugiyamaUtil(workingNodes, workingLinks)
    #===========================================================================
    
    responseString = json.dumps({"nodes" : nodes, "links" : links})
    response = Response(responseString, content_type='application/json')
    response.add_header("Access-Control-Allow-Origin", "*")
    response.add_header("Access-Control-Expose-Headers", "Access-Control-Allow-Origin")
    # response.add_header("Access-Control-Allow-Headers: Origin, X-Requested-With, Content-Type, Accept")
    return response

###########Sugiyama Hierarchy changes ends###############################

##Sugiyama method on nodes starts#####
def sugiyamaUtil(nodes, links):
    global base
    if len(nodes) > 0 :
        nodeIds = []
        
        V = []
        nodeIdIndexDict = {}
        index = 0;
        for node in nodes :
            V.append(Vertex(node['id']))
            nodeIdIndexDict[node['id']] = index
            index = index + 1 
            
        X = [(link['source'], link['target']) for link in links]
        E = [Edge(V[nodeIdIndexDict[v]], V[nodeIdIndexDict[w]]) for (v, w) in X]
        g = Graph(V, E)
        
        for v in V:
            v.view = defaultview()
        
        connectedComponents = len(g.C)
        for i in range(connectedComponents):
            connectedComponent = g.C[i]    
            sug = SugiyamaLayout(connectedComponent)
            sug.init_all()      
            sug.draw(10)
            
            localBase = -1
            for l in sug.layers:
                for n in l:
                    try : 
                        nodesDict[n.data]['x'] = n.view.xy[0]
                        if base == -1 : 
                            nodesDict[n.data]['y'] = n.view.xy[1]
                        else : 
                            nodesDict[n.data]['y'] = n.view.xy[1] + base
                        if localBase < nodesDict[n.data]['y'] :
                            localBase = nodesDict[n.data]['y']
                    except :
                        pass
                    
            base = localBase
        return nodes
    else :
        return None
         
##Sugiyama method on nodes ends#####



# ##Sugiyama method custom ranking begins
class CustomRankingSugiyamaLayout(SugiyamaLayout):
    

    def init_all(self, roots=None, inverted_edges=None, cons=False, initial_ranking=None):
        '''
        :param dict{vertex:int} initial_ranking:
            The initial ranking of each vertex if passed
        '''
        if initial_ranking is not None:
            self.initial_ranking = initial_ranking
            assert 0 in initial_ranking
            nblayers = max(initial_ranking.keys()) + 1
            self.layers = [Layer([]) for l in range(nblayers)]
            
        SugiyamaLayout.init_all(self, roots=roots, inverted_edges=inverted_edges, cons=cons)
        
    def _rank_init(self, unranked):
        assert self.dag
        
        if not hasattr(self, 'initial_ranking'):
            SugiyamaLayout._rank_init(self, unranked)
        else:
            for rank, vertices in sorted(self.initial_ranking.iteritems()):
                for v in vertices:
                    self.grx[v].rank = rank
                    # add it to its layer:
                    self.layers[rank].append(v)

@post("/custom_sugiyam_ranking")
def custom_sugiyam_ranking(request):
    global linksDict
    initialize()
    body = json.loads(request.body)
    nodes = body['nodes']
    links = body['links']
    generateLinksDictionary(links)
    print nodes
    print links
    nodeIds = []
    for node in nodes:
        nodeIds.append(node['id'])
        nodesDict[node['id']] = node
    # V = [Vertex(data) for data in range(10)]
    
    
    
    V = []
    nodeIdIndexDict = {}
    index = 0;
    rank_to_data = {0:[], 1:[], 2:[], 3:[], 4:[]}
    hierarchyDict = {'cloud':0, 'border router':1, 'distribution':2, 'access':3, 'host':4}
    for node in nodes :
        vertex = Vertex(node['id'])
        V.append(vertex)
        nodeIdIndexDict[node['id']] = index
        index = index + 1 
        role = node.get('role').lower()
        rank_to_data[hierarchyDict[role]].append(vertex)
        
    # X = [(link['source'], link['target']) for link in links]
    
    X = []
    hierarchy = ['cloud', 'border router', 'distribution', 'access', 'host']
    hierarchy.reverse()
    for link in links:
        sourceRole = nodesDict[link['source']]['role'].lower()
        targetRole = nodesDict[link['target']]['role'].lower()
        sourceRoleIndex = hierarchy.index(sourceRole)
        targetRoleIndex = hierarchy.index(targetRole)
        if sourceRoleIndex > targetRoleIndex :
            X.append((link['source'], link['target']))
        else :
            X.append((link['target'], link['source']))
            
            
    E = [Edge(V[nodeIdIndexDict[v]], V[nodeIdIndexDict[w]]) for (v, w) in X]
    
    # #
    
    
    #===========================================================================
    # rank_to_data = {0:[V[nodeIdIndexDict['1']]],
    #                 1:[V[nodeIdIndexDict['11']], V[nodeIdIndexDict['4']], V[nodeIdIndexDict['9']]],
    #                 2:[V[nodeIdIndexDict['5']], V[nodeIdIndexDict['12']], V[nodeIdIndexDict['13']]],
    #                 3:[V[nodeIdIndexDict['7']], V[nodeIdIndexDict['3']], V[nodeIdIndexDict['2']]],
    #                 4:[V[nodeIdIndexDict['8']], V[nodeIdIndexDict['15']], V[nodeIdIndexDict['14']]],
    #                 5:[V[nodeIdIndexDict['10']]],
    #                 6:[V[nodeIdIndexDict['6']]],
    #                 7:[V[nodeIdIndexDict['16']]]}
    # 
    # rank_to_data = {0:[V[nodeIdIndexDict['1']]],
    #                 1:[V[nodeIdIndexDict['11']], V[nodeIdIndexDict['4']]],
    #                 2:[V[nodeIdIndexDict['5']], V[nodeIdIndexDict['12']], V[nodeIdIndexDict['9']]],
    #                 3:[V[nodeIdIndexDict['7']]],
    #                 4:[V[nodeIdIndexDict['8']]],
    #                 5:[V[nodeIdIndexDict['10']], V[nodeIdIndexDict['2']]],
    #                 6:[V[nodeIdIndexDict['6']], V[nodeIdIndexDict['13']], V[nodeIdIndexDict['3']]],
    #                 7:[V[nodeIdIndexDict['14']], V[nodeIdIndexDict['15']], V[nodeIdIndexDict['16']]]}
    #===========================================================================
    # #
    
    #===========================================================================
    # X = [(0,1),(0,2),(1,3),(2,3),(4,0),(1,4),(4,5),(5,6),(3,6),(3,7),(6,8),(7,8),(8,9),(5,9)]
    # E = [Edge(V[v],V[w]) for (v,w) in X]
    #===========================================================================
    
    # Changes starts
    g = Graph(V, E)
    
    for v in V:
        v.view = defaultview()
        
    connectedComponent = g.C[0]
    sug = CustomRankingSugiyamaLayout(connectedComponent)
    sug.init_all()
    # sug.init_all(initial_ranking=rank_to_data)      
    sug.draw(10)
    cloudDevices = {}
    border_routerDevices = {}
    distributionDevices = {}
    accessDevices = {}
    hostDevices = {}
    workingDict = {}
    layerNumber = 0
    layers = {}
    for l in sug.layers:
        layer = []
        for n in l:
            if isinstance(n, DummyVertex) == False:
                nodeId = n.data
                role = nodesDict[n.data]['role'].lower()
                if role == 'cloud' : 
                    workingDict = cloudDevices
                elif role == 'border router' : 
                    workingDict = border_routerDevices
                elif role == 'distribution' : 
                    workingDict = distributionDevices
                elif role == 'access' :
                    workingDict = accessDevices 
                elif role == 'host' : 
                    workingDict = hostDevices
                    
                workingList = workingDict.get(layerNumber, [])
                workingList.append(nodeId)
                workingDict[layerNumber] = workingList
                layer.append(nodeId)
                nodesDict[n.data]['x'] = n.view.xy[0]
                nodesDict[n.data]['y'] = n.view.xy[1]
        layers[layerNumber] = layer
        layerNumber = layerNumber + 1
    
    
    # #assign Layers
    layerNumber = 0
    idLayerNumberDict = {}
    layerIdNumberDict = {}
    layerNumber = assignLayers(cloudDevices, layerNumber, idLayerNumberDict, layerIdNumberDict)
    layerNumber = assignLayers(border_routerDevices, layerNumber, idLayerNumberDict, layerIdNumberDict)
    layerNumber = assignLayers(distributionDevices, layerNumber, idLayerNumberDict, layerIdNumberDict)
    layerNumber = assignLayers(accessDevices, layerNumber, idLayerNumberDict, layerIdNumberDict)
    layerNumber = assignLayers(hostDevices, layerNumber, idLayerNumberDict, layerIdNumberDict)
    
    startEndRolesHierarchyList = getStartEndRolesHierarchyList(idLayerNumberDict)
    
    print cloudDevices
    print border_routerDevices
    print distributionDevices
    print accessDevices
    print hostDevices
    print layers
    
    print idLayerNumberDict
    print layerIdNumberDict
    rank_to_data = generate_rank_to_data(layerIdNumberDict, V, nodeIdIndexDict)
    layerIdNumberSetDict = generateLayerIdNumberSetDict(layerIdNumberDict)
    print layerIdNumberSetDict
    print startEndRolesHierarchyList
    
    
    for n in range(10) :
        result = makeLayers(layerIdNumberSetDict, layerIdNumberDict, idLayerNumberDict, startEndRolesHierarchyList)
        layerIdNumberSetDict = result['layerIdNumberSetDict']
        layerIdNumberDict = result['layerIdNumberDict']
        idLayerNumberDict = result['idLayerNumberDict']
        startEndRolesHierarchyList = result['startEndRolesHierarchyList']
    
    rank_to_data = generate_rank_to_data(layerIdNumberDict, V, nodeIdIndexDict)
    
    g = Graph(V, E)
    
    for v in V:
        v.view = VertexViewer(1, 1)
        
    connectedComponent = g.C[0]
    sug = CustomRankingSugiyamaLayout(connectedComponent)
    # sug.init_all()
    sug.init_all(initial_ranking=rank_to_data)      
    sug.draw(10)
    for l in sug.layers:
        layer = []
        for n in l:
            if isinstance(n, DummyVertex) == False:
                nodesDict[n.data]['x'] = n.view.xy[0]
                nodesDict[n.data]['y'] = n.view.xy[1] 
    # Changes ends
    
    responseString = json.dumps({"nodes" : nodes, "links" : links})
    response = Response(responseString, content_type='application/json')
    response.add_header("Access-Control-Allow-Origin", "*")
    response.add_header("Access-Control-Expose-Headers", "Access-Control-Allow-Origin")
    # response.add_header("Access-Control-Allow-Headers: Origin, X-Requested-With, Content-Type, Accept")
    return response 

def makeLayers(layerIdNumberSetDict, layerIdNumberDict, idLayerNumberDict, startEndRolesHierarchyList):
    layers = sorted(layerIdNumberSetDict.keys(), reverse=True)
    
    index = 0
    length = len(layers)
    while index < len(layers) - 1 :
        layerNumber = layers[index]
        layerNodesSet = layerIdNumberSetDict[layerNumber]
        layerNodes = layerIdNumberDict[layerNumber]
        upperLayerNodesSet = layerIdNumberSetDict[layerNumber - 1]
        j = 0
        while j < len(layerNodes)  :
            nodeMoved = False
            layerNode = layerNodes[j]
            connectedNodes = linksDict[layerNode]
            levelMoved = False
            if len(upperLayerNodesSet.intersection(connectedNodes)) == 0 :
                role = nodesDict[layerNode]['role'].lower()
                startEndRolesHierarchyListDummy = isMovePossible(idLayerNumberDict, startEndRolesHierarchyList, layerNumber, role, layerNode) 
                if startEndRolesHierarchyListDummy is not None:
                    nodeMoved = True
                    startEndRolesHierarchyList = startEndRolesHierarchyListDummy
                    # remove from idLayerNumberDict
                    # Already done in isMovePossible
                    # idLayerNumberDict[layerNode] = idLayerNumberDict[layerNode] - 1
                    
                    # remove from layerIdNumberDict
                    layerIdNumberDict[layerNumber].remove(layerNode)
                    layerIdNumberDict[layerNumber - 1].append(layerNode)
                    if len(layerIdNumberDict[layerNumber]) == 0 :
                        levelMoved = True
                        i = layerNumber
                        while i < len(layerIdNumberDict) - 1 :
                            layerIdNumberDict[i] = layerIdNumberDict[i + 1]
                            for n in layerIdNumberDict[i] : 
                                idLayerNumberDict[n] = idLayerNumberDict[n] - 1
                            startEndRolesHierarchyList = getStartEndRolesHierarchyList(idLayerNumberDict)
                            i = i + 1
                        del layerIdNumberDict[i]
                    
                    # remove from layerIdNumberSetDict    
                    layerIdNumberSetDict[layerNumber].remove(layerNode)
                    layerIdNumberSetDict[layerNumber - 1].add(layerNode)
                    if len(layerIdNumberSetDict[layerNumber]) == 0 :
                        levelMoved = True
                        i = layerNumber
                        while i < len(layerIdNumberSetDict) - 1 :
                            layerIdNumberSetDict[i] = layerIdNumberSetDict[i + 1]
                            i = i + 1
                        del layerIdNumberSetDict[i]
            if nodeMoved == False : 
                j = j + 1
        
        if levelMoved == False : 
            index = index + 1
        layers = sorted(layerIdNumberSetDict.keys(), reverse=True) 
        
    return {"layerIdNumberSetDict" : layerIdNumberSetDict, "layerIdNumberDict" : layerIdNumberDict, "idLayerNumberDict" : idLayerNumberDict, "startEndRolesHierarchyList" :startEndRolesHierarchyList}

def getStartEndRolesHierarchyList(idLayerNumberDict):
    startEndRolesHierarchyList = [[None, None], [None, None], [None, None], [None, None], [None, None]]
    for nodeId, lNumber in idLayerNumberDict.items() : 
        node = nodesDict[nodeId]
        roleNode = node['role'].lower()
        ind = hierarchy[roleNode]
        if startEndRolesHierarchyList[ind][0] is None : 
            startEndRolesHierarchyList[ind][0] = lNumber
            startEndRolesHierarchyList[ind][1] = lNumber
        elif lNumber > startEndRolesHierarchyList[ind][1] : 
            startEndRolesHierarchyList[ind][1] = lNumber
        elif lNumber < startEndRolesHierarchyList[ind][0] : 
            startEndRolesHierarchyList[ind][0] = lNumber
        
    return startEndRolesHierarchyList

def isMovePossible(idLayerNumberDict, startEndRolesHierarchyList, layerNumber, role, layerNode) : 
    moveToLayerNumber = layerNumber - 1
    index = hierarchy[role]
    sourceStart = startEndRolesHierarchyList[index][0]
    sourceEnd = startEndRolesHierarchyList[index][1]
    destinationStart = startEndRolesHierarchyList[index - 1][0]
    destinationEnd = startEndRolesHierarchyList[index - 1][1]
    
    if moveToLayerNumber < destinationEnd : 
        return None
    else : 
        idLayerNumberDict[layerNode] = idLayerNumberDict[layerNode] - 1 
        startEndRolesHierarchyList = getStartEndRolesHierarchyList(idLayerNumberDict)
        return startEndRolesHierarchyList  


def generateLayerIdNumberSetDict(layerIdNumberDict):
    layerIdNumberSetDict = {}
    for layerNumber, layerNodes in layerIdNumberDict.items():
        layerIdNumberSetDict[layerNumber] = Set(iterable=layerNodes)
    return layerIdNumberSetDict

def generate_rank_to_data(layerIdNumberDict, V, nodeIdIndexDict):
    rank_to_data = {}
    for layerNumber, layerItems in layerIdNumberDict.items():
        itemList = [V[nodeIdIndexDict[item]] for item in layerItems]
        rank_to_data[layerNumber] = itemList
    return rank_to_data    
    

def assignLayers(hierarchyDict, layerNumber, idLayerNumberDict, layerIdNumberDict):
    keys = sorted(hierarchyDict.keys())
    for key in keys:
        nodes = hierarchyDict[key]
        for node in nodes :
            idLayerNumberDict[node] = layerNumber
            if layerIdNumberDict.get(layerNumber) is None:
                layerIdNumberDict[layerNumber] = [node]
            else :
                layerIdNumberDict.get(layerNumber).append(node)
        layerNumber = layerNumber + 1
    return layerNumber

# ##Sugiyama method custom ranking ends
        
        
        
if __name__ == '__main__':
    run_itty()

