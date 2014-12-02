import json
import urllib
import urllib2



root_url = "http://localhost:8080/post"
values = {}

if __name__ == '__main__':
    inputDir = "/Users/ronaklakhwani/Desktop/comparision/sampleData/data/"
    outputDir = "/Users/ronaklakhwani/Desktop/comparision/clusteringData/data/"
    fileName = "testData.json"
    
    
    
    # generateData(numberOfNodes, numberOfLinks, width, height, inputDir + fileName)
    
    with open(inputDir + fileName) as f:
        catalog = json.load(f)
        
    nodes = catalog['nodes']  
    links = catalog['links']
    #nodes = json.dumps(nodes) 
    
    values['nodes'] = json.dumps(nodes)
    values['links'] = json.dumps(links)
    
    data = urllib.urlencode(values)
    req = urllib2.Request(root_url,data)
    u = urllib2.urlopen(req)
    info = json.loads(u.read())
    print info["nodes"]
    print info["links"]
    print info["nodeSet"]
    