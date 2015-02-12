import json

if __name__ == '__main__':
    with open("/Users/ronaklakhwani/Desktop/comparision/sampleData/data/simple_apicem_layout.json") as f:
        catalog = json.load(f)
    
    originalNodes = catalog['nodes']  
    originalLinks = catalog['links']
    
    nodes = [{'id' : node['id'],'x' : node['x'],'y' : node['y'],'role' : node['role']} for node in originalNodes]
    
    links = []
    linkId = 0;
    for link in originalLinks:
        generatedLink = {};
        source = link['source']
        target = link['target']
        generatedLink['source'] = source
        generatedLink['target'] = target
        generatedLink['id'] = linkId
        linkId = linkId + 1
        links.append(generatedLink)
        
    with open("/Users/ronaklakhwani/Desktop/comparision/sampleData/data/generated_simple_apicem_layout.json","w+") as f:
        #f = open("Users/ronaklakhwani/Desktop/comparision/sampleData/data/generated_data_800.json","w+")
        fileData = {'nodes' : nodes, 'links' : links} 
        json.dump(fileData, f, True)