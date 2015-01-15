import json

if __name__ == '__main__':
    with open("/Users/ronaklakhwani/Desktop/comparision/sampleData/data/data_800.json") as f:
        catalog = json.load(f)
    
    originalNodes = catalog['nodes']  
    originalLinks = catalog['links']
    
    links = []
    linkId = 0;
    for link in originalLinks:
        generatedLink = {};
        source = link['source']
        target = link['target']
        generatedLink['source'] = source['id']
        generatedLink['target'] = target['id']
        generatedLink['id'] = linkId
        linkId = linkId + 1
        links.append(generatedLink)
        
    with open("/Users/ronaklakhwani/Desktop/comparision/sampleData/data/generated_data_800.json","w+") as f:
        #f = open("Users/ronaklakhwani/Desktop/comparision/sampleData/data/generated_data_800.json","w+")
        fileData = {'nodes' : originalNodes, 'links' : links} 
        json.dump(fileData, f, True)