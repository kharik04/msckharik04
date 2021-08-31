import json
import time
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import isodate
import cvxpy as cp
from functools import reduce
from scipy.sparse import vstack, coo_matrix, csc_matrix


file_names = ['sample_scenario.json','01_dummy.json', '02_a_little_less_dummy.json',  '03_FWA_0.125.json', '04_V1.02_FWA_without_obstruction.json', '05_V1.02_FWA_with_obstruction.json', '06_V1.20_FWA.json', '07_V1.22_FWA.json', '08_V1.30_FWA.json', '09_ZUE-ZG-CH_0600-1200.json']

for fn in file_names:

    scenario = 'problem_instances/'+ fn  # adjust path to the sample instance if it is not located there
    with open(scenario) as fp:
        scenario = json.load(fp)

    service_intentions = {}
    for si in scenario['service_intentions']:
        requirements = {}
        for i,req in enumerate(si['section_requirements']):
            requirements[req['section_marker']] = {key:value for key,value in req.items()}
            if i == 0:
                requirements[req['section_marker']]['start_end'] = 'start'
            elif i == len(si['section_requirements']) - 1:
                requirements[req['section_marker']]['start_end'] = 'end'
            else: 
                requirements[req['section_marker']]['start_end'] = None
        service_intentions[si['id']] = requirements

    #get resources
    resources = {}
    for resource in scenario['resources']:
        idx = resource['id']
        if idx in resources:
            print('something is wrong')
        resources[idx] = resource['release_time']

    def from_node_id(route_path, route_section, index_in_path):
        if "route_alternative_marker_at_entry" in route_section.keys() and \
                route_section["route_alternative_marker_at_entry"] is not None and \
                len(route_section["route_alternative_marker_at_entry"]) > 0:
                    return "(" + str(route_section["route_alternative_marker_at_entry"][0]) + ")"
        else:
            if index_in_path == 0:  # can only get here if this node is a very beginning of a route
                return "(" + str(route_section["sequence_number"]) + "_beginning)"
            else:
                return "(" + (str(route_path["route_sections"][index_in_path - 1]["sequence_number"]) + "->" +
                            str(route_section["sequence_number"])) + ")"
    def to_node_id(route_path, route_section, index_in_path):
        if "route_alternative_marker_at_exit" in route_section.keys() and \
                route_section["route_alternative_marker_at_exit"] is not None and \
                len(route_section["route_alternative_marker_at_exit"]) > 0:

                    return "(" + str(route_section["route_alternative_marker_at_exit"][0]) + ")"
        else:
            if index_in_path == (len(route_path["route_sections"]) - 1): # meaning this node is a very end of a route
                return "(" + str(route_section["sequence_number"]) + "_end" + ")"
            else:
                return "(" + (str(route_section["sequence_number"]) + "->" +
                            str(route_path["route_sections"][index_in_path + 1]["sequence_number"])) + ")"


    def from_node_id(route_path, route_section, index_in_path):
        if "route_alternative_marker_at_entry" in route_section.keys() and \
                route_section["route_alternative_marker_at_entry"] is not None and \
                len(route_section["route_alternative_marker_at_entry"]) > 0:
                    return "(" + str(route_section["route_alternative_marker_at_entry"][0]) + ")"
        else:
            if index_in_path == 0:  # can only get here if this node is a very beginning of a route
                return "(" + str(route_section["sequence_number"]) + "_beginning)"
            else:
                return "(" + (str(route_path["route_sections"][index_in_path - 1]["sequence_number"]) + "->" +
                            str(route_section["sequence_number"])) + ")"
    def to_node_id(route_path, route_section, index_in_path):
        if "route_alternative_marker_at_exit" in route_section.keys() and \
                route_section["route_alternative_marker_at_exit"] is not None and \
                len(route_section["route_alternative_marker_at_exit"]) > 0:

                    return "(" + str(route_section["route_alternative_marker_at_exit"][0]) + ")"
        else:
            if index_in_path == (len(route_path["route_sections"]) - 1): # meaning this node is a very end of a route
                return "(" + str(route_section["sequence_number"]) + "_end" + ")"
            else:
                return "(" + (str(route_section["sequence_number"]) + "->" +
                            str(route_path["route_sections"][index_in_path + 1]["sequence_number"])) + ")"



    def from_node_id(route_path, route_section, index_in_path):
        if "route_alternative_marker_at_entry" in route_section.keys() and \
                route_section["route_alternative_marker_at_entry"] is not None and \
                len(route_section["route_alternative_marker_at_entry"]) > 0:
                    return "(" + str(route_section["route_alternative_marker_at_entry"][0]) + ")"
        else:
            if index_in_path == 0:  # can only get here if this node is a very beginning of a route
                return "(" + str(route_section["sequence_number"]) + "_beginning)"
            else:
                return "(" + (str(route_path["route_sections"][index_in_path - 1]["sequence_number"]) + "->" +
                            str(route_section["sequence_number"])) + ")"
    def to_node_id(route_path, route_section, index_in_path):
        if "route_alternative_marker_at_exit" in route_section.keys() and \
                route_section["route_alternative_marker_at_exit"] is not None and \
                len(route_section["route_alternative_marker_at_exit"]) > 0:

                    return "(" + str(route_section["route_alternative_marker_at_exit"][0]) + ")"
        else:
            if index_in_path == (len(route_path["route_sections"]) - 1): # meaning this node is a very end of a route
                return "(" + str(route_section["sequence_number"]) + "_end" + ")"
            else:
                return "(" + (str(route_section["sequence_number"]) + "->" +
                            str(route_path["route_sections"][index_in_path + 1]["sequence_number"])) + ")"


    route_section_keys = ['penalty', 'minimum_running_time']
    section_requirement_keys = ['start_end','min_stopping_time', 'entry_earliest', 'entry_latest',
                            'exit_earliest', 'exit_latest', 'entry_delay_weight',
                            'exit_delay_weight']

    start_time = time.time()

    # now build the graph. Nodes are called "previous_FAB -> next_FAB" within lineare abschnittsfolgen and "AK" if
    # there is an Abschnittskennzeichen 'AK' on it
    route_graphs = dict()
    for route in scenario["routes"]:# iterates over si?
        

        # set global graph settings
        G = nx.DiGraph(route_id = route["id"], name="Route-Graph for route "+str(route["id"]))

        # add edges with data contained in the preprocessed graph
        for path in route["route_paths"]:#iterate over admissible routes
            for (i, route_section) in enumerate(path["route_sections"]):
                sn = route_section['sequence_number']
                
                edge_data = {}
                for key in route_section_keys:
                    if key in route_section:
                        edge_data[key] = route_section[key]
                    else:
                        edge_data[key] = None
                    
                if 'resource_occupations' in route_section:
                    resource_occupations = {}
                    for resource in route_section['resource_occupations']:
                        idx = resource['resource']
                        R = resources[idx]
                        resource_occupations[idx] = R
                        #maximum? code below
                        R=[isodate.parse_duration(value).seconds for key,value in resource_occupations.items()]
                edge_data['resource_occupations'] = resource_occupations
                edge_data['R'] = max(R)
                
                #get section marker
                section_marker = None
                if 'section_marker' in route_section:
                    try:
                        section_marker = route_section['section_marker'][0]
                    except:
                        pass
                edge_data['section_marker'] = section_marker
                
                
                
                if section_marker in service_intentions[route['id']]:
                    for key in section_requirement_keys:
                        if key in service_intentions[route['id']][section_marker]:
                                edge_data[key] = service_intentions[route['id']][section_marker][key]
            
                edge_data['sequence_number'] = sn
                
                G.add_edge(from_node_id(path, route_section, i),
                        to_node_id(path, route_section, i),
                        edge_data = edge_data)

        route_graphs[route["id"]] = G

    G = None
    for route in route_graphs:
        if G:
            G = nx.compose(G, route_graphs[route])
        else:
            G =  route_graphs[route]
    pos = nx.spring_layout(G, scale=20)#, k=3/np.sqrt(G.order()))
    nx.draw(G, pos=pos, with_labels=False)#, k=13.8, node_color='lightgreen', node_size=800)

    print(fn)
    b = len(G.edges)/len(G.nodes)
    sumoutsquared = 0
    for node in G.nodes:
        curr = 0
        for j in G.successors(node):
            curr+=1
        outsqared = curr**2
        sumoutsquared += outsqared
    print(sumoutsquared/len(G.nodes) - b**2)