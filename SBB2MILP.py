import json
import time
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import isodate
import cvxpy as cp
from functools import reduce
from scipy.sparse import vstack, coo_matrix, csc_matrix
import itertools
import seaborn as sns


from time import strftime
from time import gmtime



def str_to_sec(s):
    s=s.split(':')
    return int(s[0])*60*60+int(s[1])*60+int(s[2])
def sec_to_str(x):
    return strftime("%H:%M:%S", gmtime(x))


def load_problem(scenario):

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

    route_section_keys = ['penalty', 'minimum_running_time']
    section_requirement_keys = ['start_end','min_stopping_time', 'entry_earliest', 'entry_latest',
                            'exit_earliest', 'exit_latest', 'entry_delay_weight',
                            'exit_delay_weight']


    start_time = time.time()

    # now build the graph. Nodes are called "previous_FAB -> next_FAB" within lineare abschnittsfolgen and "AK" if
    # there is an Abschnittskennzeichen 'AK' on it
    route_graphs = dict()
    for route in scenario["routes"]:# iterates over si?
        
        print(f"Constructing graph for service intention {route['id']}")
        # set global graph settings
        G = nx.DiGraph(route_id = route["id"], name="Route-Graph for route "+str(route["id"]))

        # add edges with data contained in the preprocessed graph
        for path in route["route_paths"]:#iterate over admissible routes
            #print('new route')
            for (i, route_section) in enumerate(path["route_sections"]):
                sn = route_section['sequence_number']
                #print("Adding Edge from {} to {} with sequence number {}".format(from_node_id(path, route_section, i), to_node_id(path, route_section, i), sn))
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

    Lats = {}
    for si in route_graphs:
        Latins = {}
        Latouts = {}
        for e in route_graphs[si].edges(data=True):
            if 'entry_latest' in e[2]['edge_data']:
                Latins[f'{e[0]},{e[1]}'] = e[2]['edge_data']['entry_latest']
            if 'exit_latest' in e[2]['edge_data']:
                Latouts[f'{e[0]},{e[1]}'] = e[2]['edge_data']['exit_latest']
            Lats[si] = {'LatIn': None, 'LatOut':None}
            Lats[si]['LatIn'] = Latins
            Lats[si]['LatOut'] = Latouts


    paths = {}
    for si in service_intentions:
        startnodes=[]
        endnodes = []
        G = route_graphs[si]
        for edge in G.edges(data=True):
            if 'start_end' in edge[2]['edge_data']:
                if edge[2]['edge_data']['start_end'] == 'start':
                    startnodes.append(edge[0])
                if edge[2]['edge_data']['start_end'] == 'end':
                    endnodes.append(edge[1])
        paths_si = []
        for s in startnodes:
            for e in endnodes:
                for path in nx.all_simple_paths(G, source=s, target=e):
                    paths_si.append([(path[i], path[i+1]) for i in range(len(path)-1)])
        paths[si] = paths_si

    si_list = []
    t_index_by_edge = {}
    delta_index_by_edge = {}
    edges_by_path = {}
    get_index_by_delta = {}
    j=0
    k=0
    for si in paths:
        pathlist = []
        ttemp = {}
        dtemp = {}
        for i,path in enumerate(paths[si]):
            srlist = []
            etemp = {}
            for sr in path:
                srlist.append((si,i,sr))
                if sr in ttemp:
                    ttemp[sr].append(j)
                else:
                    ttemp[sr] = [j]
                
                
                if sr in dtemp:
                    dtemp[sr].append(k)
                else:
                    dtemp[sr] = [k]
                
                etemp[sr] = j
                if k in get_index_by_delta:
                    get_index_by_delta[k].append(j)
                else:
                    get_index_by_delta[k] = [j]
                j+=1
            edges_by_path[k] = etemp
                
            k+=1
            pathlist.append(srlist)
        t_index_by_edge[si] = ttemp
        delta_index_by_edge[si] = dtemp
        si_list.append(pathlist)


    xindex = {}
    enum = 0
    for si in service_intentions:
        xindexsi = {}
        for i,e in enumerate(route_graphs[si].edges):
            xindexsi[e] = i+enum
        enum+=i+1
        xindex[si] = xindexsi


    betaindex = []
    for (si1, si2) in list(itertools.combinations([key for key in service_intentions],2)):
        intsc = list(set.intersection(set(route_graphs[si1].edges), set(route_graphs[si2].edges)))
        for e in intsc:
            betaindex.append((si1, si2, e))


    def recursive_len(item):
        if type(item) == list:
            return sum(recursive_len(subitem) for subitem in item)
        else:
            return 1

    t_len = recursive_len(si_list)
    x_len = 0
    delta_len = 0
    for si in service_intentions:
        x_len += len(route_graphs[si].edges)
        delta_len += len(paths[si])
    beta_len = len(betaindex)
    total_length = t_len*2 + x_len + delta_len + beta_len


    TL={}
    i = 0
    for si, si_id in zip(si_list, service_intentions):
        temp = {} 
        temp['t'] = recursive_len(si)
        temp['x'] = len(route_graphs[si_id].edges)
        temp['delta'] = len(paths[si_id])
        slacknum = 0
        for r in si:
            for rs in r:
                edge_data = route_graphs[si_id].edges[rs[2]]['edge_data']
                if 'entry_delay_weight' in edge_data and 'entry_latest' in edge_data:
                    slacknum+=1
                if 'exit_delay_weight' in edge_data and 'exit_latest' in edge_data:
                    slacknum+=1
                #if 'entry_earliest' in edge_data:
                #    slacknum+=1
                #if 'exit_earliest' in edge_data:
                #    slacknum+=1
                i+=1
        temp['slack'] = slacknum
        TL[si_id] = temp
    slack_len = sum([TL[si]['slack'] for si in service_intentions])
    total_length+=slack_len







    Mmrtmst = [1]
    MEarOut = [1]
    MEarIn = [1]
    for si in route_graphs:
        for e in route_graphs[si].edges(data=True):
            edge_data = e[2]['edge_data']
            if ('exit_earliest' in edge_data):
                EarOut = str_to_sec(edge_data['exit_earliest'])
                MEarOut.append(EarOut)
            if ('entry_earliest' in edge_data):
                EarIn = str_to_sec(edge_data['entry_earliest'])
                MEarIn.append(EarIn)
            if ('min_stopping_time' in edge_data) or ('minimum_running_time' in edge_data):
                mst = 0
                mrt = 0
                if 'min_stopping_time' in edge_data:
                    mst = isodate.parse_duration(edge_data['min_stopping_time']).seconds
                if 'minimum_running_time' in edge_data:
                    mrt = isodate.parse_duration(edge_data['minimum_running_time']).seconds
                Mmrtmst.append(mrt+mst)


    ub_numCons_nor = recursive_len(si_list)*11
    ub_numCons_coup =  4*sum([len(delta_index_by_edge[si1][e])*len(delta_index_by_edge[si2][e]) for (si1, si2, e) in betaindex])

    c = np.zeros(total_length)
    testv = [None]*total_length

    indexer = {}

    #A = lil_matrix((ub_numCons_nor, total_length))
    row = []
    col = []
    data = []
    b = np.zeros(ub_numCons_nor)
    cnum = 0
    cnums = []
    currlen = 0
    bool_idx = []
    for si, si_id in zip(si_list, service_intentions):
        tempindexer = {'height_loc': [cnum, None], 'width_loc':[currlen, None]}
        tempsum = 0
        j=0
        coltemp = []
        datatemp = []
        i =0
        s=0
        for r, path in zip(si, paths[si_id]):
            for enum, rs in enumerate(r):
                edge_data = route_graphs[si_id].edges[rs[2]]['edge_data']
                #slack
                tin_idx = currlen + i + TL[si_id]['slack'] 
                tout_idx = currlen + i + TL[si_id]['slack'] + TL[si_id]['t']
                delta_idx = currlen + j + TL[si_id]['slack'] + 2*TL[si_id]['t']
                ad_x = 0
                for t in service_intentions:
                    if t==si_id:
                        break
                    ad_x += TL[t]['x']
                x_idx = currlen + xindex[si_id][rs[2]] + TL[si_id]['slack'] + 2*TL[si_id]['t'] + TL[si_id]['delta'] -ad_x
                
                
                if 'entry_delay_weight' in edge_data and 'entry_latest' in edge_data:
                    b[cnum] = float(edge_data['entry_delay_weight'])*str_to_sec(edge_data['entry_latest'])
                    row += [cnum, cnum]
                    col += [currlen + s, tin_idx]
                    data += [-1, float(edge_data['entry_delay_weight'])]
                    cnum +=1
                    b[cnum] = 0
                    row.append(cnum)
                    col.append(currlen + s)
                    data.append(-1)
                    cnum +=1
                    
                    c[currlen + s] = 1

                    testv[currlen + s] = f'slack_{s}_{si_id}'
                    s+=1
                    
                if 'exit_delay_weight' in edge_data and 'exit_latest' in edge_data:
                    b[cnum] = float(edge_data['exit_delay_weight'])*str_to_sec(edge_data['exit_latest'])
                    row += [cnum, cnum]
                    col += [currlen + s, tout_idx]
                    data += [-1, float(edge_data['exit_delay_weight'])]
                    cnum+=1
                    
                    b[cnum] = 0
                    row.append(cnum)
                    col.append(currlen + s)
                    data.append(-1)
                    cnum += 1
                    
                    c[currlen + s] = 1
                    testv[currlen + s] = f'slack_{s}_{si_id}'
                    s+=1

                
                b[cnum] = 0
                row.append(cnum)
                col.append(tin_idx)
                data.append(-1)
                cnum+=1
                testv[tin_idx] = f'tin_{i}_{si_id}'
                
                b[cnum] = 0
                row.append(cnum)
                col.append(tout_idx)
                data.append(-1)
                cnum += 1
                testv[tout_idx] = f'tout_{i}_{si_id}'
                
                #(1)
                row += [cnum, cnum]
                col += [tin_idx, tout_idx]
                data += [1,-1]
                b[cnum] = 0
                cnum += 1
                
                # (2)
                if enum!=len(r)-1:
                    b[cnum] = 0
                    row+=[cnum, cnum]
                    col+=[tin_idx+1, tout_idx]
                    data += [1,-1]
                    cnum += 1
                    
                    row+=[cnum, cnum]
                    col+=[tin_idx+1, tout_idx]
                    data += [-1,1]
                    b[cnum] = 0
                    cnum+=1

                #(3)
                #M=10000000
                M=M=10000000##max(Mmrtmst)
                if ('min_stopping_time' in edge_data) or ('minimum_running_time' in edge_data):
                    mst = 0
                    mrt = 0
                    if 'min_stopping_time' in edge_data:
                        mst = isodate.parse_duration(edge_data['min_stopping_time']).seconds
                    if 'minimum_running_time' in edge_data:
                        mrt = isodate.parse_duration(edge_data['minimum_running_time']).seconds

                    b[cnum] = M-mrt-mst
                    row += [cnum, cnum, cnum]
                    col+=[tin_idx, tout_idx, delta_idx]
                    data+=[1,-1,M]
                    cnum += 1

                # (4)
                #M=max(MEarIn)
                if ('entry_earliest' in edge_data):
                    EarIn = str_to_sec(edge_data['entry_earliest'])
                    
                    b[cnum] = M-EarIn
                    row += [cnum, cnum]
                    col += [tin_idx, delta_idx]
                    data += [-1, M]
                    cnum += 1

                #(5)
                #M=max(MEarOut)    
                if ('exit_earliest' in edge_data):
                    EarOut = str_to_sec(edge_data['exit_earliest'])
                    
                    row +=[cnum, cnum]
                    col += [tout_idx, delta_idx]
                    data += [-1, M]
                    b[cnum] = M-EarOut
                    cnum += 1
                # (7)
                b[cnum] = 0
                row += [cnum, cnum]
                col += [delta_idx, x_idx]
                data+= [1,-1]
                cnum += 1
                
                bool_idx.append(delta_idx)
                bool_idx.append(x_idx)
                testv[x_idx] = f'x_{xindex[si_id][rs[2]] - ad_x}_{si_id}'
                testv[delta_idx] = f'delta_{j}_{si_id}'
                
                i+=1
            # (6)
            coltemp.append(delta_idx)
            datatemp.append(1)
            j+=1
        
        tempb = np.array([1])
        b = np.hstack((b[:cnum],tempb,b[cnum:]))
        row += [cnum]*len(coltemp)
        col+= coltemp
        data+=datatemp
        cnum += 1
        
        row += [cnum]*len(coltemp)
        datatemp = [-1*x for x in datatemp]
        col+=coltemp
        data+=datatemp
        tempb = -1*tempb.copy()
        b = np.hstack((b[:cnum],tempb,b[cnum:]))
        cnum += 1
        if len(cnums)==0:
            cnums.append(cnum)
        else:
            cnums.append(cnum - cnums[-1])
        currlen += TL[si_id]['slack'] + 2*TL[si_id]['t'] + TL[si_id]['delta'] + TL[si_id]['x']
        tempindexer['height_loc'][1] = cnum
        tempindexer['width_loc'][1] = currlen
        indexer[si_id] = tempindexer 

    b = b[:cnum]
    A = coo_matrix((data, (row, col)), shape=(cnum, total_length))


    #objective
    currlen = 0    
    for si, si_id in zip(si_list, service_intentions):
        for edge in route_graphs[si_id].edges(data=True):
            ad_x = 0
            for t in service_intentions:
                if t==si_id:
                    break
                ad_x += TL[t]['x']
            e = (edge[0],edge[1])
            x_idx = currlen + xindex[si_id][e] + TL[si_id]['slack'] + 2*TL[si_id]['t'] + TL[si_id]['delta'] - ad_x
            p = 0
            if 'penalty' in edge[2]['edge_data']:
                p = edge[2]['edge_data']['penalty']
                if p == None:
                    p = 0
                else:
                    p = float(p)
                c[x_idx]= p
            i+=1   


    #coupling
    cnum = 0
    #A_coup = csr_matrix((ub_numCons_coup, total_length))
    b_coup = np.zeros(ub_numCons_coup)
    row, col, data = [], [], []

    i = 0
    eps = 1
    M = 100000
    R=30
    prevsi1, prevsi2, _ = betaindex[0]
    LenSI = total_length - beta_len
    indexer_coup = {}
    tempindexer = {'height': [cnum, None], 'width': [LenSI + i, None]}
    beta_idx = []
    for (si1, si2, e) in betaindex:
        if prevsi1 != si1 or prevsi2 != si2:
            tempindexer['height'][1] = cnum
            tempindexer['width'][1] = LenSI + i
            indexer_coup[str(prevsi1)+','+str(prevsi2)] = tempindexer
            tempindexer = {'height': [cnum, None], 'width': [LenSI + i, None]}
            prevsi1 = si1
            prevsi2 = si2
            
        deltaidx1 = delta_index_by_edge[si1][e]
        deltaidx2 = delta_index_by_edge[si2][e]
        
        prevsi = []
        for siprevel in [si for si in service_intentions]:
            if siprevel==si1:
                break
            prevsi.append(siprevel)
        adjuster1 = sum([TL[siprevel]['t'] for siprevel in prevsi])
        
        prevsi = []
        for siprevel in [si for si in service_intentions]:
            if siprevel==si2:
                break
            prevsi.append(siprevel)
        adjuster2 = sum([TL[siprevel]['t'] for siprevel in prevsi])
        
        prevsi = []
        for siprevel in [si for si in service_intentions]:
            if siprevel==si1:
                break
            prevsi.append(siprevel)
        adjuster1d = sum([TL[siprevel]['delta'] for siprevel in prevsi])
        
        prevsi = []
        for siprevel in [si for si in service_intentions]:
            if siprevel==si2:
                break
            prevsi.append(siprevel)
        adjuster2d = sum([TL[siprevel]['delta'] for siprevel in prevsi])
        
        sumtil1 = 0
        prevsi = []
        for siprevel in [si for si in service_intentions]:
            if siprevel==si1:
                break
            prevsi.append(siprevel)
        sumtil1 = sum([TL[siprevel]['slack'] + 2*TL[siprevel]['t'] + TL[siprevel]['delta'] + TL[siprevel]['x'] for siprevel in prevsi])

        
        sumtil2 = 0
        prevsi = []
        for siprevel in [si for si in service_intentions]:
            if siprevel==si2:
                break
            prevsi.append(siprevel)
        sumtil2 = sum([TL[siprevel]['slack'] + 2*TL[siprevel]['t'] + TL[siprevel]['delta'] + TL[siprevel]['x'] for siprevel in prevsi])

        for idx1 in deltaidx1:
            for idx2 in deltaidx2:
                tin1_idx = edges_by_path[idx1][e] - adjuster1 + sumtil1 + TL[si1]['slack']
                tin2_idx = edges_by_path[idx2][e] - adjuster2 + sumtil2 + TL[si2]['slack']
                tout1_idx = edges_by_path[idx1][e] - adjuster1 + sumtil1 + TL[si1]['slack'] + TL[si1]['t']
                tout2_idx = edges_by_path[idx2][e] - adjuster2 + sumtil2 + TL[si2]['slack'] + TL[si2]['t']
                delta1_idx = idx1+TL[si1]['slack']+2*TL[si1]['t'] - adjuster1d + sumtil1
                delta2_idx = idx2+TL[si2]['slack']+2*TL[si2]['t'] - adjuster2d + sumtil2
                
                #print(f'tin1 {tin1_idx}, tin2 {tin2_idx}, delta1 {delta1_idx}, delta2 {delta2_idx}, beta {LenSI+i}')
                row+=[cnum,cnum, cnum, cnum, cnum]
                col+=[tin1_idx, tin2_idx, delta1_idx, delta2_idx,LenSI+i]
                data += [1,-1,M,M,M]
                b_coup[cnum]=3*M
                cnum += 1
                
                
                row+=[cnum,cnum, cnum, cnum, cnum]
                col+=[tin1_idx, tin2_idx, delta1_idx, delta2_idx,LenSI+i]
                data += [-1,1,M,M,-M]
                b_coup[cnum]=2*M-eps
                cnum += 1
                
                
                row+=[cnum,cnum, cnum, cnum, cnum]
                col+=[tin2_idx, tout1_idx, delta1_idx, delta2_idx,LenSI+i]
                data += [-1,1,M,M,M]
                b_coup[cnum]=3*M-R
                cnum += 1
                
                
                row+=[cnum,cnum, cnum, cnum, cnum]
                col+=[tin1_idx, tout2_idx, delta1_idx, delta2_idx,LenSI+i]
                data += [-1,1,M,M,-M]
                b_coup[cnum]=2*M-R
                cnum += 1
        testv[LenSI+i] = f'beta_{i}'       
        beta_idx.append(LenSI+i)
        i+=1

    bool_idx =list(set(bool_idx))
    tempindexer['height'][1] = cnum
    tempindexer['width'][1] = LenSI + i
    indexer_coup[str(si1) + ',' + str(si2)] = tempindexer
    A_coup = coo_matrix((data, (row, col)), shape=(cnum, total_length))
    b_coup = b_coup[:cnum].flatten()

    return A, b, A_coup, b_coup, c, indexer,indexer_coup, bool_idx, beta_idx, service_intentions, TL, paths, get_index_by_delta, Lats





def display_solution(x,service_intentions, TL, paths, get_index_by_delta, Lats, fsize = 1, labelsize = 200, text = False):
    sns.set_style("ticks")
    sns.set_style("whitegrid")
    x = list(x)
    slack, tin, tout, deltas, xs, betas = [],[],[],[],[],[]
    currlen = 0
    for si_id in service_intentions:
        tin += x[currlen + TL[si_id]['slack']:currlen + TL[si_id]['slack'] + TL[si_id]['t']]
        tout += x[currlen + TL[si_id]['slack'] +  TL[si_id]['t']:currlen + TL[si_id]['slack'] + 2*TL[si_id]['t']]
        deltas +=x[currlen + TL[si_id]['slack'] +  2*TL[si_id]['t']:currlen + TL[si_id]['slack'] + 2*TL[si_id]['t'] + TL[si_id]['delta']]
        currlen += TL[si_id]['slack'] + 2*TL[si_id]['t'] + TL[si_id]['delta'] + TL[si_id]['x']

    tin = np.array(tin)
    tout = np.array(tout)
    deltas = np.array(deltas)
    deltas = np.round(deltas)
    #betas = betas.value
    ad_paths = np.where(deltas==1)[0]
    paths_flat = []
    timetable = {}
    penalty = 0
    for si in paths:
        paths_flat += paths[si]
    for (si,ad_path) in zip(service_intentions, ad_paths):
        p = paths_flat[ad_path]
        timetable[si] = {'tin': [], 'tout':[], 'e':[]}
        for idx,e in zip(get_index_by_delta[ad_path],p):
            timetable[si]['tin'].append((tin[idx]))
            timetable[si]['tout'].append((tout[idx]))
            
            estr = e[0]+','+e[1]
            timetable[si]['e'].append(estr)
            print(f'si: {si}, edge: {estr}, tin: {sec_to_str(tin[idx])}, tout: {sec_to_str(tout[idx])}')
            if estr in Lats[si]['LatIn']:
                #print('LatIn', Lats[si]['LatIn'][estr])
                pen = max(0,tin[idx] - str_to_sec(Lats[si]['LatIn'][estr]))
                penalty+=pen
            if estr in Lats[si]['LatOut']:
                #print('LatOut', Lats[si]['LatOut'][estr])
                pen= max(0,tout[idx] - str_to_sec(Lats[si]['LatOut'][estr]))
                penalty+=pen
            #print(route_graphs[si].edges[e])
    import matplotlib.pyplot as plt
    #plt.figure(figsize = (15,11.5))
    fig, ax = plt.subplots(figsize = (int(15*fsize),(11.5*fsize)))
    s=labelsize

    for si in timetable:
        ax.scatter(range(len(timetable[si]['tin'])),timetable[si]['tin'], marker = 'o', label = f'{si} entry', s=s)
        ax.scatter(range(1,len(timetable[si]['tout'])+1),timetable[si]['tout'], marker = 'x', label = f'{si} exit', s=s)
    if text:
        for si in timetable:
            for i, txt in enumerate(timetable[si]['e']):
                ax.annotate(timetable[si]['e'][i], (range(len(timetable[si]['tin']))[i], timetable[si]['tin'][i] + 100))
    plt.xlabel('Section sequence')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.show()

def drawmatrix(A, S = 1):
    sns.set_style("white")
    f = A.shape[0]/A.shape[1]
    f2 = 1
    plt.figure(figsize = (S*f2, f*S))
    plt.spy(A)