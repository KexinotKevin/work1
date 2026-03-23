import pandas as pd
import numpy as np
import os.path as osp

def load_subjlist(listpath):
    with open(osp.join(listpath), 'r') as f:
        slist=f.read().split('\n')
    return slist[:-1]

def load_connectivity_matrix(filename, isheader=False):
    if isheader:
        mat = pd.read_csv(filename, header=None).values
    else:
        mat = pd.read_csv(filename).values
    return mat

def load_data(label_fp, conn_dir, atlas_type, modname, labellist, load_lb):
    labeldf = pd.read_csv(label_fp)
    # load graphs
    graph_list = []
    label_list = []
    for subj in labeldf.src_subject_id.values:
        matname = '{}.csv'.format(subj)
        mat = load_connectivity_matrix(osp.join(conn_dir, atlas_type, modname, matname))
        graph_list.append(mat)
        if load_lb:
            lbtmp = []
            for lbtype in labellist:
                lbtmp.append(int(labeldf[labeldf.src_subject_id== subj][lbtype].values[0]))
            label_list.append(lbtmp)

        # print(subj)

    graphs = np.array(graph_list)     # shape: subjnum*ROI*ROI
    # graphs = np.transpose(graphs, (2,1,0))   # shape: ROI*ROI*subjnum

    if load_lb:
        labels = np.array(label_list)     # shape: subjnum*labelnum
        # labels = np.transpose(labels, (1,0))   # shape: labelnum*subjnum
    else:
        labels = None    

    return graphs, labels

if __name__ == "__main__":
    graphs_loc = osp.join(projDir, "{}_{}_{}_graphs.npy".format(dtname, atlas, modname))
    labels_loc = osp.join(projDir, "{}_{}_labels.npy".format(dtname, lbtype))

    if not osp.exists(graphs_loc):
        load_lb = True
        if not osp.exists(labels_loc):
            load_lb = False
        graphs, labels = load_data(dataDir=dataDir,
                                conn_dir=osp.join(dataDir, "network_csv"),
                                subjlist=slist,
                                modname="{}_{}".format(atlas, modname),
                                labelfile="HCD_net_cogcomp01.csv",
                                labellist=labellist,
                                load_lb=True)
        np.save(graphs_loc, graphs)
        np.save(labels_loc, labels)
    else:
        graphs = np.load(graphs_loc)
        labels = np.load(labels_loc)

    num_nodes = graphs.shape[1]
    num_subjs = graphs.shape[0]
    num_labels = labels.shape[1]