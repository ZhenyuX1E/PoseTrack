import numpy as np

def find_view(human_number, human_counts_all_views):
    if len(human_counts_all_views)==1:
        print("no human in any view")
        return -1

    else:
        for i in range(len(human_counts_all_views)-1):
            if human_number > human_counts_all_views[i]-1 and human_number <= human_counts_all_views[i+1]-1:
                return i, human_number - human_counts_all_views[i]
            else:
                continue
        raise Exception("human array searching out of range") 

def find_view_for_cluster(cluster,human_counts_all_views):
    view_list=[]
    number_list = []
    for human_number in cluster:
        view, number = find_view(human_number, human_counts_all_views)
        view_list.append(view)
        number_list.append(number)
    return view_list, number_list