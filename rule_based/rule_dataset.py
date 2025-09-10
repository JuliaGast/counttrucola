import time
import sys
import pickle
import os.path as osp
import numpy as np
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__)))) 

from tgb.linkproppred.dataset import LinkPropPredDataset
from tgb.utils.utils import read_id_to_string_mappings

class RuleDataset:

    def __init__(self, name, dir_data = "datasets", large_data_hardcode_flag=None, very_large_data_hardcode_flag=None):
        """
        Dataset class for rule-based link prediction
        :param name: the name of the dataset
        :param dir_data: the directory where the dataset is located in the project
        """
        
        self.dataset = LinkPropPredDataset(name= name, root=dir_data, preprocess=True)

        # to get all quadruples
        self.relations = self.dataset.edge_type
        self.subjects = self.dataset.full_data["sources"]
        self.objects= self.dataset.full_data["destinations"]
        self.timestamps_orig = self.dataset.full_data["timestamps"]
        self.timestamps = None # will be mapped in the following fct
        
        self.timestamp_id2orig = {} # will be mapped in the following fct. maps timestamp id to original timestamp
        
        # will be filled later for each split
        self.entities = {}
        
        
        self.min_timestep = {} # the minimal timestep in each split
        self.max_timestep = {} # the maximal timestep in each split        
        self.ent2currentness = {}

        # to map the timestamps to 0,...,n
        self.timestamps_to_id()
        self.timesteps_set = set(self.timestamps)
        self.all_quads = np.stack((self.subjects, self.relations, self.objects, self.timestamps), axis=1)

        self.very_large_data_flag = False # for very very large datasets, this is for the update of the f-rules per timestep. if this is true, we do not store the f-rule confidences for each timestep
        self.large_data_flag = False # if this is true: we do only store the highest 200 predictions for each test query

        if not very_large_data_hardcode_flag == None:
            self.very_large_data_flag = very_large_data_hardcode_flag   # for very very large datasets, this is for the update of the f-rules per timestep. if this is true, we do not store the f-rule confidences for each timestep
        else:
            if len(self.timesteps_set) > 5000 and len(self.all_quads) > 1000000: # for very very large datasets, this is for the update of the f-rules per timestep. if this is true, we do not store the f-rule confidences for each timestep
                self.very_large_data_flag = True

        if not large_data_hardcode_flag == None:
            self.large_data_flag = large_data_hardcode_flag            
        else:
            if len(self.all_quads) > 1000000:
                self.large_data_flag = True # if this is true: we do only store the highest 200 predictions for each test query


        
        # to get the train, valid and test data 
        self.train_data = self.all_quads[self.dataset.train_mask]
        self.val_data = self.all_quads[self.dataset.val_mask]
        self.test_data = self.all_quads[self.dataset.test_mask]

        # to get the strings of the entities and the relations
        self.nodes_id_to_string, self.rels_id_to_string = read_id_to_string_mappings(self.dataset.root) 

        self.num_rels = self.dataset.num_rels
        self.num_rels_half = self.num_rels // 2

        start = time.time()

        self.rels_set = set(self.relations)
        self.inverse_rel_dict = {rel: self.get_inv_rel_id(rel) for rel in self.rels_set}

        sub_set = set(self.subjects)
        obj_set = set(self.objects)
        self.ents_set = sub_set | obj_set
        

        self.latest_timestep = max(self.timesteps_set)
        self.earliest_timestep = min(self.timesteps_set)

        # main data structure that maps head => tail => rel => ordered list of timeindices
        # allows to retrieve all timesteps at which a triple holds in constant time
        self.head_tail_rel_t = {}
        
        # this datastructure is similar to self.head_tail_rel_t but the rel is located in the middle
        self.head_rel_tail_t = {}

        # we need to have the same structure where the list of time steps is replaced by a set of time steps
        self.head_tail_rel_ts = {}
        
        # this datastruture is similar to self.head_tail_rel_ts, however, the specific tail is not stored
        # there will be an entry in this data structure if there IS SOME tail for the specfic head and relation
        self.head_rel_ts = {}
        
        # as above, however, time steps are not sored in a set but in a list
        self.head_rel_t = {}

        # for the complete dataset in case what we search for could be in different splits
        self.all_head_rel_tail_t = {}
        self.all_head_tail_rel_ts = {}

        self.rel_head_tail_t = {}
        
        # new index which starts with time index, the tails in the end are stored in a set
        # the index is created only for train
        self.time_head_rel_tails = {}

        self.head_rel_t_tail ={} # only used in explain mode

        for quad in self.train_data: self.index(int(quad[0]), int(quad[1]), int(quad[2]), int(quad[3]), "train")
        for quad in self.val_data: self.index(int(quad[0]), int(quad[1]), int(quad[2]), int(quad[3]), "val")
        for quad in self.test_data: self.index(int(quad[0]), int(quad[1]), int(quad[2]), int(quad[3]), "test")
        
        self.rel_head_tail_t
        self.tuplify()

        print(">>> loading and indexing of dataset " + "{:.3f}".format(time.time()-start) + " seconds")
        self.check_order_time_indices() #
        
        self.equivalent_rels = {}

        


    def check_for_equivalent_rels(self, threshold=0.95):
        """ for each relation check if there are equivalent relations in the dataset and write them in dict  
        if (x, rel, y) in dataset and also (x, rel2, y) in dataset, then rel and rel2 are equivalent
        we set a threshold. if this threshold is reached, we consider the relations as equivalent
        use the train set for this check
        """
        equivalent_rels_dict = {}

        r_sym_count = {}
        for x in self.head_tail_rel_t["train"]:
            for y in self.head_tail_rel_t["train"][x]:
                for r in self.head_tail_rel_t["train"][x][y]:
                    if not r in r_sym_count: r_sym_count[r] = {}
                    for t in self.head_tail_rel_t["train"][x][y][r]:
                        for r2 in self.head_tail_rel_ts['train'][x][y]:                            
                            if r == r2: continue
                            if not r2 in r_sym_count[r]: r_sym_count[r][r2] = (0,0)
                            if t in self.head_tail_rel_ts["train"][x][y][r2]:                                
                                r_sym_count[r][r2] = (r_sym_count[r][r2][0] + 1, r_sym_count[r][r2][1] + 1)
                            else:
                                if not r2 in r_sym_count[r]: r_sym_count[r][r2] = (0,0)
                                r_sym_count[r][r2] = (r_sym_count[r][r2][0], r_sym_count[r][r2][1] + 1)

        for r in r_sym_count:
            for r2 in r_sym_count[r]:
                if r == r2: continue
                if r_sym_count[r][r2][1] > 10:
                    if r_sym_count[r][r2][0] / r_sym_count[r][r2][1] > threshold:
                        if r_sym_count[r2][r][0] / r_sym_count[r2][r][1] > threshold: 
                            if r not in equivalent_rels_dict: equivalent_rels_dict[r] = []
                            if r2 not in equivalent_rels_dict: equivalent_rels_dict[r2] = []
                            equivalent_rels_dict[r].append(r2) 
                            # print(str(r) + " ("+self.rels_id_to_string[r]+")" + " and " + str(r2) +" ("+self.rels_id_to_string[r2]+")" +" are equivalent", r_sym_count[r][r2])

        for rel in equivalent_rels_dict:
            equivalent_rels_dict[rel] = tuple(equivalent_rels_dict[rel]) 

        return equivalent_rels_dict
        
    def check_order_time_indices(self):
        """ make sure that the time indices are in increasing order for each head rel tail, previous <= current
        """
        ts_len = []
        for h in self.all_head_rel_tail_t:
            for r in self.all_head_rel_tail_t[h]:
                for t in self.all_head_rel_tail_t[h][r]:
                    ts = self.all_head_rel_tail_t[h][r][t]
                    ts_len.append(len(ts))
                    previous = -1
                    for current in ts:
                        if not previous <= current: 
                            print(">>> problem detected: bad order of time indices")
                            print(">>> problematic triple (head  relation tail): " + str(h) + " " + str(r) + " " + str(t))
                            print(">>> " + str(ts))
                            exit()
                        else:
                            previous = current
        print(">>> average number of time steps for a triple: " + ("{:.3f}".format(sum(ts_len) / len(ts_len))))
        print(">>> checked order of time steps, everything is fine")
    
    def timestamps_to_id(self):
        """
        Maps the timestamps of a dataset into ids from 0 ... to |timestamps|
        also creates a dictionary to map the ids back to the original timestamps
        """
        tmp_timestamps = np.unique(self.timestamps_orig)
        tmp_timestamps2 = self.reformat_ts(tmp_timestamps)
        tmp_dict = dict()
        i = 0
        for new, old in zip(tmp_timestamps2, tmp_timestamps):
            tmp_dict[old] = new
            self.timestamp_id2orig[new] = old

        self.timestamps = np.array([tmp_dict[x] for x in self.timestamps_orig])

    def tkg_granularity_lookup(self, dataset_name, ts_distmean, ts_distmin):
        """ lookup the granularity of the dataset, and return the corresponding granularity
        """
        if 'icews' in dataset_name or 'polecat' in dataset_name:
            if '18' in dataset_name or '14' in dataset_name:
                return ts_distmean            
            else:
                return 86400
        elif 'wiki' in dataset_name or 'yago' in dataset_name:
            return 31536000
        else:
            print("TODO: work on ts reformatting") #ts_distmean
            return ts_distmin #ts_distmean
            

    def compute_maxminmean_distances(self, unique_sorted_timestamps):
        """ compute the maximum, minimum and mean distances between timestamps, where the timestamps are in a sorted list"""
        differences = []
        
        # Iterate over the list and compute the differences between successive elements
        for i in range(len(unique_sorted_timestamps) - 1):
            diff = unique_sorted_timestamps[i+1] - unique_sorted_timestamps[i]
            differences.append(diff)
        
        # Calculate the mean of the differences
        mean_diff = sum(differences) / len(differences)
        
        return np.max(differences), np.min(differences), np.mean(differences)

    def reformat_ts(self, timestamps):
        """ reformat timestamps s.t. they start with 0, and have stepsize 1.
        :param timestamps: np.array() with timestamps
        returns: np.array(ts_new)
        """
        dataset_name = self.dataset.name
        all_ts = list(set(timestamps))
        all_ts.sort()
        ts_min = np.min(all_ts)
        if 'tkgl' in dataset_name:
            ts_distmax, ts_distmin, ts_distmean = self.compute_maxminmean_distances(all_ts)
            if ts_distmean != ts_distmin:
                ts_dist = self.tkg_granularity_lookup(dataset_name, ts_distmean, ts_distmin)
                if ts_dist - ts_distmean > 0.1*ts_distmean:
                    print('PROBLEM: the distances are somehwat off from the granularity of the dataset. using original mean distance')
                    ts_dist = ts_distmin #ts_distmean
                    print("TODO: work on ts reformatting") #ts_distmean
            else:
                ts_dist = ts_distmean
        ts_new = []
        timestamps2 = timestamps - ts_min
        ts_new = np.ceil(timestamps2/ts_dist).astype(int)

        return np.array(ts_new)


    def index(self, head, rel, tail, t, split):
        """
        Adds each ingoing temporal triple to several index structures which are required for creating the input examples for the function learning.
        :param head: the subject/head of a quadruple
        :param rel: the relation of a quadruple
        :param tail: the object/tail of a quadruple
        :param t: the timestamp of a quadruple
        :param split: the used split train, val or test
        """
        # filling list based main data structure         
        
        if not split in self.min_timestep:
            self.min_timestep[split] = 10000000
            self.max_timestep[split] = 0
        if t < self.min_timestep[split]: self.min_timestep[split] = t
        if t > self.max_timestep[split]: self.max_timestep[split] = t
        
        if not split in self.rel_head_tail_t:
            self.rel_head_tail_t[split] = {}
        if not rel in self.rel_head_tail_t[split]:
            self.rel_head_tail_t[split][rel] = {}
        if not head in self.rel_head_tail_t[split][rel]:
            self.rel_head_tail_t[split][rel][head] = {}
        if not tail in self.rel_head_tail_t[split][rel][head]:
            self.rel_head_tail_t[split][rel][head][tail] = []
        self.rel_head_tail_t[split][rel][head][tail].append(t)
        
        if not split in self.entities:
            self.entities[split] = set()
        self.entities[split].add(head)
        self.entities[split].add(tail)
        
        if not split in self.head_tail_rel_t:
            self.head_tail_rel_t[split] = {}
        if not head in self.head_tail_rel_t[split]:
            self.head_tail_rel_t[split][head] = {}
        if not tail in self.head_tail_rel_t[split][head]:
            self.head_tail_rel_t[split][head][tail] = {}
        if not rel in self.head_tail_rel_t[split][head][tail]:
            self.head_tail_rel_t[split][head][tail][rel] = []
        self.head_tail_rel_t[split][head][tail][rel].append(t)

        # filling list based main data structure but rel is in the middle
        if not split in self.head_rel_tail_t:
            self.head_rel_tail_t[split] = {}
        if not head in self.head_rel_tail_t[split]:
            self.head_rel_tail_t[split][head] = {}
        if not rel in self.head_rel_tail_t[split][head]:
            self.head_rel_tail_t[split][head][rel] = {}
        if not tail in self.head_rel_tail_t[split][head][rel]:
            self.head_rel_tail_t[split][head][rel][tail] = []
        self.head_rel_tail_t[split][head][rel][tail].append(t)

        # filling set based main data structure 
        if not split in self.head_tail_rel_ts:
            self.head_tail_rel_ts[split] = {}
        if not head in self.head_tail_rel_ts[split]:
            self.head_tail_rel_ts[split][head] = {}
        if not tail in self.head_tail_rel_ts[split][head]:
            self.head_tail_rel_ts[split][head][tail] = {}
        if not rel in self.head_tail_rel_ts[split][head][tail]:
            self.head_tail_rel_ts[split][head][tail][rel] = set()
        self.head_tail_rel_ts[split][head][tail][rel].add(t)

        # filling 'exists tail' datastructure (list variant)
        if not split in self.head_rel_t:
            self.head_rel_t[split] = {}
            self.head_rel_ts[split] = {}
        if not head in self.head_rel_t[split]:
            self.head_rel_t[split][head] = {}
            self.head_rel_ts[split][head] = {}
        if not rel in self.head_rel_t[split][head]:
            self.head_rel_t[split][head][rel] = []
            self.head_rel_ts[split][head][rel] = set()
        if not t in self.head_rel_ts[split][head][rel]:
            # this check is required to avoid duplicates in the list
            # which might happen due to the "exists"
            self.head_rel_t[split][head][rel].append(t)
        self.head_rel_ts[split][head][rel].add(t)
        
        ########### for the complete dataset ###########

        # #filling list based main data structure 
        if not split == "explain":
            if not head in self.all_head_rel_tail_t:
                self.all_head_rel_tail_t[head] = {}
            if not rel in self.all_head_rel_tail_t[head]:
                self.all_head_rel_tail_t[head][rel] = {}
            if not tail in self.all_head_rel_tail_t[head][rel]:
                self.all_head_rel_tail_t[head][rel][tail] = []
            self.all_head_rel_tail_t[head][rel][tail].append(t)
            
            if not head in self.all_head_tail_rel_ts:
                self.all_head_tail_rel_ts[head] = {}
            if not tail in self.all_head_tail_rel_ts[head]:
                self.all_head_tail_rel_ts[head][tail] = {}
            if not rel in self.all_head_tail_rel_ts[head][tail]:
                self.all_head_tail_rel_ts[head][tail][rel] = set()
            self.all_head_tail_rel_ts[head][tail][rel].add(t)
        

        if split == "train":
            if not t in self.time_head_rel_tails:
                self.time_head_rel_tails[t] = {}
            if not head in self.time_head_rel_tails[t]:
                self.time_head_rel_tails[t][head] = {}
            if not rel in self.time_head_rel_tails[t][head]:
                self.time_head_rel_tails[t][head][rel] = set()
            self.time_head_rel_tails[t][head][rel].add(tail)
            
    

        # if split == 'explain':
        #     # filling 'exists tail' datastructure (list variant)
        #     if not split in self.head_rel_t_tail:
        #         self.head_rel_t_tail[split] = {}
        #     if not head in self.head_rel_t_tail[split]:
        #         self.head_rel_t_tail[split][head] = {}
        #     if not rel in self.head_rel_t_tail[split][head]:
        #         self.head_rel_t_tail[split][head][rel] = {}
        #     if not t in self.head_rel_t_tail[split][head][rel]:
        #         # this check is required to avoid duplicates in the list
        #         # which might happen due to the "exists"
        #         self.head_rel_t_tail[split][head][rel][t] = []
        #     self.head_rel_t_tail[split][head][rel][t].append(tail)
        

    def create_explain_gt_index(self, mode):
        """ create a dict that is used in explain mode.
        It contains the ground truth objects for each query - to show the user which objects besides the test quadruples object are true at this time"""
        self.head_rel_t_tail['explain'] = {}
        split = 'explain'   
        self.head_rel_t[split] = {}
        # self.head_rel_ts[split] = {}
        # self.head_rel_tail_t[split] = {}
        
        if mode =='val':
            quads = self.val_data
        elif mode == 'test':
            quads = self.test_data
        for quad in quads:
            head = int(quad[0])
            rel = int(quad[1])
            tail = int(quad[2])
            t = int(quad[3])

            if not head in self.head_rel_t_tail[split]:
                self.head_rel_t_tail[split][head] = {}
            if not rel in self.head_rel_t_tail[split][head]:
                self.head_rel_t_tail[split][head][rel] = {}
            if not t in self.head_rel_t_tail[split][head][rel]:
                # this check is required to avoid duplicates in the list
                # which might happen due to the "exists"
                self.head_rel_t_tail[split][head][rel][t] = []
            self.head_rel_t_tail[split][head][rel][t].append(tail)
            
            # if not head in self.head_rel_ts[split]:
            #     self.head_rel_ts[split][head] = {}
            # if not rel in self.head_rel_ts[split][head]:
            #     self.head_rel_ts[split][head][rel] = set()

            # self.head_rel_ts[split][head][rel].add(t)


            # if not head in self.head_rel_tail_t[split]:
            #     self.head_rel_tail_t[split][head] = {}
            # if not rel in self.head_rel_tail_t[split][head]:
            #     self.head_rel_tail_t[split][head][rel] = {}
            # if not tail in self.head_rel_tail_t[split][head][rel]:
            #     self.head_rel_tail_t[split][head][rel][tail] = []
            # self.head_rel_tail_t[split][head][rel][tail].append(t)

    def create_apply_index(self, testset_dict):
        """ create a dict that is used in apply.
        It does not contain the ground truth objects
        subject: rel: set(timestamps)
        """
        apply_dict_head_rel_ts = {}
        for head in testset_dict:
            if not head in apply_dict_head_rel_ts:
                apply_dict_head_rel_ts[head] = {}
            for rel in testset_dict[head]:
                if not rel in apply_dict_head_rel_ts[head]:
                    apply_dict_head_rel_ts[head][rel] = set()
                for tail in testset_dict[head][rel]:
                    for t in testset_dict[head][rel][tail]:
                        apply_dict_head_rel_ts[head][rel].add(t)
        return apply_dict_head_rel_ts

    def tuplify(self):        
        for head in self.all_head_rel_tail_t:
            for rel in self.all_head_rel_tail_t[head]:
                for tail in self.all_head_rel_tail_t[head][rel]:
                   self.all_head_rel_tail_t[head][rel][tail] = tuple(self.all_head_rel_tail_t[head][rel][tail])
        
        for split in ("val", "test", "train"):
            for head in self.head_rel_tail_t[split]:
                for rel in self.head_rel_tail_t[split][head]:
                    for tail in self.head_rel_tail_t[split][head][rel]:
                        self.head_rel_tail_t[split][head][rel][tail] = tuple(self.head_rel_tail_t[split][head][rel][tail])
                        
            for head in self.head_tail_rel_t[split]:
                for tail in self.head_tail_rel_t[split][head]:
                    for rel in self.head_tail_rel_t[split][head][tail]:
                        self.head_tail_rel_t[split][head][tail][rel] = tuple(self.head_tail_rel_t[split][head][tail][rel])

    def is_true_head_tail(self, head, rel, tail, time, split):
        """ 
        we know already that head and tail are true at this time for this split
        Looks up the index structure to check of a triple is true at a given time within a certain split.
        :param head: the subject/head of a quadruple
        :param rel: the relation of a quadruple
        :param tail: the object/tail of a quadruple
        :param t: the timestamp of a quadruple
        :param split: the used split train, val or test
        """
        if not rel in self.head_tail_rel_ts[split][head][tail]: return False
        if not time in self.head_tail_rel_ts[split][head][tail][rel]: return False
        return True
    
    def is_true(self, head, rel, tail, time, split):
        """
        Looks up the index structure to check of a triple is true at a given time within a certain split.
        :param head: the subject/head of a quadruple
        :param rel: the relation of a quadruple
        :param tail: the object/tail of a quadruple
        :param t: the timestamp of a quadruple
        :param split: the used split train, val or test
        """
        if not head in self.head_tail_rel_ts[split]: return False
        if not tail in self.head_tail_rel_ts[split][head]: return False
        if not rel in self.head_tail_rel_ts[split][head][tail]: return False
        if not time in self.head_tail_rel_ts[split][head][tail][rel]: return False
        return True

    ## moved to apply_dataset
    # def is_true_all(self, head, rel, tail, time):
    #     """
    #     Looks up the index structure to check of a triple is true at a given time in the whole dataset.
    #     :param head: the subject/head of a quadruple
    #     :param rel: the relation of a quadruple
    #     :param tail: the object/tail of a quadruple
    #     :param t: the timestamp of a quadruple
    #     """
    #     if not head in self.all_head_tail_rel_ts: return False
    #     if not tail in self.all_head_tail_rel_ts[head]: return False
    #     if not rel in self.all_head_tail_rel_ts[head][tail]: return False
    #     if not time in self.all_head_tail_rel_ts[head][tail][rel]: return False
    #     return True
        
    ## moved to apply_dataset
    # def get_heads_all(self, rel, tail):
    #     """
    #     Returns a dictionary with all heads => [t1, t2, ...] for which there is some (head, rel, tail, t) in the dataset.
    #     """
        
    #     rel_inv = self.get_inv_rel_id(rel)
        
    #     if not tail in self.all_head_rel_tail_t: return {}
    #     if not rel_inv in self.all_head_rel_tail_t[tail]: return {}
    #     return self.all_head_rel_tail_t[tail][rel_inv]
        
    ## moved to apply_dataset
    # def get_t_when_true_all(self, head, rel, tail):
    #     """
    #     Returns all timesteps within the whole dataset for which the triple stated via the parameters is true.
    #     """
    #     if not head in self.all_head_rel_tail_t: return []
    #     if not rel in self.all_head_rel_tail_t[head]: return []
    #     if not tail in self.all_head_rel_tail_t[head][rel]: return []
    #     return self.all_head_rel_tail_t[head][rel][tail]


    def get_inv_rel_id(self, rel_id):
        """
        Returns the inverse relation id for a given relation id in both directions.
        If the relation id is smaller than the number of non-inverse relations (i.e. relations divided by 2), the inverse relation id is the relation id plus the number of non-inverse relations (relations divided by 2).
        If the relation id is greater than the number of non-inverse relations (i.e. relations divided by 2), the inverse relation id is the relation id minus the number of non-inverse relations (relations divided by 2).
        """
        if rel_id >= self.num_rels_half:
            return rel_id - self.num_rels_half # map inverse to original
        else:
            return rel_id + self.num_rels_half # map original to inverse


    def sanity_check_window_size(self, window_size):
        """ 
        Sanity check for the window size. If the window size is larger than half of the dataset, a warning is printed.
        """
        if window_size/len(self.timesteps_set) > 0.5:
            print(f'!!!!!!!!!!!!!!!!!!!!!!WARNING!!!!!!!!!!!!!!!!!!!!!!')
            print(f'The learn window size  LEARN_WINDOW_SIZE {window_size} is larger than half of the dataset timesteps. It makes up {window_size/len(self.timesteps_set)} of the number of timesteps in the full dataset.')
            print('This might lead to problems, as the learn window size is too large for the dataset.')
            print(f'We recommend to reduce the learn window size to max {int(len(self.timesteps_set)/3)}.')
            exit()

    def blocked_by_recurrency_train(self, head, rel, tail, j, i, offset):
        if offset == -1:
            if self.is_true(head, rel, tail, i-1, "train"):
                return True
            else:
                return False
        if offset < -1:
            return False
        # the remaining cases are 0 or positive values for offset
        for k in range(max(0,j-offset), i):
            if self.is_true(head, rel, tail, k, "train"): return True
        return False