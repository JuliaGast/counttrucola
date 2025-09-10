class ApplyDataset:
    def __init__(self, all_head_rel_tail_t, all_head_tail_rel_ts, head_rel_ts, num_quads,  inverse_rel_dict):
        """
        Dataset class to be used only during apply; 
        only contains a subset of the data, i.e. only the index structures that are needed for the apply process.
        This is important for the apply multithreading, because otherwise the whole dataset would be duplicated (by number of threads) in memory.

        """

        self.num_quads = num_quads
        self.all_head_rel_tail_t = all_head_rel_tail_t
        self.all_head_tail_rel_ts = all_head_tail_rel_ts
        self.inverse_rel_dict = inverse_rel_dict
        self.head_rel_ts = head_rel_ts

        # not needed for apply:
        # self.rel_head_tail_t
        # head_tail_rel_t
        # head_rel_tail_t
        # head_rel_t



    def is_true_all(self, head, rel, tail, time):
        """
        Looks up the index structure to check of a triple is true at a given time in the whole dataset.
        :param head: the subject/head of a quadruple
        :param rel: the relation of a quadruple
        :param tail: the object/tail of a quadruple
        :param t: the timestamp of a quadruple
        """
        if not head in self.all_head_tail_rel_ts: return False
        if not tail in self.all_head_tail_rel_ts[head]: return False
        if not rel in self.all_head_tail_rel_ts[head][tail]: return False
        if not time in self.all_head_tail_rel_ts[head][tail][rel]: return False
        return True

    def get_t_when_true_all(self, head, rel, tail):
        """
        Returns all timesteps within the whole dataset for which the triple stated via the parameters is true.
        """
        if not head in self.all_head_rel_tail_t: return []
        if not rel in self.all_head_rel_tail_t[head]: return []
        if not tail in self.all_head_rel_tail_t[head][rel]: return []
        return self.all_head_rel_tail_t[head][rel][tail]

    def get_heads_all(self, rel, tail):
        """
        Returns a dictionary with all heads => [t1, t2, ...] for which there is some (head, rel, tail, t) in the dataset.
        """
        
        rel_inv = self.inverse_rel_dict[rel]
        
        if not tail in self.all_head_rel_tail_t: return {}
        if not rel_inv in self.all_head_rel_tail_t[tail]: return {}
        return self.all_head_rel_tail_t[tail][rel_inv]

    def blocked_by_recurrency_all(self, head, rel, tail, j, i, offset):
        if offset < 0: return False
        for k in range(max(0,j-offset), i):
            if self.is_true_all(head, rel, tail, k, "train"): return True
        return False
