import re
import numpy as np


class RuleC:
# for asking about the object in c rules (the one represented by the C)
# e.g. query: climbs(monkey3,?,T)
# climbs(X,l3,T) <- climbs(X,l2,U) & exists Z with climbs(X, Z, T) 

    def __init__(self, relh, ch, relb, cb, parameters, relh_string, ch_string, relb_string, cb_string):
        self.relh = relh
        self.ch = ch
        self.relb = relb 
        self.cb = cb
        self.parameters = parameters # [lmbda, alpha, phi, rho, beta, gamma, function] 
        self.relh_string = relh_string
        self.ch_string   = ch_string
        self.relb_string = relb_string
        self.cb_string   = cb_string
    
    def get_rule_key(self):
        return (self.relh, self.ch, self.relb, self.cb)
        
    def get_string_repre(self):
        return "F\t"+self.relh_string + "(X," + str(self.ch_string) + ",T)" + " <= " + self.relb_string + "(X," + str(self.cb_string) + ",U)" # F for forward
    
    def get_id_repre(self):
        return "F\t"+str(self.relh) + "(X," + str(self.ch) + ",T)" + " <= " + str(self.relb) + "(X," + str(self.cb) + ",U)"

class RuleCBackward:
# for asking about the subject in c rules (the part not represented by the C)
# e.g. query: climbs(?,l3,T)
# climbs(X,I3,T) <- climbs(X, I2, U) & exists Z with climbs(Z, I3, T) 
    def __init__(self, relh, ch, relb, cb, parameters, relh_string, ch_string, relb_string, cb_string):
        self.relh = relh
        self.ch = ch
        self.relb = relb 
        self.cb = cb
        self.parameters = parameters # [lmbda, alpha, phi, rho, beta, gamma, function] 
        self.relh_string = relh_string
        self.ch_string   = ch_string
        self.relb_string = relb_string
        self.cb_string   = cb_string
    
    def get_rule_key(self):
        return (self.relh, self.ch, self.relb, self.cb, 'b')
        
    def get_string_repre(self):
        return "B\t"+self.relh_string + "(X," + str(self.ch_string) + ",T)" + " <= " + self.relb_string + "(X," + str(self.cb_string) + ",U)" # B for backward
    
    def get_id_repre(self):
        return "B\t"+str(self.relh) + "(X," + str(self.ch) + ",T)" + " <= " + str(self.relb) + "(X," + str(self.cb) + ",U)"


class Rule2:
    def __init__(self, relh, relb, parameters, relh_string, relb_string):
        """
        :param relh: the relation in the head of the rule
        :param relb: the relation in the body of the rule
        :param parameters: list. the learned parameters of the rule [lmbda, alpha, phi, rho, beta, gamma, function] 
        :param rel_id_to_string: a dictionary that maps the id of a relation to its string represen tation
        """
        self.relh = relh
        self.relb = relb 
        self.relh_string = relh_string
        self.relb_string = relb_string
        self.parameters = parameters

    def get_rule_key(self):
        return (self.relb, self.relh)
        
    def get_string_repre(self):
        """
        :return repre: returns the representation of the rule with relations as strings
        """
        repre = "F\t"
        var1 = "(X,Y,T)"
        var2 = "(X,Y,U)"
        repre += self.relh_string + var1 + " <= " + self.relb_string + var2
        
        return repre
    
    def get_id_repre(self):
        """
        :return repre: returns the representation of the rule with relations as ids
        """
        repre = "F\t"
        var1 = "(X,Y,T)"
        var2 = "(X,Y,U)"
        repre += str(self.relh) + var1 + " <= " + str(self.relb) + var2
        
        return repre

class Rule1(Rule2):
    def __init__(self, rel, parameters, rel_string):
        """
        :param rel: the relation in the head and the body of the rule
        :param parameters:list. the learned parameters of the rule  [lmbda, alpha, phi, rho, beta, gamma, function] 
        :param rel_id_to_string: a dictionary that maps the id of a relation to its string representation
        """
        super().__init__(rel, rel, parameters, rel_string, rel_string)


class RuleSet:
    def __init__(self, rel_id_to_string, node_id_to_string):
        """
        :param rel_id_to_string: a dictionary that maps the id of a relation to its string representation
        """
        self.rel_id_to_string = rel_id_to_string
        self.node_id_to_string = node_id_to_string
        self.rules = []
    
    def add_rule(self, rule):
        """
        :params rule: the rule that needs to be added to the ruleset
        """
        self.rules.append(rule)


    def write_strings(self, path):
        """
        This function writes out a file with the string representation of the rules
        :param path: the path that the rules have to be written to
        """
        file = open(path, "w", encoding='utf-8')
        for rule in self.rules:
            rule_str = ''
            for i in range(len(rule.parameters)):
                rule_str += str(rule.parameters[i]) + "\t"
            rule_str += rule.get_string_repre()+"\n"
            file.write(rule_str)
        file.close()

    def write_ids(self, path):
        """
        This function writes out a file with the id representation of the rules
        :param path: the path that the rules have to be written to
        """
        file = open(path, "w")
        for rule in self.rules:
            rule_str = ''
            for i in range(len(rule.parameters)):
                rule_str += str(rule.parameters[i]) + "\t"
            rule_str += rule.get_id_repre()+"\n"
            file.write(rule_str)
        file.close()


    def read_rules(self, path, all_rule_types_false=False):
        """ read all rules stored in the file stored at path
        :param path: path to the file with rules
        :param all_rule_types_false: if True, then no rules are used.
        :return: rules_dict dictionary with rules
        :return: number_of_rules [int] number of rules read
        """
        
        rules_xy = {} # data structure for the apply that contains all xy rules, which have no constants
        rules_c = {} # data structure for the apply that contains all c rules, which have no constants
        rules_c_backward = {}
        number_of_rules = 0
        if all_rule_types_false:
            print("all_rule_types_false is True, so no rules are read - if you want any rules, you need to modify your config file")            
            return rules_xy, rules_c, rules_c_backward, number_of_rules
        
        f = open(path, "r")
        for line in f:
            values = line.split("\t")
            param = values[:-2]  #TODO hier ist es falsch wenn wir noch ein B oder ein F haben
            # if 'B' in param[-1] or 'F' in param[-1]:
            #     param = param[:-1]
            backward_forward = values[-2]
            if backward_forward == 'B':
                backward_flag = True
            else:
                backward_flag = False
            rule = values[-1]
            head, body = rule.split("<=")
            head, body = head.strip(), body.strip()
            rel_head = int(re.findall("[0-9]+\\(", head)[0][:-1])
            rel_body = int(re.findall("[0-9]+\\(", body)[0][:-1])
            rule = None
            p_list = list(np.zeros(len(param)))
            for index, par in enumerate(param[0:]): p_list[index] = float(par)
            p_list.append(param[-1])
            if "(X,Y,T)" in head: # its a cyclic rule without constants, e.g. 666(X,Y,T) <= 777(X,Y,U)
                if rel_head not in rules_xy: rules_xy[rel_head] = {}
                rules_xy[rel_head][rel_body] = tuple(p_list)
                if rel_head == rel_body:
                    rule = Rule1(rel_head, param, self.rel_id_to_string[rel_head])
                else:
                    rule = Rule2(rel_head, rel_body, param, self.rel_id_to_string[rel_head], self.rel_id_to_string[rel_body])
            else: # its a rule with constants, e.g., e.g. 666(X,13,T) <= 777(X,567,U)                
                if rel_head not in rules_c: rules_c[rel_head] = {}
                if rel_head not in rules_c_backward: rules_c_backward[rel_head] = {}
                # the structure of the rules_c dictionary needs to fit to the way how these rules are applied
                # ... todo ...

                c_head = int(re.findall("X,[0-9]+,T", head)[0][2:-2])
                c_body = int(re.findall("X,[0-9]+,U", body)[0][2:-2])
                relh_string = self.rel_id_to_string[rel_head]
                ch_string = self.node_id_to_string[c_head][0]
                relb_string = self.rel_id_to_string[rel_body]
                cb_string = self.node_id_to_string[c_body][0]
                if backward_flag:
                    rules_c_backward[rel_head][(rel_body, c_head, c_body)] = tuple(p_list)
                    rule = RuleCBackward(rel_head, c_head, rel_body, c_body, param, relh_string, ch_string, relb_string, cb_string)
                else:
                    rules_c[rel_head][(rel_body, c_head, c_body)] = tuple(p_list)              
                    rule = RuleC(rel_head, c_head, rel_body, c_body, param, relh_string, ch_string, relb_string, cb_string)
            self.add_rule(rule)
            number_of_rules +=1
        f.close()
        return rules_xy, rules_c, rules_c_backward, number_of_rules
    
    def add_params(self, params):
        """
        This function adds the learned parameters to the rules
        :param params: the needed parameters to be added
        """
        for rule in self.rules:
            rk = rule.get_rule_key()

            rule.parameters = params[rk]


    
    

