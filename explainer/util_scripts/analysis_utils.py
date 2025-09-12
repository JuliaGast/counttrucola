from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np
import os
import pandas as pd

import util_scripts.dataset_utils as du

def plot_relation_pie(dataset, rel_df, figs_dir, dataset_name, num_slices=10):
    """
    Plot a pie chart showing the distribution of relations. store in os.path.join(figs_dir, f"rel_pie_{dataset_name}.png")) and pdf
    :param dataset: RuleDataset object 
    :param rel_df: DataFrame containing relation statistics - created with util_scripts.stats_utils.compute_stats()
    :param figs_dir: Directory to save the figures
    :param dataset_name: Name of the dataset, e.g. tkgl-icews14
    :param num_slices: Number of top relations to show separately in the pie chart, all others are grouped into "others"
    """
    ## make pie chart to show relation distribution
    #code from tgb 2.0  https://github.com/JuliaGast/TGB2/blob/main/stats_figures/create_edges_figures.py
    # pie chart colors

    colors2= ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99'] #from https://colorbrewer2.org/#type=qualitative&scheme=Paired&n=11

    # colors2= ['#8e0152', '#c51b7d', '#de77ae', '#f1b6da', '#fde0ef', '#f7f7f7', '#e6f5d0', '#b8e186', '#7fbc41', '#4d9221', '#276419']
    # from https://colorbrewer2.org/#type=diverging&scheme=PiYG&n=11 color blind friendly 


    k=num_slices # how many slices in the cake +1

    rels_string_to_id = {}
    for id, string_name in dataset.rels_id_to_string.items():
        rels_string_to_id[string_name] = id

    ### A) pie charts #plot top k relations accordung to the number of occurences plus a slice for "others"
    plot_names = list(rel_df['rel_string_word'].iloc[:k]) 
    plot_values = list(rel_df['number_total_occurences'].iloc[:k])
    all_others = np.sum(rel_df['number_total_occurences'].iloc[k:]) #slice for "others" (sum of all other relations occurences)
    plot_values.append(all_others)
    plot_names.append('Others')
    # for the pie chart labels to be more readable (i.e. force line break if words are long)
    plot_names_multi_line= []
    for name in plot_names: # add some \n to make the labels more fittable to the pie chart
        if type(name) == str:
            if name in rels_string_to_id:
                name = str(rels_string_to_id[name])+': '+name
            words = name.split()
            words = name.split('_')
            newname = words[0]
            if len(words) > 1:
                for i in range(len(words)-1):
                    if not '(' in words[i+1]:
                        if len(words[i]) > 4:
                            newname+='\n'
                        else:
                            newname+=' ' 
                        newname+=words[i+1]
        else:
            newname = str(name) #then only plot the int as is. 
        plot_names_multi_line.append(newname)


    plt.figure(figsize=(7, 7))
    wedges, texts, autotexts =plt.pie(plot_values,autopct=lambda pct: f"{pct:.0f}%" if pct > 1.5 else '', startangle=140, colors=colors2, labeldistance=2.2) #repeated_colors)
    # Increase the font size of the percentage values
    for autotext in autotexts:
        autotext.set_fontsize(20) #15
    plt.axis('equal')  
    # Move the percentage labels further outside
    for autotext, wedge in zip(autotexts, wedges):
        angle = (wedge.theta2 - wedge.theta1) / 2 + wedge.theta1
        x = np.cos(np.deg2rad(angle))
        y = np.sin(np.deg2rad(angle))
        distance = 0.85  # Adjust this value to move the labels further or closer to the center
        autotext.set_position((x * distance, y * distance))
    # Set the labels for each pie slice
    # plt.legend(wedges, plot_names_multi_line, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=14)
    plt.legend(wedges, plot_names_multi_line, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=14)
    save_path = (os.path.join(figs_dir, f"rel_pie_{dataset_name}.png"))
    plt.savefig(save_path, bbox_inches='tight')
    save_path = (os.path.join(figs_dir, f"rel_pie_{dataset_name}.pdf"))
    plt.savefig(save_path, bbox_inches='tight')



def plot_edges_per_ts(stats_df, num_triples_dict, figs_dir, dataset_name, log_flag=False, bars_list=[20]):
    """ plot the number of edges over time (line chart with min-max range). save in figs_dir,f"num_triples_discretized_{num_bars}_{dataset_name}2.png")) and pdf
    :param stats_df: dataframe with dataset statistics
    :param num_triples_dict: dictionary with number of triples per timestep
    :param figs_dir: directory to save the figures
    :param dataset_name: name of the dataset
    :param log_flag: optional boolean whether to plot the y-axis in log scale (additional plot)
    :param bars_list: optional list with number of bars to discretize the timesteps into (e.g. [20,50,100])
    """
    ## make plots with dataset stats over time
    #code from tgb 2.0  https://github.com/JuliaGast/TGB2/blob/main/stats_figures/create_edges_figures.py
    # specify params
    granularity ={} #for labels
    granularity['tkgl-polecat'] = 'days'
    granularity['tkgl-icews14'] = 'days'
    granularity['tkgl-icews18'] = 'days'
    granularity['tkgl-gdelt'] = '15 mins'
    granularity['tkgl-icews'] = 'days'
    granularity['tkgl-smallpedia'] = 'years'
    granularity['tkgl-wikidata'] = 'years'
    granularity['tkgl-wikiold'] = 'years'
    granularity['tkgl-yago'] = 'years'


    # colors from tgb logo
    colortgb = '#60ab84'
    colortgb2 = '#eeb641'
    colortgb3 = '#dd613a'
    fontsize =12
    labelsize=12

    n_Triplesnodes_list_all  = num_triples_dict
    n_Triples_list = n_Triplesnodes_list_all['num_triples']
    

    start_date = stats_df.loc['first_ts_string', 0]
    end_date = stats_df.loc['last_ts_string', 0]

    for num_bars in bars_list:
        # Create the 'figs' directory if it doesn't exist
        if not os.path.exists(figs_dir):
            os.makedirs(figs_dir)
        if num_bars < 100:
            capsize=2
            capthick=2
            elinewidth=2
        else:
            capsize=1 
            capthick=1
            elinewidth=1
        ts_discretized_mean, ts_discretized_sum, ts_discretized_min, ts_discretized_max, start_indices, end_indices, mid_indices = du.discretize_values(n_Triples_list, num_bars)
        
        # line chart
        plt.figure()
        plt.tick_params(axis='both', which='major', labelsize=labelsize)
        mins = np.array(ts_discretized_min)
        maxs = np.array(ts_discretized_max)
        means = np.array(ts_discretized_mean)
        # plt.bar(mid_indices, ts_discretized_mean, width=(len(n_Triples_list) // num_bars), label='Mean', color =colortgb)
        plt.step(mid_indices, ts_discretized_mean, where='mid', linestyle='-', label ='Mean Value', color=colortgb, linewidth=2)
        #plt.scatter(mid_indices, ts_discretized_mean, label ='Mean Value', color=colortgb)
        plt.errorbar(mid_indices, maxs, yerr=[maxs-mins, maxs-maxs], fmt='none', alpha=0.9, color='grey',capsize=capsize, capthick=capthick, elinewidth=elinewidth, label='Min-Max Range')
        plt.xlabel(f'Ts. [{granularity[dataset_name]}] from {start_date} to {end_date}', fontsize=fontsize)
        plt.ylabel('Number of Triples', fontsize=fontsize)
        plt.legend()
        plt.tight_layout()
        #plt.title(dataset_name+ ' - Number of Triples aggregated across multiple timesteps')
        # plt.show()
        save_path2 = (os.path.join(figs_dir,f"num_triples_discretized_{num_bars}_{dataset_name}2.png"))
        plt.savefig(save_path2, bbox_inches='tight')
        save_path2 = (os.path.join(figs_dir,f"num_triples_discretized_{num_bars}_{dataset_name}2.pdf"))
        plt.savefig(save_path2, bbox_inches='tight')

        try:
            # try log scale
            if log_flag:
                plt.figure()
                plt.tick_params(axis='both', which='major', labelsize=labelsize)
                mins = np.array(ts_discretized_min)
                maxs = np.array(ts_discretized_max)
                means = np.array(ts_discretized_mean)
                # plt.bar(mid_indices, ts_discretized_mean, width=(len(n_Triples_list) // num_bars), label='Mean', color =colortgb)
                plt.step(mid_indices, ts_discretized_mean, where='mid', linestyle='-', label ='Mean Value', color=colortgb)
                #plt.scatter(mid_indices, ts_discretized_mean, label ='Mean Value', color=colortgb)
                plt.errorbar(mid_indices, maxs, yerr=[maxs-mins, maxs-maxs], fmt='none', alpha=0.9, color='grey',capsize=capsize, capthick=capthick, elinewidth=elinewidth, label='Min-Max Range')
                plt.xlabel(f'Timestep [{granularity[dataset_name]}] from {start_date} to {end_date}', fontsize=fontsize)
                plt.ylabel('Number of Triples', fontsize=fontsize)
                #plt.title(dataset_name+ ' - Number of Triples aggregated across multiple timesteps')
                plt.yscale('log')
                plt.legend(fontsize=fontsize)
                plt.show()
                save_path2 = (os.path.join(figs_dir,f"num_triples_discretized_{num_bars}_{dataset_name}2log.png"))
                plt.savefig(save_path2, bbox_inches='tight')
                save_path2 = (os.path.join(figs_dir,f"num_triples_discretized_{num_bars}_{dataset_name}2log.pdf"))
                plt.savefig(save_path2, bbox_inches='tight')
        except:
            print('Could not plot log scale')
        # plt.close('all')
            


def plot_mrr_per_ts(mrrperts, eval_mode, dataset_name, figs_dir):
    """ plot a line chart with the MRR over time (per timestep). save in figs_dir, figs_dir,f"mrr_over_time_{dataset_name}.png")) and pdf
    :param mrrperts: dictionary with MRR per timestep (output of eval.py)
    :param eval_mode: string indicating the evaluation mode (e.g. 'test' or 'valid')
    :param dataset_name: name of the dataset
    :param figs_dir: directory to save the figures    
    """

    colortgb = '#60ab84'

    ## make a figure that plots the mrr over time
    # Plot MRR over time (per timestep)
    timesteps = sorted(mrrperts.keys())
    mrrs = [mrrperts[ts][0] for ts in timesteps]
    counts = [mrrperts[ts][1] for ts in timesteps]

    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, mrrs, marker='o', color=colortgb, label='MRR')
    plt.xlabel('Timestamp')
    plt.ylabel('MRR')
    plt.title(f'MRR over Time ({eval_mode} set) for dataset {dataset_name}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(figs_dir, f"mrr_over_time_{dataset_name}.png")
    plt.savefig(save_path, bbox_inches='tight')
    save_path = os.path.join(figs_dir, f"mrr_over_time_{dataset_name}.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_mrr_per_relation(rel_df, figs_dir, dataset_name, num_rels_plot=10, head_tail_flag=True, mode='num_occurences', mrrperrel={}):
    """ plot a scatter plot with the MRR per relation (head and tail separately if head_tail_flag=True) for the num_rels_plot most frequent relations. 
    color (and sort)  the points according to the recurrency degree or the number of occurences.
    :param rel_df: DataFrame containing relation statistics - created with util_scripts.stats_utils.compute_stats() and extended to contain mrr per rel
    :param figs_dir: Directory to save the figures
    :param dataset_name: Name of the dataset, e.g. tkgl-icews14
    :param num_rels_plot: Number of most frequent relations to plot
    :param head_tail_flag: Boolean whether to plot head and tail direction separately (True) or the mean value (False)
    :param mode: String indicating whether to color the points according to 'recurrency_degree' or 'num_occurences' (number of occurences)    
    :param mrrperrel: optional dictionary with MRR per relation for other methods to compare in the plot (e.g. {'TLogic': mrrperrel_tlogic, 'regcn': mrrperrel_regcn} where mrrperrel_tlogic is the output of eval.py for TLogic)
    """
    rel_df = rel_df.sort_values(by='number_total_occurences', ascending=False).reset_index(drop=True)
    rels_sorted =  np.array(rel_df['relation'])[0:num_rels_plot] # the 10 most frequent relations
    mask = rel_df['relation'].isin(rels_sorted)
    markers = ['*', 's', '^', 'D', 'v', 'P', '*'] # add as many as you want

    mrr_per_rel_freq_others ={}
    for method in mrrperrel.keys():
        mrr_per_rel_freq_others[method] = [] # list of mrr values for each relation - three lists for three methods
    selected_df = rel_df[mask] #only the parts of the dataframe that contain the top ten relations according to number of occurences

    if mode == 'recurrency_degree':
        selected_df_sorted = selected_df.sort_values(by='recurrency_degree', ascending=False) # Sort selected_df by 'recurrency_degree' column in descending order

    elif mode =='num_occurences':      
        selected_df_sorted = selected_df.sort_values(by='number_total_occurences', ascending=False)

    else:
        print('mode not recognized, using num_occurences/ choose from recurrency_degree or num_occurences')
        selected_df_sorted = selected_df.sort_values(by='number_total_occurences', ascending=False)

    rels_to_plot = list(selected_df_sorted['relation'])
    labels = np.array(selected_df_sorted['relation'])# only plotting the id for space reasons
    mrr_per_rel_freq = [] # list of mrr values for each relation - three lists for three methods

    lab = []
    lab_ht = []
    lab_rel = []
    # rel_oc_dict[rel] = count_occurrences
    count_occurrences_sorted = []
    rec_degree_sorted = []
    for index, r in enumerate(rels_to_plot):   
        if head_tail_flag:
            lab_ht.append('h')
            lab_ht.append('t')
            lab_rel.append(str(labels[index])+'') # add spaces to make the labels longer
        else:
            lab_rel.append(str(labels[index])+'') # add spaces to make the labels longer
        lab.append(labels[index])


        if head_tail_flag: # if we do head and tail separately we need the value for head and tail direction
            mrr_per_rel_freq.append(selected_df_sorted['head'].iloc[index])
            mrr_per_rel_freq.append(selected_df_sorted['tail'].iloc[index])
            count_occurrences_sorted.append(selected_df_sorted['number_total_occurences'].iloc[index])#append twice for head and tail
            count_occurrences_sorted.append(selected_df_sorted['number_total_occurences'].iloc[index])
            rec_degree_sorted.append(selected_df_sorted['recurrency_degree'].iloc[index]) #append twice for head and tail
            rec_degree_sorted.append(selected_df_sorted['recurrency_degree'].iloc[index])
            for method in mrrperrel.keys():
                if method+'_head' in selected_df_sorted.columns and method+'_tail' in selected_df_sorted.columns:
                    mrr_per_rel_freq_others[method].append(selected_df_sorted[method+'_head'].iloc[index])
                    mrr_per_rel_freq_others[method].append(selected_df_sorted[method+'_tail'].iloc[index])
        else:# if we do  NOT head and tail separately we need the mean value for head and tail direction
            mrr_per_rel_freq.append(np.mean([selected_df_sorted['head'].iloc[index], selected_df_sorted['tail'].iloc[index]]))
            count_occurrences_sorted.append(selected_df_sorted['number_total_occurences'].iloc[index])#append twice for head and tail
            rec_degree_sorted.append(selected_df_sorted['recurrency_degree'].iloc[index])
            for method in mrrperrel.keys():
                if method+'_head' in selected_df_sorted.columns and method+'_tail' in selected_df_sorted.columns:
                    mrr_per_rel_freq_others[method].append(np.mean([selected_df_sorted[method+'_head'].iloc[index], selected_df_sorted[method+'_tail'].iloc[index]]))


        # these are the x-values of the ticks. in case we plot head and tail separately, we need to have two ticks per relation
        x_values = []
        x_values_rel = []
        for i in range(0,num_rels_plot):
            if head_tail_flag:
                x_values.append(i*2+0.3)
                x_values.append(i*2+0.8)
            else:
                x_values.append(i*2+0.3)
            x_values_rel.append(i*2+0.4)

        lab_lines = lab_rel #labels, for now
        

    plt.figure(figsize=(10, 5)) 
    if mode == 'recurrency_degree':  
        a = rec_degree_sorted
        sca = plt.scatter(x_values, mrr_per_rel_freq,   marker='o',s=40,    c = a, alpha=1,  edgecolor='grey', cmap='viridis', norm=Normalize(vmin=0, vmax=1), label='rucola')
    elif mode =='num_occurences':
        a = count_occurrences_sorted 
        # sca = plt.scatter(x_values, mrr_per_rel_freq,   marker='o',s=40,    c = a, alpha=1, edgecolor='grey', norm=LogNorm(), cmap='viridis', label='rucola')     
        sca = plt.scatter(x_values, mrr_per_rel_freq,   marker='o',s=40,    c = a, alpha=1, edgecolor='grey', cmap='viridis', label='rucola')  

    index = 0
    for method, values_method in mrr_per_rel_freq_others.items():
        plt.scatter(x_values, values_method,  marker=markers[index],  s=60,    c = 'grey', alpha=1, label= method)
        index +=1
        if index >= len(markers):
            index = 0
    plt.ylabel('MRR', fontsize=14) 
    plt.xlabel('Relation', fontsize=14, labelpad=15)  # increase padding (default ~10)
    # plt.legend(fontsize=14)
    cbar =plt.colorbar(sca)
    max_lim = max(mrr_per_rel_freq) + 0.2*max(mrr_per_rel_freq)
    plt.ylim([0,max_lim])
    cbar.ax.yaxis.label.set_color('gray')

    if len(mrr_per_rel_freq_others.keys()) >0:
        plt.legend(fontsize=12)

    if head_tail_flag:
        # Major ticks
        plt.xticks(ticks=x_values, labels=lab_ht, fontsize=9)
        # Add second row manually below the axis
        for x, label in zip(x_values_rel, lab_lines):
            plt.text(x, -0.05, label, ha='center', va='top', fontsize=13, transform=plt.gca().get_xaxis_transform())
    else:
        plt.xticks(x_values_rel, lab_lines,  size=14)
        plt.tick_params(axis='x',  rotation=90,  length=0)
    plt.yticks(size=13)
    # Create a locator for the second set of x-ticks
    # plt.secondary_xaxis('top', x_values_rel)

    plt.grid()
    if mode == 'recurrency_degree':
        plt.title('MRR per Relation (Colored by Recurrency Degree)')
        if len(list(mrrperrel.keys())) > 0:
            save_path = (os.path.join(figs_dir, f"rel_mrrperrel_recdeg_{dataset_name}_{list(mrrperrel.keys())[0]}.png"))
            save_path = (os.path.join(figs_dir, f"rel_mrrperrel_recdeg_{dataset_name}_{list(mrrperrel.keys())[0]}.pdf"))
        else:
            save_path = (os.path.join(figs_dir, f"rel_mrrperrel_recdeg_{dataset_name}.png"))
            save_path = (os.path.join(figs_dir, f"rel_mrrperrel_recdeg_{dataset_name}.pdf"))
    elif mode =='num_occurences':
        plt.title('MRR per Relation (Colored by number of Occurrences)')
        if len(list(mrrperrel.keys())) > 0:
            save_path = (os.path.join(figs_dir, f"rel_mrrperrel_occ_{dataset_name}_{list(mrrperrel.keys())[0]}.png"))
            save_path = (os.path.join(figs_dir, f"rel_mrrperrel_occ_{dataset_name}_{list(mrrperrel.keys())[0]}.pdf"))
        else:
            save_path = (os.path.join(figs_dir, f"rel_mrrperrel_occ_{dataset_name}.png"))

    plt.savefig(save_path, bbox_inches='tight')
    print('saved in ', save_path)
