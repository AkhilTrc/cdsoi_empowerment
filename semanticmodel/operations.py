import json
import numpy as np
import pandas as pd
import seaborn as sns
import fastText.python.fasttext_module.fasttext as fasttext 
import fasttext.util
import matplotlib as mpl
import matplotlib.pyplot as plt

class Operations():
    def __init__(self, vector_version, split_version, n_elements):
        """Initialize the Operations class
        """
        self.empowerment = list()
        self.vector_version = vector_version
        self.split_version = split_version
        self.n_elements = n_elements

    def get_gametree(self):
        # print info for user
        print('\nGet game tree.')

        # load raw gametree
        with open('data/raw/elements.json', encoding='utf8') as infile:
            old_gametree = json.load(infile)

        # initialize element storage for elements
        elements = set()

        # get all elements
        for key, value in old_gametree.items():
            parents = key.split(',')
            results = value
            elements.update(parents, results)
        elements.difference_update({'water', 'fire', 'earth', 'air'})   
        elements = ['water', 'fire', 'earth', 'air'] + list(elements)

        # initialize game tree
        gametree = dict()
        for element_id, element in enumerate(elements):
            gametree[element_id] = {'name': element, 'parents': []}

        # fill game tree
        for key, value in old_gametree.items():
            parents = key.split(',')
            parents = sorted([elements.index(parents[0]), elements.index(parents[1])])
            results = value
            for result in results:
                gametree[elements.index(result)]['parents'].append(parents)

        # write edited library to JSON file
        with open('data/Gametree.json', 'w') as filehandle:
            json.dump(gametree, filehandle, indent=4, sort_keys=True)

        # write elements to JSON file
        with open('data/Elements.json', 'w') as filehandle:
            json.dump(elements, filehandle, indent=4, sort_keys=True)

    def get_tables(self, table_type='combination', expand=False):
        """Returns table depending on the type. Can either be combination, parent or child table.
        """
        
        if table_type != 'combination' and table_type != 'parent' and table_type != 'child':
            raise ValueError('Undefined Version: "{}". Use "combination", "parent" or "child" instead.'.format(table_type))
        
        # print info for user
        print('\nGet table of type {} for version {}.'.format(table_type))
        
        # get game tree
        with open('data/Gametree.json', encoding='utf8') as infile:
            gametree = json.load(infile)
            gametree = {int(k):v for k,v in gametree.items()}

        # initialize search table and parent respectively child table
        table = dict()
        table_csv = list()

        # traverse through game tree 
        for element in gametree:
            for parent in gametree[element]["parents"]:
                # sort parents
                parent = sorted(parent)

                if table_type == 'combination':
                    # assign resulting elements to combination elements
                    if parent[0] not in table:
                        table[parent[0]] = {} 
                    if parent[1] not in table:
                        table[parent[1]] = {} 
                    if parent[1] not in table[parent[0]]:
                        table[parent[0]][parent[1]] = [element]
                    elif element not in table[parent[0]][parent[1]]:
                        table[parent[0]][parent[1]].append(element)
                    if parent[0] not in table[parent[1]]:
                        table[parent[1]][parent[0]] = [element]
                    elif element not in table[parent[1]][parent[0]]:
                        table[parent[1]][parent[0]].append(element)
                    
                    table_csv.append({'first': parent[0], 'second': parent[1], 'result': element})

                elif table_type == 'parent':
                    # add resulting element to parent list for each combination element
                    if parent[0] not in table: 
                        table[parent[0]] = {element}
                    else:
                        table[parent[0]].update([element])
                    if parent[1] not in table:
                        table[parent[1]] = {element}
                    else:
                        table[parent[1]].update([element])
                
                elif table_type == 'child': 
                    # add parents to child list for resulting element
                    if element not in table:
                        table[element] = set(parent)
                    else:
                        table[element].update(parent)

        if table_type == 'parent' or table_type == 'child':
            # adjust to list to successfully write to JSON file 
            for element in table:
                table[element] = list(table[element])
            
            # transform into DataFrame   
            table_csv = pd.DataFrame({key:pd.Series(value) for key, value in table.items()})
        else:
            table_csv = pd.DataFrame(table_csv)  
            if expand is True:                      # Includes unsuccesful elements. 
                table_csv = self.expand_combination_table(table_csv)    # Haven't included this function yet.

        # write to JSON file
        with open('data/{}Table.json'.format(table_type.capitalize()), 'w') as filehandle:
            json.dump(table, filehandle, indent=4, sort_keys=True)
        
        # replace nan values with -1
        table_csv = table_csv.fillna(-1)
        
        # write to csv file
        table_csv.to_csv('data/{}Table.csv'.format(table_type.capitalize()), index=False, float_format='%.f')

    def get_wordvectors(self, vector_version, dim):
        """Loads and stores fastText wordvectors of elements.
        """

        # print info for user
        print('\nGet word vectors from fastText model {} with dimension {}.'.format(vector_version, dim))
        
        # load pretrained fastText model
        if vector_version == 'crawl':
            model = fasttext.load_model('data/fasttext/crawl-300d-2M-subword.bin')
        elif vector_version == 'ccen':
            model = fasttext.load_model('data/fasttext/cc.en.300.bin')
        elif vector_version == 'wiki':
            model = fasttext.load_model('data/fasttext/wiki.en.bin')
        else:
            raise ValueError('Undefined version: "{}". Use "wiki", "crawl" or "ccen" instead.'.format(vector_version))

        # reduce dimensionality
        fasttext.util.reduce_model(model, dim)

        # load elements
        with open('data/Elements.json', encoding='utf8') as infile:
            elements = json.load(infile)

        # get element word vectors
        element_vectors = np.empty((0,model.get_dimension()), int)
        for i in range(len(elements)):
            element_vector = model[elements[i]]
            element_vectors = np.r_[element_vectors, np.reshape(element_vector, (1,-1))]

        # write element word vectors to text file for later usage
        np.savetxt('data/ElementVectors-{}{}.txt'.format(vector_version, dim), element_vectors)

    def get_similarities(self):
        """Initializes similarity class.
        """
        # print info for user
        print('\nPlot similarity values for version with {} vectors.'.format(self.vector_version))
        
        # set general settings for plotting 
        # TODO: change font to Open Sans
        sns.set_theme(context='paper', style='ticks', font='Arial', font_scale=2, rc={'lines.linewidth': 2, 'grid.linewidth':0.6, 'grid.color': '#9d9d9d', 
                                                                                      'axes.linewidth':0.6, 'axes.edgecolor': '#9d9d9d'})      
        self.colors = ['#bf3409']
        sns.set_palette(sns.color_palette(self.colors))
        
        # plot values
        # self.plot_similarity_histogram()

        if self.vector_version == 'ccen100' or self.vector_version == 'ccen300' or self.vector_version == 'crawl100' or self.vector_version == 'crawl300' or self.vector_version == 'wiki100' or self.vector_version == 'wiki300':
            vectors = np.loadtxt('data/{}ElementVectors-{}.txt'.format(self.vector_version))
        else:
            raise ValueError('Undefined vector_version: "{}". Use "ccen100", "ccen300", "crawl100", "crawl300", "wiki100" or "wiki300" instead.'.format(self.vector_version))
        
        """

        FOR PLOTTING SIMILARITY VALUES...

        """

    def get_empowerment(self):
        empowerment = list()

        # set general settings for plotting 
        # TODO: change font to Open Sans
        sns.set_theme(context='paper', style='ticks', font='Arial', font_scale=2, rc={'lines.linewidth': 2, 'grid.linewidth':0.6, 'grid.color': '#9d9d9d', 
                                                                                      'axes.linewidth':0.6, 'axes.edgecolor': '#9d9d9d'})      
        self.colors = ['#ffc640']
        sns.set_palette(sns.color_palette(self.colors))

        """Plot Empowerment value distribution (as a Historgram).
        """
        plt.figure(figsize=(12,5))

        for i in range(2):
            # import parent table 
            if i == 0:
                with open('data/ParentTable.json', encoding='utf8') as infile:
                    parents = json.load(infile)
                    parents = {int(k):v for k,v in parents.items()}
            else:
                # Parent table where each parent has its own dict entry consisting of all resulting children.
                with open('data/ChildrenEmpowermentTable-{}-{}.json'.format(self.split_version, self.vector_version), encoding='utf8') as infile:
                    parents = json.load(infile)
                    parents = {int(k):v for k,v in parents.items()}

            # get array of empowerment values
            empowerment = np.zeros(self.n_elements)
            for element_id in range(self.n_elements):
                if element_id in parents:
                    empowerment[element_id] = len(parents[element_id])
                else:
                    empowerment[element_id] = 0
            self.empowerment.append(empowerment)

            # plot data as histogram (density)
            plt.subplot(1,2,i+1)
            ax = sns.histplot(data=empowerment, kde=True, bins=20)
            #x.set_xscale('log')
            #ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
            #ax.xaxis.get_major_formatter().set_scientific(False)
            #plt.minorticks_off()

            # plot mean
            if len(ax.lines) != 0:
                kdeline = ax.lines[0]
                mean = empowerment.mean()
                print(mean)
                height = np.interp(mean, kdeline.get_xdata(), kdeline.get_ydata())
                ax.vlines(mean, 0, height, ls='dashed', color='#444444', linewidth=1)
                
            # set titles, labels
            #plt.xlim(left=0)
            plt.xlabel('Empowerment')
            plt.ylabel('Count')
            plt.tight_layout()
            
        # save figure
        filename = 'cdsoi_empowerment/alldata/figures/EmpowermentHistogram-{}-{}'.format(self.split_version, self.vector_version)
        plt.savefig('{}.svg'.format(filename))
        plt.savefig('{}.png'.format(filename))
        plt.savefig('{}.pdf'.format(filename))
        plt.close()

        """Plots empowerment value distribution (as a Scatterplot).
        """
        # make scatter/regression plot 
        g = sns.jointplot(x=self.empowerment[0], y=self.empowerment[1], kind="reg")
        g.ax_joint.set_xscale('log')
        g.ax_joint.set_yscale('log')
        g.ax_joint.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        g.ax_joint.xaxis.get_major_formatter().set_scientific(False)
        g.ax_joint.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        g.ax_joint.yaxis.get_major_formatter().set_scientific(False)
        
        # set titles, labels
        plt.xlabel('True empowerment')
        plt.ylabel('Predicted empowerment')
        plt.tight_layout()
        
        # save figure
        filename = 'cdsoi_empowerment/alldata/figures/EmpowermentRegression-{}-{}'.format(self.split_version, self.vector_version)
        plt.savefig('{}.svg'.format(filename))
        plt.savefig('{}.png'.format(filename))
        plt.savefig('{}.pdf'.format(filename))
        plt.close()

