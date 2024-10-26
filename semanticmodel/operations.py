import json
import numpy as np
import pandas as pd
import seaborn as sns
import fastText.python.fasttext_module.fasttext as fasttext 
import fasttext.util

class Operations():
    def __init__():
        """Initialize the Operations class
        """

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
                # table_csv = self.expand_combination_table(table_csv)

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

    def get_similarities(self, vector_version='crawl300'):
        """Initializes similarity class.
        """
        # print info for user
        print('\nPlot similarity values for version with {} vectors.'.format(vector_version))
        
        # set general settings for plotting 
        # TODO: change font to Open Sans
        sns.set_theme(context='paper', style='ticks', font='Arial', font_scale=2, rc={'lines.linewidth': 2, 'grid.linewidth':0.6, 'grid.color': '#9d9d9d', 
                                                                                      'axes.linewidth':0.6, 'axes.edgecolor': '#9d9d9d'})      
        self.colors = ['#bf3409']
        sns.set_palette(sns.color_palette(self.colors))
        
        # plot values
        # self.plot_similarity_histogram()

        if vector_version == 'ccen100' or vector_version == 'ccen300' or vector_version == 'crawl100' or vector_version == 'crawl300' or vector_version == 'wiki100' or vector_version == 'wiki300':
            vectors = np.loadtxt('data/{}ElementVectors-{}.txt'.format(vector_version))
        else:
            raise ValueError('Undefined vector_version: "{}". Use "ccen100", "ccen300", "crawl100", "crawl300", "wiki100" or "wiki300" instead.'.format(vector_version))
        
        """

        FOR PLOTTING SIMILARITY VALUES...

        """

    def get_empowerment(self, vector_version, similarity_version):
        empowerment = list()

        # set general settings for plotting 
        # TODO: change font to Open Sans
        sns.set_theme(context='paper', style='ticks', font='Arial', font_scale=2, rc={'lines.linewidth': 2, 'grid.linewidth':0.6, 'grid.color': '#9d9d9d', 
                                                                                      'axes.linewidth':0.6, 'axes.edgecolor': '#9d9d9d'})      
        self.colors = ['#ffc640']
        sns.set_palette(sns.color_palette(self.colors))
        
        # plot values
        # self.plot_empowerment_histogram()
        # self.plot_empowerment_scatter_plot()

        """
        
        FOR PLOTTING EMPOWERMENT VALUES...
        
        """