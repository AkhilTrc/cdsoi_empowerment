from operations import Operations
from custom_empowerment import CustomEmpowerment

if __name__ == '__main__':
    """ Main function to run the Operations and create gametree, tables, vectors; and calculate empowerment values. 
    """
    n_elements = 1000
    emp = CustomEmpowerment('crawl300', 'data', n_elements)

    op = Operations('crawl300', 'data', n_elements)
    op.get_gametree()

    op.get_tables('combination', expand=True)
    op.get_tables('parent')
    op.get_tables('child')

    op.get_wordvectors('crawl', 300)
    # vector.get_wordvectors('crawl', 100)
    # vector.get_wordvectors('wiki', 300)
    # vector.get_wordvectors('wiki', 100)
    # vector.get_wordvectors('ccen', 300)
    # vector.get_wordvectors('ccen', 100)
    
    op.get_similarities()
    op.get_empowerment()