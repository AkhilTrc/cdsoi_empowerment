from operations import Operations

if __name__ == '__main__':
    op = Operations()
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
    
    op.get_similarities('crawl300')

    op.get_empowerment('data', 'crawl300')