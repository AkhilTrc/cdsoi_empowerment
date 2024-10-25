import pandas as pd

if __name__ == '__main__':
    split_version = "data"
    vector_version = "crawl300"
    empowerment_info = pd.read_csv('cdsoi_empowerment/alldata/tables/EmpowermentTable-{}-{}.csv'.format(split_version, vector_version), dtype={'first': int, 'second': int, 'predResult': int, 'empComb': float, 'empChild': float, 'binComb': float, 'binChild': float})
