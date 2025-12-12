"""

    Multi-criteria optimisation project
    Fougeroux Alex & Robert Paul-Aime

"""
import pandas as pd
from .utils.const import DEFAULT_PATH, DEFAULT_FILE, DEFAULT_FILE_LIST


def get_df(file_name = DEFAULT_FILE, trace = False):
    """
        extract the data from the csv files in the data dir

        input:
            (String): file_name
            (Bool): trace : feedback of failure or success
        output:
            (pandas: data frame): file_df
    """
    try:
        file_df = pd.read_csv(DEFAULT_PATH + file_name,index_col = "Date")
        # file_df.index = pd.to_date_time(df.index)
        if trace:
            print("successfully extracted " + DEFAULT_PATH + file_name)
        return file_df
    except:
        if trace:
            print("failed to extract " + DEFAULT_PATH + file_name) 
        pass


def get_all_df(file_names = DEFAULT_FILE_LIST, trace = False):
    """
        extract all the data from csv files in the data dir
        
        input:
            (list[String]): file_names
            (Bool): trace : feedback of failure or success
        output:
            (list[pandas: data frame]): file_df_list
    """
    file_df_list = []
    if trace:
        print("proceed to extract:", end = " ")
        for file_name in file_names:
            print(file_name, end = ", ")
        print("")
    for file_name in file_names:
        file_df_list.append(get_df(file_name, trace))
    return file_df_list

""" test get_df 
df = get_df(trace = True)
print(type(df),df.head())
"""

"""test get_all_df
dfs = get_all_df(trace = True)
print(dfs[3].head())
"""
