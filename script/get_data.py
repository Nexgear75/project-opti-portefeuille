"""

    Multi-criteria optimisation project
    Fougeroux Alex & Robert Paul-Aime

"""
import pandas as pd

DEFAULT_PATH = "data/"
DEFAULT_FILE = "Energy.csv"
DEFAULT_FILE_LIST = [
    "Energy.csv",
    "Financials.csv",
    "Health_Care.csv",
    "Industrials.csv",
    "Information_Technology.csv",
    "Materials.csv",
    "Real_Estate.csv",
    "Utilities.csv"
]

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
        file_df = pd.read_csv(DEFAULT_PATH + file_name)
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
