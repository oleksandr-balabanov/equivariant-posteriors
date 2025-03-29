import pandas
import glob
import matplotlib.pyplot as plt

dfs = [pandas.read_pickle(path) for path in glob.glob("checkpoints/*.df.pickle")]
df = pandas.concat(dfs, ignore_index=True)
df.to_pickle("database.df.pickle")
df.to_csv("database.csv")
