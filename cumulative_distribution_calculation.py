# a universal class for cdfs and pdfs? difficult though possible, with many functions depending
# on what i want to look at?
# here I subset the dataframes based on stimulus, then I choose the variable i want to calculatwe pdf cdf for.
# What I can DO IS add the streak length here. or anywhere else whatsoever. and then have the streak length in the form
# of a dataframe and then take the right dataframe and the right variable name and you end up using the same function
# very beautiful!
import datetime
import h5py
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
import tables
from tqdm import tqdm
import math

class cdf_pdf_calculation:
    def __init__(
            self,
            path_to_input =r"C:\Users\ag-bahl\Desktop\plaids\data_preprocessed_plaids.hdf5",
            list_dfs = ['gray'],
            list_dfs_str = ['gray'],
            variable = 'ring_membership',
            compare_which = [],
            ):
        # Set user input
        self.path_to_input = Path(path_to_input)
        self.list_dfs = list_dfs
        self.list_dfs_str = list_dfs_str
        self.variable = variable
        self.compare_which = compare_which
    def run(self, **kwargs):
        """Main function to perform sanity check."""
        # Update class attributes if given
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.load_df()
        self.subset_dfs()
        self.cdf_pdf_together()

      #  return self.return_df()

    def load_df(self):
        """Load bout-level dataframe from folder."""
        self.df = pd.read_hdf(self.path_to_input)
        self.df = self.df[np.isfinite(self.df['radius'])]
        self.df.sort_values(['fish_ID', 'start_time_absolute'], ascending=True,inplace = True)
        self.df.reset_index(inplace = True)

    def subset_dfs(self):
        self.list_dfs = []
        self.list_dfs_str = self.df['stimulus_name'].unique().tolist()
        for i in self.list_dfs_str:
            self.list_dfs.append(self.df[self.df['stimulus_name'] == i])
# anything that is used again becomes self?        return list_dfs, list_dfs_str

    def add_combinations_subsets(self): # combine subset dataframes into a new dataframe
        for combo in range(len(self.ls_dfs)):
            self.list_dfs.append(pd.concat(self.ls_dfs[combo]))
            self.list_dfs_str.append(self.ls_dfs_str[combo])

# anything that is used again becomes self?           return list_dfs,list_dfs_str

    def pdf_cdf(self, list_var, plot_which, label_1):
        plt.rcParams["figure.figsize"] = (20, 10)
        count, bins_count = np.histogram(list_var, bins=10)
        pdf = count / sum(count)
        cdf = np.cumsum(pdf)
        ls = [cdf, pdf]
        plt.plot(bins_count[1:], ls[plot_which], label=label_1)

    def cdf_pdf_together(self):
        zero_one = [0, 1]
        plot_which = ['CDF', 'PDF']
        for pl in zero_one:
            for i in range(len(self.list_dfs)):
                df_temp = self.list_dfs[i]
                list_values = df_temp[self.variable].tolist()
                self.pdf_cdf(list_values, zero_one[pl], self.list_dfs_str[i])
            h = max(list_values)-1 # maximum of  ring membership - 1
            if variable == 'radius':
                h = 1
            sampl = np.random.uniform(low=0, high=h, size=(len(df_temp),))
            self.pdf_cdf(sampl, zero_one[pl], ' Random')
            plt.legend()
            plt.title(str(plot_which[pl]) + ' ' + self.variable, size=20)
            plt.show()
  #  def cdf_pdf_separate(self):
   #     if len(self.separate_which)>0:
    #        for subdf in self.separate_which:


# add compare pairs of subsets and which
# add the histogram thing...
# test it tomorrow :/
x = cdf_pdf_calculation()

x.run()






