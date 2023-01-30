import datetime
import h5py
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
import tables
from tqdm import tqdm
#%matplotlib inline
import random
import math
# TODO PLOT HISTOGRAMS AS A LINE INSTEAD OF JUST PDF CDF SO PEOPLE UNDERSTAND IT BETTER BUT YOU ALSO GET A GRIP OF WHAT IT LOOKS LIKE..
#  and see again how you defined the random data, they could also be data retrieved not from the exact same distribution but from a random distribution with the same variab

import numpy as np
from scipy.stats import ranksums

plt.rcParams["figure.figsize"] = (8, 5)

pd.options.mode.chained_assignment = None  # default='warn'
# here for sure

# TODO : name of the folder becomes title of the graph
# TODO : ADD a function to combine left right extra!
# TODO : Merge without flipping
class StreakLength:
    def __init__(
            self,
            path_to_input1=r"C:\Users\ag-bahl\Desktop\data\sine_45_wv_0_15\data_preprocessed.hdf5",
            angle_value = 5, #meeeeeeeeeeeeeeh you need complete redefinition of left right straight...

            ):

        # Set user input
        self.path_to_input1 = Path(path_to_input1)
        self.angle_value = angle_value
        self.bout_angle_threshold = angle_value

    def run(self, **kwargs):
        """Main function to perform sanity check."""
        # Update class attributes if given
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.load_df()
        self.label_bouts()
        self.subset_dfs()
    #    self.add_combinations_subsets()
        self.looper_function_edited()
       # self.plot_pref(separate = True)

      #  return self.return_df()

    def load_df(self):
        """Load bout-level dataframe from folder."""
        self.df = pd.read_hdf(self.path_to_input1)
        self.df['duration'] = self.df['duration'].astype(np.float64) # temp
        self.df['distance_change'] = self.df['distance_change'].astype(np.float64)
        self.df['bout_orientation'] = self.df['bout_orientation'].astype(np.float64)
        print(self.df['stimulus_name'].unique().tolist())
    def label_bouts(self):

        self.df['time'] = self.df['end_time']
        self.df['left_bouts'] = np.nan
        self.df['right_bouts'] = np.nan
        self.df['straight_bouts'] = np.nan
        self.df['bout_orientation'] = np.nan
        self.df['bout_orientation_absolute'] = np.nan
        self.df['left_bouts_absolute'] = np.nan
        self.df['right_bouts_absolute'] = np.nan
        index_names = self.df.index
        print(len(index_names))
        self.df.reset_index(inplace=True,drop=True)  # Need to go back to a normal column-structure for this
        # bigger than | right
        self.df.loc[self.df['estimated_orientation_change'] > self.bout_angle_threshold, "bout_orientation"] = 1
        self.df.loc[self.df['estimated_orientation_change'] > self.bout_angle_threshold, "right_bouts"] = 1
        self.df.loc[self.df['estimated_orientation_change'] > self.bout_angle_threshold, "left_bouts"] = 0
        self.df.loc[self.df['estimated_orientation_change'] > self.bout_angle_threshold, "straight_bouts"] = 0
        # smaller | than left
        self.df.loc[self.df['estimated_orientation_change'] < - self.bout_angle_threshold, "bout_orientation"] = -1
        self.df.loc[self.df['estimated_orientation_change'] < - self.bout_angle_threshold, "right_bouts"] = 0
        self.df.loc[self.df['estimated_orientation_change'] < - self.bout_angle_threshold, "left_bouts"] = 1
        self.df.loc[self.df['estimated_orientation_change'] < - self.bout_angle_threshold, "straight_bouts"] = 0
        # absolute value | straight
        self.df.loc[abs(self.df['estimated_orientation_change']) < self.bout_angle_threshold, "bout_orientation"] =0
        self.df.loc[abs(self.df['estimated_orientation_change']) < self.bout_angle_threshold, "right_bouts"] =0
        self.df.loc[abs(self.df['estimated_orientation_change']) < self.bout_angle_threshold, "left_bouts"] =0
        self.df.loc[abs(self.df['estimated_orientation_change']) < self.bout_angle_threshold, "straight_bouts"] =1

        # absolutes:
        # left
        self.df.loc[self.df['estimated_orientation_change'] < 0, "left_bouts_absolute"] = 1
        self.df.loc[self.df['estimated_orientation_change'] < 0, "right_bouts_absolute"] = 0
        self.df.loc[self.df['estimated_orientation_change'] < 0, "bout_orientation_absolute"] = -1
        # right
        self.df.loc[self.df['estimated_orientation_change'] > 0, "left_bouts_absolute"] = 0
        self.df.loc[self.df['estimated_orientation_change'] > 0, "right_bouts_absolute"] = 1
        self.df.loc[self.df['estimated_orientation_change'] > 0, "bout_orientation_absolute"] = 1
        print('done labeling bouts')
      #  return self.df
    def subset_dfs(self):
        self.list_dfs = []
        self.list_dfs_str = self.df['stimulus_name'].unique().tolist()
        for i in self.list_dfs_str:
            self.list_dfs.append(self.df[self.df['stimulus_name'] == i])  # TODO use mutlindexing here and .ax
    # anything that is used again becomes self?        return list_dfs, list_dfs_str

    def add_combinations_subsets(self):  # combine subset dataframes into a new dataframe
        ls_dfs =[self.list_dfs[0],self.list_dfs[1],self.list_dfs[2]] #,self.list_dfs[3],self.list_dfs[4]]
        ls_dfs_str = '45 degree up and plaids with 45 degree orientation'
        self.list_dfs.append(pd.concat(ls_dfs))
        self.list_dfs_str.append(ls_dfs_str)

    def streak_length(self,df,rand = False):
        if rand:
            list_of_directions = df
        else:
            list_of_directions = df['bout_orientation'].tolist()

        list_of_streaks = []  # i save here the length of the streak, defined as a_streak
        a_streak = 1  # we keep track of each streak length here, and when the streak is done, we append it to the list_of_streaks. streak length is 1 if only one bout was done in that direction
        for element in range(1, len(list_of_directions)):  # we start from one bcs we compare with 0th element
            if (list_of_directions[element - 1] == list_of_directions[element]):
                a_streak += 1
            else:
                list_of_streaks.append(a_streak)
                a_streak = 1
        list_of_streaks.append(a_streak)

    #    list_of_streaks = [x for x in list_of_streaks if x <= 20]
        plt.rcParams["figure.figsize"] = (20, 10)
       # plt.hist(list_of_streaks, bins=20)
        #plt.title('streak length, all directions',size = 20)
        #plt.show()
        return list_of_streaks
     #   return list_of_streaks
    def streak_length_orientation(self,df, angle_value):
        nr_mesups = 0
        list_of_directions = df['bout_orientation'].tolist()

        list_of_streaks_left = []  # i save here the length of the streak, defined as a_streak
        list_of_streaks_right = []
        list_of_streaks_straight = []
        left_streak = 1
        straight_streak = 1
        right_streak = 1  # we keep track of each streak length here, and when the streak is done, we append it to the list_of_streaks. streak length is 1 if only one bout was done in that direction
        # we create a new list of directions for each trial. and the new list makes it impossible for them to concat!
        for element in range(1, len(list_of_directions)):  # we start from one bcs we compare with 0th element
            if (list_of_directions[element - 1] == list_of_directions[element]):
                if list_of_directions[element]==-1.0:
                    left_streak +=1
                elif list_of_directions[element]==1.0:
                    right_streak +=1
                elif list_of_directions[element]==0.0:
                    straight_streak+=1
                else:
                    print(' now i dont understand was happenin')
            else:
                if ((left_streak>1) & (right_streak==1) & (straight_streak==1)):
                    list_of_streaks_left.append(left_streak)
                    left_streak = 1
                elif ((left_streak==1) & (right_streak>1) & (straight_streak==1)):
                    list_of_streaks_right.append(right_streak)
                    right_streak = 1
                elif ((left_streak==1) & (right_streak==1) & (straight_streak>1)):
                    list_of_streaks_straight.append(straight_streak)
                    straight_streak = 1
                elif ((left_streak==1) & (right_streak==1) & (straight_streak==1)):
                    if (list_of_directions[element - 1] == -1):
                        list_of_streaks_left.append(left_streak)
                    elif (list_of_directions[element - 1] == 1):
                        list_of_streaks_right.append(right_streak)
                    elif (list_of_directions[element - 1] == 0):
                        list_of_streaks_straight.append(straight_streak)
                    else:
                        print(' now i also dont understand was happenin')

                else:
                    nr_mesups+=1
      #  print('nr_mesups ' + str(nr_mesups))
      #  print('df length'+ str(len(df)))
       # list_of_streaks.append(a_streak)
        list_of_streaks_left = [x for x in list_of_streaks_left if x <= 20]
        list_of_streaks_right = [x for x in list_of_streaks_right if x <= 20]
        list_of_streaks_straight = [x for x in list_of_streaks_straight if x <= 20]
        ls_together = list_of_streaks_left+list_of_streaks_right

        return list_of_streaks_left,list_of_streaks_right,list_of_streaks_straight
       # list_of_streaks = [x for x in list_of_streaks if x <= 20]
    def plotter_left_right(self,ls_str,title):

        for elm in range(len(ls_str)):
            count_l, bins_count_l = np.histogram(ls_str[elm], bins=np.arange(0, 15, 0.5, dtype=int))
            pdf_l = count_l / sum(count_l)
            cdf_l = np.cumsum(pdf_l)

            plt.plot(bins_count_l[1:], cdf_l, label=lb[elm], linewidth=1)
        #    plt.scatter(bins_count_l[1:], cdf_l, label=lb[elm], s=5)

            plt.title(title,size = 20)
        plt.legend()
        plt.scatter([1,1],[0.68,1],s = 0.1,color = 'dimgray')
        plt.savefig("C:/Users/ag-bahl/Desktop/labmeeting_prep/motion_forward/" + str(title) + ".png" )
        plt.show()
    def plotter_angles(self,ls_str,title, lb): # ls_lefts - has 4 elements, gratings plaids1,2,3
        for elm in range(len(ls_str)):
            count_l, bins_count_l = np.histogram(ls_str[elm], bins=np.arange(0, 15, 1, dtype=int))
            pdf_l = count_l / sum(count_l)
            cdf_l = np.cumsum(pdf_l)

            plt.plot(bins_count_l[1:], cdf_l, label=lb[elm], linewidth=1)
        #    plt.scatter(bins_count_l[1:], cdf_l, label=lb[elm], s=5)
            plt.title(title,size = 10)

        plt.legend()
        #plt.savefig("C:/Users/ag-bahl/Desktop/labmeeting_prep/motion_forward/" + str(title) + ".png" )
        plt.show()
        print(title + " variables compared are :" +lb[0] + " and " + lb[2]  )
        print(ranksums(ls_str[0], ls_str[2]))

    def looper_function_edited(self): # the plotter angle now receives from looper function! 4 different lists, for each variable!
        # the subset list_dfsa contains dataframes for each stimulus! thats good. we subset from it again!
        # store them in separate lists, lefts for grating, lefts for plaid 45 etc... then you plot lefts rights straights separately! and a random shuffled version of each
        ls_lefts = []
        ls_rights = []
        ls_straights = []
        ls_dfs =[self.list_dfs[0],self.list_dfs[1],self.list_dfs[2]]#,self.list_dfs[3]]#,self.list_dfs[4]] # forward_0 = 4,6,11,12,5 | leftup_45 = 0,7,9,13,5 | left_90 = 2, 8, 10, 14,5
        ls_dfs_str = 'motion_up45 and plaids with angle 45 '
        new_subset = pd.concat(ls_dfs) # one df with 4 stimuli!
        stimuli_new_s = new_subset['stimulus_name'].unique().tolist() # this will also become the list of titles!
        print(stimuli_new_s)
        for subset in range(len(ls_dfs)): # here we loop through the stimuli, and plot separately for each orientation left right straight!
            ss_df = new_subset[new_subset['stimulus_name']==stimuli_new_s[subset]]
            list_of_streaks_left, list_of_streaks_right, list_of_streaks_straight = self.streak_length_orientation(ss_df,self.angle_value)
            ls_lefts.append(list_of_streaks_left)
            ls_rights.append(list_of_streaks_right)
            ls_straights.append(list_of_streaks_straight)
        ls_ori = ['LEFT', 'RIGHT', 'STRAIGHT']
        lsls = [ls_lefts,ls_rights,ls_straights]

        for i in range(3):
            self.plotter_angles(lsls[i],title = ls_ori[i]+ ' Streaks distribution, angle used '+ str(self.angle_value) +' degrees, ',lb = stimuli_new_s)
# 2 116 357
    def looper_function(
            self):  # the plotter angle now receives from looper function! 4 different lists, for each variable!
        # the subset list_dfsa contains dataframes for each stimulus! thats good. we subset from it again!
        ls_dfs = [self.list_dfs[0], self.list_dfs[7], self.list_dfs[9], self.list_dfs[13]]
        ls_dfs_str = '45 degree up and plaids with 45 degree orientation'

        for subset in range(len(self.list_dfs)):  # here we
            print(self.list_dfs_str[subset])

            rand_vals = (np.random.randint(2, size=len(self.list_dfs[subset]))).tolist()
            list_of_streaks_rand = self.streak_length(rand_vals, rand=True)

            list_of_streaks_subset = self.streak_length(self.list_dfs[subset])
            list_of_streaks_left, list_of_streaks_right, list_of_streaks_straight = self.streak_length_orientation(
                self.list_dfs[subset], self.angle_value)
            ls_str = [list_of_streaks_left, list_of_streaks_right, list_of_streaks_straight, list_of_streaks_rand,
                      list_of_streaks_subset]
            self.plotter_angles(ls_str,title=str(self.list_dfs_str[subset]) + ' angle value ' + str(self.angle_value))


# def strak length to left to the right and straight(+-2 degrees)! and also absolute valyue
x = StreakLength()
x.run()

### archive
"""    def add_combinations_subsets(self):  # combine subset dataframes into a new dataframe
        self.ls_dfs =[self.list_dfs[0],self.list_dfs[7],self.list_dfs[9],self.list_dfs[13]]
        for combo in range(len(self.ls_dfs)):
            self.list_dfs.append(pd.concat(self.ls_dfs[combo]))
            self.list_dfs_str.append(self.ls_dfs_str[combo])"""