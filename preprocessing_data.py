import datetime
import h5py
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
import tables
from tqdm import tqdm
import math

class PreprocessingData:
    def __init__(
            self,
            path_to_input = r"C:\Users\ag-bahl\Desktop\gray_four_directions\data_combined.hdf5",
            path_to_output =r"C:\Users\ag-bahl\Desktop\gray_four_directions\gray_four_directions_data.hdf5",
            bout_angle_threshold = 2,
            ):
        # Set user input
        self.path_to_input = Path(path_to_input)
        self.path_to_output = Path(path_to_output)
        self.bout_angle_threshold = bout_angle_threshold
        self.nr_rings = 10

    def run(self, **kwargs):
        """Main function to perform sanity check."""
        # Update class attributes if given
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.load_df()
        self.fish_ids()
    #    self.radius()
        self.flip_flops()
        self.label_bouts()
#        self.ring_membership()
        self.store_df()
      #  return self.return_df()

    def load_df(self):
        """Load bout-level dataframe from folder."""
        self.df = pd.read_hdf(self.path_to_input)
        self.df.sort_values(['fish_ID', 'start_time_absolute'], ascending=True,inplace = True)
        self.df.reset_index(inplace=True)

    def store_df(self):
        """Store bout-level dataframe with excluded trials to folder."""
        print(f"Storing sanity check to {self.path_to_output}...")
        # Store dataframe as hdf5 file
        self.df.to_hdf(str(self.path_to_output), key="all_events", complevel=9)

    def fish_ids(self):
        self.df['fish_ID'] = self.df.loc[:, 'folder_name'] # fish_name folder_name
        old_fish_name = self.df['folder_name'].unique().tolist()
        new_fish_name = list(range(0, len(old_fish_name)))
        self.df['fish_ID'].replace(old_fish_name, new_fish_name, inplace=True)
        print('done with fish_ids')
#        return self.df

    def flip_flops(self):
        # put them in a stupid list first?
        nr = 0
        flip_list = []
        for row in tqdm(range(0,len(self.df))):
            if self.df['stimulus_name'][row] == 'motion_rightward': # ((self.df['stimulus_name'][row] == 'motion_rightdown45') or (self.df['stimulus_name'][row] =='motion_rightup45') or(
                flip_list.append(-self.df['estimated_orientation_change'][row])
                nr+=1
            else:
                flip_list.append(self.df['estimated_orientation_change'][row])
        print(len(self.df), nr)
        self.df['estimated_orientation_change_flipped'] = flip_list
        print("done flip flops")
      #  return df


    def label_bouts(self):
        self.df['time'] = self.df['end_time']
        self.df['left_bouts'] = np.nan
        self.df['right_bouts'] = np.nan
        self.df['straight_bouts'] = np.nan
        self.df['bout_orientation'] = np.nan
        self.df['bout_orientation_absolute'] = np.nan
        self.df['left_bouts_absolute'] = np.nan
        self.df['right_bouts_absolute'] = np.nan

        for index in tqdm(range(len(self.df))):
            ori_change = self.df['estimated_orientation_change_flipped'].iloc[index]
            if ori_change > self.bout_angle_threshold:
                self.df['left_bouts'].iloc[index] = 1
                self.df['right_bouts'].iloc[index] = 0
                self.df['straight_bouts'].iloc[index] = 0
                self.df['bout_orientation'].iloc[index] = 1

            elif ori_change < -self.bout_angle_threshold:
                self.df['left_bouts'].iloc[index] = 0
                self.df['right_bouts'].iloc[index] = 1
                self.df['straight_bouts'].iloc[index] = 0
                self.df['bout_orientation'].iloc[index] = -1

            elif abs(ori_change) < self.bout_angle_threshold:
                self.df['left_bouts'].iloc[index] = 0
                self.df['right_bouts'].iloc[index] = 0
                self.df['straight_bouts'].iloc[index] = 1
                self.df['bout_orientation'].iloc[index] = 0

            else:
                self.df['left_bouts'].iloc[index] = np.nan
                self.df['right_bouts'].iloc[index] = np.nan
                self.df['straight_bouts'].iloc[index] = np.nan
                self.df['bout_orientation'].iloc[index] = np.nan
            if ori_change>0:
                self.df['bout_orientation_absolute'].iloc[index] = 1
                self.df['left_bouts_absolute'].iloc[index] = 1
                self.df['right_bouts_absolute'].iloc[index] = 0
            elif ori_change<0:
                self.df['bout_orientation_absolute'].iloc[index] = -1
                self.df['left_bouts_absolute'].iloc[index] = 0
                self.df['right_bouts_absolute'].iloc[index] = 1
            else:
                self.df['bout_orientation_absolute'].iloc[index] = np.nan
                self.df['left_bouts_absolute'].iloc[index] = np.nan
                self.df['right_bouts_absolute'].iloc[index] = np.nan
        print('labelling done')
      #  return self.df
    def radius(self):
        x = np.square(np.array(self.df['end_x_position'].tolist()))
        y = np.square(np.array(self.df['end_y_position'].tolist()))
        r = np.sqrt(np.add(x, y))
        self.df['radius'] = r
        print('radius done')

    def ring_membership(self):
        c = 1 / (math.sqrt(self.nr_rings))
        radi_ls = [0, c]
        for i in range(2, self.nr_rings + 1):
            radi_ls.append(c * (math.sqrt(i)))
        a = np.empty((len(self.df),))
        a[:] = np.nan
        ring_membership = a.tolist()
        radius_l = self.df['radius'].tolist()
        ring = 0
        for couples in range(1, len(radi_ls)):
            for val in range(len(self.df)):
                if ((radius_l[val] >= radi_ls[couples - 1]) and (radius_l[val] <= radi_ls[couples])):
                    ring_membership[val] = ring
            ring += 1
        self.df['ring_membership'] = ring_membership

        #        return ls_cs
# to be added - stimnulus gray light versus stimulus actual.
# to be added - the preprocessing plots!!!

x = PreprocessingData()
x.run()


