import datetime
import h5py
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
import tables
from tqdm import tqdm
import math
# add to sanity check -
#TODO fix this - df['distance_change'] = df['distance_change'].astype(np.float64)
# TODO - ADD A VARIABLE THAT SHOWS THE WAVELENGTH,CONTRAST AND EVERYTHING ELSE.. (THEY ARE TO COME FROM THE STIMULUS NAME... IN THE FUTURE
# TODO - add a variable that shows a graph or whatever, a big plot of sample size and whatever the sanity check was
# TODO - as a sanity check, I see how long the experiment lasted for each fish. and a graph of fish activity through time to see if it dies. and then plot it all together!
# TODO - MULTIINDEXING EVERYTHING???????????
# supposed to show...
# flip dicts stay here:
"""for sine gratings: 
            flip_dict={'motion_rightdown45': 'motion_leftdown45',
                       'motion_rightward': 'motion_leftward',
                       'motion_rightup45': 'motion_leftup45', },
                       
    for plaids:
            flip_dict =  {'45_plaid_-45':'45_plaid_45',
                          '45_plaid_-90':'45_plaid_90',
                          '60_plaid_-45':'60_plaid_45',
                          '60_plaid_-90':'60_plaid_90',
                          '75_plaid_-45':'75_plaid_45',
                          '75_plaid_-90':'75_plaid_90',}
    for stationary dot hole :
    flip_dict = {'dir0_coh100_den1200_speed0.3_liftime0.011111111111111112_brightness0.5_size0.01_innerr0_outerr1': 'dir1_coh100_den1200_speed0.3_liftime0.011111111111111112_brightness0.5_size0.01_innerr0_outerr1',
                  'dir0_coh100_den1200_speed0.3_liftime0.011111111111111112_brightness0.5_size0.01_innerr0.2_outerr1':'dir1_coh100_den1200_speed0.3_liftime0.011111111111111112_brightness0.5_size0.01_innerr0.2_outerr1',
                  'dir0_coh100_den1200_speed0.3_liftime0.011111111111111112_brightness0.5_size0.01_innerr0.5_outerr1': 'dir1_coh100_den1200_speed0.3_liftime0.011111111111111112_brightness0.5_size0.01_innerr0.5_outerr1',
                    }
    for close looped dot hole:
    flip_dict = {     'dir0_coh100_den1200_speed0.3_liftime0.011111111111111112_brightness0.5_size0.01_innerr0_outerr1':'dir1_coh100_den1200_speed0.3_liftime0.011111111111111112_brightness0.5_size0.01_innerr0_outerr1',
                  'dir0_coh100_den1200_speed0.3_liftime0.011111111111111112_brightness0.5_size0.01_innerr0.1_outerr1':'dir1_coh100_den1200_speed0.3_liftime0.011111111111111112_brightness0.5_size0.01_innerr0.1_outerr1',
                  'dir0_coh100_den1200_speed0.3_liftime0.011111111111111112_brightness0.5_size0.01_innerr0.2_outerr1':'dir1_coh100_den1200_speed0.3_liftime0.011111111111111112_brightness0.5_size0.01_innerr0.2_outerr1',
                  'dir0_coh100_den1200_speed0.3_liftime0.011111111111111112_brightness0.5_size0.01_innerr0.3_outerr1':'dir1_coh100_den1200_speed0.3_liftime0.011111111111111112_brightness0.5_size0.01_innerr0.3_outerr1',
                  'dir0_coh100_den1200_speed0.3_liftime0.011111111111111112_brightness0.5_size0.01_innerr0.4_outerr1':'dir1_coh100_den1200_speed0.3_liftime0.011111111111111112_brightness0.5_size0.01_innerr0.4_outerr1',
                  'dir0_coh100_den1200_speed0.3_liftime0.011111111111111112_brightness0.5_size0.01_innerr0.5_outerr1': 'dir1_coh100_den1200_speed0.3_liftime0.011111111111111112_brightness0.5_size0.01_innerr0.5_outerr1',
                    }
            """

class PreprocessingData:
    def __init__(
            self,
            path_to_folder = r"C:\Users\ag-bahl\Desktop\data_processed\eight_directions",
            bout_angle_threshold = 2,
            ori_change_col = 'estimated_orientation_change',
            flip_dict={"bla":"bla" }
            ):
        # Set user input
        self.path_to_folder = Path(path_to_folder)
        self.bout_angle_threshold = bout_angle_threshold
        self.nr_rings = 10
        self.ori_change_col = ori_change_col
        self.flip_dict = flip_dict
        self.path_to_input_file = self.path_to_folder.joinpath('data_combined_checked.hdf5')
        self.path_to_output_file = self.path_to_folder.joinpath('data_preprocessed.hdf5')
    def run(self, **kwargs):
        """Main function to perform sanity check."""
        # Update class attributes if given
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.load_df()
        self.flip_flops()
        self.fish_ids()
        self.radius()
        self.label_bouts()
        self.ring_membership()
        self.store_df()
      #  return self.return_df()

    def load_df(self):
        """Load bout-level dataframe from folder."""
        self.df = pd.read_hdf(self.path_to_input_file)
        self.df.sort_values(['folder_name', 'start_time_absolute'], ascending=True,inplace = True)
        self.df.reset_index(inplace = True)

    def store_df(self):
        """Store bout-level dataframe with excluded trials to folder."""
        self.df.sort_values(['folder_name', 'start_time_absolute'], ascending=True,inplace = True)
        self.df.reset_index(inplace=True,drop=True)
#        self.df.drop(['index'],inplace = True,axis = 1)
        print(f"Storing preprocessed file to {self.path_to_output_file}...")
        # Store dataframe as hdf5 file
        self.df.to_hdf(str(self.path_to_output_file), key="all_events", complevel=9)
        print('storage done')


    def fish_ids(self):
        # TODO - WHAT THE HEP IS THIS BELOW?
        self.df['fish_ID'] = self.df.loc[:, 'folder_name'] # fish_name folder_name
        old_stimulus_name = ['dir0_coh100_den1200_speed0.3_liftime0.011111111111111112_brightness0.5_size0.01_innerr0_outerr1',
                'dir0_coh100_den1200_speed0.3_liftime0.011111111111111112_brightness0.5_size0.01_innerr0.1_outerr1',
                'dir0_coh100_den1200_speed0.3_liftime0.011111111111111112_brightness0.5_size0.01_innerr0.2_outerr1',
                'dir0_coh100_den1200_speed0.3_liftime0.011111111111111112_brightness0.5_size0.01_innerr0.3_outerr1',
                'dir0_coh100_den1200_speed0.3_liftime0.011111111111111112_brightness0.5_size0.01_innerr0.4_outerr1',
                 'dir0_coh100_den1200_speed0.3_liftime0.011111111111111112_brightness0.5_size0.01_innerr0.5_outerr1',
                ]

        new_stimulus_name = ['inner0_outer_1',
                'inner_0.1_outer_1',
                'inner_0.2_outer_1',
                'inner_0.3_outer_1',
                'inner_0.4_outer_1',
                'inner_0.5_outer_1',]
        old_fish_name = self.df['folder_name'].unique().tolist()
        new_fish_name = list(range(0, len(old_fish_name)))  # here you need a loc[].blabla! ! !
        self.df['fish_ID'].replace(old_fish_name, new_fish_name, inplace=True)
        self.df['stimulus_name'].replace(old_stimulus_name, new_stimulus_name, inplace=True)
        print('done with fish_ids')
#        return self.df

    def flip_flops(self):
        index_names = self.df.index
        print(len(index_names))
        self.df.reset_index(inplace=True,drop=True)  # Need to go back to a normal column-structure for this
        for flip_name in list(self.flip_dict.keys()):
            self.df.loc[self.df['stimulus_name'] == flip_name, "estimated_orientation_change"] *= -1
            self.df.loc[self.df['stimulus_name'] == flip_name, "stimulus_name"] = self.flip_dict[flip_name]
        self.df.set_index(index_names, inplace=True)
        self.df.sort_index(inplace=True)
        print('done flipping data')

    #  return df
    def label_bouts(self):
        bout_angle_threshold = 2

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
        print(' the radi of each ring from 0 to 9: '+str(radi_ls))
        print('ring done')
        #        return ls_cs
# to be added - stimnulus gray light versus stimulus actual.
# to be added - the preprocessing plots!!!





x = PreprocessingData()
x.run()



