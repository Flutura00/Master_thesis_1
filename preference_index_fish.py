import datetime
import h5py
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
import tables
from tqdm import tqdm
import random
#pd.options.mode.chained_assignment = None  # default='warn'

import seaborn as sns

# orientation change per stimulus
# todo = preference in terms of response frequency toward that angle, and preference in terms average angle of orientation...
# todo = probability correct? in y axis.
# todo = i plot 12 or sth plots per image?
class TimePreferenceIndex:
    def __init__(
            self,
            path_to_input1=r"C:\Users\ag-bahl\Desktop\data_processed\sine_forward_sine_45_plaid_45_wavelength_0_15_not_symmetric\data_preprocessed.hdf5",
            bin_size = 1,
            subset = False,
            pref = 'preference_absolute',
            plot_together = True,
            plot_sem = False,
            ):

        # Set user input
        self.path_to_input1 = Path(path_to_input1)
        self.bin_size = bin_size
        self.subset = subset
        self.pref = pref
        self.plot_together =plot_together
        self.plot_sem = plot_sem

    def run(self, **kwargs):
        """Main function to perform sanity check."""
        # Update class attributes if given
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.load_df()
        self.bin_data()
        self.mean_sem_df()
        self.plot_pref(separate = False)
      #  self.each_fish(separate = True)
     #   self.plot_pref(separate = True)

      #  return self.return_df()

    def load_df(self):
        """Load bout-level dataframe from folder."""
        self.df = pd.read_hdf(self.path_to_input1)
        self.df['duration'] = self.df['duration'].astype(np.float64) # temp
        self.df['distance_change'] = self.df['distance_change'].astype(np.float64)
        self.df['bout_orientation'] = self.df['bout_orientation'].astype(np.float64)
    def bin_data(self): # we group by sum! good!
        if self.subset != False:
            self.df = pd.concat(self.subset)
        time_index = self.df['end_time']
        self.df['binned_time'] = time_index - time_index % self.bin_size + self.bin_size / 2
        self.df = self.df.reset_index()
        self.bin_df = self.df.groupby(['fish_ID', 'stimulus_name', 'binned_time']).sum() # for every fish for every stimulus for every second!fatline = sem, and cloud around it
        self.bin_df['total_bouts'] = self.bin_df.left_bouts + self.bin_df.right_bouts + self.bin_df.straight_bouts
        #Analysed part:
        self.bin_df['preference_index'] = (-self.bin_df.left_bouts + self.bin_df.right_bouts + self.bin_df.straight_bouts) / self.bin_df.total_bouts#(self.bin_df.bout_orientation/self.bin_df.total_bouts)*100# (-self.bin_df.left_bouts + self.bin_df.right_bouts + self.bin_df.straight_bouts)/self.bin_df.total_bouts
        self.bin_df['preference_absolute'] =(-self.bin_df.left_bouts_absolute + self.bin_df.right_bouts_absolute) / self.bin_df.total_bouts#  (self.bin_df.bout_orientation_absolute/self.bin_df.total_bouts)*100
        self.bin_df['percentage_left'] = (self.bin_df.left_bouts / self.bin_df.total_bouts)*100
        self.bin_df['percentage_right'] = (self.bin_df.right_bouts / self.bin_df.total_bouts)*100
        self.bin_df['percentage_straight'] = (self.bin_df.straight_bouts / self.bin_df.total_bouts)*100
        self.bin_df['orientation'] = self.bin_df.estimated_orientation_change/self.bin_df.total_bouts # in terms of angle!
        print(self.bin_df)

        print('done data binning')
       #self.bin_df['left_bouts_absolute'] = (self.bin_df.left_bouts_absolute / self.bin_df.total_bouts)*100
       # self.bin_df['right_bouts_absolute'] = (self.bin_df.right_bouts_absolute / self.bin_df.total_bouts)*100
    def mean_sem_df(self):

        self.time_marker = None
        num_fish = len(self.bin_df.index.unique('fish_ID'))
        print(' number of fish ' +str(num_fish) )
        self.t_stamp = self.bin_df.index.unique('binned_time')
        self.bin_df = self.bin_df.reset_index()

        self.mean_df = self.bin_df.groupby(['stimulus_name', 'binned_time']).mean()
        self.sem_df = self.bin_df.groupby(['stimulus_name', 'binned_time']).std() / (num_fish) ** 0.5
        print('done mean sem')
    def plot_pref(self,separate):
        plt.rcParams["figure.figsize"] = (15, 12)

    #    colors = ['dimgray','red','magenta','orange','green','blue','yellow','black','maroon']# sns.color_palette("hls", 9)
        i = 0
        if self.plot_together == True:
            legend = []
        #   for variable in
        for stim in self.mean_df.index.unique('stimulus_name'):
            legend.append(str(stim))
            plot_mean_df = self.mean_df.xs(stim, level='stimulus_name')
            plot_sem_df = self.sem_df.xs(stim, level='stimulus_name')

            plt.plot(plot_mean_df[self.pref], marker='o', linewidth=3, markersize=2,label = stim)#, color = colors[i])
            if self.plot_sem:
                plt.fill_between(self.t_stamp, plot_mean_df[self.pref] + plot_sem_df[self.pref],
                                 plot_mean_df[self.pref] - plot_sem_df[self.pref],
                                 alpha=0.25)# ,color = colors[i])
            i+=1
            if separate:
                if self.time_marker != None:
                    for t in self.time_marker:
                        plt.axvline(x=t, color='grey', linestyle='--', alpha=0.4, label='_nolegend_')
                plt.scatter([0,0],[-1,1],s = 0.1, color = 'dimgray')
                plt.legend(legend)
         #       plt.legend(legend,bbox_to_anchor=(1.04, 0), loc="upper left", borderaxespad=0)
                plt.title(str(stim), size=20)
                plt.xlabel('Time in seconds', size=20)
                plt.ylabel(str(self.pref), size=20)
                plt.show()
        if separate == False:
            if self.time_marker != None:
                for t in self.time_marker:
                    plt.axvline(x=t, color='grey', linestyle='--', alpha=0.4, label='_nolegend_')
            plt.legend(legend, loc = 'upper left')#bbox_to_anchor=(1.04, 0), loc="upper left", borderaxespad=0)
        #    plt.legend(legend,loc='center left', bbox_to_anchor=(1, 0.5))
        #    plt.plot([0,50],[0.6,0.6],color = 'black')
        #    plt.plot([0,50],[0.5,0.5],color = 'black')
        #    plt.plot([40,40],[0,0.6],color = 'black')

            plt.title('Sine gratings forward sine 45 plaid 45', size=20)
            plt.xlabel('Time in seconds', size=20)
            plt.ylabel(str(self.pref), size=20)
            plt.show()

    def each_fish(self,separate):
        for fish in self.df['fish_ID'].unique().tolist():

            new_df  = self.df[self.df['fish_ID']== fish]

            print(' the fish we want '+ str(fish))
            if self.subset != False:
                new_df = pd.concat(self.subset)
            time_index = new_df['end_time']
            new_df['binned_time'] = time_index - time_index % self.bin_size + self.bin_size / 2
#            new_df = new_df.reset_index()
            bin_df = new_df.groupby(['fish_ID', 'stimulus_name', 'binned_time']).sum()
            bin_df['total_bouts'] = bin_df.left_bouts + bin_df.right_bouts + bin_df.straight_bouts
            # Analysed part:
            bin_df['preference_index'] = (-bin_df.left_bouts + bin_df.right_bouts + bin_df.straight_bouts) / bin_df.total_bouts
            bin_df['preference_absolute'] = (-bin_df.left_bouts_absolute + bin_df.right_bouts_absolute) / bin_df.total_bouts
            bin_df['percentage_left'] = (bin_df.left_bouts / bin_df.total_bouts) * 100
            bin_df['percentage_right'] = (bin_df.right_bouts / bin_df.total_bouts) * 100
            bin_df['percentage_straight'] = (bin_df.straight_bouts / bin_df.total_bouts) * 100
            bin_df['orientation'] = bin_df.bout_orientation / bin_df.total_bouts
            print('done data binning')

            self.time_marker = None
            num_fish = len(bin_df.index.unique('fish_ID'))
            print(' number of fish ' + str(num_fish))
            self.t_stamp = bin_df.index.unique('binned_time')
            bin_df = bin_df.reset_index()

            mean_df = bin_df.groupby(['stimulus_name', 'binned_time']).mean()
            sem_df = bin_df.groupby(['stimulus_name', 'binned_time']).std() / (num_fish) ** 0.5
            print('done mean sem')
            plt.rcParams["figure.figsize"] = (15, 12)

            #    colors = ['dimgray','red','magenta','orange','green','blue','yellow','black','maroon']# sns.color_palette("hls", 9)
            i = 0
            if self.plot_together == True:
                legend = []
            #   for variable in
            for stim in mean_df.index.unique('stimulus_name'):
                legend.append(str(stim))
                plot_mean_df = mean_df.xs(stim, level='stimulus_name')
                plot_sem_df = sem_df.xs(stim, level='stimulus_name')

                plt.plot(plot_mean_df[self.pref], marker='o', linewidth=3, markersize=2,
                         label=stim)  # , color = colors[i])
                if self.plot_sem:
                    plt.fill_between(self.t_stamp, plot_mean_df[self.pref] + plot_sem_df[self.pref],
                                     plot_mean_df[self.pref] - plot_sem_df[self.pref],
                                     alpha=0.25)  # ,color = colors[i])
                i += 1
                if separate:
                    if self.time_marker != None:
                        for t in self.time_marker:
                            plt.axvline(x=t, color='grey', linestyle='--', alpha=0.4, label='_nolegend_')
                    plt.scatter([0, 0], [-1, 1], s=0.1, color='dimgray')
                    plt.legend(legend)
                    #       plt.legend(legend,bbox_to_anchor=(1.04, 0), loc="upper left", borderaxespad=0)
                    plt.title(str(stim), size=20)
                    plt.xlabel('Time in seconds', size=20)
                    plt.ylabel(str(self.pref), size=20)
                    plt.show()
            if separate == False:
                if self.time_marker != None:
                    for t in self.time_marker:
                        plt.axvline(x=t, color='grey', linestyle='--', alpha=0.4, label='_nolegend_')
                plt.legend(legend, loc='upper left')  # bbox_to_anchor=(1.04, 0), loc="upper left", borderaxespad=0)
                #    plt.legend(legend,loc='center left', bbox_to_anchor=(1, 0.5))
                #    plt.plot([0,50],[0.6,0.6],color = 'black')
                #    plt.plot([0,50],[0.5,0.5],color = 'black')
                #    plt.plot([40,40],[0,0.6],color = 'black')

                plt.title('Sine_forward_sine_45_plaid_45, fish: '+str(fish), size=20)
                plt.xlabel('Time in seconds', size=20)
                plt.ylabel(str(self.pref), size=20)
                plt.show()

x = TimePreferenceIndex(plot_sem = True)
x.run()
x = TimePreferenceIndex(plot_sem = False)
x.run()