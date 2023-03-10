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

 ###########################################################################################
 # THIS MODEL DOES NOT TAKE FROM THE DISTRIBUTION. IT EXTRACTS FROM A PROBABILITY FUNCTION, RANDOM NUMBERS FROM 0 TO 1 AND THIS IS HOW ITS RANDOMNES ARISES.
 # THEN THERE  ARE SPECIAL PROBABILITYIES FOR EACH COMBINATION OF 2 CONSECUTIVBE BOUTS, SRTAIGHT-STRAIGHT, S-L,S-R,R-S,R-R-, R-L,L-S,L-L,L-R
 ###########################################################################################
# TODO: A model that looks at
# TODO: Make it in a confined circular space and have a bout length and plot the trajectory.
# TODO: PLOT  a lot of stuff together with labels for comparison
# TODO: YOU NEED TO SHOW SOMEWHERE IN THE GRAPH ALL THE PARAMETERS AND STUFF, WHAT YOU DID WITH THE MODEL...
# TODO PLOT HISTOGRAMS AS A LINE INSTEAD OF JUST PDF CDF SO PEOPLE UNDERSTAND IT BETTER BUT YOU ALSO GET A GRIP OF WHAT IT LOOKS LIKE..
#  and see again how you defined the random data, they could also be data retrieved not from the exact same distribution but from a random distribution with the same variab

# TODO: Katja 100 shuffles, and 100 simulations and 100 of everything to see what sample size makes sure that i get reliable data so i dont have to do my
#  experiments 100 times lol...

# TODO: toplot:  Distribution of bouts, distribution of streak lengths, and then cdfs... and pdfs...

# TODO: Some stats to see the significance of the difference between bouts or just find oiut how many bouts you need for significant differences..
# TODO: Understand the relationship between there being more straight bouts and the streaks that come from this phenomenon alone (shuffled data)
# make plots the same for the sake of the heavens..
# compare it with streak length of random data and of shuffled data ,not of data without computing streak lengtrh........................

# TODO: How you do stats with distributions? aparently its richard time...

# TODO: Average and std of whatever 100 simulations

# TODO: SOMETIHG TO RUN IT 100 TIMES..
"""axis limits 
ax = plt.gca()
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])"""
import numpy as np
from scipy.stats import ranksums

plt.rcParams["figure.figsize"] = (20, 20)

pd.options.mode.chained_assignment = None  # default='warn'
# for 2 milion iterations they differ by  0.25%, for 200 000 differ by 1%... so on so forth...
# but then streak lengths seem a bit very different, what about streak lengths for random data, without lookin at the model?
print('200 000 bouts as usual')

class Model1:
    def __init__(
            self,
            prob_ss,  # straight bout has value 0 / so it is going to be a straight after a straight with probability 1/3...
            prob_sl,  # left bout has value -1
            prob_sr,  # left bout has value -1

            prob_ll,  # left bout has value -1
            prob_ls, # straight bout has value 0 / so it is going to be a straight after a straight with probability 1/3...
            prob_lr,# left bout has value -1

            prob_rr,  # left bout has value -1
            prob_rl,  # straight bout has value 0 / so it is going to be a straight after a straight with probability 1/3...
            prob_rs,  # left bout has value -1

            # this order will always be the same, so if prob straight is 0.1 and left 0.7 and right 0.2 than it will be 0.1, 0.8, 1
            # these probabilites basically means, how likely is the animal to stay in one state for some time, given different parameter values...

            bout_value = 1,  # you do a few (ex. 50) iteration of starting with 1,0,-1 to show that it does not matter.
            probability_value = 1/2,
            nr_iterations = 200000, # 2 milion!!!!!
            prob_bout_change = 1/2,  # 1. probability that the new bout is left vs probability that the new bout is right, given that it changes? or
            # 2. Probability that the new bout is a different value from before, so if it is 1, then the next value has 2 choices, 0, -1 with different probabilities..
            # yeah the second ones says, probability that we have a different bout.????????????????????????????????????????
            # probability that we care about the probability? if we do then we see what happens depending on the values we set, if we dont we just pick randomly again..

            variable='streak length',

    ):

        # Set user input
        self.prob_ss = prob_ss
        self.prob_sl = prob_sl
        self.prob_sr = prob_sr

        self.prob_ll = prob_ll
        self.prob_ls = prob_ls
        self.prob_lr = prob_lr

        self.prob_rr = prob_rr
        self.prob_rl = prob_rl
        self.prob_rs = prob_rs

        self.bout_value = bout_value
        self.probability_value = probability_value
        self.nr_iterations = nr_iterations
        self.prob_bout_change = prob_bout_change
        self.variable = variable


    def run(self):
        """Main function to perform sanity check."""
        self.simulated_bouts = self.model_simulation()  # 1.1 simulate bouts, give it a name
        self.simulated_bouts_shuffled = np.copy(self.simulated_bouts)
        np.random.shuffle(self.simulated_bouts_shuffled)

       # self.simulated_bouts_shuffled = random.sample(self.simulated_bouts, len(self.simulated_bouts))  # 1.2 shuffle bouts, give it a name
        self.random_bouts = np.random.randint(low = -1, high=2, size=self.nr_iterations) # 1.3 simulate random bouts ,give it a name # TODO a good start would be to explore different ways that you can generate random data, like, what type of distribution and stuff..

#         return list_of_streaks_left ,list_of_streaks_right ,list_of_streaks_straight
        self.sim_left,self.sim_right,self.sim_straight = self.streak_length_orientation(self.simulated_bouts) #2. calcualte the streak length of these bouts, each of the three   and compare them.
        self.sh_sim_left,self.sh_sim_right,self.sh_sim_straight = self.streak_length_orientation(self.simulated_bouts_shuffled)
        self.rand_left,self.rand_right,self.rand_straight = self.streak_length_orientation(self.random_bouts)
        list_all9 = [self.sim_left,self.sim_right,self.sim_straight,
                    self.sh_sim_left,self.sh_sim_right,self.sh_sim_straight,
                    self.rand_left,self.rand_right,self.rand_straight]
        # 1. We plot bouts in histograms all together in one image:(use axes)
        self. histogram_plotter(self.simulated_bouts, self.simulated_bouts_shuffled, self.random_bouts)

        # 2. We plot streak lengths in histograms all together in one image, use axes (all streaks together,  left right straight!

        self.streak_length_together_histogram(data = list_all9)
        # 3. We do cdfs and pdfs of streak lengths all together

        #4. We do point 2 and point 3 for left right straight separately, but together in terms of simulated data, shuffled data, and random data...

        self.streak_length_separate_histogram(data=list_all9)

    # pdf cdf
        self.cdf_pdf_together(data =self.sim_left + self.sim_right + self.sim_straight)

    #    self.cdf_pdf_together(data=self.sim_left + self.sim_right + self.sim_straight)


    #    self.cdf_pdf_together(data=self.sim_left + self.sim_right + self.sim_straight)
# straight bout has value but a straight bout after a left bout has a different prob than a straight after a straight! I think as a start I should make a model that is simpler, in which all probabilities are the same...just for a beginning plsss...
# all combos are ss,sl,sr, rs,rr,rl, ls,lr,ll


    # TODO - The angle thing is still to go. and the position in the arena with size one given swim bout size blablabla...

    def model_simulation(self):

        ls_values = []
        bout_value = 0
        for i in tqdm(range(0,self.nr_iterations)):
            # whatever initial_value is does not matter bcs symmetric? but you build it as if it wasn't symmetric idiot!
            probability_value = random.random() # this is just a random number between 0 and 1

            # now initial_value? we ask is it straight or left or right? bcs depending on what it is then probability on how it changes is different!
            # let's say our bout is straight

            if bout_value == 0: # if there was a straight bout before then: we have an S.
                # now we go through everything as if we have an initial straight bout
                if probability_value <= self.prob_ss: # SS
                    bout_value = 0
                elif probability_value <= (self.prob_ss + self.prob_sl):
                    bout_value = -1
                elif probability_value <= (self.prob_ss + self.prob_sl + self.prob_sr):
                    bout_value = +1


                # if self.prob_ss < probability_value <= self.prob_sl:
                #     bout_value = -1
                #
                # if probability_value >= self.prob_sl:
                #     bout_value = +1

                # #if (probability_value <= self.prob_sl) and (probability_value > self.prob_ss): # SL
                #     bout_value = -1
                #
                # elif probability_value>=self.prob_sl: # SR
                #     bout_value = +1

                else:
                    print("Confusion that never stops \
                            Closing walls and ticking clocks\
                          Gonna come back and take you home ")
            elif bout_value == -1: # if there was a left bout before then:
                # now we go through everything as if we have an initial straight bout
                if probability_value <= self.prob_ll:
                    bout_value = -1
                elif probability_value <= (self.prob_ll + self.prob_lr):
                    bout_value = +1
                elif probability_value <= (self.prob_ll + self.prob_lr + self.prob_ls):
                    bout_value = 0
                #
                # # now we go through everything as if we have an initial straight bout
                # if probability_value <self.prob_ls: # SS
                #     bout_value = 0
                #
                # elif ((probability_value<=self.prob_ll) and (probability_value>self.prob_ls)): # SL
                #     bout_value = -1
                #
                # elif probability_value>=self.prob_ll: # SR
                #     bout_value = +1

                else:
                    print("I could not stop that you now know \
                            Singin' come out upon my seas \
                            Cursed missed opportunities "
                            )
                # left bout

            elif bout_value == +1: # if there was right bout before then:
                # now we go through everything as if we have an initial straight bout
                if probability_value <= self.prob_rr:
                    bout_value = +1
                elif probability_value <= (self.prob_rr + self.prob_rl):
                    bout_value = -1
                elif probability_value <= (self.prob_rr + self.prob_rl + self.prob_rs):
                    bout_value = 0
                #
                # if probability_value <self.prob_rs: # SS
                #     bout_value = 0
                #
                # elif ((probability_value<self.prob_rl) and (probability_value>self.prob_rs)): # SL
                #     bout_value = -1
                #
                # elif probability_value>self.prob_rl: # SR
                #     bout_value = +1

                else:
                    print("Am I a part of the cure \
                            Or am I part of the disease? Singin \
                            ...")
            else:
                print('Nothing else compares...')
            ls_values.append(bout_value)

        return ls_values

    def streak_length_orientation(self,bouts):
        nr_mesups = 0
        list_of_directions = bouts

        list_of_streaks_left = []  # i save here the length of the streak, defined as a_streak
        list_of_streaks_right = []
        list_of_streaks_straight = []
        left_streak = 1
        straight_streak = 1
        right_streak = 1  # we keep track of each streak length here, and when the streak is done, we append it to the list_of_streaks. streak length is 1 if only one bout was done in that direction
        # we create a new list of directions for each trial. and the new list makes it impossible for them to concat!
        print('started with streak length')
        for element in tqdm(range(0, len(list_of_directions))):  # we start from one bcs we compare with 0th element
            if (list_of_directions[element - 1] == list_of_directions[element]):
                if list_of_directions[element ] == -1.0:
                    left_streak +=1
                elif list_of_directions[element ] == 1.0:
                    right_streak +=1
                elif list_of_directions[element ] == 0.0:
                    straight_streak+=1
                else:
                    print(' now i dont understand was happenin')
            else:
                if ((left_streak >1) & (right_streak == 1) & (straight_streak == 1)):
                    list_of_streaks_left.append(left_streak)
                    left_streak = 1
                elif ((left_streak == 1) & (right_streak >1) & (straight_streak == 1)):
                    list_of_streaks_right.append(right_streak)
                    right_streak = 1
                elif ((left_streak == 1) & (right_streak == 1) & (straight_streak >1)):
                    list_of_streaks_straight.append(straight_streak)
                    straight_streak = 1
                elif ((left_streak == 1) & (right_streak == 1) & (straight_streak == 1)):
                    if (list_of_directions[element - 1] == -1):
                        list_of_streaks_left.append(left_streak)
                    elif (list_of_directions[element - 1] == 1):
                        list_of_streaks_right.append(right_streak)
                    elif (list_of_directions[element - 1] == 0):
                        list_of_streaks_straight.append(straight_streak)
                    else:
                        print(' now i also dont understand was happenin')

                else:
                    nr_mesups +=1
        #  print('nr_mesups ' + str(nr_mesups))
        #  print('df length'+ str(len(df)))
        # list_of_streaks.append(a_streak)
        #list_of_streaks_left = [x for x in list_of_streaks_left if x <= 20]
        #list_of_streaks_right = [x for x in list_of_streaks_right if x <= 20]
        #list_of_streaks_straight = [x for x in list_of_streaks_straight if x <= 20]
        ls_together = list_of_streaks_left +list_of_streaks_right
        print('done streak lengths')
        return list_of_streaks_left ,list_of_streaks_right ,list_of_streaks_straight

    def histogram_plotter(self,simulated,shuffled,random):
        print('histogram plotter started')
        plt.rcParams["figure.figsize"] = (10, 4)

        fig, axs = plt.subplots(1,3)
        fig.suptitle('bout histograms')
        counts0, bins, bars = axs[0].hist(simulated)#,bins = np.arange(-2,2,5))
        axs[0].set_title('simulated data')

        counts1, bins, bars = axs[1].hist(shuffled)#,bins = np.arange(-2,2,5))
        axs[1].set_title('shuffled data')

        counts2, bins, bars = axs[2].hist(random)#,bins = np.arange(-2,2,5))
        axs[2].set_title('random rand',size = 15)
        ls_min = [counts0.min(),counts1.min(),counts2.min()]
        ls_max = [counts0.max(),counts1.max(),counts2.max()]
        print('ls_max:::::::::::::::::::>')
        print(ls_max)
        val_max = max(ls_max)
        val_min = min(ls_min)
    #    val_min = self.nr_iterations/5

        axs[0].set_ylim([val_min, val_max+(val_max/100)]) # automatize it so that whatever max value is for all you get that plus a troshe
        axs[1].set_ylim([val_min, val_max+(val_max/100)])
        axs[2].set_ylim([val_min, val_max+(val_max/100)])

        fig.tight_layout(pad=1)

        plt.show()
        print('histogram plotter is done')
    def streak_length_together_histogram(self,data):

        print('streak together started')
        plt.rcParams["figure.figsize"] = (10, 4)
        fig, axs = plt.subplots(1,3)
        fig.suptitle('streak lengths histograms')

        counts0, bins, bars =  axs[0].hist(data[0]+data[1]+data[2])
        axs[0].set_title('simulated data streaks ')

        counts1, bins, bars =  axs[1].hist(data[3]+data[4]+data[5])
        axs[1].set_title('simulated data shuffled streaks ')

        counts2, bins, bars =  axs[2].hist(data[6]+data[7]+data[8])
        axs[2].set_title('random data streaks ')
        ls_min = [counts0.min(),counts1.min(),counts2.min()]
        ls_max = [counts0.max(),counts1.max(),counts2.max()]
        val_max = max(ls_max)
        val_min = min(ls_min)
        axs[0].set_ylim([val_min, val_max+(val_max/100)]) # automatize it so that whatever max value is for all you get that plus a troshe
        axs[1].set_ylim([val_min, val_max+(val_max/100)])
        axs[2].set_ylim([val_min, val_max+(val_max/100)])

        plt.show()
        print('streak together is done')
    def streak_length_separate_histogram(self,data):
        plt.rcParams["figure.figsize"] = (20, 20)

        fig, axs = plt.subplots(3,3)
       # fig.suptitle('streak length plot')
        counter = 0
        # sim, then shuffled sim, then random data
        print('streak separate started')
        ls_min = []
        ls_max = []
        for rows in range(0,3):
            for columns in range(0,3):
                counts, bins, bars = axs[rows, columns].hist(data[counter])  # ,bins = np.arange(-2,2,5))
                ls_min.append(counts.min())
                ls_max.append(counts.max())
                if rows == 0:
                    ttl1 = 'Simulated data '
                elif rows == 1:
                    ttl1 = ' Simulated shuffled  '
                elif rows == 2:
                    ttl1 = 'Random data '
                else:
                    print('confusion never stops...')
                if columns == 0:
                    ttl2 = ' left streaks '
                elif columns == 1:
                    ttl2 = ' right streaks '
                elif columns == 2:
                    ttl2 = ' straight streaks '
                else:
                    print('confusion never stops2...')
                axs[rows, columns].set_title(ttl1+ttl2 +" number of streaks considered "+ str(len(data[counter])), size = 10)
                counter += 1
        val_max = max(ls_max)
        val_min = min(ls_min)
        for rows in range(0, 3):
            for columns in range(0, 3):
                axs[rows,columns].set_ylim([val_min, val_max+(val_max/100)])

        fig.tight_layout(pad=1)
        plt.show()
        print('streak separated is done')
    def histogram_range_find(self,data):
        counts, bins, bars = plt.hist(data)
        max_val = counts.max()
        min_val = counts.min()
        return min_val, max_val
        # put it on a list...
 #   def cdf_pdf_streaks(self,data):

    # TODO : the pdf cdf functions as well as a function for a transparent line histogram:
    def pdf_cdf(self, list_var, plot_which,title):
        plt.rcParams["figure.figsize"] = (20, 10)
        count, bins_count = np.histogram(list_var, bins=10)
        pdf = count / sum(count)
        cdf = np.cumsum(pdf)
        ls = [cdf, pdf]
        plt.plot(bins_count[1:], ls[plot_which])

    def cdf_pdf_together(self,data):
        zero_one = [0,1]
        plot_which = ['CDF', 'PDF']
        #for pl in zero_one:

        self.pdf_cdf(list_var = data, plot_which=zero_one[1],title =  'Model data')
        #    sampl = np.random.uniform(low=-1, high=2, size=(self.nr_iterations))
        #    self.pdf_cdf(list_var = sampl, plot_which= zero_one[pl], title = 'Random')
        #    random.shuffle(data) # TODO YOU NEED STREAKS FOR RANDOM AND SHUFFLED DATA, NOT VALUES!!!!
        #    self.pdf_cdf(list_var = ls_values, plot_which= zero_one[pl],title =  'Shuffle')
         #   plt.legend(['Model data', 'Random','Shuffle'])
        plt.title(str(plot_which[1]) + ' 10 simulations and '+str(self.nr_iterations)+ ' bouts '+ self.variable, size=20)
        plt.show()


x = Model1(prob_ss = 0.5, prob_sl=0.4, prob_sr=0.1,
           prob_ll = 0.9, prob_lr=0.05, prob_ls=0.05,
           prob_rr = 0.5, prob_rl=0.4, prob_rs=0.1)

values = x.model_simulation()
print(values)
#sdf
for i in range(0,1):
    print(i)
    x.run()

