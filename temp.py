


# from analysis.behavior_modelling.environments import classical_phototaxis, classical_phototaxis_stim_tuple

from streak_length import StreakLength
theclass = StreakLength()
# if you want to return the output of run
#return theclass.run()
# if you want to return run itself to be used later
function =  theclass.plotter_left_right

print(function([1,0,-1,-1,0],'title'))