'''
Hylaa check code for quickzonoreach
'''

import math

from hylaa.hybrid_automaton import HybridAutomaton
from hylaa.settings import HylaaSettings, PlotSettings
from hylaa.core import Core
from hylaa.stateset import StateSet
from hylaa import lputil

def make_automaton():
    'make the hybrid automaton'

    ha = HybridAutomaton('Hylaa Output (hylaa_check.py)')

    # mode one: x' = y + u1, y' = -x + + u1 + u2
    # u1 in [-0.5, 0.5], u2 in [-1, 0]
    m1 = ha.new_mode('m1')
    m1.set_dynamics([[0, 1], [-1, 0]])

    b_mat = [[1, 0], [1, 1]]
    b_constraints = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    b_rhs = [0.5, 0.5, 0, 1]
    m1.set_inputs(b_mat, b_constraints, b_rhs)

    return ha

def make_init(ha):
    'make the initial states'

    # initial set has x0 = [-5, -4], y = [0, 1], c = 0, a = 1
    mode = ha.modes['m1']
    init_lpi = lputil.from_box([(-5, -4), (0, 1)], mode)
    
    init_list = [StateSet(init_lpi, mode)]

    return init_list

def make_settings():
    'make the reachability settings object'

    # see hylaa.settings for a list of reachability settings
    settings = HylaaSettings(math.pi / 4, math.pi) # step size = pi/4, time bound pi
    settings.plot.plot_mode = PlotSettings.PLOT_IMAGE
    settings.stdout = HylaaSettings.STDOUT_NORMAL
    settings.plot.filename = "hylaa.png"

    settings.plot.label.title_size = 18
    settings.plot.plot_size = (6, 6)

    return settings

def run_hylaa():
    'main entry point'

    ha = make_automaton()

    init_states = make_init(ha)

    settings = make_settings()

    Core(ha, settings).run(init_states)

if __name__ == "__main__":
    run_hylaa()
