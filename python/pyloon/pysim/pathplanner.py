import numpy as np
import matplotlib.pyplot as plt
import copy

from mpl_toolkits.mplot3d import Axes3D

import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0],'..'))

from pyutils import pyutils
from pyutils.pyutils import parsekw, hash3d, hash4d, rng, downsize

from optiloon import loonpathplanner

class PathPlanner:
    def __init__(self, *args, **kwargs):
        planner = parsekw(kwargs,'planner','mpcfast')
        if planner == 'mpcfast':
            self.planner = loonpathplanner.MPCWAPFast(**kwargs)
            self.planner.__delta_p_between_jetstreams__(5.0)
        elif planner == 'mpc':
            self.planner = loonpathplanner.MPCWAP(**kwargs)
            self.planner.__delta_p_between_jetstreams__(5.0)
        elif planner == 'ldp':
            self.planner = loonpathplanner.LocalDynamicProgrammingPlanner(**kwargs)
            self.planner.__delta_p_between_jetstreams__(5.0)
        elif planner == 'wap':
            self.planner = loonpathplanner.WindAwarePlanner(**kwargs)
            self.planner.__delta_p_between_jetstreams__(5.0)
        elif planner == 'montecarlo':
            self.planner = loonpathplanner.MonteCarloPlanner(**kwargs)
        elif planner == 'pic':
            self.planner = loonpathplanner.PlantInvertingController(**kwargs)
        elif planner == 'naive':
            self.planner = loonpathplanner.NaivePlanner(**kwargs)
        elif planner == 'molchanov':
            self.planner = loonpathplanner.MolchanovEtAlPlanner(**kwargs)
        elif planner == 'stayaloft':
            self.planner = loonpathplanner.JustStayAloft(**kwargs)

    def plan(self, *args, **kwargs):
        return self.planner.plan(loon=kwargs.get('loon'),
                            u=kwargs.get('u'),
                            T=kwargs.get('T'),
                            pstar=kwargs.get('pstar'),
                            depth=kwargs.get('depth'))
