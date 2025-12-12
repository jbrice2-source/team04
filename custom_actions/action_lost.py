# action_lost.py
# To be placed: /root/mdk-230105/share/python/miro2/core/action/
# 
#
# Not used in final report (spoken about in abstract)
# BioRescue – LOST launch (Alternate approach)
# High-priority BG action used to suspend demo/forebrain behaviour
# when the robot is “lost”.
#
# Activation:
#       /team04/lost_active == True  →  priority_uninterruptable (Avoids demo interuption)
#       /team04/lost_active == False →  0.0
#
# All movement + distress signalling
# is controlled externally.
# 
import rospy
import numpy as np
import miro2 as miro
from . import action_types
class ActionLost(action_types.ActionTemplate):
    # ---------------------------------------------------------
    # REQUIRED: finalize()
    # ---------------------------------------------------------
    def finalize(self):
        """
        Called automatically during ActionTemplate __init__().
        We define static properties of this action here.
        """
        self.name = "lost"
        # This is a pure override state, not a movement action
        self.retreatable = False
        self.move_away = False
        self.modulate_by_wakefulness = False
        # High sustained priority when selected
        self.ongoing_priority = self.pars.action.priority_uninterruptable


    def compute_priority(self):
        """
        Priority is defined entirely by an external ROS parameter.
        """
        flag = rospy.get_param("/team04/lost_active", False)
        if flag:
            return self.pars.action.priority_uninterruptable
        return 0.0


    def start(self):
        """
        Called when the BG selects this action.
        LOST action contains no motor pattern.
        """
        miro.lib.print_info("[LOST ACTION] STARTED")
        self.clock.stop()    # ensure no service() calls


    def service(self):
        """
        LOST action never uses motor-pattern service cycles.
        Everything is external.
        """
        return


    def event_stop(self):
        miro.lib.print_info("[LOST ACTION] STOPPED")

