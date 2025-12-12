# action_helper.py
# To be placed: /root/mdk-230105/share/python/miro2/core/action/
#
# Not used in final report (spoken about in abstract)
# BioRescue – Helper launch (Alternate approach)
#
# Priority logic:
#   /team04/helper_active == False  → priority = 0.0
#   /team04/helper_active == True   → priority = priority_uninterruptable (Avoids demo interuption)
#
# External supervisor drives real robot behaviour
#
# Add these to the node_action.py (path:/Users/abdurrahmanibrahim/Downloads/mdk-230105/share/python/miro2/core/node_action.py)
#   from action.action_rescue import ActionRescue # Added imports
#   from action.action_lost import ActionLost # Added imports
#   
#                 self.actions = [
#                 ActionRescue(self), # OUR NEW RESCUE ACTION
#                 ActionLost(self) # OUR NEW LOST ACTION
#                 ]
import rospy
import miro2 as miro
from . import action_types
class ActionRescue(action_types.ActionTemplate):
    def finalize(self):
        self.name = "helper"
        # Behaviour is externally driven, so we disable internal modulation
        self.retreatable = False
        self.move_away = False
        self.modulate_by_wakefulness = False
        self.ongoing_priority = self.pars.action.priority_uninterruptable
        self.interface = miro.lib.RobotInterface()
    def compute_priority(self):
        flag = rospy.get_param("/team04/helper_active", False)
        if flag:
            return self.pars.action.priority_uninterruptable
        return 0.0
    def start(self):
        """
        RESCUE SELECTED BY BG:
        - freeze internal patterns
        - raise helper confirmation signal
        - permit external supervisor (Helper.py) to take control
        """
        miro.lib.print_info("[RESCUE ACTION] ENTER")
        # Stop pattern cycle
        self.clock.stop()
        # Freeze internal motion
        self.interface.post_cmd_vel(0.0, 0.0)
        # Turn on illumination (rescue marker)
        illum = [255, 100, 0] # amber
        self.interface.post_illum(illum)
        # Notify Helper.py the BG has yielded control
        rospy.set_param("/team04/helper_mode_active", True)


    def service(self):
        return
    def event_stop(self):
        miro.lib.print_info("[RESCUE ACTION] EXIT")
        # Restore LEDs
        self.interface.post_illum([0,0,0])
        rospy.set_param("/team04/helper_mode_active", False)

