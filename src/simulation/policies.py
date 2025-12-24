import math

class AutoscalerPolicy: # base class for all autoscaling policies
    def __init__(self, node_capacity=16, target_util=0.60, cooldown_window=3, min_instances=1):
        self.node_capacity = node_capacity
        self.target_util = target_util
        self.cooldown_window = cooldown_window 
        self.last_scale_change_step = -1    
        self.last_action_type = "NONE"         
        self.min_instances = min_instances

    def calculate_required_nodes(self, load_value): # Used by Predictive Policy to jump straight to the right number.
        if load_value == 0:
            return 0
            
        capacity_per_node = self.node_capacity * self.target_util
        required_nodes = math.ceil(load_value / capacity_per_node)
        return required_nodes
    # allow immediate scale up but not scale down
    def apply_stability_logic(self, proposed_target, current_nodes, current_step):
        if proposed_target > current_nodes:
            self.last_scale_change_step = current_step
            self.last_action_type = "UP"
            return proposed_target

        elif proposed_target < current_nodes:
            steps_since_last_change = current_step - self.last_scale_change_step
            
            if steps_since_last_change < self.cooldown_window:
                return current_nodes
            self.last_scale_change_step = current_step
            self.last_action_type = "DOWN"
            return proposed_target

        return current_nodes

    def get_decision(self, current_step, current_load, predicted_load, current_nodes):
        raise NotImplementedError("Subclasses must implement get_decision")


class ReactivePolicy(AutoscalerPolicy):
    def get_decision(self, current_step, current_load, predicted_load, current_nodes):
        """
        MIDDLE GROUND LOGIC: Geometric Scaling.
        If overloaded -> Scale up by 40%.
        If underloaded -> Scale down by 10%.
        
        This creates a 'Sawtooth' catch-up pattern. 
        It's not as dumb as +1, but not as smart as instant calculation.
        """
        current_capacity = current_nodes * self.node_capacity
        if current_capacity == 0: 
            utilization = 1.0 
        else:
            utilization = current_load / current_capacity

        high_threshold = self.target_util
        low_threshold = self.target_util * 0.6  

        proposed_target = current_nodes

        if utilization > high_threshold:
            # geometric scaling up logic 
            scale_factor = 1.40 
            proposed_target = math.ceil(current_nodes * scale_factor)
            proposed_target = max(proposed_target, current_nodes + 1)

        elif utilization < low_threshold:
            scale_factor = 0.90
            proposed_target = math.floor(current_nodes * scale_factor)
            proposed_target = max(self.min_instances, proposed_target)
        
        final_nodes = self.apply_stability_logic(proposed_target, current_nodes, current_step)
        return final_nodes


class PredictivePolicy(AutoscalerPolicy):
    def __init__(self, safety_buffer=1.15, **kwargs):
        super().__init__(**kwargs)
        self.safety_buffer = safety_buffer 

    def get_decision(self, current_step, current_load, predicted_load, current_nodes):
        """
        SMART LOGIC: Calculated Scaling.
        Predicts exactly how many nodes are needed and jumps there instantly.
        """
        # calculated future needs
        safe_prediction = predicted_load * self.safety_buffer
        future_needs = self.calculate_required_nodes(safe_prediction)
        # calculated current needs (for hyrbrid)
        current_needs = self.calculate_required_nodes(current_load)
        proposed_target = max(future_needs, current_needs)
        final_nodes = self.apply_stability_logic(proposed_target, current_nodes, current_step)
        
        return final_nodes