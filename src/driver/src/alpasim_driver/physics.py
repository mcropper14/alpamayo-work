import math
from typing import Dict, Any, Tuple


"""
Convert an object's world-frame position into ego-frame coordinates
"""
def world_to_ego(ego_state, obj_x, obj_y):
    ego_x, ego_y, heading = ego_state["x"], ego_state["y"], ego_state["heading"]
    dx = obj_x - ego_x
    dy = obj_y - ego_y
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    longitudinal = dx * cos_h + dy * sin_h
    lateral = -dx * sin_h + dy * cos_h
    return longitudinal, lateral


"""
Project obstacle velocity onto ego vehicle's forward axis to get obstacle's forward speed wrt world frame
"""
def obstacle_forward_speed(ego_state, obstacle):
    heading = ego_state["heading"]
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    vx = obstacle.get("vx", 0.0)
    vy = obstacle.get("vy", 0.0)
    return vx * cos_h + vy * sin_h


"""
Relative closing speed along ego forward direction (positive means ego is nearing the obstacle)
"""
def relative_forward_speed(ego_speed, obstacle_speed):
    return ego_speed - obstacle_speed


"""
Closing speed of an obstacle behind ego (positive means it is catching up with ego)
"""
def rear_closing_speed(ego_speed, obstacle_speed):
    return obstacle_speed - ego_speed


"""
Compute time to collision (TTC) in seconds.
"""
def compute_time_to_collision(distance, relative_speed):
    if relative_speed <= 1e-6:
        return float("inf")
    return distance / relative_speed


"""
Compute stopping distance
stopping_distance = reaction_dist + braking_dist
                  = v * t_reaction + v^2 / (2a)
"""
def compute_stopping_distance(speed, reaction_time=0.5, max_decel=6.0):
    if max_decel <= 0:
        raise ValueError("max_decel must be positive")
    speed = max(speed, 0.0)
    reaction_distance = speed * reaction_time
    braking_distance = (speed ** 2) / (2.0 * max_decel)
    return reaction_distance + braking_distance


"""
Compute the deceleration magnitude required to stop within the given distance after accounting for reaction distance
"""
def compute_required_deceleration(speed, distance, reaction_time=0.5):
    speed = max(speed, 0.0)
    remaining_distance = distance - speed * reaction_time
    if remaining_distance <= 0:
        return float("inf")
    if speed <= 1e-6:
        return 0.0
    return (speed ** 2) / (2.0 * remaining_distance)


"""
Solve for the maximum speed v such that (v * reaction_time + v^2 / (2 * max_decel) + safety_buffer <= distance)
"""
def compute_max_safe_speed(distance, reaction_time=0.5, max_decel=6.0, safety_buffer=2.0):
    if max_decel <= 0:
        raise ValueError("max_decel must be positive!")
    
    usable_distance = max(distance - safety_buffer, 0.0)
    # Quadratic form:
    # (1 / (2a)) v^2 + reaction_time * v - usable_distance = 0
    A = 1.0 / (2.0 * max_decel)
    B = reaction_time
    C = -usable_distance
    discriminant = B * B - 4.0 * A * C
    if discriminant < 0:
        return 0.0
    v_pos = (-B + math.sqrt(discriminant)) / (2.0 * A)
    return max(v_pos, 0.0)


    """
    Lightweight geometric filter for whether an obstacle is relevant to the ego path.
        longitudinal: obstacle distance ahead in ego frame
        lateral: obstacle left/right offset in ego frame
        lane_half_width: half-width of the path corridor in meters
        prediction_horizon: max forward distance to consider in meters
    """
def is_obstacle_in_path(longitudinal, lateral, lane_half_width=1.75, prediction_horizon=60.0):
    return ((longitudinal > 0.0) and (longitudinal <= prediction_horizon) and (abs(lateral) <= lane_half_width))