"""Configuration for LIMX DYNAMICS robots.

The following configurations are available:

* :obj:`POINTFOOT_CFG`: LIMX DYNAMICS Pointfoot PF_P441C robot

Reference: https://github.com/limxdynamics
"""

import os

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg


##
# USD PATH
##


current_dir = os.path.dirname(__file__)
POINTFOOT_CFG_usd_path = os.path.join(current_dir, "../usd/Limx/PF_P441C/PF_P441C.usd")


##
# Configuration
##



POINTFOOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=POINTFOOT_CFG_usd_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={
            "abad_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "knee_R_Joint": 0.0,
            "foot_L_Joint": 0.0,
            "foot_R_Joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "abad_L_Joint",
                "abad_R_Joint",
                "hip_L_Joint",
                "hip_R_Joint",
                "knee_L_Joint",
                "knee_R_Joint",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                "abad_L_Joint": 40.0,
                "abad_R_Joint": 40.0,
                "hip_L_Joint": 40.0,
                "hip_R_Joint": 40.0,
                "knee_L_Joint": 40.0,
                "knee_R_Joint": 40.0,
            },
            damping={
                "abad_L_Joint": 2.5,
                "abad_R_Joint": 2.5,
                "hip_L_Joint": 2.5,
                "hip_R_Joint": 2.5,
                "knee_L_Joint": 2.5,
                "knee_R_Joint": 2.5,
            },
        ),
    },
)
