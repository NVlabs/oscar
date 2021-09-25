from .franka import Franka
from .franka_gripper import FrankaGripper
from .franka_pitcher import FrankaPitcher

# Define mappings from agent name to agent class
AGENT_MAPPING = {
    "franka": Franka,
    "franka_gripper": FrankaGripper,
    "franka_pitcher": FrankaPitcher,
}

AGENTS = set(AGENT_MAPPING.keys())
