from .gridworld_env import GridworldEnv
from gym.envs.registration import register

register(
    id="Gridworld-v0",
    entry_point=__name__ + ":GridworldEnv",
    max_episode_steps=100,
)

register(
    id="Gridworld-H1-v0",
    entry_point=__name__ + ":GridworldEnv",
    max_episode_steps=100,
    kwargs={"plan":4, "generate_goal":False, "random_start":True}
)
register(
    id="Gridworld-H2-v0",
    entry_point=__name__ + ":GridworldEnv",
    max_episode_steps=100,
    kwargs={"plan":5, "generate_goal":False, "random_start":True}
)
register(
    id="Gridworld-H3-v0",
    entry_point=__name__ + ":GridworldEnv",
    max_episode_steps=100,
    kwargs={"plan":6, "generate_goal":False, "random_start":True}
)