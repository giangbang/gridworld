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

# sparse reward environments
register(
    id="Gridworld-H1-s-v0",
    entry_point=__name__ + ":GridworldEnv",
    max_episode_steps=100,
    kwargs={"plan":4, "generate_goal":False, "random_start":True, "sparse_reward":True}
)
register(
    id="Gridworld-H2-s-v0",
    entry_point=__name__ + ":GridworldEnv",
    max_episode_steps=100,
    kwargs={"plan":5, "generate_goal":False, "random_start":True, "sparse_reward":True}
)
register(
    id="Gridworld-H3-s-v0",
    entry_point=__name__ + ":GridworldEnv",
    max_episode_steps=100,
    kwargs={"plan":6, "generate_goal":False, "random_start":True, "sparse_reward":True}
)

# maze environments
register(
    id="Gridworld-Maze1-v0",
    entry_point=__name__ + ":GridworldEnv",
    max_episode_steps=100,
    kwargs={"plan":30, "generate_goal":False, "random_start":True}
)
register(
    id="Gridworld-Maze2-v0",
    entry_point=__name__ + ":GridworldEnv",
    max_episode_steps=100,
    kwargs={"plan":31, "generate_goal":False, "random_start":True}
)
register(
    id="Gridworld-Maze3-v0",
    entry_point=__name__ + ":GridworldEnv",
    max_episode_steps=100,
    kwargs={"plan":32, "generate_goal":False, "random_start":True}
)

# sparse reward maze environments
register(
    id="Gridworld-Maze1-s-v0",
    entry_point=__name__ + ":GridworldEnv",
    max_episode_steps=100,
    kwargs={"plan":30, "generate_goal":False, "random_start":True, "sparse_reward":True}
)
register(
    id="Gridworld-Maze2-s-v0",
    entry_point=__name__ + ":GridworldEnv",
    max_episode_steps=100,
    kwargs={"plan":31, "generate_goal":False, "random_start":True, "sparse_reward":True}
)
register(
    id="Gridworld-Maze3-s-v0",
    entry_point=__name__ + ":GridworldEnv",
    max_episode_steps=100,
    kwargs={"plan":32, "generate_goal":False, "random_start":True, "sparse_reward":True}
)
