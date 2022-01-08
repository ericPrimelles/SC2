from pysc2.env import sc2_env
from pysc2.lib import features
from  pysc2.env import available_actions_printer

def get_environment(map_name, screen_size=32, minimap_size=32, step_mul=10, visualize=False):
    env = sc2_env.SC2Env(
                        map_name=map_name,
                        players=[sc2_env.Agent(sc2_env.Race.terran)],
                        agent_interface_format=features.AgentInterfaceFormat(
                            feature_dimensions=features.Dimensions(screen=screen_size, minimap=minimap_size),
                            use_feature_units=True
                        ),
                        step_mul=step_mul,
                        visualize=visualize
                        )

    return available_actions_printer.AvailableActionsPrinter(env)