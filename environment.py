"""
Script encargado de generar el ambiente.

"""
# Ambiente
from pysc2.env import sc2_env
from pysc2.lib import features
from  pysc2.env import available_actions_printer

def get_environment(map_name, feature_dimensions, step_mul=10, visualize=False, realtime=False):
    env = sc2_env.SC2Env(
                        map_name=map_name,
                        players=[sc2_env.Agent(sc2_env.Race.terran)],
                        agent_interface_format=features.AgentInterfaceFormat(
                            feature_dimensions=features.Dimensions(screen=feature_dimensions, minimap=feature_dimensions),
                            use_feature_units=True
                        ),
                        step_mul=step_mul,
                        visualize=visualize
                        )

    return available_actions_printer.AvailableActionsPrinter(env)