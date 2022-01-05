
from absl import flags
from Agents.beacon_agent import BeaconAgent
from Agents.battle_agent import BattleAgent
from pysc2 import maps
from pysc2.env import available_actions_printer
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
FLAGS = flags.FLAGS
flags.DEFINE_bool("render", False, "Whether to render with pygame.")
flags.DEFINE_bool("train", True, "Whether we are training or running")
flags.DEFINE_integer("screen_resolution", 32,
                     "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 32,
                     "Resolution for minimap feature layers.")

flags.DEFINE_integer("game_steps_per_episode", 0, "Game steps per episode.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")

flags.DEFINE_string("agent", "BeaconAgent", "Which agent to run")
flags.DEFINE_enum("agent_race", None, [str(i) for i in list(sc2_env.Race)], "Agent's race.")
flags.DEFINE_enum("bot_race", None, [str(i) for i in list(sc2_env.Race)], "Bot's race.")
flags.DEFINE_enum("difficulty", None, [str(i) for i in list(sc2_env.Difficulty)],
                  "Bot's strength.")

flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")
flags.DEFINE_bool("load_checkpoint", False, "Whether or not to load checkpoint from previous training session")
flags.DEFINE_bool("load_params", False, "Whether or not to load parameters from previous training session")
flags.DEFINE_string("load_file", f'./data/MoveToBeacon/beacon_13149steps_32dim', "file to load params from")
flags.DEFINE_string("save_file", '', "file to save params to / load from if not loading from checkpoint")

flags.DEFINE_integer("max_episodes", 10000, "Maximum number of episodes to train on")
flags.DEFINE_string("map", "MoveToBeacon", "Name of a map to use.")
# flags.DEFINE_string("map", "DefeatRoachesAntiSuicideMarineDeath0", "Name of a map to use.")
# flags.DEFINE_string("map", "DefeatZerglingsAndBanelings", "Name of a map to use")

class Runner(object):
    def __init__(self, agent_name, env, train, map):
        self.agent_name = agent_name
        self.agent = 0
        self.env = env
        self.train = train  # True: entrenar agente, False: se carga agente entrenado

        self.episode = 1 
        self.last_10_ep_rewards = []
        self.map = map





        
    def run(self, episodes):



        if not FLAGS.save_file:
            save_name = f'./data/{self.map}/{FLAGS.max_episodes}eps_{FLAGS.agent}'
        else:
            save_name = FLAGS.save_file

        self.agent_selector(save_name=save_name, load_name=FLAGS.load_file)
        if self.train:
            self.agent.train(self.env, self.train, episodes)
        else:
            self.agent.evaluate(self.env)

    def agent_selector(self, save_name, load_name):
        save_name = f'./data/{self.map}/{10000}eps_{self.agent_name}'
        if self.agent_name == 'Beacon':


            self.agent = BeaconAgent(save_name=save_name, load_name=load_name)

        if self.agent_name == 'Battle':
            self.agent = BattleAgent(save_name=save_name, load_name=load_name)