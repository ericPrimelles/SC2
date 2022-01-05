from absl import flags
from Agents.beacon_agent import BeaconAgent
from Agents.battle_agent import BattleAgent

FLAGS = flags.FLAGS
flags.DEFINE_string("agent", "BeaconAgent", "Which agent to run")
flags.DEFINE_string("load_file", f'./data/MoveToBeacon/beacon_13149steps_32dim', "file to load params from")
flags.DEFINE_string("save_file", '', "file to save params to / load from if not loading from checkpoint")

class Runner(object):
    def __init__(self, agent_name, env, train, map_name):
        self.agent_name = agent_name
        self.agent = 0
        self.env = env
        self.train = train  # True: entrenar agente, False: se carga agente entrenado

        self.episode = 1 
        self.last_10_ep_rewards = []
        self.map_name = map_name
        
    def run(self, episodes):
        if not FLAGS.save_file:
            save_name = f'{episodes}eps_{self.agent_name}'
        else:
            save_name = FLAGS.save_file

        self.agent_selector(save_name=save_name, load_name=FLAGS.load_file)
        if self.train:
            self.agent.train(self.env, self.train, episodes)
        else:
            self.agent.evaluate(self.env)

    def agent_selector(self, save_name, load_name):
        #save_name = f'./data/{self.map}/{10000}eps_{self.agent_name}'
        if self.agent_name == 'Beacon':
            print('\n\n\nSelecting beacon\n\n\n')
            self.agent = BeaconAgent(save_name=save_name, load_name=load_name)

        if self.agent_name == 'Battle':
            print('\n\n\nSelecting beacon\n\n\n')
            self.agent = BattleAgent(save_name=save_name, load_name=load_name)