

from Agents.beacon_agent import BeaconAgent
from Agents.battle_agent import BattleAgent

class Runner(object):
    def __init__(self, agent_name, env, train, map, load_name=''):
        self.agent_name = agent_name
        self.agent = 0
        self.env = env
        self.train = train  # True: entrenar agente, False: se carga agente entrenado

        self.episode = 1 
        self.last_10_ep_rewards = []
        self.map = map
        self.load_name = load_name




        
    def run(self, episodes):
        self.agent_selector()

        if self.train:
            self.agent.train(self.env, self.train, episodes)
        else:
            self.agent.evaluate(self.env)

    def agent_selector(self):
        save_name = f'./data/{self.map}/{10000}eps_{self.agent_name}'
        if self.agent_name == 'Beacon':


            self.agent = BeaconAgent(save_name=save_name, load_name='')

        if self.agent_name == 'Battle':
            self.agent = BattleAgent(save_name=save_name, load_name='')