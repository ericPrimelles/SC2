"""
Script del  ejecutor. Es el encargado de seleccionar y cargar el agente,  además crea el agente

"""
from Agents.beacon_agent import BeaconAgent
from Agents.battle_agent import BattleAgent


# Clase del ejecutor
class Runner(object):
    def __init__(self, agent_name, env, map_name, FLAGS):
        self.agent_name = agent_name  # Nombre del agente
        self.agent = 0  # Placeholder para el agente
        self.env = env  # Agente
        self.map_name = map_name  # Nombre del mapa a utilizar, se usa para los archivos de guardado
        self.FLAGS = FLAGS  # Se pasan las banderas para su utilización

    # Método de ejecución
    def run(self, episodes):
        if not self.FLAGS.save_file: # Si no hay una dirección de guardado almacenada
            save_name = f'{episodes}eps_{self.agent_name}' # Crea una por defecto
        else:
            save_name = self.FLAGS.save_file # Si no toma la almacenada

        self.agent_selector(save_name=save_name, load_name=self.FLAGS.load_file) # Selecciona un agente.
        if self.FLAGS.train: # Si la bandera de entrenamiento está activada
            self.agent.train(self.env, self.FLAGS.train, episodes) # Entrena
        else:
            self.agent.evaluate(self.env) # Si no ejecuta en modo greedy

    # Selector de agente, toma como argumentos las banderas de guardado y cargado
    def agent_selector(self, save_name, load_name):
        #save_name = f'./data/{self.map}/{10000}eps_{self.agent_name}'
        if self.agent_name == 'Beacon': # Si es un agente para el mapa Move to Beacon

            self.agent = BeaconAgent(self.FLAGS, save_name=save_name, load_name=load_name) # Carga un agente Beacon

        if self.agent_name == 'Battle': # Si es un agente para el mapa Defeat Roaches
            self.agent = BattleAgent(self.FLAGS, save_name=save_name, load_name=load_name) # Carga un agente de batalla
