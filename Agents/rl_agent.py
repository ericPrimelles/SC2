""""
Script base para los agentes, contiene los métodos abstractos sobrecargables desde las implementaciones futuras
"""
# Computation
from abc import ABC, abstractmethod # ABC se usa para crear clases abstractas
from collections import deque
import numpy as np
import pickle
import copy

# Ambiente
from pysc2.agents.base_agent import BaseAgent # Método abstracto base para la implementacin de agentes
from pysc2.lib import actions

# Files
from Utils.epsilon import Epsilon
from Utils.replay_memory import ReplayMemory

# Modelos
import torch
import torch.nn as nn
import torch.optim as optim

# Acciones base
_NO_OP = actions.FUNCTIONS.no_op.id # No hacer nada
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id # Moverse en la pantalla

class BaseRLAgent(BaseAgent, ABC):
    def __init__(self, FLAGS, save_name='./data/', load_name=None, save_joblib=None):
        super(BaseRLAgent, self).__init__()
        self.FLAGS = FLAGS
        self.training = False # Fijando training a False
        self.max_frames = FLAGS.max_frames # Maximos pasos antes de terminar un episodio
        self._epsilon = Epsilon(start= FLAGS.epsilon_start,
                                end=FLAGS.epsilon_end,
                                update_increment=FLAGS.epsilon_decrement) # Iniciando el manejador de Epsilon
        self.gamma = FLAGS.gamma # Factor de descuento

        # Hiperparámetros
        #self.train_q_per_step = 4
        self.train_q_batch_size = FLAGS.batch_size
        self.steps_before_training = FLAGS.steps_before_training
        self.target_q_update_frequency = FLAGS.target_update
        self.lr = FLAGS.lr

        f = open(save_name + "-Hyperparameters.txt", 'w')
        text = f'Episodios: {self.FLAGS.episodes} \n' \
            f'Max Frames: {self.FLAGS.max_frames}\n' \
            f'Epsilon Start: {self.FLAGS.epsilon_start} \n' \
            f'Epsilon End: {self.FLAGS.epsilon_end}\n' \
            f'Epsilon decreament: {self.FLAGS.epsilon_decrement}\n' \
            f'Batch Size: {self.FLAGS.batch_size}\n' \
            f'Gamma: {self.FLAGS.gamma}\n' \
            f'Steps before training: {self.FLAGS.steps_before_training}\n' \
            f'Target uodate: {self.FLAGS.target_update}\n' \
            f'Learning rate: {self.FLAGS.lr}\n'
        f.write(text)
        f.close()

        self.save_name = save_name # Guardando la ruta de guardado
        if load_name is None: # Si no hay ruta de cargado se iguala a la de salvado
            self.load_name = self.save_name
        else: # Si no se extrae desde las banderas
            self.load_name = load_name

        # Creación del modelo
        self._Q = None # Red principal
        self._Qt = None # Red Target
        self._optimizer = None # Placeholder del optimizador
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Device
        self._criterion = nn.MSELoss() # Función de pérdida
        self._memory = ReplayMemory(100000) # Iniicialización del Buffer (Uniforme)

        # Contenedores y herramientas
        self._loss = deque(maxlen=int(1e5))
        self._max_q = deque(maxlen=int(1e5))
        self.loss = []
        self.max_q = []
        self.reward = []
        self._action = None
        self._screen = None
        self._screen_size = self.FLAGS.feature_size
        self.n_episodes = 0
        self.features = None

    def initialize_model(self, model):
        """"
        Función para inizializar el modelo, se invoca desde cada agente especializado
        """
        self._Q = model # Red principal
        self._Qt = copy.deepcopy(self._Q) # Se copia a la target desde la principal
        # Agregando al device
        self._Q.to(self.device)
        self._Qt.to(self.device)
        # Se inicializa el optimizador
        self._optimizer = optim.Adam(self._Q.parameters(), lr=self.lr)

    def load_model_checkpoint(self, load_params=True):
        """"
        Función para cargar modelos
        """
        # Agregando compatibilidad para CPU y GPU
        if torch.cuda.is_available():  # Si hay disponibilidad de GPU
            self._Q.load_state_dict(torch.load(self.load_name + '.pth')) # Se carga normal
        else: # Si no
            self._Q.load_state_dict(torch.load(self.load_name + '.pth', map_location=torch.device('cpu'))) # Se descerializan los modelos
                                                                                                           # para ser accesible desde CPU
        for key in self._Q.state_dict():
            print(self._Q.state_dict()[key])
            print(self._Qt.state_dict()[key])
        if load_params: # Se cargan los datos
            saved_data = pickle.load(open(f'{self.load_name}' + '_data.pkl', 'rb'))
            self.loss = saved_data['loss']
            self.max_q = saved_data['max_q']
            self._epsilon._value = saved_data['epsilon']
            self.reward = saved_data['reward']
            self.n_episodes = saved_data['n_episodes']

    def get_env_action(self, action, obs, command=_MOVE_SCREEN):
        """"
        Función para obtener una acción desde el ambiente
        """
        action = np.unravel_index(action, [1, self._screen_size, self._screen_size]) # Computa el indice multidimensional de un arreglo
        target = [action[2], action[1]] # Se extrae el objetivo desde el indicie multidimensional


        if command in obs.observation["available_actions"]: # Si la acción seleccionada se encuentra entre las acciones posibles
            return actions.FunctionCall(command, [[0], target]) # Se ejecuta pasando como parámetros el target
        else:
            return actions.FunctionCall(_NO_OP, []) # Si no, no se hace nada

    def save_data(self, episodes_done=0):
        """"
        Función para salvar información
        """

        save_data = {'loss': self.loss,
                     'max_q': self.max_q,
                     'epsilon': self._epsilon._value,
                     'reward': self.reward,
                     'n_episodes': self.n_episodes} # Se crea un dict con la información

        if episodes_done > 0:
            save_name = self.save_name + f'_checkpoint{episodes_done}' # Si se ha hecho algo, se guarda la ruta de guradado y se a;ade el nombre del archivo
        else:
            save_name = self.save_name
        torch.save(self._Q.state_dict(), save_name + '.pth') # Se utiliza torch para hacer checkpoint del modelo
        pickle.dump(save_data, open(f'{save_name}_data.pkl', 'wb')) # Se salva con pickle los datos


    def evaluate(self, env, max_episodes=10000, load_dict=True):
        """"
        Función para correr el modelo sin entrenar.
        """
        if load_dict:
            self.load_model_checkpoint(load_params=False) # Se cargan los parámetros
        self._epsilon.isTraining = False # se fija Epsilon a cero, política Greedy
        while True:
            self.run_loop(env, self.max_frames, max_episodes=max_episodes, evaluate_checkpoints=0) # Se hace correr el agente hasta que se detenga manualmente

    def train(self, env, training=True, max_episodes=10000):
        """"
        Función para entrenar un agente
        """
        self._epsilon.isTraining = training # Se fija el epsilon para entrenar
        self.run_loop(env, self.max_frames, max_episodes=max_episodes) # Se hace correr el loop del agente por n episodios
        if self._epsilon.isTraining: # si al final sigue entrenando
            self.save_data() # se guardan los datos

    @abstractmethod
    def run_loop(self, env, max_frames, max_episodes, evaluate_checkpoints):
        """"
        Método run loop, para ser heredado y sobrecargado
        """
        pass

    def get_action(self, s, unsqueeze=True):
        """"
        Implementación Epsilon Greedy
        """
        if np.random.rand() > self._epsilon.value():
            s = torch.from_numpy(s).to(self.device) # Convierte el estado a un tensor de torch
            if unsqueeze: # Si se quiere realizar unsqueeze (Insertar una componente de dimension uno en cierta posición)
                s = s.unsqueeze(0).float() # Se retorna un tensor [1 X N X M X ..... X Z] (batch size 1)
            else:
                s = s.float() # Si no, simplemente retorna la acción convertida de tensor a float
            with torch.no_grad(): # Sin calcular gradiente
                self._action = self._Q(s).squeeze().cpu().data.numpy() # Se obtine la predicción desde el estado
            return self._action.argmax() # Se obtiene la acción Greedy
        # explore
        else:
            # Política aleatoria
            action = 0
            target = np.random.randint(0, self._screen_size, size=2)
            return action * self._screen_size * self._screen_size + target[0] * self._screen_size + target[1] # Se retorna el targer

    def train_q(self, squeeze=False):
        """"
        Función para entrenar los predictores
        """
        if self.train_q_batch_size >= len(self._memory): # Si no hay suficientes registros en el buffer para completar un batch no se realiza el entrenamiento
            return

        s, a, s_1, r, done = self._memory.sample(self.train_q_batch_size) # Se obtiene un batch de observaciones desde el Batch

        # Se malean a la forma de torch
        s = torch.from_numpy(s).to(self.device).float()
        a = torch.from_numpy(a).to(self.device).long().unsqueeze(1)
        s_1 = torch.from_numpy(s_1).to(self.device).float()
        r = torch.from_numpy(r).to(self.device).float()
        done = torch.from_numpy(1 - done).to(self.device).float()

        if squeeze: # Si se quieren eliminar las dimensiones de orden 1
            s = s.squeeze()
            s_1 = s_1.squeeze()

        Q = self._Q(s).view(self.train_q_batch_size, -1) # Se transforma la dimension de la predicción principal al tamaño del batch
        Q = Q.gather(1, a) # Almacena los elementos a lo largo de una dimensión

        Qt = self._Qt(s_1).view(self.train_q_batch_size, -1) # Se obtiene la predicción del target con dimension modificada

        # double Q
        best_action = self._Q(s_1).view(self.train_q_batch_size, -1).max(dim=1, keepdim=True)[1] # Se obtiene la mejor accion prediciendo el estado proximo
        y = r + done * self.gamma * Qt.gather(1, best_action) # Se calcula el valor de retorno segun la fromula de Q-Learning

        loss = self._criterion(Q, y) # Se calcula la périda
        self._loss.append(loss.sum().cpu().data.numpy()) # Se registra
        self._max_q.append(Q.max().cpu().data.numpy().reshape(-1)[0]) # Se guarda la mayor predicción
        self._optimizer.zero_grad()  # Se retorna el gradiente de todos los tensores optimizados a cero
        loss.backward() # Se retropropaga
        self._optimizer.step() # Lleva a cabo una instancia de optimización