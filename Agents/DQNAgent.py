""""
Un agente especializado en el mapa Move to Beacon
"""

import copy
import numpy as np

import time
from Agents.rl_agent import BaseRLAgent
from Models.DRL_Models import DQNModel
from Utils.replay_memory import Transition

from pysc2.lib import actions


# Acciones
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id # Accion atack screen, sirve para moverse por la pantalla y atacr enemigos
_SELECT_ARMY = actions.FUNCTIONS.select_army.id # Acción para seleccionar al ejercito

class DQNAgent(BaseRLAgent):

    def __init__(self, FLAGS, save_name=None, load_name=None):
        super(DQNAgent, self).__init__(FLAGS, save_name=save_name, load_name=load_name)


        self.initialize_model(DQNModel()) # LLamado a la funcion de inicialización seleccionando el modelo específico

        # Parámetros
        self.features = 5 # Para seleccionar las features
        self.train_q_per_step = 4 # Cantidad de pasos tras los que se realiza un entrenamiento

    def run_loop(self, env, max_frames=0, max_episodes=10000, save_checkpoints=500, evaluate_checkpoints=10):
        """
        Loop principal del agente
        """
        # Inicializando los parámetros
        total_frames = 0
        start_time = time.time()

        action_spec = env.action_spec() # Se extraen el espacio de acciones desde el ambiente
        observation_spec = env.observation_spec() # Y el espacio de observación


        self.setup(observation_spec, action_spec) # Con ellos se configura el agente base de pysc2
        try:
            while self.n_episodes < max_episodes: # Durante los episodiso

                obs = env.reset()[0] # Se reinicia el ambiente

                select_army = actions.FunctionCall(_SELECT_ARMY, [[False]]) # Se selecciona el ejercito por defecto para no tener que hacerlo despues
                obs = env.step([select_army])[0] # Se ejecuta la acción de selecctionar el ejército

                self.reset() # Se reinicia el agente
                episode_reward = 0 # Se inicia el contador de recompensas

                while True:
                    total_frames += 1 # Contador de pasos

                    self.obs = obs.observation["feature_screen"][self.features] # Se extrae la característica del feature_screen en este caso
                                                                                # Las relativas a los jugadores
                    s = np.expand_dims(self.obs, 0) # Se crea ele estado redimensionando la observación

                    if max_frames and total_frames >= max_frames:
                        print("max frames reached")
                        return # Fin de episodio
                    if obs.last(): # Si es la ultima iteración
                        # Resumen
                        print(f"Episode {self.n_episodes + 1}:\t total frames: {total_frames} Epsilon: {self._epsilon.value()}")
                        # Aumentar epsilon ( Para decrementar)
                        self._epsilon.increment()
                        break

                    # Si no
                    action = self.get_action(s) # Se obtiene una acción desde el estado siguiendo la politica EG
                    env_actions = self.get_env_action(action, obs, command=_ATTACK_SCREEN) # Se obtiene la accion desde el ambiente
                    obs = env.step([env_actions])[0] # Y se ejecuta
                    r = obs.reward # Se obtiene la recompensa
                    episode_reward += r # Se aumenta el contador
                    s1 = np.expand_dims(obs.observation["feature_screen"][self.features], 0) # Se obtiene el proximo estado
                    done = r > 0 # Y se guarda el logrado si la recompensa es mayor que 0
                    if self._epsilon.isTraining: # Si se esta entrenando
                        transition = Transition(s, action, s1, r, done) # Se almacena la experiencia
                        self._memory.push(transition) # Se guarda en el  buffer

                    if total_frames % self.train_q_per_step == 0 and total_frames > self.steps_before_training and self._epsilon.isTraining:
                        self.train_q() # Se entrena Q si se encuentra en uno de los pasos de entrenamiento (1/4 por defecto)

                    if total_frames % self.target_q_update_frequency == 0 and total_frames > self.steps_before_training and self._epsilon.isTraining:
                        self._Qt = copy.deepcopy(self._Q) # Si se cumple la frecuencia de actualización, se actualiza la target network

                if evaluate_checkpoints > 0 and ((self.n_episodes % evaluate_checkpoints) - (evaluate_checkpoints - 1) == 0 or self.n_episodes == 0):
                    print('Evaluating...') # Episodio de evaluación siguiendo la politica greedy
                    self._epsilon.isTraining = False  # Se pasa el epsilon a flaso para asegurarnos de que siga la politica greedy
                    self.run_loop(env, max_episodes=max_episodes, evaluate_checkpoints=0) # Se corre el loop evaluando
                    self._epsilon.isTraining = True # Se vuelve a entrenar
                if evaluate_checkpoints == 0:  # Esto solo ocurre cuando esta evaluando
                    self.reward.append(episode_reward) # Se almacena la recompensa
                    print(f'Evaluation Complete: Episode reward = {episode_reward}')
                    break

                self.n_episodes += 1 # Aumenta contador de episodios
                if len(self._loss) > 0:
                    self.loss.append(self._loss[-1]) # Se almacena la pérdida
                    self.max_q.append(self._max_q[-1]) # y la maxima predicción
                if self.n_episodes % save_checkpoints == 0: # Si se llega a un checkpoint

                    if self.n_episodes > 0:

                        self.save_data(episodes_done=self.n_episodes) # Se salvan los datos

        except KeyboardInterrupt:
            pass
        finally:
            print("finished")
            elapsed_time = time.time() - start_time
            print("Took %.3f seconds for %s steps: %.3f fps" % (
                elapsed_time, total_frames, total_frames / elapsed_time))

