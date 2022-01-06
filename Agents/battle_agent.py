import copy
import numpy as np
import time
from joblib import dump

from pysc2.lib import actions, features
from Utils.replay_memory import Transition

from Models.nn_models import DefeatRoachesDQN, DefeatRoachesD3QN
from Agents.rl_agent import BaseRLAgent

# Características y acciones
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index # Screen de las características relativas a los jugadores
_NO_OP = actions.FUNCTIONS.no_op.id # Sin operación
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id # Mover y Atacar
_SELECT_ARMY = actions.FUNCTIONS.select_army.id # Seleccionar ejercito
_UNIT_TYPE = 6 # Screen tipo de unidad
_UNIT_HIT_POINTS = 8 # Screen puntos de ataque
FUNCTIONS = actions.FUNCTIONS # Funciones de las acciones

class BattleAgent(BaseRLAgent):
    """
    Agent where the entire army is selected
    """

    def __init__(self, save_name=None, load_name=None):
        super(BattleAgent, self).__init__(save_name=save_name, load_name=load_name)
        self.initialize_model(DefeatRoachesDQN(3, screen_size=self._screen_size)) # Se inicializa el modelo
        self.steps_before_training = 5000 # Pasos antes del entrenamiento
        self.obs = None # Placeholder de la observación
        self.features = [_PLAYER_RELATIVE, _UNIT_TYPE, _UNIT_HIT_POINTS] # Características
        self.train_q_per_step = 1 # Se entrena q en cada paso

    def run_loop(self, env, max_frames=0, max_episodes=10000, save_checkpoints=500, evaluate_checkpoints=10):
        """Loop Principal. Solo se comentan los cambios con respecto a Beacon"""
        total_frames = 0
        start_time = time.time()

        action_spec = env.action_spec()
        observation_spec = env.observation_spec()

        self.setup(observation_spec, action_spec)
        try:
            while self.n_episodes < max_episodes:

                obs = env.reset()[0]

                select_army = actions.FunctionCall(_SELECT_ARMY, [[False]])
                obs = env.step([select_army])[0]

                self.reset()
                episode_reward = 0

                while True:
                    total_frames += 1

                    self.obs = obs.observation["feature_screen"][self.features] # Se extraen varias carácterísticas desde la pantalla
                    s = np.expand_dims(self.obs, 0)

                    if max_frames and total_frames >= max_frames:
                        print("max frames reached")
                        return
                    if obs.last():
                        print(f"Episode {self.n_episodes + 1}:\t total frames: {total_frames} Epsilon: {self._epsilon.value()}")
                        self._epsilon.increment()
                        break

                    action = self.get_action(s, unsqueeze=False)
                    env_actions = self.get_env_action(action, obs, command=_ATTACK_SCREEN)
                    try:
                        obs = env.step([env_actions])[0] # Se intenta realizar una acción
                        r = obs.reward - 10 # Se penaliza el no atacar
                    except ValueError as e: # De ser una acción no valida
                        print(e)
                        obs = env.step([actions.FunctionCall(_NO_OP, [])])[0] # no se hace nada
                        r = obs.reward - 1000 # Se penaliza al agente fuertemente
                    episode_reward += r # se aumenta el contador de episodios
                    s1 = np.expand_dims(obs.observation["feature_screen"][self.features], 0)
                    done = r > 0
                    if self._epsilon.isTraining:
                        transition = Transition(s, action, s1, r, done)
                        self._memory.push(transition)

                    if total_frames % self.train_q_per_step == 0 and total_frames > self.steps_before_training and self._epsilon.isTraining:
                        self.train_q(squeeze=True)

                    if total_frames % self.target_q_update_frequency == 0 and total_frames > self.steps_before_training and self._epsilon.isTraining:
                        self._Qt = copy.deepcopy(self._Q)

                if evaluate_checkpoints > 0 and ((self.n_episodes % evaluate_checkpoints) - (evaluate_checkpoints - 1) == 0 or self.n_episodes == 0):
                    print('Evaluating...')
                    self._epsilon.isTraining = False  # we need to make sure that we act greedily when we evaluate
                    self.run_loop(env, max_episodes=max_episodes, evaluate_checkpoints=0)
                    self._epsilon.isTraining = True
                if evaluate_checkpoints == 0:  # this should only activate when we're inside the evaluation loop
                    self.reward.append(episode_reward)
                    dump(self.reward, 'battle_evaluation_reward.joblib')
                    print(f'Evaluation Complete: Episode reward = {episode_reward}')
                    break

                self.n_episodes += 1
                if len(self._loss) > 0:
                    self.loss.append(self._loss[-1])
                    self.max_q.append(self._max_q[-1])
                if self.n_episodes % save_checkpoints == 0:
                    if self.n_episodes > 0:
                        self.save_data(episodes_done=self.n_episodes)

        except KeyboardInterrupt:
            pass
        finally:
            print("finished")
            elapsed_time = time.time() - start_time
            try:
                print("Took %.3f seconds for %s steps: %.3f fps" % (
                    elapsed_time, total_frames, total_frames / elapsed_time))
            except:
                print("Took %.3f seconds for %s steps" % (elapsed_time, total_frames))
