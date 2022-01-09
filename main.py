"""
Script principal desde el que se ejecuta el programa. En el aparece la configuración del ambiente y la ejecución de
este y los agentes.

"""

# Cabeceras
from absl import app
from absl import flags

# Files
from environment import get_environment
from runner import Runner

#  Banderas con parametros de ejecución
FLAGS = flags.FLAGS
flags.DEFINE_bool("train", True, "Whether we are training or running")
flags.DEFINE_string("agent", "DQNAgent", "Which agent to run")
flags.DEFINE_string("load_file", f'./data/MoveToBeacon/beacon_13149steps_32dim', "file to load params from")
flags.DEFINE_string("save_file", 'save_results/', "file to save params to / load from if not loading from checkpoint")
flags.DEFINE_integer('episodes', 10000, 'Num of episodes')
flags.DEFINE_string('map', 'MoveToBeacon', 'Map to be played')
flags.DEFINE_integer('stepMult', 8, 'Speed of render')
flags.DEFINE_integer('feature_size', 32, 'Minimap and screen size')
flags.DEFINE_bool ('visualize', False, 'Visualize the feature screen')
flags.DEFINE_string('agent_name', 'Beacon', 'Agent to play')
flags.DEFINE_integer('max_frames', 10000000, 'Max steps per episode')
flags.DEFINE_float('epsilon_start', 1, 'Start value for Epsilon')
flags.DEFINE_float('epsilon_end', 0.1, 'End value for Epsilon')
flags.DEFINE_float('epsilon_decrement', 0.0001, 'Decrement of Epsilon per episode')
flags.DEFINE_integer('batch_size', 256, 'Size of the training batch')
flags.DEFINE_float('gamma', 0.99, 'Discount factor')
flags.DEFINE_integer('steps_before_training', 5000, 'Steps before start training')
flags.DEFINE_integer('target_update', 10000, 'Target actualization interval')
flags.DEFINE_float('lr', 1e-8, 'Learning rate for neuronal networks')
flags.DEFINE_bool ('dueling', False, 'Dueling DQN selector')

# Configuración para la ejecución
def main(unused_argv):

    # Creación del ambiente
    env = get_environment(
                        map_name=FLAGS.map,
                        feature_dimensions=FLAGS.feature_size,
                        step_mul=FLAGS.feature_size,
                        visualize=FLAGS.visualize,

                        )

    #Instancia de ejecutor
    runner = Runner(
                    agent_name= FLAGS.agent_name,
                    env=env,
                    map_name=FLAGS.map,
                    FLAGS=FLAGS

                    )

    #Ejecución
    runner.run(episodes=FLAGS.episodes)


if __name__ == "__main__":
    app.run(main) # Correr la app
