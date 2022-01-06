"""
Script principal desde el que se ejecuta el programa. En el aparece la configuración del ambiente y la ejecución de
este y los agentes.

"""

# Cabeceras
from absl import app
from absl import flags

from environment import get_environment
from runner import Runner

#  Banderas con parametros de ejecución
FLAGS = flags.FLAGS  # kkkkk
flags.DEFINE_bool("train", True, "Whether we are training or running")
flags.DEFINE_string("agent", "BeaconAgent", "Which agent to run")
flags.DEFINE_string("load_file", f'./data/MoveToBeacon/beacon_13149steps_32dim', "file to load params from")
flags.DEFINE_string("save_file", '', "file to save params to / load from if not loading from checkpoint")


# Configuración para la ejecución


def main(unused_argv):
    _CONFIG = dict(
        episodes=10000,  # Episodios

        map_name='MoveToBeacon',  # Nombre del mapa
        screen_size=32,  # Tamaño de la pantalla
        minimap_size=32,  # Tamaño del minimapa
        step_mul=8,  # multiplicador de pasos
        visualize=False,  # visualización de las características

        agent_name='Beacon',  # Tipo de agente
        train=FLAGS.train,  # Indicador de entrenamiento

    )

    # Creación del ambiente
    env = get_environment(
                        map_name=_CONFIG['map_name'],
                        screen_size=_CONFIG['screen_size'],
                        minimap_size=_CONFIG['minimap_size'],
                        step_mul=_CONFIG['step_mul'],
                        visualize=_CONFIG['visualize']
                        )
    #Instancia de ejecutor
    runner = Runner(
                    agent_name= _CONFIG['agent_name'],
                    env=env,
                    map_name=_CONFIG['map_name'],
                    FLAGS=FLAGS
                    )

    #Ejecución
    runner.run(episodes=_CONFIG['episodes'])


if __name__ == "__main__":
    app.run(main) # Correr la app
