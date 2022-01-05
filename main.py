from absl import app

from environment import get_environment
from runner import Runner

_CONFIG = dict(
    episodes=10000,
    actions=8,

    map_name='MoveToBeacon',
    screen_size=32,
    minimap_size=32,
    step_mul=8,
    visualize=False,    
    
    method='DQN',
    gamma=0.99, 
    epsilon=1.0,
    lr=1e-4, 
    loss='mse', 
    batch_size=64,
    epsilon_decrease=0.005,
    epsilon_min=0.05, 
    update_target=2000,
    num_episodes=5000, 
    max_memory=100000,
    agent_name= 'Beacon',

    train=True,
    load_path='./graphs/train_PGAgent_190226_1942'
)

def main(unused_argv):

    env = get_environment(
                        map_name=_CONFIG['map_name'],
                        screen_size=_CONFIG['screen_size'],
                        minimap_size=_CONFIG['minimap_size'],
                        step_mul=_CONFIG['step_mul'],
                        visualize=_CONFIG['visualize']
                        )

    runner = Runner(
                    agent_name= _CONFIG['agent_name'],
                    env=env,
                    train=_CONFIG['train'],
                    map_name=_CONFIG['map_name']
                    )

    runner.run(episodes=_CONFIG['episodes'])


if __name__ == "__main__":
    app.run(main)
