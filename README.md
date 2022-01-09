# Deep Reinforcement Learning en StarCraft 2
Este trabajo evalúa el desempeño de agentes de aprendizaje por refuerzo profundo en el ambiente de StarCraft 2. Con este fin se utiliza la bibloteca PySC2, provista por [Deep Mind](https://github.com/deepmind/pysc2) y cuya documentación se puede encontrar en: https://github.com/deepmind/pysc2

## Requerimientos para la ejecución
Estos son los requerimientos necesarios para ejecutar el código:
- Python 3
- Pythorch 1.1.10 (En este código, otras versiones pueden ser compatibles)
- Torchvision
- PySC2
- StartCraft 2

Los siguientes requerimientos no son mandatorios pero si altamente sugeridos:
- Disponibilidad de GPU 
- CUDA

## Ejecución del código

Para ejecutar el código se debe correr el archivo [main](main.py) desde la consola utilizando el siguinete comando:
```
python main.py
```
El flujo de ejecución de este comando y los hiperparámetros de los agentes y los modelos pueden ser modificados utilizando el conjunto de banderas mostradas a continuación.


|Nombre de la bandera| Tipo   | Valor por defecto | Otros valores posibles (**único posible**) |Descripción                                                             |
|--------------------|--------|-------------------|--------------------------------------------|------------------------------------------------------------------------|
| agent_name         | String | DQN          | **D3QN**                              | Tipo de agente a entrenar                                              |
| batch_size         | Int    | 256               | Enteros                                    | Tamaño del batch para entrenamiento                                    |
| episodes           | Int    | 10,000            | Enteros                                    | Número de episodios en los que se ejecutará el programa                |
| epsilon_decrement  | Float  | 0.0001            | (0, 1]                                     | Factor de decrecimiento de Epsilon para Epsilon-Greedy                 |
| epsilon_end        | Float  | 0.01              | (0, 1]                                     | Valor mínimo permitido para Epsilon                                    |
| epsilon_start      | Float  | 1                 | (0, 1]                                     | Valor en el que inicia Epsilon                                         | 
| feature_size       | Int    | 32                | Enteros                                    | Dimensión de la matriz de carácterisitca *(32 x 32)*                   |
| gamma              | Float  | 0.99              | 0 < x < 1                                  | Factor de decrecimineto del aprendizaje                                |
| help               | Esp    | -                 | -                                          | Imprime por pantalla la ayuda de las banderas            |
| load_file          | String | -                 | String                                     | Path para cargar los modelos entrenados y salvados                     |
| lr                 | Float  | 0.00000001        | (0, 1]                                     | Learning Rate para ajuste de modelos neuronales                        |
| map                | String | MoveToBeacon      | **DefeatRoaches**                          | Mapa utilizado para entrenar a los agentes                             |
| max_frames         | Int    | 10000000          | Enteros                                    | Límite de pasos permitidos en un episodio                              |
| realtime           | Bool   | False             | **True**                                   | Interruptor de ejecución en tiempo real (velocidad del render)         |
| save_file          | String | save_results/     | String                                     | Path para salvar los chekpoints realizados al modelo en  entrenamiento |
| step_mult          | Int    | 8                 | Enteros                                    | Multiplicador de pasos del ambiente. Regula la velocidad del render    |
| steps_before_training | Int    | 5000              | Enteros                                    | Pasos para recopilar información del ambiente antes de entrenar        |
| target_update      | Int    | 10000             | Enteros                                    | Intervalo para actualizar la target network                            |
| train              | Bool   | True              | **False**                                  | Interruptor para entrenar o evaluar los modelos                        |
### Ejemplos
- Para entrenar un agente Dueling Deep Q - Network en el mapa Move To Beacon y un batch size de 32:
```
python main.py --agent_name=D3QN --map=MoveToBeacon --batch_size=32
```
Como se puede observar, banderas como la bandera se omiten pues se utiliza su valor por defecto.

- Para evaluar un agente Double Deep Q  - Network con en el mapa de Defeat Roaches:
```
python main.py --agent_name=DQN --map=DefeatRoaches --train=False --load_file=<path to file>
```
En este caso load file es una bandera de definición obligatoria.

- Para invocar la ayuda:
```
python main.py --help
```
