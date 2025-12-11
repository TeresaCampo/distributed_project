# Multiple Agent RL
Proposal of "Dstributed AI" exam project.

Gridworld cooperative game.


<img src="readme_resources/demo.gif" alt="First demo" width="400"/>

### Environment implementation
Environment has been implemented basing on [ParallelEnv](https://pettingzoo.farama.org/api/parallel/ ) of PettingZoo.
It implements all the necessary methods and has all the required class variables. The compliance with the ParallelEnv requirements has been tested with the function [test_my_env](./src/test_parallel_env.py) that exploits [testing function provided in PettingZoo documentation](https://pettingzoo.farama.org/content/environment_tests/) .


### Elements
Variable number of agents (in the simulation 2).
One button, while pressed makes the gate open (self.gate_open = True).
One gate, when open vecomes a walkable cell otherwise it behaves as a wall.

### Observation of each agent
*[my_cur_pos,(other_cur_pos) * number of other agents, button_pos, gate_pos, gate_open]*

Where:
- my_cur_pos, is the absolute position (x,y) of the agent itself
- other_cur_pos, is the absolute position (x,y) of another agent
- button_pos, is the absolute position (x,y) of the button
- gate_pos, is the absolute position (x,y) of the gate
- gate_open, is an int value that represent wheather the gate is open (1) or close (0)




### Goal of the agents
The agents goal is to walk in the other room, behind the gate.
The reward function has not been defined yet. Now I focused on the environment only.


## To try it on your pc
First install the required dependencies
- **Solution 1**: :

    Install dependencies from requirements.txt

    These requirements match with Python 3.10.19
    ```bash
    pip install requirements.txt
    ```
- **Solution 2**

    Create conda environment with environment.yml
    ```bash
    conda env create -f environment.yml
    ```

Then [execute the file](./src/myenv_3observations_rewards.py) containing the last version of the project.