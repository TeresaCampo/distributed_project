import numpy as np
import functools
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv

class CustomGridWorld(ParallelEnv):
    metadata = {"render_modes": ["human", "ansi"], "name": "custom_grid_v0"}

    def __init__(self):
        # 1. Configurazione Base
        self.grid_size = 5
        self.possible_agents = ["agent_0", "agent_1"]
        self.agents = self.possible_agents[:]
        
        # 2. Definizione delle Componenti Fisse (Posizione e Tipo)
        # Esempio: Componente A (Ricompensa), Componente B (Penalità/Trappola)
        self.fixed_components = {
            "gold":  {"pos": np.array([4, 4]), "reward": 10},
            "trap":  {"pos": np.array([2, 2]), "reward": -5}
        }
        
        # 3. Definizione degli Ostacoli (Mappa personalizzata)
        # 1 = Muro, 0 = Libero
        self.grid_map = np.zeros((self.grid_size, self.grid_size))
        self.grid_map[1, 1] = 1 # Esempio ostacolo
        self.grid_map[1, 2] = 1 # Esempio ostacolo

        # 4. Spazi di Azione e Osservazione
        # Azioni: 0=Fermo, 1=Su, 2=Giù, 3=Sinistra, 4=Destra
        self.action_spaces = {
            agent: spaces.Discrete(5) for agent in self.possible_agents
        }
        
        # Osservazione: Per semplicità, diamo la posizione (x, y) dell'agente
        self.observation_spaces = {
            agent: spaces.Box(low=0, high=self.grid_size-1, shape=(2,), dtype=np.int32) 
            for agent in self.possible_agents
        }

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        
        # Posiziona gli agenti (Esempio: Angolo in alto a sx e basso sx)
        self.agent_positions = {
            "agent_0": np.array([0, 0]),
            "agent_1": np.array([0, 4])
        }
        
        observations = {a: self.agent_positions[a] for a in self.agents}
        infos = {a: {} for a in self.agents}
        return observations, infos

    def step(self, actions):
        # Se non ci sono agenti, restituisci tutto vuoto (richiesto da PettingZoo)
        if not self.agents:
            return {}, {}, {}, {}, {}

        rewards = {a: 0 for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}

        # Logica di Movimento per ogni agente
        for agent, action in actions.items():
            current_pos = self.agent_positions[agent].copy()
            target_pos = current_pos.copy()

            # Calcola movimento
            if action == 1: target_pos[1] -= 1 # Su
            elif action == 2: target_pos[1] += 1 # Giù
            elif action == 3: target_pos[0] -= 1 # Sinistra
            elif action == 4: target_pos[0] += 1 # Destra
            
            # Controllo 1: Limiti della Griglia
            target_pos = np.clip(target_pos, 0, self.grid_size - 1)
            
            # Controllo 2: Ostacoli (Muri)
            if self.grid_map[target_pos[0], target_pos[1]] == 1:
                target_pos = current_pos # Rimbalza/Resta fermo

            # Aggiorna posizione
            self.agent_positions[agent] = target_pos

            # --- Logica Componenti Personalizzate ---
            # Controlla se l'agente è finito sopra una componente
            for comp_name, comp_data in self.fixed_components.items():
                if np.array_equal(target_pos, comp_data["pos"]):
                    rewards[agent] += comp_data["reward"]
                    # Esempio: Se prendi l'oro, l'episodio finisce per quell'agente
                    if comp_name == "gold":
                        terminations[agent] = True

        # Aggiorna osservazioni finali
        observations = {a: self.agent_positions[a] for a in self.agents}
        
        # Rimuovi agenti che hanno finito (standard PettingZoo)
        self.agents = [a for a in self.agents if not terminations[a]]

        return observations, rewards, terminations, truncations, infos

    def render(self):
        # Visualizzazione testuale semplice
        grid = np.full((self.grid_size, self.grid_size), ".", dtype=str)
        
        # Disegna componenti
        grid[self.fixed_components["gold"]["pos"][0], self.fixed_components["gold"]["pos"][1]] = "G"
        grid[self.fixed_components["trap"]["pos"][0], self.fixed_components["trap"]["pos"][1]] = "T"
        
        # Disegna ostacoli
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.grid_map[x, y] == 1: grid[x, y] = "#"

        # Disegna agenti
        for agent, pos in self.agent_positions.items():
            char = "0" if agent == "agent_0" else "1"
            grid[pos[0], pos[1]] = char
            
        print("\n".join([" ".join(row) for row in grid.T])) # Trasposto per asse x/y intuitivo
        print("-" * 10)

    # Metodi di utility richiesti da PettingZoo
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]
    



# ----------------------------test
# Crea l'ambiente
env = CustomGridWorld()

observations, infos = env.reset()

# Ciclo di gioco (loop)
for _ in range(10):
    # Scegli azioni casuali per ogni agente attivo
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    
    # Esegui lo step
    observations, rewards, terminations, truncations, infos = env.step(actions)
    
    # Visualizza
    env.render()
    
    # Se non ci sono più agenti, esci
    if not env.agents:
        break