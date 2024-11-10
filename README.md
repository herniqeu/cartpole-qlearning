# Análise Técnica e Implementação do Problema CartPole usando Q-Learning e Equações de Bellman

## 1. Visão Geral

O problema CartPole representa um sistema dinâmico não-linear de quarta ordem, caracterizado por quatro variáveis de estado:
- x: Posição do carrinho
- ẋ: Velocidade do carrinho
- θ: Ângulo do pêndulo
- θ̇: Velocidade angular do pêndulo

### 1.1 Dinâmica do Sistema
O sistema é regido pelas seguintes equações diferenciais:

```math
ẍ = \frac{F + ml\ddot{\theta}\cos{\theta} - ml\dot{\theta}^2\sin{\theta}}{M + m}

\ddot{\theta} = \frac{g\sin{\theta} - \ddot{x}\cos{\theta}}{l}
```

Onde:
- m: massa do pêndulo
- M: massa do carrinho
- l: comprimento do pêndulo
- F: força aplicada ao carrinho
- g: aceleração gravitacional

## 2. Fundamentação Matemática

### 2.1 Formulação MDP
O problema é modelado como um MDP definido pela tupla (S, A, P, R, γ), onde:
- S: Espaço de estados S ⊆ ℝ⁴
- A: Espaço de ações A = {0, 1}
- P: P(s'|s,a): S × A × S → [0,1]
- R: R(s,a): S × A → ℝ
- γ: Fator de desconto γ ∈ [0,1]

### 2.2 Q-Learning e Equação de Bellman
A equação de Bellman para a função valor-ação ótima Q*(s,a) é:

```math
Q^*(s,a) = \mathbb{E}_{s'}\left[R(s,a) + \gamma \max_{a'} Q^*(s',a')\right]
```

O algoritmo Q-learning aproxima esta função através de atualizações iterativas:

```math
Q(s,a) \leftarrow Q(s,a) + \alpha\left[R + \gamma \max_{a'} Q(s',a') - Q(s,a)\right]
```

Esta atualização é implementada na classe `CartPoleQLearningAgent`:

```python
def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
    current_state = self.discretize_state(state)
    next_state = self.discretize_state(next_state)
    
    # Implementação da equação de Bellman
    future_q_value = (not done) * np.max(self.q_table[next_state])
    temporal_difference = (
        reward + self.gamma * future_q_value - self.q_table[current_state][action]
    )
    
    self.q_table[current_state][action] += self.lr * temporal_difference
```

## 3. Implementação Técnica

### 3.1 Discretização do Espaço de Estados
A discretização transforma o espaço contínuo S ⊆ ℝ⁴ em um espaço discreto Ŝ através de uma função de discretização φ: S → Ŝ:

```math
φ(s)_i = \left\lfloor\frac{s_i - min_i}{max_i - min_i} \cdot n_{bins}\right\rfloor
```

Implementada como:

```python
def _create_bins(self) -> Dict[int, np.ndarray]:
    bins = {
        0: np.linspace(-4.8, 4.8, self.n_bins),    # x
        1: np.linspace(-4, 4, self.n_bins),        # ẋ
        2: np.linspace(-0.418, 0.418, self.n_bins),# θ
        3: np.linspace(-4, 4, self.n_bins)         # θ̇
    }
    return bins

def discretize_state(self, state: np.ndarray) -> Tuple:
    discretized = []
    for i, value in enumerate(state):
        bin_index = np.digitize(value, self.bins[i]) - 1
        discretized.append(bin_index)
    return tuple(discretized)
```

### 3.2 Política ε-greedy
A política de seleção de ações segue uma distribuição de probabilidade:

```math
π(a|s) = \begin{cases} 
ε/|A| + (1-ε) & \text{se } a = \arg\max_{a'} Q(s,a') \\
ε/|A| & \text{caso contrário}
\end{cases}
```

Implementada como:

```python
def get_action(self, state: np.ndarray) -> int:
    if np.random.random() < self.epsilon:
        return self.env.action_space.sample()
    
    discretized_state = self.discretize_state(state)
    return int(np.argmax(self.q_table[discretized_state]))
```

### 3.3 Decaimento de ε
O parâmetro ε decai exponencialmente segundo:

```math
ε_t = \max(ε_{min}, ε_0 \cdot d^t)
```

Onde:
- ε₀: epsilon inicial
- d: taxa de decaimento
- t: número do episódio
- ε_{min}: valor mínimo de epsilon

```python
def decay_epsilon(self):
    self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
```

## 4. Análise de Convergência

### 4.1 Condições de Convergência
O Q-learning converge para Q* sob as seguintes condições:
1. Visitação suficiente: ∑ₜ I(sₜ=s, aₜ=a) = ∞
2. Taxa de aprendizado decrescente: ∑ₜ αₜ = ∞, ∑ₜ αₜ² < ∞
3. Recompensas limitadas: |R(s,a)| ≤ Rₘₐₓ < ∞

### 4.2 Métricas de Avaliação
A performance é avaliada através de três métricas principais:

1. Recompensa média por episódio:
```math
\bar{R}_n = \frac{1}{n}\sum_{i=1}^n R_i
```

2. Taxa de sucesso:
```math
S_n = \frac{\text{episódios com duração máxima}}{n}
```

3. Erro TD médio:
```math
\text{TD}_{\text{error}} = |R + \gamma \max_{a'} Q(s',a') - Q(s,a)|
```

Implementadas em:

```python
def evaluate_agent(agent: CartPoleQLearningAgent, n_episodes: int = 5) -> float:
    total_rewards = []
    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward
            
        total_rewards.append(episode_reward)
    
    return np.mean(total_rewards)
```

## 5. Resultados Experimentais

### 5.1 Hiperparâmetros Utilizados
```python
learning_rate = 0.1
discount_factor = 0.95
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
n_bins = 10
```

### 5.2 Análise de Convergência
A convergência foi analisada através da evolução temporal de três métricas:

1. Valor Q médio: 
```math
\bar{Q}_t = \frac{1}{|S||A|}\sum_{s,a} Q_t(s,a)
```

2. Variância dos valores Q:
```math
\text{Var}(Q_t) = \frac{1}{|S||A|}\sum_{s,a} (Q_t(s,a) - \bar{Q}_t)^2
```

3. Taxa de exploração efetiva:
```math
ε_{\text{eff}} = \frac{\text{ações aleatórias}}{\text{total de ações}}
```

## 6. Conclusões e Extensões Teóricas

### 6.1 Limitações Matemáticas
1. Erro de discretização:
```math
E_d = \sup_{s \in S} \|s - \hat{s}\|
```
onde ŝ é o estado discretizado.

2. Erro de aproximação da função Q:
```math
E_Q = \|Q^* - Q_\theta\|_\infty
```