# LunarLander model

The LunarLander environment is shared by SAC and PPO.

## 1. Problem and observations

A lander descends toward a central landing area. It must regulate horizontal and vertical motion and attitude, avoid crashing, and settle on the ground. Gravity, engine rules, the terrain-generation distribution, and the reward are fixed, and wind is disabled. Initial impulses and terrain are randomized in each episode.

Gymnasium's eight observations are normalized horizontal/vertical position, horizontal/vertical velocity, body angle, angular velocity, and two leg-contact indicators. The simulator also contains terrain and contact state that are not part of this observation.

Two actuator-command states are appended in **both** conditions, so each policy has a 10-dimensional input and a two-dimensional continuous output. These states expose the memory introduced by the lag filter.

## 2. Actions and engine-command response

The policy requests $a_t=(a_t^{main},a_t^{side})\in[-1,1]^2$. With $\Delta t=1/50$ s,

$$\beta=\begin{cases}1,&\tau=0,\\1-\exp(-\Delta t/\tau),&\tau>0,\end{cases}
\qquad u_t=u_{t-1}+\beta(a_t-u_{t-1}).$$

The filtered command $u_t$ is passed to the unmodified environment. Reset sets $u_{-1}=(-1,0)$: engines off. The observation after each step contains $u_t$. For $\tau=0$, transitions and rewards exactly match the standard environment under the same actions and seed.

The main engine is off for $u^{main}\leq0$; otherwise its power is $(u^{main}+1)/2$. Side engines fire only for $|u^{side}|>0.5$, with power $|u^{side}|$. Filtering commands therefore delays threshold crossings; it is **not** equivalent to exponentially filtering physical thrust. At $\tau=0.10$ s, $\beta\approx0.18$: a command step reaches 63% of its new value after five simulation frames (0.1 s) and 95% after about fifteen frames.

## 3. Reward and episode outcomes

The built-in reward is unchanged: shaping terms for position, velocity, angle, and leg contact, minus engine-use costs of $0.30p_{main}+0.03p_{side}$ per step. A crash or departure beyond the horizontal bounds gives a terminal reward of −100; a lander that comes to rest gives +100. The 1000-step time limit is a truncation, and both SAC and PPO bootstrap through it.

Outcomes are reported as **settled landing**, failure (crash or out of bounds), and timeout. Settled landing follows the simulator's terminal condition, in which the lander body comes to rest; it does not require touching down on the pad or below a specific speed. Mean return and the fraction of episodes with return ≥ 200 are also reported; the latter is a return threshold, distinct from the landing rate.

Engine-use integrals are proxies in power-seconds, not fuel mass. First-contact speed and angle are measured after the physics step in which a leg first touches the ground, so they are not exact pre-impact values.

## 4. Learning algorithms

Both algorithms use the Stable-Baselines3 implementations with two 256-unit ReLU hidden layers and $\gamma=0.99$; their objectives are described in the [navigation model](navigation.md#5-learning-algorithms). SAC uses a 300,000-transition replay buffer and begins updates after 5000 steps. PPO uses 1000-step rollouts, 10 epochs, minibatches of 250, clip range 0.2, and GAE with $\lambda=0.95$. Training stops early when validation B return has not improved by at least 2.0 for 10 consecutive evaluations after 100,000 steps.
