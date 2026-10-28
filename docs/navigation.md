# Navigation model

The navigation environment is shared by SAC and PPO.

## 1. Navigation problem and information structure

Let the workspace be $\mathcal W=[0,10]^2$, with static obstacle set $\mathcal O$, goal $g\in\mathcal W$, and robot radius $R=0.20$ m. An episode samples circular and axis-aligned rectangular obstacles, a collision-free initial pose, and a goal satisfying $\|p_0-g\|_2\geq4$ m. Circles have radii in $[0.35,0.70]$ m; rectangle widths and heights lie in $[0.55,1.30]$ m. Obstacles may overlap.

A conservative grid-connectivity test accepts only physically connected start–goal configurations. It tests whole-cell clearance using an additional half-cell-diagonal allowance, which can reject narrow feasible routes. It does not certify that a particular controller will complete the route. Neither sampling nor physical reachability depends on the observation margin.

At decision time $t$, the simulator's state can be represented as

$$s_t=(q_t,g,\mathcal O,v_{t-1}^{\mathrm{last}},\omega_{t-1}^{\mathrm{last}},t),\qquad q_t=(x_t,y_t,\theta_t).$$

The policy does not observe this full state. It receives a local noisy observation $o_t$ and selects $a_t\sim\pi_\phi(\cdot\mid o_t)$. The environment is therefore partially observed from the policy's perspective. The feedforward policy has no explicit recurrent memory.

**Information flow:** true state → noisy observation → policy waypoint → fixed tracking controller → physical motion and collision checks → reward and next observation. The tracker uses true simulated pose; geometry is available to the simulator for sensing, rewards, and collision checking, but is not supplied directly to the policy.

## 2. Robot dynamics and waypoint execution

The robot follows unicycle kinematics,

$$\dot x=v\cos\theta,\qquad\dot y=v\sin\theta,\qquad\dot\theta=\omega.$$

For each integration increment $\Delta t$, midpoint integration gives

$$\begin{aligned}
x^+&=x+v\Delta t\cos(\theta+\omega\Delta t/2),\\
y^+&=y+v\Delta t\sin(\theta+\omega\Delta t/2),\\
\theta^+&=\operatorname{wrap}(\theta+\omega\Delta t).
\end{aligned}$$

The policy action is a normalized robot-frame waypoint offset:

$$a_t=(a_t^{\mathrm f},a_t^{\mathrm l})\in[-1,1]^2,\qquad
w_t=p_t+1.5\,R(\theta_t)a_t,$$

where $R(\theta)$ is the planar rotation matrix. Each coordinate is scaled by 1.5 m; the diagonal offset can therefore reach $1.5\sqrt2$ m. A negative forward component requests a waypoint behind the robot, not backward velocity.

At each controller tick, define $d_w=\|w_t-p\|_2$ and

$$e_\theta=\operatorname{wrap}\!\left(\operatorname{atan2}(w_{t,y}-y,w_{t,x}-x)-\theta\right).$$

The fixed controller computes

$$v=\min(0.8,0.9d_w)\max(0,\cos e_\theta),\qquad
\omega=\operatorname{clip}(2.5e_\theta,-1.5,1.5).$$

One RL action executes for at most 0.5 s, with ten 0.05 s controller ticks. Each tick is subdivided to limit translation to 0.02 m and rotation to 0.04 rad per integration increment. The waypoint stays fixed within an action and is replaced at the next decision. A colliding increment is rejected and the episode terminates. These discrete checks do not guarantee continuous swept-path safety.

There is no motor lag, acceleration limit, or wheel slip. The experiment isolates waypoint planning rather than motor control or localization.

## 3. Observation model and obstacle margin

For margin $m\geq0$, a circle $(c_x,c_y,r)$ is perceived with radius $r+m$. A rectangle $(x,y,w,h)$ is perceived as $(x-m,y-m,w+2m,h+2m)$. Rectangle corners use square expansion, a conservative approximation to a rounded offset. **Arena walls are not inflated.** Let this perceived obstacle set be $\mathcal O^{(m)}$.

The physical obstacle set $\mathcal O$ remains unchanged. The robot radius is already accounted for in collision checking; $m$ is an additional observation transformation, not a change in robot size or collision penalty.

There are 36 equally spaced LiDAR beams over $2\pi$, starting at the robot center. If $D_i(q_t,\mathcal O^{(m)})$ is the distance to the nearest perceived surface or wall along beam $i$,

$$\ell_{t,i}=\operatorname{clip}\bigl(D_i(q_t,\mathcal O^{(m)})+\epsilon_{t,i},0,5\bigr),\quad
\epsilon_{t,i}\sim\mathcal N(0,0.02^2).$$

If the sensor origin lies inside an inflated obstacle, all LiDAR readings are set to zero, so the interior of an inflated obstacle is never reported as free space.

Noisy localization is modeled by

$$\hat p_t=p_t+\xi_t,\quad \xi_t\sim\mathcal N(0,0.01^2I_2),\qquad
\hat\theta_t=\operatorname{wrap}(\theta_t+\zeta_t),\quad\zeta_t\sim\mathcal N(0,0.01^2).$$

Define $\hat d_t=\|g-\hat p_t\|_2$ and $\hat\beta_t=\operatorname{wrap}(\operatorname{atan2}(g_y-\hat y_t,g_x-\hat x_t)-\hat\theta_t)$. The observation is

$$o_t=\left[\frac{\ell_{t,1}}5,\ldots,\frac{\ell_{t,36}}5,
\operatorname{clip}\!\left(\frac{\hat d_t}{15},0,1\right),
\frac{\hat\beta_t}{\pi},\frac{v_{t-1}^{\mathrm{last}}}{0.8},
\frac{\omega_{t-1}^{\mathrm{last}}}{1.5}\right]\in\mathbb R^{40}.$$

The velocity entries are last commanded velocities, not noisy measured velocities. Sensor errors are independent across observations; they do not model accumulated odometry drift. The margin is fixed for each agent and is used consistently during its training and evaluation.

## 4. Reward and episode outcomes

Let $d_t=\|g-p_t\|_2$ denote the true goal distance. The reward for one high-level action is

$$r_t=10(d_t-d_{t+1})-0.05-20\,\mathbf1_{\mathrm{collision}}+50\,\mathbf1_{\mathrm{success}}.$$

Progress is rewarded, time has a small per-decision cost, and physical collisions are penalized. A detour may temporarily incur negative progress reward. There is no explicit clearance, turning, or control-effort penalty.

An episode has exactly one outcome:

| Outcome | Condition | Gymnasium signal |
|---|---|---|
| Success | True goal distance is less than 0.35 m | `terminated=True` |
| Collision | A proposed increment intersects a physical obstacle or wall | `terminated=True` |
| Timeout | 160 actions are exhausted without success or collision | `truncated=True` |

Timeout is an external time limit, not an additional terminal penalty. Both SAC and PPO bootstrap the value estimate at time-limit truncations and treat success and collision as true terminal states.

## 5. Learning algorithms

Both algorithms use the Stable-Baselines3 implementations; this project implements the environments and the experiment pipeline. Policy and value networks have two 256-unit ReLU hidden layers, and the discount factor is $\gamma=0.99$. Evaluation uses deterministic policy actions without learning.

**SAC** maximizes an entropy-regularized discounted return,

$$J(\phi)=\mathbb E\left[\sum_t\gamma^t\left(r_t+\alpha\,\mathcal H(\pi_\phi(\cdot\mid o_t))\right)\right],$$

using replay sampling, twin critics, target networks with Polyak coefficient 0.005, and automatic adjustment of the entropy coefficient $\alpha$.

**PPO** collects on-policy rollouts of 1000 decisions and performs 10 epochs of minibatch updates (minibatch size 250) on the clipped surrogate objective

$$L^{\mathrm{CLIP}}(\phi)=\mathbb E_t\left[\min\left(\rho_t(\phi)\hat A_t,\ \operatorname{clip}(\rho_t(\phi),1-\epsilon,1+\epsilon)\hat A_t\right)\right],\qquad
\rho_t(\phi)=\frac{\pi_\phi(a_t\mid o_t)}{\pi_{\phi_{\mathrm{old}}}(a_t\mid o_t)},$$

with $\epsilon=0.2$ and advantages $\hat A_t$ from generalized advantage estimation with $\lambda=0.95$. The value-loss coefficient is 0.5, the entropy coefficient is 0, and gradients are clipped to norm 0.5.
