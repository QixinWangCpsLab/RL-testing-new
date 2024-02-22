### Create Experiment Environment

Create a new conda environment before running testing_script

```
conda create -n "SB3Testing" python=3.10
conda activate SB3Testing

cd ...

pip install -e .
```


### Bug Library
#### stable_baseline3/dqn/dqn.py
在提供的 `dqn.py` 文件中，我们可以引入几个不容易被察觉的错误。请注意，这些错误通常不是好的编程实践，而是为了完成您的特定要求。下面是一些可能的错误：

0. **初始化方法中的默认参数错误**：
   将 `tau` 的默认值设置为超出其应有范围的数值，例如 `2.0`，这可能会导致目标网络的更新不正确。

    ```python
    tau: float = 1.0,
    ->
    tau: float = 2.0,  # should be within 0 and 1, buggy
    ```

1. **梯度裁剪的遗漏**：
   在 `train` 方法中，故意省略 `th.nn.utils.clip_grad_norm_()` 调用，这可能会导致梯度爆炸问题但并不容易被注意到。

    ```python
    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
    ->
    # th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)  # buggy
    ```

2. **错误的批量归一化统计更新**：
   在 `_on_step` 方法中，将批量归一化统计信息的更新权重从 `1.0` 改为 `tau`，这是不正确的，因为批量归一化的统计信息应该完全复制，而不是通过 Polyak 更新。

    ```python
    polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
    ->
    polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, self.tau)  # 错误的方法
    ```

3. **错误的环境步骤计数**：
   在 `_on_step` 方法中，错误地计算环境步数。例如，如果 `target_update_interval` 是一个小于 `n_envs` 的数值，将永远不会执行 `polyak_update`。

    ```python
    if self._n_calls % max(self.target_update_interval // self.n_envs, 1) == 0:
    
    ->
   
    if self._n_calls % max(0 // self.n_envs, 1) == 0:
    ```

4. **探索率调度错误**：
   在探索率调度的定义中，错误地交换 `exploration_initial_eps` 和 `exploration_final_eps` 的值，导致探索率的变化与预期相反。

    ```python
    self.exploration_schedule = get_linear_fn(
        self.exploration_final_eps,  # 应该是 self.exploration_initial_eps
        self.exploration_initial_eps,  # 应该是 self.exploration_final_eps
        self.exploration_fraction,
    )
   
    self.exploration_initial_eps,
    self.exploration_final_eps, #
    ->
    self.exploration_final_eps,
    self.exploration_initial_eps,
    ```




### Gym
在 OpenAI Gym 的 `FrozenLake` 环境中，观测（observations）和动作（actions）是以整数形式编码的。

#### 观测（Observations）

`FrozenLake` 环境是一个基于网格的世界，其中每个格子代表一个冰面（F）、一个洞（H）、起点（S）或目标点（G）。观测值是一个整数，表示代理（agent）当前所在格子的线性索引。如果环境的网格是 `4x4` 的布局，那么观测值会在 `0` 到 `15` 之间，如下所示：

```
SFFF       (S: starting point, safe)
FHFH       (F: frozen surface, safe)
FFFH       (H: hole, fall to your doom)
HFFG       (G: goal, where the frisbee is located)
```

在这个例子中，索引按照从左到右、从上到下的顺序分配：

```
0  1  2  3
4  5  6  7
8  9  10 11
12 13 14 15
```

### 动作（Actions）

动作是以整数形式编码的，代表代理可以采取的方向移动。在 `FrozenLake` 环境中，通常有四种动作：

- `0`: 向左移动
- `1`: 向下移动
- `2`: 向右移动
- `3`: 向上移动

你可以通过环境的 `action_space` 属性来查看动作的范围：

```python
import gym
env = gym.make('FrozenLake-v0')
print(env.action_space)
```

默认情况下，`FrozenLake-v0` 环境是确定性的，意味着执行特定动作会导致预期的状态转移。然而，在 `FrozenLake-v0` 的随机版本（通过将 `is_slippery` 参数设置为 `True` 来创建），执行动作可能会导致非预期的滑动，这意味着代理可能不会按照预期方向移动。

在编写针对 `FrozenLake` 环境的强化学习算法时，你需要基于这些整数值来处理观测和动作。