"""
PPO (Proximal Policy Optimization) 纯 PyTorch 手写实现
======================================================
从零实现 PPO-Clip 算法核心逻辑，帮助理解原理。

参考论文: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
         https://arxiv.org/abs/1707.06347

安装依赖:
    pip install torch gymnasium numpy

运行:
    python ppo/ppo_pure_torch.py
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# ============================================================
# 1. 定义 Actor-Critic 网络
# ============================================================
class ActorCritic(nn.Module):
    """Actor-Critic 网络：共享特征层，分别输出策略(Actor)和价值(Critic)"""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64):
        super().__init__()
        # Actor: 策略网络 π(a|s) —— 输出动作的概率分布
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),  # 输出 logits（未归一化概率）
        )
        # Critic: 价值网络 V(s) —— 输出状态价值
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),  # 标量价值
        )

    def get_action_and_value(self, obs: torch.Tensor, action=None):
        """采样动作 + 返回 log_prob、entropy、value"""
        logits = self.actor(obs)
        dist = Categorical(logits=logits)       # Softmax → 离散分布
        action = action if action is not None else dist.sample()
        return (
            action,
            dist.log_prob(action),               # log π(a|s)
            dist.entropy(),                      # 策略熵 H(π)，鼓励探索
            self.critic(obs).squeeze(-1),        # V(s)
        )

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """对已有动作重新评估（用于 PPO 更新时计算 ratio）"""
        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        return (
            dist.log_prob(actions),
            dist.entropy(),
            self.critic(obs).squeeze(-1),
        )


# ============================================================
# 2. Rollout Buffer：存储一段 trajectory 的数据
# ============================================================
class RolloutBuffer:
    """简单的轨迹缓存，收集一个 epoch 的交互数据"""

    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def store(self, obs, action, log_prob, reward, done, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self):
        self.obs, self.actions, self.log_probs = [], [], []
        self.rewards, self.dones, self.values = [], [], []


# ============================================================
# 3. GAE (Generalized Advantage Estimation) 优势估计
# ============================================================
def compute_gae(
    rewards: list,
    values: list,
    dones: list,
    next_value: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple:
    """
    计算 GAE 优势估计和 returns。
    
    公式:
        δ_t = r_t + γ V(s_{t+1}) (1 - d_t) - V(s_t)
        A_t = Σ_{l=0}^{T-t} (γλ)^l δ_{t+l}
    
    GAE 是 TD 误差的指数加权移动平均，λ=0 退化为 TD(0)，λ=1 退化为 Monte Carlo。
    """
    advantages = []
    gae = 0.0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]

        next_non_terminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages.insert(0, gae)

    advantages = np.array(advantages, dtype=np.float32)
    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


# ============================================================
# 4. PPO 核心：策略更新（Clip 版本）
# ============================================================
def ppo_update(
    agent: ActorCritic,
    optimizer: optim.Optimizer,
    obs: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    n_epochs: int = 10,
    batch_size: int = 64,
    clip_eps: float = 0.2,       # PPO-Clip 核心超参数 ε
    vf_coef: float = 0.5,        # 价值函数损失权重
    ent_coef: float = 0.01,      # 熵正则化权重（鼓励探索）
    max_grad_norm: float = 0.5,  # 梯度裁剪阈值
):
    """
    PPO-Clip 策略更新。
    
    核心思想：限制新旧策略的比例 ratio = π_new / π_old，
    使其在 [1-ε, 1+ε] 范围内，防止策略更新幅度过大。
    
    目标函数 (clipped surrogate objective):
        L^CLIP = E[ min( ratio * A, clip(ratio, 1-ε, 1+ε) * A ) ]
    
    总损失:
        L = -L^CLIP + c1 * L^VF + c2 * H
    其中 L^VF 是价值函数的 MSE 损失，H 是熵正则项。
    """
    n_samples = obs.shape[0]
    indices = np.arange(n_samples)

    for epoch in range(n_epochs):
        np.random.shuffle(indices)

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            mb_idx = indices[start:end]

            # 获取当前策略对这批数据的评价
            new_log_probs, entropy, values = agent.evaluate_actions(
                obs[mb_idx], actions[mb_idx]
            )

            # ── PPO 核心：计算 ratio 和 clipped ratio ──
            # ratio = exp(log π_new - log π_old) = π_new / π_old
            ratio = torch.exp(new_log_probs - old_log_probs[mb_idx])

            adv = advantages[mb_idx]
            # 优势归一化（稳定训练）
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # PPO-Clip 目标：取 min 保证策略不会偏离太远
            surr1 = ratio * adv                          # 未裁剪
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv  # 裁剪后
            policy_loss = -torch.min(surr1, surr2).mean()

            # 价值函数损失：MSE(V(s), R_t)
            value_loss = ((values - returns[mb_idx]) ** 2).mean()

            # 熵正则：鼓励探索，防止策略过早收敛
            entropy_loss = entropy.mean()

            # 总损失
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)  # 梯度裁剪
            optimizer.step()

    # 返回最后一次的指标用于日志
    return {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy_loss.item(),
    }


# ============================================================
# 5. 主训练循环
# ============================================================
def main():
    # ── 超参数 ──
    ENV_NAME = "CartPole-v1"
    TOTAL_TIMESTEPS = 50_000
    STEPS_PER_EPOCH = 2048       # 每次更新前收集的步数
    GAMMA = 0.99                 # 折扣因子
    GAE_LAMBDA = 0.95            # GAE λ 参数
    LR = 3e-4                    # Adam 学习率
    N_EPOCHS = 10                # 每次更新的 epoch 数
    BATCH_SIZE = 64              # 小批量大小
    CLIP_EPS = 0.2               # PPO 裁剪范围 ε

    # ── 初始化 ──
    env = gym.make(ENV_NAME)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    agent = ActorCritic(obs_dim, act_dim)
    optimizer = optim.Adam(agent.parameters(), lr=LR)
    buffer = RolloutBuffer()

    print("=" * 60)
    print("PPO 纯 PyTorch 实现 - 从零理解 PPO-Clip")
    print("=" * 60)
    print(f"环境: {ENV_NAME} | 观测维度: {obs_dim} | 动作数: {act_dim}")
    print(f"总步数: {TOTAL_TIMESTEPS} | 每轮步数: {STEPS_PER_EPOCH}")
    print(f"Clip ε: {CLIP_EPS} | GAE λ: {GAE_LAMBDA} | γ: {GAMMA}")
    print("-" * 60)

    # ── 训练循环 ──
    obs, _ = env.reset()
    global_step = 0
    episode_reward = 0
    episode_rewards = []
    update_count = 0

    while global_step < TOTAL_TIMESTEPS:
        # ── 收集数据 ──
        for _ in range(STEPS_PER_EPOCH):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

            with torch.no_grad():
                action, log_prob, _, value = agent.get_action_and_value(obs_tensor)

            action_np = action.item()
            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            buffer.store(
                obs, action_np, log_prob.item(),
                reward, float(done), value.item(),
            )

            global_step += 1
            episode_reward += reward

            if done:
                episode_rewards.append(episode_reward)
                episode_reward = 0
                next_obs, _ = env.reset()

            obs = next_obs

        # ── 计算优势 GAE ──
        with torch.no_grad():
            next_value = agent.get_action_and_value(
                torch.FloatTensor(obs).unsqueeze(0)
            )[3].item()

        advantages, returns = compute_gae(
            buffer.rewards, buffer.values, buffer.dones,
            next_value, GAMMA, GAE_LAMBDA,
        )

        # ── 转换为 Tensor ──
        obs_t = torch.FloatTensor(np.array(buffer.obs))
        actions_t = torch.LongTensor(buffer.actions)
        old_log_probs_t = torch.FloatTensor(buffer.log_probs)
        advantages_t = torch.FloatTensor(advantages)
        returns_t = torch.FloatTensor(returns)

        # ── PPO 策略更新 ──
        stats = ppo_update(
            agent, optimizer,
            obs_t, actions_t, old_log_probs_t,
            advantages_t, returns_t,
            n_epochs=N_EPOCHS, batch_size=BATCH_SIZE,
            clip_eps=CLIP_EPS,
        )

        update_count += 1
        if update_count % 5 == 0:
            recent = episode_rewards[-20:]
            mean_r = np.mean(recent) if recent else 0
            print(
                f"[更新 {update_count:3d}] "
                f"步数: {global_step:>6d} | "
                f"近20轮均奖: {mean_r:>7.1f} | "
                f"策略损失: {stats['policy_loss']:.4f} | "
                f"价值损失: {stats['value_loss']:.4f} | "
                f"熵: {stats['entropy']:.4f}"
            )

        buffer.clear()

    # ── 最终评估 ──
    print("\n" + "=" * 60)
    print("训练完成！最终评估（10 个 episode）：")
    print("=" * 60)

    eval_rewards = []
    for ep in range(10):
        obs, _ = env.reset()
        done = False
        total_r = 0
        while not done:
            with torch.no_grad():
                action = agent.get_action_and_value(
                    torch.FloatTensor(obs).unsqueeze(0)
                )[0].item()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_r += reward
        eval_rewards.append(total_r)
        print(f"  Episode {ep + 1}: 奖励 = {total_r:.0f}")

    print(f"\n平均奖励: {np.mean(eval_rewards):.1f} ± {np.std(eval_rewards):.1f}")

    # ── 保存模型 ──
    torch.save(agent.state_dict(), "ppo/ppo_pure_torch_model.pt")
    print("模型已保存到 ppo/ppo_pure_torch_model.pt")


if __name__ == "__main__":
    main()
