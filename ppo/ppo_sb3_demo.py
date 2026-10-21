"""
PPO (Proximal Policy Optimization) 最简示例
==========================================
使用 Stable-Baselines3 官方框架，在 CartPole 环境上训练 PPO 智能体。

安装依赖:
    pip install stable-baselines3 gymnasium

运行:
    python ppo/ppo_sb3_demo.py
"""

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy


def main():
    # ── 1. 创建环境（CartPole: 倒立摆，经典控制任务）──
    env = make_vec_env("CartPole-v1", n_envs=4)

    # ── 2. 创建 PPO 模型 ──
    # MlpPolicy: 使用全连接网络处理观测
    # verbose=1: 打印训练日志
    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=2048,         # 每次更新前收集的步数
        batch_size=64,        # 小批量大小
        n_epochs=10,          # 每次更新的 epoch 数
        gamma=0.99,           # 折扣因子
        learning_rate=3e-4,   # 学习率
        clip_range=0.2,       # PPO 裁剪范围 ε
        verbose=1,
    )

    # ── 3. 训练 ──
    print("=" * 50)
    print("开始训练 PPO ...")
    print("=" * 50)
    model.learn(total_timesteps=50_000)

    # ── 4. 保存模型 ──
    model.save("ppo/ppo_cartpole_model")
    print("\n模型已保存到 ppo/ppo_cartpole_model")

    # ── 5. 评估 ──
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"\n评估结果: 平均奖励 = {mean_reward:.1f} ± {std_reward:.1f}")

    # ── 6. 可视化（单次运行展示）──
    env_single = env.envs[0]
    obs, info = env_single.reset()
    total_reward = 0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_single.step(action)
        done = terminated or truncated
        total_reward += reward

    print(f"单次运行总奖励: {total_reward}")


if __name__ == "__main__":
    main()
