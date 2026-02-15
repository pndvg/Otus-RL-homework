import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from src.dyna_q.agent import DynaQAgent


def run_experiment(
    n_planning_steps,
    num_episodes=50,
    alpha=0.1,
    gamma=0.99,
    max_epsilon=1.0,
    min_epsilon=0.01,
    decay_rate=0.0005,
):
    env = gym.make("FrozenLake-v1", is_slippery=True, map_name="8x8")

    # Инициализируем с max_epsilon, но будем управлять затуханием вручную
    agent = DynaQAgent(
        action_space=env.action_space,
        alpha=alpha,
        gamma=gamma,
        epsilon=max_epsilon,
        planning_steps=n_planning_steps,
    )

    rewards_per_episode = []

    pbar = tqdm(range(num_episodes), desc=f"Dyna-Q(n={n_planning_steps})")
    for i in pbar:
        # Затухание epsilon
        agent.epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(
            -decay_rate * i
        )

        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0

        while not (done or truncated):
            action = agent.choose_action(state)
            next_state, reward, done, truncated, _ = env.step(action)

            # Обновление агента
            agent.update(state, action, reward, next_state)

            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)

        if (i + 1) % 500 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            pbar.set_postfix(
                {"ср_нагр_100": f"{avg_reward:.3f}", "eps": f"{agent.epsilon:.3f}"}
            )

    return rewards_per_episode, agent


def main():
    # Параметры
    num_episodes = 10000
    alpha = 0.1
    gamma = 0.99

    # Эксперименты
    planning_steps_list = [0, 5, 50]
    results = {}

    print(
        f"Запуск обучения на FrozenLake-v1 8x8 (Скользкая) на {num_episodes} эпизодов."
    )

    for n in planning_steps_list:
        rewards, agent = run_experiment(n, num_episodes, alpha, gamma)
        results[n] = rewards

        # Вывод финальных метрик
        avg_reward_1000 = np.mean(rewards[-1000:])
        print(
            f"Результат для n={n}: Ср. награда (последние 1000): {avg_reward_1000:.4f}"
        )

        # Сохранение агента
        model_dir = os.path.join("results", "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"dyna_q_n{n}.pkl")
        agent.save(model_path)
        print(f"Модель сохранена в {model_path}")

    # Построение графиков
    plt.figure(figsize=(10, 6))

    for n, rewards in results.items():
        # Сглаживание наград (окно 500 для 10к эпизодов)
        window = 500
        if len(rewards) >= window:
            smoothed_rewards = np.convolve(
                rewards, np.ones(window) / window, mode="valid"
            )
            plt.plot(smoothed_rewards, label=f"n={n}")
        else:
            plt.plot(rewards, label=f"n={n}")

    plt.title(f"Dyna-Q на Stochastic FrozenLake 8x8 ({num_episodes} эп.)")
    plt.xlabel("Эпизод")
    plt.ylabel("Средняя награда (Сглаженная)")
    plt.legend()
    plt.grid(True)

    os.makedirs("results", exist_ok=True)
    output_path = os.path.join("results", "dyna_q_comparison.png")
    plt.savefig(output_path)
    print(f"График сохранен в {output_path}")


if __name__ == "__main__":
    main()
