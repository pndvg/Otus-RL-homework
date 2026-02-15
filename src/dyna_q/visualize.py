import gymnasium as gym
import numpy as np
import pickle
import os
import argparse
import time
from src.dyna_q.agent import DynaQAgent


def visualize(model_path, num_episodes=5, delay=0.5):
    if not os.path.exists(model_path):
        print(f"Ошибка: Файл модели не найден по пути {model_path}")
        return

    env = gym.make(
        "FrozenLake-v1", is_slippery=True, map_name="8x8", render_mode="human"
    )

    # Инициализируем агента с фиктивными параметрами (мы загрузим Q-таблицу)
    agent = DynaQAgent(env.action_space)
    agent.load(model_path)
    print(f"Модель загружена из {model_path}")
    print(f"Размер Q-таблицы: {len(agent.q_table)}")

    # Устанавливаем epsilon в 0 для жадной оценки
    agent.epsilon = 0.0

    for i in range(num_episodes):
        print(f"Запуск эпизода {i + 1}")
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        step = 0

        while not (done or truncated):
            env.render()
            action = agent.choose_action(state)
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            step += 1
            time.sleep(delay)

        print(f"Эпизод {i + 1} завершен. Шагов: {step}, Награда: {total_reward}")
        time.sleep(1.0)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Визуализация агента Dyna-Q")
    parser.add_argument(
        "--model",
        type=str,
        default="results/models/dyna_q_n50.pkl",
        help="Путь к файлу модели",
    )
    parser.add_argument(
        "--episodes", type=int, default=3, help="Количество эпизодов для запуска"
    )
    parser.add_argument(
        "--delay", type=float, default=0.1, help="Задержка между шагами"
    )

    args = parser.parse_args()
    visualize(args.model, args.episodes, args.delay)
