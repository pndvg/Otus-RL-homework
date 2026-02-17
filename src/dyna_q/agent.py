import numpy as np
import random
import pickle
from src.dyna_q.model import StochasticEnvModel


class DynaQAgent:
    def __init__(
        self, action_space, alpha=0.1, gamma=0.95, epsilon=0.1, planning_steps=5
    ):
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.planning_steps = planning_steps

        # Q-таблица: словарь, отображающий состояние в значения действий
        # Мы не знаем размер пространства состояний заранее, поэтому используем defaultdict (реализовано через dict с проверкой)
        self.q_table = {}
        self.model = StochasticEnvModel()

    def get_q(self, state, action):
        state = int(state)
        action = int(action)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space.n)
        return self.q_table[state][action]

    def set_q(self, state, action, value):
        state = int(state)
        action = int(action)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space.n)
        self.q_table[state][action] = value

    def choose_action(self, state):
        state = int(state)
        if random.random() < self.epsilon:
            return self.action_space.sample()

        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space.n)

        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        state = int(state)
        action = int(action)
        next_state = int(next_state)

        # 1. Прямое обучение (Q-Learning) на реальном опыте
        max_next_q = np.max(self.q_table.get(next_state, np.zeros(self.action_space.n)))
        current_q = self.get_q(state, action)

        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.set_q(state, action, new_q)

        # 2. Обучение модели
        self.model.add(state, action, next_state, reward)

        # 3. Планирование (имитация опыта)
        self.plan()

    def plan(self):
        for _ in range(self.planning_steps):
            # Сэмплируем ранее наблюдаемое состояние и действие
            s_sim, a_sim = self.model.get_random_previously_seen_state_action()
            if s_sim is None:
                break

            # Симулируем следующее состояние и награду используя модель
            next_s_sim, r_sim = self.model.sample(s_sim, a_sim)

            if next_s_sim is not None:
                # Обновляем Q-значение используя симулированный опыт
                max_next_q_sim = np.max(
                    self.q_table.get(next_s_sim, np.zeros(self.action_space.n))
                )
                current_q_sim = self.get_q(s_sim, a_sim)
                new_q_sim = current_q_sim + self.alpha * (
                    r_sim + self.gamma * max_next_q_sim - current_q_sim
                )
                self.set_q(s_sim, a_sim, new_q_sim)

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self, filepath):
        with open(filepath, "rb") as f:
            self.q_table = pickle.load(f)
