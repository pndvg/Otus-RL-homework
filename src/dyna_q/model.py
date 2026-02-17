import random
from collections import defaultdict


class StochasticEnvModel:
    """
    Выученная модель стохастической среды.
    Хранит счетчики переходов: T(s, a, s') -> count
    и суммы наград: R(s, a, s') -> sum_r

    Это позволяет нам сэмплировать следующее состояние и награду из оценочного распределения.
    """

    def __init__(self):
        # transitions[state][action][next_state] = count
        # переходы[состояние][действие][след_состояние] = количество
        self.transitions = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        # rewards[state][action][next_state] = sum_rewards
        # награды[состояние][действие][след_состояние] = сумма_наград
        self.rewards = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        # отслеживаем посещенные пары состояние-действие для случайного сэмплирования
        self.seen_state_actions = set()
        self.seen_state_actions_list = []

    def add(self, state, action, next_state, reward):
        """
        Обновляет модель новым кортежем опыта (s, a, s', r).
        """
        self.transitions[state][action][next_state] += 1
        self.rewards[state][action][next_state] += reward

        if (state, action) not in self.seen_state_actions:
            self.seen_state_actions.add((state, action))
            self.seen_state_actions_list.append((state, action))

    def sample(self, state, action):
        """
        Сэмплирует следующее состояние и награду на основе выученных вероятностей.
        """
        if state not in self.transitions or action not in self.transitions[state]:
            return None, 0.0

        next_states_counts = self.transitions[state][action]
        # total_count = sum(next_states_counts.values())

        # Сэмплируем следующее состояние на основе счетчиков (взвешенный случайный выбор)
        states = list(next_states_counts.keys())
        counts = list(next_states_counts.values())

        next_state = random.choices(states, weights=counts, k=1)[0]

        # Оцениваем ожидаемую награду для этого перехода (s, a, s')
        # Мы возвращаем среднюю награду, наблюдаемую для этого конкретного перехода.
        count = next_states_counts[next_state]
        reward_sum = self.rewards[state][action][next_state]
        avg_reward = reward_sum / count

        return next_state, avg_reward

    def get_random_previously_seen_state_action(self):
        """
        Возвращает случайную пару (состояние, действие), которая была замечена хотя бы раз.
        Используется для шага планирования в Dyna-Q.
        """
        if not self.seen_state_actions_list:
            return None, None
        return random.choice(self.seen_state_actions_list)
