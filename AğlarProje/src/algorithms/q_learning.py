# -*- coding: utf-8 -*-
"""
QLearningRouter
===============

Bu modül, çok kriterli (QoS-aware) ağ yönlendirme problemi için
Q-Learning tabanlı bir çözüm sunar.

Amaç:
- Ağırlıklı toplam maliyet fonksiyonunu minimize eden
  en uygun yolun öğrenilmesi.

Özellikler:
- Çok amaçlı optimizasyon (Delay, Reliability, Resource)
- Bant genişliği kısıtı (Bandwidth constraint)
- Döngü önleme (Loop avoidance)
- Kararlı ve tekrarlanabilir öğrenme süreci
"""

import random
from typing import List, Optional, Dict, Any
import networkx as nx

from src.metrics import calculate_path_attributes, calculate_weighted_cost


class QLearningRouter:
    """
    Q-Learning tabanlı QoS farkındalıklı yönlendirme algoritması.
    """

    def __init__(
        self,
        graph: nx.Graph,
        weights: Dict[str, float],
        alpha: float = 0.15,
        gamma: float = 0.92,
        initial_epsilon: float = 1.0,
        min_epsilon: float = 0.01,
        epsilon_decay: float = 0.99,
        episodes: int = 4000,
        max_steps: int = 100,
        step_penalty: float = -2.0,
        progress_bonus: float = 0.5
    ):
        # Ağ topolojisi
        self.graph = graph

        # Ağırlıklı Toplam Yöntemi (Weighted Sum Method) ağırlıkları
        self.weights = weights

        # Q-Learning hiperparametreleri
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.episodes = episodes
        self.max_steps = max_steps

        # Ödül fonksiyonu parametreleri
        self.step_penalty = step_penalty
        self.progress_bonus = progress_bonus

        # Q-tablosu: Q[state][action]
        self.Q: Dict[int, Dict[int, float]] = {
            node: {nbr: 0.0 for nbr in graph.neighbors(node)}
            for node in graph.nodes
        }

        # Eğitim sırasında bulunan en iyi yol
        self.best_path: Optional[List[int]] = None
        self.best_cost: float = float("inf")

    # --------------------------------------------------
    # Eğitim (Training)
    # --------------------------------------------------
    def train(self, src: int, dst: int, demand_bw: float = 0) -> None:
        """
        Verilen kaynak (src) ve hedef (dst) düğümleri için
        Q-Learning eğitimi gerçekleştirir.
        """

        for _ in range(self.episodes):
            state = src
            visited = {state}
            path = [state]
            steps = 0
            min_bw_on_path = float("inf")

            while state != dst and steps < self.max_steps:
                steps += 1

                # Döngü oluşturmayan geçerli aksiyonlar
                actions = [a for a in self.Q[state] if a not in visited]
                if not actions:
                    break

                # Epsilon-greedy aksiyon seçimi
                if random.random() < self.epsilon:
                    action = random.choice(actions)
                else:
                    action = max(actions, key=lambda a: self.Q[state][a])

                # Bant genişliği darboğazı takibi
                bw = self.graph.edges[state, action].get("bandwidth", 0)
                min_bw_on_path = min(min_bw_on_path, bw)

                next_state = action
                path.append(next_state)
                visited.add(next_state)

                # Temel adım cezası (kısa yolları teşvik eder)
                reward = self.step_penalty

                # Hedefe doğru ilerleme bonusu (heuristic)
                reward += self.progress_bonus

                # Hedefe ulaşıldığında
                if next_state == dst:
                    attrs = calculate_path_attributes(self.graph, path)
                    cost = calculate_weighted_cost(attrs, self.weights)
                    cost = max(cost, 1e-6)

                    # Bant genişliği kısıtı kontrolü
                    if min_bw_on_path >= demand_bw:
                        reward = 2000.0 / cost
                        if cost < self.best_cost:
                            self.best_cost = cost
                            self.best_path = path.copy()
                    else:
                        reward = -1.0

                # Q-değer güncellemesi
                future_q = max(self.Q[next_state].values(), default=0.0)
                self.Q[state][action] += self.alpha * (
                    reward + self.gamma * future_q - self.Q[state][action]
                )

                state = next_state

            # Keşif oranını azalt
            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.epsilon_decay

    # --------------------------------------------------
    # En İyi Yolun Alınması (GUI Uyumlu)
    # --------------------------------------------------
    def get_best_path(self, src: int = None, dst: int = None) -> Optional[List[int]]:
        """
        Eğitim sonucunda bulunan en iyi yolu döndürür.

        Not:
        - src ve dst parametreleri GUI uyumluluğu içindir.
        - Yol, eğitim sırasında zaten öğrenilmiştir.
        """
        return self.best_path

    # --------------------------------------------------
    # Yol Metriklerinin Hesaplanması
    # --------------------------------------------------
    def get_path_metrics(
        self,
        path: Optional[List[int]],
        demand_bw: float = 0
    ) -> Dict[str, Any]:
        """
        Bulunan yol için QoS metriklerini hesaplar.
        """

        if not path or len(path) < 2:
            return {"valid": False, "error": "Geçerli bir yol bulunamadı"}

        attrs = calculate_path_attributes(self.graph, path)

        # Toplam güvenilirlik (çarpımsal)
        total_reliability = 1.0
        for i in range(len(path) - 1):
            total_reliability *= self.graph.edges[
                path[i], path[i + 1]
            ].get("reliability", 1.0)

        for n in path:
            total_reliability *= self.graph.nodes[n].get(
                "reliability", 1.0
            )

        # Darboğaz bant genişliği
        min_bw = min(
            self.graph.edges[path[i], path[i + 1]].get("bandwidth", float("inf"))
            for i in range(len(path) - 1)
        )

        if min_bw < demand_bw:
            return {"valid": False, "error": "Bant genişliği kısıtı sağlanmadı"}

        return {
            "valid": True,
            "path": path,
            "total_delay": attrs["total_delay"],
            "total_reliability": total_reliability,
            "reliability_cost": attrs["reliability_cost"],
            "resource_cost": attrs["resource_cost"],
            "total_cost": calculate_weighted_cost(attrs, self.weights),
            "bottleneck_bw": min_bw,
            "hop_count": len(path) - 1,
            "node_count": len(path)
        }
    # --------------------------------------------------
    # Çoklu Çalıştırma (Independent Runs)
    # --------------------------------------------------
    def run_multiple(
        self,
        src: int,
        dst: int,
        demand_bw: float = 0,
        runs: int = 5
    ) -> Optional[List[int]]:
        """
        Q-Learning algoritmasını birden fazla bağımsız çalıştırır
        ve en düşük toplam maliyete sahip yolu seçer.

        Her çalıştırma:
        - Q-tablosunu sıfırdan başlatır
        - Epsilon değerini başlangıç seviyesine alır
        - Öğrenme sürecini bağımsız yürütür
        """

        global_best_cost = float("inf")
        global_best_path: Optional[List[int]] = None

        for _ in range(runs):
            # Her çalıştırma için öğrenme durumunu sıfırla
            self.Q = {
                node: {nbr: 0.0 for nbr in self.graph.neighbors(node)}
                for node in self.graph.nodes
            }
            self.best_path = None
            self.best_cost = float("inf")
            self.epsilon = 1.0  # Keşif oranını yeniden başlat

            # Eğitimi çalıştır
            self.train(src, dst, demand_bw)

            # Bu çalıştırmanın sonucu
            if self.best_path is not None:
                attrs = calculate_path_attributes(self.graph, self.best_path)
                cost = calculate_weighted_cost(attrs, self.weights)

                if cost < global_best_cost:
                    global_best_cost = cost
                    global_best_path = self.best_path.copy()

        # Tüm çalıştırmalar arasındaki en iyi yol
        self.best_path = global_best_path
        self.best_cost = global_best_cost

        return global_best_path

