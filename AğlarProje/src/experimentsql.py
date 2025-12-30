"""
src/experiments.py - BSM307 Güz 2025 Dönem Projesi

Bu dosya sadece Q-Learning algoritması için deney düzeneğini uygular (PDF sayfa 7).

- Her talep ve her ağırlık seti için en az 5 bağımsız çalıştırma
- Ortalama, standart sapma, en iyi/en kötü maliyet, çalışma süresi
- Sonuçlar kök dizindeki 'results/' klasörüne CSV olarak kaydedilir
"""

import time
import os
import pandas as pd
from statistics import mean, stdev
from typing import List, Dict

# Proje modüllerini doğru yol ile import et
from src.network_generator import NetworkManager
from src.algorithms.q_learning import QLearningRouter
from src.metrics import calculate_path_attributes, calculate_weighted_cost


def run_qlearning_experiments(
    num_runs: int = 5,
    episodes: int = 5000,
    weight_sets: List[Dict[str, float]] = None,
    output_file: str = "results/qlearning_experiment_results.csv"
) -> pd.DataFrame:
    """
    Q-Learning deneylerini çalıştırır ve sonuçları CSV'ye kaydeder.
    """
    if weight_sets is None:
        weight_sets = [
            {'w_delay': 0.5, 'w_reliability': 0.3, 'w_resource': 0.2},
            {'w_delay': 0.7, 'w_reliability': 0.2, 'w_resource': 0.1},
            {'w_delay': 0.1, 'w_reliability': 0.7, 'w_resource': 0.2},
            {'w_delay': 0.2, 'w_reliability': 0.2, 'w_resource': 0.6},
        ]

    print("Ağ ve talepler yükleniyor...")
    nm = NetworkManager()
    graph, demands = nm.load_from_csv(
        'data/BSM307_317_Guz2025_TermProject_NodeData.csv',
        'data/BSM307_317_Guz2025_TermProject_EdgeData.csv',
        'data/BSM307_317_Guz2025_TermProject_DemandData.csv'
    )

    print(f"{len(demands)} talep bulundu. {len(weight_sets)} ağırlık seti ile {num_runs} tekrar başlıyor...\n")

    results = []

    for weight_idx, weights in enumerate(weight_sets, 1):
        print(f"--- Ağırlık Seti {weight_idx}/{len(weight_sets)}: {weights} ---")

        for demand in demands:
            demand_id = demand['id']
            src = demand['src']
            dst = demand['dst']
            demand_bw = demand['bandwidth_needed']

            print(f"  Talep {demand_id}: {src} → {dst} (B ≥ {demand_bw} Mbps)")

            run_costs = []
            run_times = []
            run_delays = []
            run_rel_costs = []
            run_res_costs = []
            valid_count = 0

            for run in range(1, num_runs + 1):
                start_time = time.time()

                ql = QLearningRouter(
                    graph=graph,
                    weights=weights,
                    alpha=0.1,
                    gamma=0.9,
                    initial_epsilon=1.0,
                    episodes=episodes,
                    max_steps=500
                )

                ql.train(src=src, dst=dst, demand_bw=demand_bw)

                best_path = ql.get_best_path(src=src, dst=dst)

                end_time = time.time()
                runtime = end_time - start_time
                run_times.append(runtime)

                if best_path and len(best_path) > 1:
                    attrs = calculate_path_attributes(graph, best_path)
                    cost = calculate_weighted_cost(attrs, weights)

                    # Darboğaz kontrolü
                    min_bw = min(
                        graph.edges[best_path[i], best_path[i+1]]['bandwidth']
                        for i in range(len(best_path)-1)
                    )

                    if min_bw >= demand_bw:
                        run_costs.append(cost)
                        run_delays.append(attrs['total_delay'])
                        run_rel_costs.append(attrs['reliability_cost'])
                        run_res_costs.append(attrs['resource_cost'])
                        valid_count += 1
                    else:
                        run_costs.append(float('inf'))
                else:
                    run_costs.append(float('inf'))

            # İstatistikler
            valid_costs = [c for c in run_costs if c < float('inf')]

            row = {
                'demand_id': demand_id,
                'src': src,
                'dst': dst,
                'demand_bw': demand_bw,
                'weight_set': weight_idx,
                'weights': str(weights),
                'num_runs': num_runs,
                'valid_runs': valid_count,
                'average_runtime_sec': mean(run_times),
                'total_runtime_sec': sum(run_times)
            }

            if valid_costs:
                row.update({
                    'best_cost': min(valid_costs),
                    'worst_cost': max(valid_costs),
                    'average_cost': mean(valid_costs),
                    'std_dev_cost': stdev(valid_costs) if len(valid_costs) > 1 else 0.0,
                    'average_delay': mean(run_delays),
                    'average_reliability_cost': mean(run_rel_costs),
                    'average_resource_cost': mean(run_res_costs),
                })
            else:
                row.update({
                    'best_cost': None,
                    'worst_cost': None,
                    'average_cost': None,
                    'std_dev_cost': None,
                    'average_delay': None,
                    'average_reliability_cost': None,
                    'average_resource_cost': None,
                })

            results.append(row)

    # Sonuçları kaydet (kök dizindeki results/ klasörüne)
    df_results = pd.DataFrame(results)
    
    # results klasörü yoksa oluştur
    os.makedirs('results', exist_ok=True)
    
    df_results.to_csv(output_file, index=False)
    print(f"\nTamamlandı! Sonuçlar '{output_file}' dosyasına kaydedildi.")
    print(f"Toplam {len(df_results)} satır sonuç.")

    return df_results


if __name__ == "__main__":
    # Test için küçük değerler, final için artır
    df = run_qlearning_experiments(
        num_runs=5,
        episodes=3000,  # Test için 3000, finalde 5000-10000 yap
        output_file="results/qlearning_results.csv"
    )
    print(df.head())