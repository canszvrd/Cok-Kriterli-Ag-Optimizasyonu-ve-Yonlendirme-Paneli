import tkinter as tk
from tkinter import ttk, scrolledtext
import networkx as nx
import time
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from algorithms.ga import GenetikAlgorithm
# Kendi yazdığımız modüller
from network_generator import NetworkManager
from metrics import calculate_path_attributes, calculate_weighted_cost
from algorithms.q_learning import QLearningRouter
from algorithms.aco_optimizer import ACORouter

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class NetworkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ağ Optimizasyon ve Metrik Paneli")
        self.root.geometry("1200x800")

        # NETWORK MANAGER
        self.nm = NetworkManager()
        self.graph = None
        self.demands = None

        self.pos = None

        # ================== ANA PANEL ==================
        main_pane = tk.PanedWindow(
            root, orient=tk.HORIZONTAL,
            sashrelief=tk.RAISED, sashwidth=6
        )
        main_pane.pack(fill="both", expand=True, padx=10, pady=10)

        left_frame = tk.Frame(main_pane, relief=tk.SUNKEN, borderwidth=2)
        right_frame = tk.Frame(main_pane, relief=tk.SUNKEN, borderwidth=2)

        main_pane.add(left_frame)
        main_pane.add(right_frame)
        main_pane.paneconfigure(left_frame, width=400)
        main_pane.paneconfigure(right_frame, stretch="always")

        # ================== SOL TARAF ==================
        tk.Label(
            left_frame, text="Ağ Metrik Simülasyonu",
            font=("Arial", 16, "bold")
        ).pack(pady=10)

        # === GRAF GÖRSELLEŞTİRME ALANI ===
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=False)
        self.canvas.mpl_connect("scroll_event", self.on_zoom)
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_drag)
        self.canvas.mpl_connect("button_release_event", self.on_release)

        self._panning = False

        self.status_label = tk.Label(
            left_frame, text="Durum: Başlatılıyor...", fg="blue"
        )
        self.status_label.pack(pady=5)

        # -------- Algoritma seçimi --------
        algo_frame = tk.LabelFrame(
            left_frame, text="1. Adım: Algoritma Seçimi",
            padx=10, pady=10
        )
        algo_frame.pack(fill="x", padx=10, pady=5)

        self.algo_var = tk.StringVar(value="Q-Learning")
        ttk.Combobox(
            algo_frame,
            textvariable=self.algo_var,
            state="readonly",
            values=( "Q-Learning", "ACO", "Genetik")
        ).pack(fill="x")

        # -------- Ağırlıklar --------
        weight_frame = tk.LabelFrame(
            left_frame, text="2. Adım: Ağırlık Ayarları",
            padx=10, pady=10
        )
        weight_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(weight_frame, text="Gecikme").grid(row=0, column=0, sticky="w")
        self.slider_delay = tk.Scale(weight_frame, from_=0, to=1, resolution=0.1, orient="horizontal")
        self.slider_delay.set(0.4)
        self.slider_delay.grid(row=0, column=1)

        tk.Label(weight_frame, text="Güvenilirlik").grid(row=1, column=0, sticky="w")
        self.slider_rel = tk.Scale(weight_frame, from_=0, to=1, resolution=0.1, orient="horizontal")
        self.slider_rel.set(0.3)
        self.slider_rel.grid(row=1, column=1)

        tk.Label(weight_frame, text="Kaynak").grid(row=2, column=0, sticky="w")
        self.slider_bw = tk.Scale(weight_frame, from_=0, to=1, resolution=0.1, orient="horizontal")
        self.slider_bw.set(0.3)
        self.slider_bw.grid(row=2, column=1)

        # -------- Talep seçimi --------
        demand_frame = tk.LabelFrame(
            left_frame, text="3. Adım: Talep Seçimi",
            padx=10, pady=10
        )
        demand_frame.pack(fill="x", padx=10, pady=5)

        self.demand_mode = tk.StringVar(value="manual")
        tk.Radiobutton(demand_frame, text="Manuel",
                       variable=self.demand_mode, value="manual",
                       command=self.toggle_demand_mode).grid(row=0, column=0, sticky="w")
        tk.Radiobutton(demand_frame, text="Tek Talep",
                       variable=self.demand_mode, value="single",
                       command=self.toggle_demand_mode).grid(row=1, column=0, sticky="w")
        tk.Radiobutton(demand_frame, text="Tüm Talepler",
                       variable=self.demand_mode, value="all",
                       command=self.toggle_demand_mode).grid(row=2, column=0, sticky="w")

        self.entry_src = tk.Entry(demand_frame, width=8)
        self.entry_dst = tk.Entry(demand_frame, width=8)
        self.entry_bw = tk.Entry(demand_frame, width=8)

        self.entry_src.insert(0, "0")
        self.entry_dst.insert(0, "5")
        self.entry_bw.insert(0, "")

        tk.Label(demand_frame, text="Source").grid(row=3, column=1)
        self.entry_src.grid(row=3, column=1)

        tk.Label(demand_frame, text="Target").grid(row=4, column=0)
        self.entry_dst.grid(row=4, column=1)

        tk.Label(demand_frame, text="Bandwidth (opsiyonel)").grid(row=5, column=0)
        self.entry_bw.grid(row=5, column=1)

        self.demand_combo = ttk.Combobox(demand_frame, state="readonly", width=40)
        self.demand_combo.grid(row=6, column=0, columnspan=2)

        tk.Button(
            left_frame, text="HESAPLA VE GÖSTER",
            bg="#4CAF50", fg="white",
            command=self.calculate_score
        ).pack(pady=20)

        # ================== SAĞ TARAF ==================
        tk.Label(
            right_frame, text="Sonuçlar",
            font=("Arial", 14, "bold")
        ).pack(pady=10)

        self.result_text = scrolledtext.ScrolledText(
            right_frame, font=("Consolas", 10)
        )
        self.result_text.pack(fill="both", expand=True)

        self.load_network_data()
        self.toggle_demand_mode()

    # --------------------------------------------------
    def toggle_demand_mode(self):
        if self.demand_mode.get() == "single" and self.demands:
            self.demand_combo["values"] = [
                f"ID {d['id']}: {d['src']} → {d['dst']} (B≥{d['bandwidth_needed']})"
                for d in self.demands
            ]
            self.demand_combo.current(0)

    def load_network_data(self):
        self.graph, self.demands = self.nm.load_from_csv(
            "data/BSM307_317_Guz2025_TermProject_NodeData.csv",
            "data/BSM307_317_Guz2025_TermProject_EdgeData.csv",
            "data/BSM307_317_Guz2025_TermProject_DemandData.csv"
        )
        self.status_label.config(
            text=f"Yüklendi | Node: {len(self.graph.nodes)} | Talep: {len(self.demands)}",
            fg="green"
        )

    # --------------------------------------------------
    def calculate_score(self):
        self.result_text.delete(1.0, tk.END)

        weights = {
            "w_delay": self.slider_delay.get(),
            "w_reliability": self.slider_rel.get(),
            "w_resource": self.slider_bw.get()
        }

        if self.demand_mode.get() == "manual":
            bw_text = self.entry_bw.get().strip()
            demand_bw = int(bw_text) if bw_text.isdigit() else 0
            demands = [{
                "src": int(self.entry_src.get()),
                "dst": int(self.entry_dst.get()),
                "bandwidth_needed": demand_bw
            }]
        elif self.demand_mode.get() == "single":
            demands = [self.demands[self.demand_combo.current()]]
        else:
            demands = self.demands

        for d in demands:
            self.root.update()
            start_time = time.time()

            # --- Q-LEARNING ---
            if self.algo_var.get() == "Q-Learning":
                best_metrics = None
                best_cost = float("inf")

                for _ in range(5):
                    router = QLearningRouter(self.graph, weights)
                    router.train(d["src"], d["dst"], d["bandwidth_needed"])
                    path = router.get_best_path(d["src"], d["dst"])
                    metrics = router.get_path_metrics(path, d["bandwidth_needed"])

                    if metrics and metrics.get("valid", False):
                        if metrics["total_cost"] < best_cost:
                            best_cost = metrics["total_cost"]
                            best_metrics = metrics
                metrics = best_metrics

            # --- ACO (Ant Colony Optimization) ---
            elif self.algo_var.get() == "ACO":
                best_metrics = None
                best_cost = float("inf")

                for _ in range(5):
                    router = ACORouter(self.graph, weights, ant_count=20, iterations=20)
                    router.train(d["src"], d["dst"], d["bandwidth_needed"])
                    path = router.get_best_path(d["src"], d["dst"])
                    if path and len(path) > 1:
                        metrics = router.get_path_metrics(path, d["bandwidth_needed"])

                        if metrics and metrics.get("valid", False):
                            if metrics["total_cost"] < best_cost:
                                best_cost = metrics["total_cost"]
                                best_metrics = metrics

                metrics = best_metrics

            # --- GENETİK ---
            elif self.algo_var.get() == "Genetik":
                best_metrics = None
                best_cost = float("inf")

                ga = GenetikAlgorithm()
                for _ in range(1):
                    path = ga.genetik_calistir(self.graph, d["src"], d["dst"], d["bandwidth_needed"], weights)
                    if path and len(path) > 1:
                        metrics = ga.get_path_metrics(path, d["bandwidth_needed"])

                        if metrics and metrics.get("valid", False):
                            if metrics["total_cost"] < best_cost:
                                best_cost = metrics["total_cost"]
                                best_metrics = metrics

                metrics = best_metrics

            # --- DIJKSTRA ---
            else:
                try:
                    # Dijkstra по умолчанию ищет кратчайший путь (по весу 1, т.е. хопам)
                    path = nx.shortest_path(self.graph, d["src"], d["dst"])
                    metrics = self.calculate_metrics_manual(path, weights)
                except nx.NetworkXNoPath:
                    metrics = None

            runtime = time.time() - start_time

            if metrics and metrics.get("valid", False):
                self.result_text.insert(
                    tk.END,
                    f"🧠 Algoritma: {self.algo_var.get()}\n"
                    f"📌 Talep: {d['src']} → {d['dst']}\n"
                    f"{'-' * 70}\n"
                    f"🛣️  Seçilen Yol:\n   {metrics['path']}\n\n"
                    f"📊 Performans Metrikleri:\n"
                    f"   • Toplam Gecikme: {metrics['total_delay']:.2f}\n"
                    f"   • Toplam Güvenilirlik: {metrics['total_reliability']:.6f}\n"
                    f"   • Kaynak Maliyeti: {metrics['resource_cost']:.4f}\n"
                    f"   • Toplam Maliyet: {metrics['total_cost']:.4f}\n"
                    f"   • Darboğaz Bant Genişliği: {metrics['bottleneck_bw']} Mbps\n\n"
                    f"⏱️  Çalışma Süresi: {runtime:.4f} saniye\n"
                    f"{'=' * 80}\n\n"
                )
                self.draw_graph_with_path(self.graph, metrics["path"])

            else:
                self.result_text.insert(
                    tk.END,
                    f"🧠 Algoritma: {self.algo_var.get()}\n"
                    f"📌 Talep: {d['src']} → {d['dst']}\n"
                    f"❌ Yol bulunamadı (veya kısıtlar sağlanmadı)\n"
                    f"{'=' * 80}\n\n"
                )

    # --------------------------------------------------
    def calculate_metrics_manual(self, path, weights):
        attrs = calculate_path_attributes(self.graph, path)
        cost = calculate_weighted_cost(attrs, weights)

        total_reliability = 1.0
        for i in range(len(path) - 1):
            total_reliability *= self.graph.edges[path[i], path[i + 1]]['reliability']
        for n in path:
            total_reliability *= self.graph.nodes[n]['reliability']

        min_bw = min(
            self.graph.edges[path[i], path[i + 1]]['bandwidth']
            for i in range(len(path) - 1)
        )

        return {
            "valid": True,
            "path": path,
            "total_delay": attrs["total_delay"],
            "total_reliability": total_reliability,
            "resource_cost": attrs["resource_cost"],
            "total_cost": cost,
            "bottleneck_bw": min_bw
        }

    def draw_graph_with_path(self, graph, path=None):
        self.ax.clear()

        if self.pos is None:
            self.pos = nx.spring_layout(graph, seed=42, k=0.6, iterations=50)

        pos = self.pos

        # Node'lar
        nx.draw_networkx_nodes(
            graph,
            pos,
            node_size=25,
            node_color="#8ecae6",
            edgecolors="black",
            linewidths=0.2,
            ax=self.ax
        )

        # Linkler (GRİ)
        nx.draw_networkx_edges(
            graph,
            pos,
            edge_color="gray",
            alpha=0.05,  # <-- ÇOK ÖNEMLİ
            width=0.5,
            ax=self.ax
        )

        # En iyi yol (KIRMIZI)
        if path and len(path) > 1:
            source = path[0]
            destination = path[-1]
            middle_nodes = path[1:-1]

            path_edges = list(zip(path[:-1], path[1:]))

            # Orta yol node'ları (kırmızı)
            nx.draw_networkx_nodes(
                graph, pos,
                nodelist=middle_nodes,
                node_color="red",
                node_size=70,
                ax=self.ax
            )

            # Source (yeşil)
            nx.draw_networkx_nodes(
                graph, pos,
                nodelist=[source],
                node_color="green",
                node_size=120,
                ax=self.ax
            )

            # Destination (mor)
            nx.draw_networkx_nodes(
                graph, pos,
                nodelist=[destination],
                node_color="purple",
                node_size=120,
                ax=self.ax
            )

            # Yol kenarları (kırmızı)
            nx.draw_networkx_edges(
                graph, pos,
                edgelist=path_edges,
                width=3,
                edge_color="red",
                ax=self.ax
            )
            # Label (S ve D)
            nx.draw_networkx_labels(
                graph,
                pos,
                labels={
                    source: f"S ({source})",
                    destination: f"D ({destination})"
                },
                font_size=9,
                font_weight="bold",
                ax=self.ax
            )

        self.ax.set_title("Ağ Grafiği (En İyi Yol Kırmızı)")
        self.ax.axis("off")
        self.canvas.draw()

    def on_zoom(self, event):
        base_scale = 1.2
        ax = self.ax

        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            return

        xdata = event.xdata
        ydata = event.ydata
        if xdata is None or ydata is None:
            return

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        new_width = (xlim[1] - xlim[0]) * scale_factor
        new_height = (ylim[1] - ylim[0]) * scale_factor

        relx = (xlim[1] - xdata) / (xlim[1] - xlim[0])
        rely = (ylim[1] - ydata) / (ylim[1] - ylim[0])

        ax.set_xlim([xdata - new_width * (1 - relx),
                     xdata + new_width * relx])
        ax.set_ylim([ydata - new_height * (1 - rely),
                     ydata + new_height * rely])

        self.canvas.draw_idle()

    def on_press(self, event):
        if event.button == 1 and event.xdata is not None:
            self._panning = True
            self._pan_start = (event.xdata, event.ydata)

    def on_drag(self, event):
        if not self._panning or event.xdata is None or event.ydata is None:
            return

        dx = self._pan_start[0] - event.xdata
        dy = self._pan_start[1] - event.ydata

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        self.ax.set_xlim(xlim[0] + dx, xlim[1] + dx)
        self.ax.set_ylim(ylim[0] + dy, ylim[1] + dy)

        self._pan_start = (event.xdata, event.ydata)
        self.canvas.draw_idle()

    def on_release(self, event):
        self._panning = False


if __name__ == "__main__":
    root = tk.Tk()
    app = NetworkApp(root)
    root.mainloop()