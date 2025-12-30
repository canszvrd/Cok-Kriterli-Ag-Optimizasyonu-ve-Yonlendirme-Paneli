import random
import math
import networkx as nx
from typing import List, Dict, Any, Optional

# Metrik fonksiyonlarını içe aktar
from src.metrics import calculate_path_attributes, calculate_weighted_cost

class ACORouter:
    """
    Ant Colony Optimization (Karınca Kolonisi Algoritması).
    
    Ağ yönlendirme için en doğal ve güçlü algoritmalardan biridir.
    Mantık:
    - Sanal karıncalar ağ üzerinde gezinir.
    - İyi yollar üzerine "feromon" (koku) bırakırlar.
    - Sonraki karıncalar, feromonu yüksek olan yolları seçmeye meyillidir.
    - Bant genişliği (Bandwidth) kısıtına uymayan yollara girilmez.
    """

    def __init__(
        self,
        graph: nx.Graph,
        weights: Dict[str, float],
        # Hiperparametreler
        ant_count: int = 20,     # Her iterasyondaki karınca sayısı
        iterations: int = 30,    # Kaç tur döneceği
        alpha: float = 1.0,      # Feromonun önemi (Tarihçe)
        beta: float = 2.0,       # Sezgisel bilginin önemi (Maliyet/Uzaklık)
        evaporation: float = 0.5,# Buharlaşma oranı (Eski bilgiyi unutma hızı)
        q: float = 100.0         # Bırakılacak feromon miktarı sabiti
    ):
        self.graph = graph
        self.weights = weights
        self.ant_count = ant_count
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.q = q
        
        self.nodes = list(graph.nodes())
        
        # En iyi sonuçları saklamak için
        self.best_path: Optional[List[int]] = None
        self.best_cost: float = float("inf")

        # Feromon Haritası: {(u, v): feromon_miktarı}
        # Başlangıçta tüm kenarlarda az miktarda feromon var
        self.pheromones = {}
        self._initialize_pheromones()

    def _initialize_pheromones(self):
        """Tüm kenarlara başlangıç feromonu ekler."""
        for u, v in self.graph.edges():
            self.pheromones[(u, v)] = 1.0
            self.pheromones[(v, u)] = 1.0 # Yönsüz graf olduğu için tersi de aynı

    # --------------------------------------------------
    # 1. EĞİTİM (Training) - Standart Yapı
    # --------------------------------------------------
    def train(self, src: int, dst: int, demand_bw: float = 0) -> None:
        """
        ACO algoritmasını çalıştırır.
        """
        # Her yeni talepte en iyiyi sıfırla, feromonları taze tut ama silme (isteğe bağlı)
        # self._initialize_pheromones() # Eğer her talep bağımsız olsun istersen aç
        
        self.best_path = None
        self.best_cost = float("inf")

        for _ in range(self.iterations):
            all_paths = [] # Bu turda bulunan tüm geçerli yollar
            
            # --- Karıncaları Gönder ---
            for _ in range(self.ant_count):
                path = self._construct_path(src, dst, demand_bw)
                
                if path:
                    # Yol maliyetini hesapla
                    cost = self._calculate_path_cost(path)
                    all_paths.append((path, cost))
                    
                    # Global en iyiyi güncelle
                    if cost < self.best_cost:
                        self.best_cost = cost
                        self.best_path = path[:]

            # --- Feromon Güncelleme (Evaporation + Deposit) ---
            self._update_pheromones(all_paths)

    # --------------------------------------------------
    # 2. EN İYİ YOL (Get Best Path)
    # --------------------------------------------------
    def get_best_path(self, src: int = None, dst: int = None) -> Optional[List[int]]:
        return self.best_path

    # --------------------------------------------------
    # 3. METRİKLER (Metrics)
    # --------------------------------------------------
    def get_path_metrics(self, path: Optional[List[int]], demand_bw: float = 0) -> Dict[str, Any]:
        """Yolun detaylı metriklerini döndürür (GUI için)."""
        if not path or len(path) < 2:
            return {"valid": False, "error": "Yol bulunamadı"}

        # Darboğaz bant genişliği
        min_bw = float('inf')
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            bw = self.graph.edges[u, v].get('bandwidth', 0)
            if bw < min_bw: min_bw = bw

        if min_bw < demand_bw:
            return {"valid": False, "error": "Bant genişliği yetersiz"}

        attrs = calculate_path_attributes(self.graph, path)
        cost = calculate_weighted_cost(attrs, self.weights)
        
        # Güvenilirlik hesabı
        total_reliability = 1.0
        for i in range(len(path) - 1):
            total_reliability *= self.graph.edges[path[i], path[i+1]].get('reliability', 1.0)
        for n in path:
            total_reliability *= self.graph.nodes[n].get('reliability', 1.0)

        return {
            "valid": True,
            "path": path,
            "total_delay": attrs["total_delay"],
            "total_reliability": total_reliability,
            "resource_cost": attrs["resource_cost"],
            "total_cost": cost,
            "bottleneck_bw": min_bw,
            "hop_count": len(path) - 1
        }

    # --------------------------------------------------
    # YARDIMCI: Bir Karıncanın Yol İnşa Etmesi
    # --------------------------------------------------
    def _construct_path(self, src, dst, demand_bw):
        """
        Bir karınca src'den başlar ve olasılıksal olarak dst'ye gitmeye çalışır.
        """
        current = src
        path = [current]
        visited = {current}
        
        # Sonsuz döngü koruması (maksimum adım sayısı)
        for _ in range(len(self.nodes) * 2):
            if current == dst:
                return path
            
            neighbors = list(self.graph.neighbors(current))
            
            # 1. Filtreleme: Ziyaret edilmemiş VE Bant genişliği yeten komşular
            valid_neighbors = []
            for n in neighbors:
                if n not in visited:
                    bw = self.graph.edges[current, n].get('bandwidth', 0)
                    if bw >= demand_bw:
                        valid_neighbors.append(n)
            
            if not valid_neighbors:
                return None # Çıkmaz sokak (Dead end), karınca öldü
            
            # 2. Olasılık Hesabı (Rulet Tekerleği)
            probabilities = []
            denominator = 0.0
            
            for n in valid_neighbors:
                # Tau (Feromon): Geçmiş tecrübe
                tau = self.pheromones.get((current, n), 1.0)
                
                # Eta (Sezgisel): Maliyetin tersi (Daha düşük gecikme = Daha yüksek çekicilik)
                # Basitlik için sadece gecikmeyi kullanabiliriz veya 1.0 diyebiliriz.
                # Burada kenarın 'delay' değerini kullanıyoruz.
                edge_delay = self.graph.edges[current, n].get('delay', 1.0)
                eta = 1.0 / (edge_delay + 0.1) # 0'a bölme hatası olmasın diye +0.1
                
                prob = (tau ** self.alpha) * (eta ** self.beta)
                probabilities.append(prob)
                denominator += prob
            
            if denominator == 0:
                next_node = random.choice(valid_neighbors)
            else:
                # Rulet seçimi
                probabilities = [p / denominator for p in probabilities]
                next_node = random.choices(valid_neighbors, weights=probabilities, k=1)[0]
            
            current = next_node
            path.append(current)
            visited.add(current)
            
        return None # Hedefe ulaşamadı

    # --------------------------------------------------
    # YARDIMCI: Maliyet Hesabı
    # --------------------------------------------------
    def _calculate_path_cost(self, path):
        attrs = calculate_path_attributes(self.graph, path)
        return calculate_weighted_cost(attrs, self.weights)

    # --------------------------------------------------
    # YARDIMCI: Feromon Güncelleme
    # --------------------------------------------------
    def _update_pheromones(self, all_paths):
        """
        1. Buharlaşma (Evaporation)
        2. Yeni feromon ekleme (Deposit)
        """
        # 1. Buharlaşma: Mevcut tüm feromonları azalt
        for edge in self.pheromones:
            self.pheromones[edge] *= (1.0 - self.evaporation)
            # Alt sınır koyalım ki feromon tamamen sıfırlanmasın (keşif devam etsin)
            if self.pheromones[edge] < 0.01:
                self.pheromones[edge] = 0.01

        # 2. Ekleme: Bu turdaki karıncaların geçtiği yolları ödüllendir
        for path, cost in all_paths:
            # Maliyet ne kadar düşükse, ödül o kadar büyük olsun
            deposit = self.q / (cost + 0.0001)
            
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                # Yönsüz graf olduğu için iki yöne de ekle
                if (u, v) in self.pheromones:
                    self.pheromones[(u, v)] += deposit
                if (v, u) in self.pheromones:
                    self.pheromones[(v, u)] += deposit