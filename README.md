Ağ Optimizasyonu ve Performans Analizi (GA - ACO - Q-Learning)
Bu proje, karmaşık ağ yapıları üzerinde farklı optimizasyon algoritmalarının performansını test etmek ve ağ verimliliğini artırmak amacıyla geliştirilmiştir. Proje kapsamında Genetik Algoritma (GA), Karınca Kolonisi Optimizasyonu (ACO) ve Q-Learning algoritmaları kullanılarak en uygun ağ çözümleri aranmaktadır.

📋 Proje Özeti
Algoritmalar: Genetik Algoritma, Karınca Kolonisi (ACO) ve Q-Learning entegrasyonu.

Veri Yönetimi: Ağ düğümleri (nodes), bağlantıları (edges) ve talepler (demands) harici veri dosyalarından dinamik olarak yüklenir.

Analiz: Belirlenen ağ metrikleri üzerinden algoritmaların başarı oranları karşılaştırılır.

🚀 Çalıştırma Adımları
Projeyi kendi ortamınızda çalıştırmak için aşağıdaki adımları sırasıyla uygulayınız:

1. Gereksinimleri Yükleyin Terminali açarak proje dizinine gidin ve gerekli kütüphaneleri yükleyin:

Bash

pip install -r requirements.txt
2. Projeyi Başlatın Ana giriş dosyasını çalıştırarak algoritma süreçlerini ve GUI (varsa) arayüzünü başlatın:

Bash

python main.py
3. Seed Bilgisi (Tekrarlanabilirlik) Algoritmaların her çalıştırıldığında aynı tutarlı sonuçları üretmesi için kod içerisinde sabit bir seed değeri kullanılmıştır:

Seed Değeri: 42

Bu değer, rastlantısal süreçlerin (mutasyon, yol seçimi vb.) akademik olarak doğrulanabilir ve yeniden üretilebilir olmasını sağlar.

📂 Dosya Yapısı Hakkında
src/algorithms/: Algoritmaların temel mantığını içeren dosyalar.

data/: Ağ topolojisini oluşturan Excel ve pickle verileri.

main.py: Projeyi ayağa kaldıran ana kontrol mekanizması.
