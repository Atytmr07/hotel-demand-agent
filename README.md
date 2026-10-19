# Otel Talep Tahmini ve Kapasite/Fiyat Planlama Ajanı (IE 4023)

Süeda Yurdakonar / Emre Atay Tümer. Proje önerisi: otel doluluk talebini tahmin eden, modeli veriye göre seçen ve LP ile kapasite/fiyat planı öneren çok ajanlı sistem.

## Veri
Antonio, N., De Almeida, A., Nunes, L. (2019). "Hotel booking demand datasets." Data in Brief, 22, 41–49. https://doi.org/10.1016/j.dib.2018.11.126

## Mimari karar
Hesaplama Python'da, orkestrasyon n8n + Gemini'de. Ajanların araçları (`data`, `forecast`, `optimize`) saf Python fonksiyonları olarak yazılır; n8n'e daha sonra bir FastAPI katmanıyla HTTP uç noktası olarak açılır. Böylece çekirdek n8n'siz test edilebilir.
Model seçimi LLM'e bırakılmaz: validasyon MAPE/RMSE'ye göre kod seçer, LLM sonucu yorumlar ve yönlendirir.

## Klasörler
- `src/hotel_agent/` çekirdek modüller (`data.py` hazır)
- `data/raw/hotel_bookings.csv` Antonio vd. (2019), tidytuesday kopyası, 119.390 satır
- `tests/` pytest, `docs/` notlar, `reports/` çıktılar

## Kurulum
```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pytest
```

## Yol haritası
1. [x] Veri ajanı: rezervasyon, günlük dolu oda serisi
2. [x] Seri uçlarını kırp (15 Tem 2015 – 31 Ağu 2017, 779 gün)
3. [x] Tahmin ajanı (`forecast.py`): seasonal-naive, ARIMA, SARIMA, RF, XGBoost; 28 gün ufuk, 4 katlı ileri-yönlü validasyon, MAPE ile seçim.
   Sonuç (net doluluk): Resort → RandomForest (MAPE 2.29), City → XGBoost (3.32). İlk SARIMA(1,0,1)x(1,1,1,7) zayıftı (5.8 / 11.0); airline tipi SARIMA(1,1,1)x(0,1,1,7) ile 3.8 / 4.0'a indi. Seçim 4 aday arasından aynı katlarla yapıldı, hafif seçim yanlılığı var.
4. [x] Optimizasyon ajanı (`optimize.py`): günlük overbooking limiti MILP (PuLP/CBC); talep hatası ve gelme oranı senaryolu, min() ikili değişkenle tam modellendi. Günler bağımsız olduğundan her gün ayrı çözülür; sonuç kaba kuvvetle (tüm limitler denenerek) doğrulandı (`tests/test_optimize.py`).
   Not: ilk sürümde 28 günlük tek model 60 sn zaman sınırına takılıp tam optimal çözülmüyordu; önceki rakamlar bu yüzden geçersiz, aşağıdakiler düzeltilmiş çözücüyle.
5. [x] Geriye dönük kıyas (`backtest.py`, son 112 gün, `python -m hotel_agent.backtest`). Gerçekleşen gelir, overbooking'siz politikaya göre:
   | | Resort | City |
   |---|---|---|
   | overbooking yok | 1,17 M (taban) | 1,55 M (taban) |
   | sabit %10 | +9,8% | +9,9% |
   | kural: kapasite / ort. gelme oranı | **+37,5%** | **+60,4%** |
   | MILP | +32,4% | +57,6% |

   **Önemli:** MILP, gelme oranı kuralını varsayılan varsayımlarda geçemedi (Resort -3,7%, City -1,7%). Yalnızca daha az walk üretiyor (City 5 vs 13). Kazanç büyüklüğü, veri setindeki ~%40 iptal oranından ve varsayımlardan (kapasite = net doluluğun %90'lık dilimi, walk maliyeti = 1,5 × ADR, talebin kapasiteden bağımsız gözlendiği) kaynaklanıyor; mutlak yüzdeler gerçek otel için yorumlanmamalı.
   **Duyarlılık (`reports/sensitivity.csv`, `python -m hotel_agent.sensitivity`)**: MILP'in geliri, gelme oranı kuralına göre (%), kapasite dilimi 0,9:
   | walk maliyeti (x ADR) | 0,5 | 1 | 1,5 | 3 | 5 | 10 |
   |---|---|---|---|---|---|---|
   | Resort | +3,3 | -1,2 | -3,7 | -8,2 | -12,0 | -14,3 |
   | City | +2,6 | -0,1 | -1,7 | -6,6 | -9,7 | -14,4 |

   Yorum: MILP yalnızca walk maliyeti düşükken (~0,5 × ADR) kuralı geçiyor, ~1 × ADR'de başa baş, üstünde kaybediyor. Gerçekleşen walk kuralda bile çok az (Resort 0, City 13), yani senaryolar riski gerçekte olduğundan büyük gösteriyor olabilir (test edilmedi, hipotez). Kapasite dilimi sonucu az etkiliyor.
   Sıradaki: senaryo üretiminin kalibrasyonu (örn. conformal aralıklar), gerçekçi walk maliyeti için kaynak.
6. [ ] Raporlama/sohbet ajanı: saf Python orkestratör + Gemini fonksiyon çağırma (n8n şart değil), yerel model/kural tabanlı yedek, Streamlit arayüz
7. [ ] Rapor ve sunum

## Açık noktalar
- Gelir kıyası için `adr` (günlük oda fiyatı) ve iptal bilgisini nasıl kullanacağımıza karar vermeli.
- Kaynakların (OptiMUS, NOVA tezi vb.) rapora konmadan önce doğrulanması.
