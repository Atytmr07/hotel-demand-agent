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
   Sonuç: Resort → RandomForest (MAPE 2.29), City → XGBoost (3.32). SARIMA(1,0,1)x(1,1,1,7) beklenenden kötü çıktı, parametreleri gözden geçirilmeli.
4. [ ] Optimizasyon ajanı: tek problemle başla (overbooking seviyesi, PuLP). Not: veri setinde kapasite yok, varsayım olarak belgelenmeli
5. [ ] Değerlendirme: sabit tek modeller ve kural tabanlı politika ile kıyas (gelir, geçmiş veride simülasyon)
6. [ ] Raporlama/sohbet ajanı (Gemini) ve n8n akışı
7. [ ] Rapor ve sunum

## Açık noktalar
- Gelir kıyası için `adr` (günlük oda fiyatı) ve iptal bilgisini nasıl kullanacağımıza karar vermeli.
- Kaynakların (OptiMUS, NOVA tezi vb.) rapora konmadan önce doğrulanması.
