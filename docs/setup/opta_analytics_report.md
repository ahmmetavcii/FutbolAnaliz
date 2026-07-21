# Opta-benzeri otomatik video analizi — durum raporu

Ürün adı: **Opta-benzeri otomatik video analizi** (resmî Opta verisi değildir).

## Production pipeline

`configs/pipeline/opta_analytics.yaml`

Sıra: ingest → shot_classification → detection → tracking → track_quality → reid → team_identity → camera_motion → calibration → ball_state → possession → metrics → analytics_render → **ball_tracking → touch_inference → action_inference → opta_analytics**

## Runtime’dan çıkarılanlar

- sn-trackeval (yalnızca evaluation/adapter testleri)
- sn-echoes (yalnızca dataset reader / lisanslı offline kullanım)
- Tam TrackLab runtime ve kullanılmayan SoccerNet dataset reader’ları production stage listesinde yok

## Smoke çıktıları

- Iniesta: `/home/ahmet/workspace/opta_analytics_smoke/run_20260720_154652_37f681`
- Football: `/home/ahmet/workspace/opta_analytics_smoke/run_20260720_154807_15747b`

Kısa klipler doğruluk kanıtı değildir; altyapı/smoke testi olarak değerlendirilir.

Gerçek doğruluk için gerekli: etiketli pas / dribbling / tackle-duel / aerial klipleri + 5–10 dk oyun bölümü.
