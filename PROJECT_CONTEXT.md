# PROJECT_CONTEXT — FutbolAnaliz

Bu belge, repoya ilk kez bakan bir geliştiricinin veya ajanın **ne inşa
edildiğini, neden böyle tasarlandığını, nerede durduğunu ve neyin iddia
edilmediğini** hızlıca anlaması için yazılmıştır.

Kısa kullanım kılavuzu için: [README.md](./README.md)

---

## 1. Ürün özeti

**FutbolAnaliz** (`football-analytics`), yayın / antrenman videosundan:

1. oyuncu ve top tespiti + takibi,
2. saha düzlemi kalibrasyonu ve metre cinsinden hareket metrikleri,
3. takım / rol / (kısmi) jersey kimliği,
4. possession ve top durumu,
5. full-match chunk orchestration + export / panel,
6. Opta-**benzeri** otomatik aksiyon çıkarımı (pas, touch, vb. adayları)

üreten, **sözleşme (contract) öncelikli** bir Python pipeline’ıdır.

Önemli: Bu sistem resmi Opta verisi değildir. “Opta-benzeri” ifadesi ürün
hedefini (event / aksiyon timeline’ı) anlatır; lisanslı Opta feed’i yoktur.

---

## 2. Tasarım ilkeleri

### 2.1 Contract-first

- Canonical tabular çıktılar PyArrow şemalarıyla tanımlıdır:
  `src/football_analytics/contracts/schemas.py`
- Ortak alanlar: `schema_version`, `run_id`, `match_id`, `frame_id`,
  `timestamp_ms`, `source_method`, `confidence`, `valid`
- Bilinmeyen / güvenilmeyen değerler **nullable** veya `valid=false` bırakılır
- Stage’ler girdi/çıktıyı doğrular; checksum + config-hash manifesto yazar

### 2.2 Streaming ve resume

- Video stage’leri frame’leri stream eder; tüm maçı RAM’e yüklemez
- Chunk üst sınırı tipik olarak 300 saniye (config ile)
- `PipelineRunner` (`orchestration/runner.py`) stage listesini yürütür
- `--resume-run-dir`: tamamlanmış stage artifact checksum’ları geçerliyse skip
- `--rerun-from <stage>`: o stage’den itibaren zorla yeniden hesaplar

### 2.3 Conservative analytics (uydurma yok)

- Kalibrasyon yoksa fiziksel hız/mesafe iddia edilmez
- Top recall düşükse ball state null / kısa tahmin penceresi sonra kesilir
- Possession state machine’dir; ground-truth değildir
- Jersey / rol / event eşikleri altında kalanlar `unresolved` / boş kalır
- Smoke clip sonuçları **doğruluk kanıtı** olarak sunulmaz

### 2.4 Clean-room + lisans izolasyonu

- Referans fikirler yeniden implemente edilir; upstream kaynak dosya
  kopyalanmaz (`docs/research/reference_football_analysis_audit.md`)
- GPL bileşenler (PnLCalib, sn-gamestate, …) **out-of-process / izole env**
- MIT adapter’lar ve canonical parquet bu repoda yaşar
- Commit pin’leri: `external_repos.lock.yaml`
- Dağıtım politikası: `THIRD_PARTY_NOTICES.md`

### 2.5 Ortam ayrımı (WSL2)

| Konum | İçerik |
|---|---|
| `/home/ahmet/projects/football-analytics` | Aktif kod (bu repo) |
| `/home/ahmet/projects/soccernet` | Dış SoccerNet clone’ları |
| `/home/ahmet/models` | Model ağırlıkları |
| `/home/ahmet/workspace/runs` | Pipeline koşu çıktıları |
| `/mnt/c/football_data` | Büyük video / dataset / result arşivi |

Path sözleşmesi: `configs/system/paths.yaml`  
Başka makinede çalıştırırken bu path’leri güncelleyin.

---

## 3. Mimari genel bakış

```text
                    ┌─────────────────────────────────────┐
   video.mp4 ─────► │  scripts/run_pipeline.py             │
                    │  PipelineRunner (orchestration)      │
                    └──────────────┬──────────────────────┘
                                   │ stage list (YAML)
           ┌───────────────────────┼───────────────────────┐
           ▼                       ▼                       ▼
     stages/*                 analytics/*            visualization/*
     (I/O + gates)            (saf hesaplar)         (overlay render)
           │                       │
           └──────────┬────────────┘
                      ▼
            contracts/schemas.py  →  *.parquet + manifests
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   full_match/     opta/        evaluation/
   (chunk/resume)  (aksiyon)    (GT metrikleri)
```

Full-match yolu, mevcut MVP-2 pipeline’ını yeniden yazmak yerine
`ExistingPipelineAdapter` ile chunk bazında `run_pipeline.py` çağırır
(`configs/full_match/existing_pipeline_adapter.yaml`).

---

## 4. Pipeline katmanları

### 4.1 MVP-1 — Tracking

- Config: `configs/pipeline/mvp1_tracking.yaml`
- Detection: Ultralytics YOLO (`yolo11n` doğrulanmış baseline)
- Tracking: ByteTrack
- Çıktı: detections / tracks parquet + annotated preview (opsiyonel)

### 4.2 MVP-2 — Spatial analytics

- Config: `configs/pipeline/mvp2_spatial_analytics.yaml`
- Stage sırası (tipik):

  1. `ingest`
  2. `shot_classification` — broadcas shot / scene-cut heuristic
  3. `detection`
  4. `tracking`
  5. `track_quality` — kısa/jittery track eleme
  6. `reid` — sn-reid / OSNet özellikleri (opsiyonel backend)
  7. `team_identity` — temporal shirt-colour clustering
  8. `camera_motion` — forward/backward LK + RANSAC affine
  9. `calibration` — provider zinciri (PnLCalib / manual / metadata…)
  10. `field_coordinates`
  11. `ball_state`
  12. `possession`
  13. `metrics` — player/team speed & distance
  14. `analytics_render`
  15. `event_detection` (altyapı; gerçek olay iddiası ayrı kapı)

Rapor: `docs/setup/mvp2_spatial_analytics_report.md`  
Sınırlar: `docs/setup/mvp2_known_limitations.md`

### 4.3 Opta-benzeri aksiyon katmanı

- Config: `configs/pipeline/opta_analytics.yaml`
- MVP-2 üzerine ek stage’ler:
  - `ball_tracking`
  - `touch_inference`
  - `action_inference`
  - `opta_analytics`
- Publishability bayrakları: `configs/quality/publishability_thresholds.yaml`
- Doğruluk için GT şablonları:
  `configs/evaluation/short_clip_gt_template/` (CSV/schema; frame PNG’ler git dışı)

Rapor: `docs/setup/opta_analytics_report.md`  
Remediation: `docs/setup/opta_accuracy_remediation_report.md`

### 4.4 Full-match orchestration

Paket: `src/football_analytics/full_match/`

- Video probe, chunking, scheduler, progress/health, consolidation
- Resume / recompute / postprocess / export
- Panel: `apps/full_match_panel.py` + `src/football_analytics/panel/`
- Multi-camera altyapı: `src/football_analytics/multicamera/`
- Roles: `src/football_analytics/roles/`
- Events: `src/football_analytics/events/`
- Export: `src/football_analytics/export/`

Durum (özet, 2026-07-18 civarı):

| Bayrak | Durum |
|---|---|
| `REAL_SHORT_VIDEO_PIPELINE_PASS` | Evet (kısa clip) |
| `MULTICAMERA_INFRASTRUCTURE_PASS` | Evet |
| `EVENT_DETECTION_INFRASTRUCTURE_PASS` | Evet |
| `GLOBAL_ID_SINGLE_CAMERA_PASS` | Evet |
| `FULL_MATCH_90MIN_PASS` | **İddia edilmiyor** |
| `MULTICAMERA_REAL_VIDEO_PASS` | **İddia edilmiyor** |
| `MATCH_EVENTS_REAL_VIDEO_PASS` | **İddia edilmiyor** |

Detay: `docs/setup/full_match_readiness/`

---

## 5. Paket sözlüğü (`src/football_analytics`)

| Paket | Sorumluluk |
|---|---|
| `contracts` | Canonical PyArrow şemaları ve kolon listeleri |
| `stages` | Pipeline stage implementasyonları (I/O + validation + gates) |
| `analytics` | Saf hesaplar: homography helpers, speed, possession, heatmaps… |
| `orchestration` | `PipelineRunner`, stage sırası, resume mantığı |
| `video` | Streaming decode / chunk yardımcıları |
| `geometry` | BBox / foot-point / görünürlük geometrisi |
| `visualization` | Analytics overlay renderer |
| `integrations` | SoccerNet / TrackEval / calibration adapter köprüleri |
| `adapters` | Dış format dönüşümleri |
| `jersey` | Forma numarası tanıma (clean-room) |
| `roles` | Oyuncu / hakem / kaleci rol sınıflandırma altyapısı |
| `multicamera` | Sync, fusion, duplicate suppression, coverage |
| `full_match` | Uzun maç scheduler + adapter |
| `events` | Event evidence / detector / review / clip özetleri |
| `opta` | Opta-benzeri aksiyon toplama ve raporlama |
| `evaluation` | Ball / identity / action GT metrikleri |
| `export` | JSON / CSV / Parquet / XLSX / video / tactical map |
| `panel` | Panel veri/komut köprüsü |
| `utils` | YAML I/O, checksum, path helpers |
| `datasets` / `api` | Dataset registry ve API iskeleti |

CLI script’leri `scripts/` altında toplanır; uygulama girişi `apps/`.

---

## 6. Kalibrasyon stratejisi

Kalibrasyon stage’i bir **provider zinciri** kullanır (config
`provider_priority`). Tipik sıra:

1. `sn_calibration` — blocked olabilir (ağırlık / lisans / indirme)
2. `pnlcalib` — GPL worker (`scripts/pnlcalib_worker.py`), izole env
3. `manual_json` — `scripts/create_manual_calibration.py` ile üretilen JSON
4. `metadata` — video manifest içinde kalibrasyon varsa
5. Demo four-point — yalnızca açık demo/fallback; production iddiası yok

Kurallar:

- Worker yalnızca image↔pitch correspondence döner
- Homography fit + coverage / reprojection / confidence kapıları **in-process**
- Geçmeyen frame’ler invalid kalır; hold `hold_max_frames` ile sınırlıdır
- Referans repodaki sabit 4 piksel / 23.32 m ölçek **kullanılmaz**

Örnek manuel kalibrasyon:
`configs/calibration/manual_football_frame100.json`

---

## 7. Kimlik, jersey, global ID

| Sinyal | Yöntem | Dürüst sınır |
|---|---|---|
| Team identity | Temporal shirt-colour clustering | Benzer forma / gölge / kısa track → unknown |
| Role | Heuristic + specialist classifiers | Kit referansı yoksa `unknown_person` |
| Re-ID | sn-reid OSNet + hard-negative calibration + position/roster gates | `reid_status=SOLVED` on smoke; Market1501 still weak — GT IDF1 optional |
| Jersey OCR/cls | Clean-room jersey recognizer | Eşik altı → unresolved |
| Global ID | Multicamera / single-cam map | Short-video pass ≠ 90 dk multi-cam pass |

Jersey config: `configs/jersey/jersey_recognition_v1.yaml`  
Eğitim/inference script’leri: `train_jersey_recognizer.py`,
`run_jersey_inference.py`, `evaluate_jersey_recognizer.py`

---

## 8. Evaluation ve doğruluk

Kısa clip üzerinde unit-test yeşili **gerçek accuracy değildir**.

GT şablon şeması: `configs/evaluation/short_clip_gt_template/`

- `gt_players.csv`, `gt_ball.csv`, `gt_touches.csv`, `gt_passes.csv`, …
- `schema.json`, `README.md`
- Frame PNG klasörleri bilinçli olarak `.gitignore`’dadır (repo şişmesin)

Metrik script’leri:

- `scripts/evaluate_ball_tracking.py`
- `scripts/evaluate_global_identity.py`
- `scripts/create_ball_gt_review_sample.py` / `annotate_ball_gt.py`
- Evaluation paketi: `src/football_analytics/evaluation/`

Doğruluk iddiası için gerekenler (özet):

- Etiketli pas / dribbling / tackle / aerial klipleri
- 5–10 dk ana kamera oyun bölümü
- Ball visibility + player identity etiketleri

---

## 9. Konfigürasyon haritası

| Path | Amaç |
|---|---|
| `configs/pipeline/*.yaml` | Ana pipeline tanımları |
| `configs/full_match/*.yaml` | 1/2/4 kamera + low_memory + adapter |
| `configs/events/` | Goal / event detection parametreleri |
| `configs/quality/` | Publishability eşikleri |
| `configs/calibration/` | Manuel kalibrasyon JSON |
| `configs/jersey/` | Jersey model/config |
| `configs/remediation/` | Düşük bellek / SoccerNet remediation |
| `configs/soccernet_install/` | Kurulum yardımcı config’ler |
| `configs/system/paths.yaml` | Makine path sözleşmesi |
| `model_registry.yaml` | Model SHA256 + kaynak URL |
| `dataset_registry.yaml` | Dataset kayıt iskeleti |
| `external_repos.lock.yaml` | Dış repo commit kilitleri |

---

## 10. Script envanteri (sık kullanılanlar)

| Script | İş |
|---|---|
| `run_pipeline.py` | Ana MVP / Opta pipeline |
| `validate_run_outputs.py` / `validate_mvp2_outputs.py` | Çıktı doğrulama |
| `prepare_full_match.py` / `run_full_match.py` / `resume_full_match.py` | Full-match |
| `validate_full_match_run.py` / `export_full_match_results.py` | Full-match QA/export |
| `create_manual_calibration.py` / `pnlcalib_worker.py` | Kalibrasyon |
| `bootstrap_external_repos.py` | SoccerNet clone pin sync |
| `check_project.py` / `check_all_envs.py` | Sağlık kontrolü |
| `run_panel_analysis.py` / `apps/full_match_panel.py` | Panel |
| `evaluate_*.py` / `annotate_*.py` | Evaluation / GT |
| `test_all_soccernet_components.py` | SoccerNet bileşen smoke |

Tam liste için `scripts/` dizinine bakın.

---

## 11. Test stratejisi

- Framework: `pytest` (`pyproject.toml` → `testpaths = ["tests"]`)
- Odak: şema sözleşmeleri, resume/checksum, geometry, analytics birimleri,
  full_match / multicamera / roles / events / panel regresyonları,
  SoccerNet adapter uyumluluğu
- Import-time’da model indirme veya ağır GPU işi **yapılmaz**
- Gerçek video smoke’ları workspace altında tutulur; repoya binary commit
  edilmez

---

## 12. Bilinen sınırlar (özet)

1. Detector COCO `yolo11n` — futbola özel sınıf seti yok (hakem/kaleci class yok)
2. Shot classifier heuristic — eğitilmiş broadcast model değil
3. `sn_calibration` sık blocked; PnLCalib coverage kısmi olabilir
4. Ball recall broadcast’te düşük; interpolasyon sınırlı, backfill yok
5. Possession tahmindir; contested/unknown bilinçlidir
6. 90 dk full-match + gerçek multi-cam + zengin event seti henüz claim edilmez
7. Jersey confident rate kısa clip’lerde düşük kalabilir

Detaylı liste: `docs/setup/mvp2_known_limitations.md` ve
`docs/setup/full_match_readiness/unresolved_limitations.md`

---

## 13. Manuel / dış adımlar

Bunlar kod push’u ile gelmez; makine başına yapılır:

1. SoccerNet NDA / password gerektiren dataset’ler
2. Google Drive kalibrasyon / teamspotting ağırlıkları
3. sn-gamestate Zenodo pretrained
4. `ai-dev` conda env + CUDA PyTorch pin’leri
5. WSL bellek ayarı (isteğe bağlı `.wslconfig`)

Kaynak: `docs/setup/manual_actions_required.md`

---

## 14. Git’e ne girer / ne girmez?

### Girer

- Tüm `src/`, `scripts/`, `apps/`, `tests/`
- YAML/JSON config’ler, GT CSV/schema şablonları
- `docs/` raporları, `patches/`, `third_party/licenses` + manifest
- `model_registry.yaml`, `external_repos.lock.yaml`, notices, LICENSE
- Küçük JSON/CSV smoke raporları (`artifacts/` altında binary hariç)

### Bilinçli olarak girmez (`.gitignore`)

- `*.mp4`, `*.pt`, `*.pth`, büyük image dump’lar (`frames/`)
- `logs/`, `runs/`, local `.venv/`, `.env`
- Model ağırlıkları ve ham maç videoları

Ağırlık bütünlüğü `model_registry.yaml` SHA256 alanlarıyla doğrulanır.

---

## 15. Başka makinede ayağa kaldırma checklist

1. Repo’yu clone et
2. Python 3.10 + `ai-dev` (veya eşdeğer CUDA PyTorch) kur
3. `pip install -e .`
4. `configs/system/paths.yaml` ve pipeline YAML mutlak path’lerini düzelt
5. YOLO / OSNet / PnLCalib ağırlıklarını indir → `model_registry.yaml` ile doğrula
6. İhtiyaç varsa SoccerNet repo’larını `external_repos.lock.yaml` commit’lerine pin’le
7. `python scripts/check_project.py`
8. Kısa clip ile `mvp1` → `mvp2` → (opsiyonel) `opta_analytics` smoke
9. `pytest`

---

## 16. Doküman indeksi

| Belge | Ne zaman oku? |
|---|---|
| `README.md` | Kurulum ve komutlar |
| `PROJECT_CONTEXT.md` | Mimari ve kararlar (bu dosya) |
| `THIRD_PARTY_NOTICES.md` | Lisans / redistribution |
| `docs/setup/mvp2_*` | Spatial analytics durumu |
| `docs/setup/full_match_readiness/` | Full-match claim matrisi |
| `docs/setup/opta_*` | Aksiyon katmanı / accuracy remediation |
| `docs/setup/soccernet_*` | Dış repo audit ve blocker remediation |
| `docs/research/reference_football_analysis_audit.md` | Referans repo clean-room auditi |

---

## 17. Kısa sözlük

| Terim | Anlam |
|---|---|
| Stage | Pipeline’daki tek iş birimi (manifest’li) |
| Canonical artifact | Şema versiyonlu parquet/JSON çıktı |
| Provider | Kalibrasyon vb. için değiştirilebilir backend |
| Clean-room | Fikir yeniden yazımı; kaynak kopyası yok |
| Publishability | Opta çıktısının yayın eşiğini geçip geçmediği |
| Smoke | Altyapı kanıtı; accuracy kanıtı değil |
| Resume | Checksum-doğrulanmış stage atlama |

---

*Son güncelleme bağlamı: 2026-07 civarı MVP-2 / full-match readiness /
Opta smoke çalışmaları. Durum bayrakları zamanla değişebilir; claim’ler için
`docs/setup/` altındaki en güncel rapora bakın.*
