# FutbolAnaliz (football-analytics)

Contract-first, reproducible football computer-vision and analytics pipelines.

Bu repo, maç videosundan oyuncu/top takibi, saha kalibrasyonu, topografik
metrikler, kimlik/jersey sinyalleri, olay çıkarımı ve Opta-benzeri aksiyon
özetlerini üreten bir analiz stack’idir. Kaynak kod `src/football_analytics`
altındadır; ağır modeller, veri setleri ve koşu çıktıları repo dışında tutulur.

**GitHub:** https://github.com/ahmmetavcii/FutbolAnaliz

> Proje mimarisi, tasarım kararları, sınırlar ve entegrasyon politikası için
> ayrıntılı belge: **[PROJECT_CONTEXT.md](./PROJECT_CONTEXT.md)**

---

## Ne yapar?

| Katman | Ne üretir? |
|---|---|
| MVP-1 Tracking | Detection + multi-object tracking (YOLO + ByteTrack) |
| MVP-2 Spatial Analytics | Shot sınıfı, kalibrasyon, saha koordinatı, hız/mesafe, top, possession, takım metrikleri |
| Full-match orchestration | Chunk’lı uzun maç koşusu, resume, panel, export |
| Opta-benzeri aksiyon katmanı | Top takibi, touch / pass / aksiyon çıkarımı, yayınlanabilirlik bayrakları |
| Evaluation | Ball / identity / action ground-truth şablonları ve metrikler |

Pipeline (özet):

```text
ingest → shot_classification → detection → tracking → track_quality
      → reid → team_identity → camera_motion → calibration
      → field_coordinates → ball_state → possession → metrics
      → analytics_render → (opta: ball_tracking → touch → action → opta_analytics)
```

Her stage girdi/çıktı sözleşmesini doğrular, checksum + config-hash manifest
yazar. `--resume-run-dir` geçerli stage’leri atlar; `--rerun-from <stage>`
belirtilen stage’den yeniden hesaplar.

---

## Hızlı başlangıç

### Gereksinimler

- Linux / **WSL2** (geliştirme hedefi)
- Python **3.10** (proje pin’i: `>=3.10,<3.11`)
- Conda env: `ai-dev` (PyTorch CUDA stack bu env’de sabitlenir)
- GPU önerilir (ör. RTX 4060 Laptop); CPU ile de smoke mümkün ama yavaştır
- Disk: modeller `~/models`, büyük veri `/mnt/c/football_data`, koşular `~/workspace/runs`

### Kurulum

```bash
git clone https://github.com/ahmmetavcii/FutbolAnaliz.git
cd FutbolAnaliz

# Conda ortamı (makinedeki mevcut ai-dev tercih edilir)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ai-dev

# Paket (editable)
pip install -e .

# Proje sağlık kontrolü
python scripts/check_project.py
```

Ortam dondurulmuş bağımlılıklar için:

- `environment.yml` — hafif proje iskeleti
- `requirements/ai-dev-conda-full.yml` — tam `ai-dev` snapshot
- `requirements/ai-dev-freeze.txt` — pip freeze

Yolları kendi makinenize göre güncelleyin:

- `configs/system/paths.yaml`
- `model_registry.yaml` (model dosya yolları)
- Pipeline YAML içindeki `model.path`, `reid.sn_reid_root` vb. mutlak path’ler

### MVP-1 (detection + tracking)

```bash
python scripts/run_pipeline.py \
  --config configs/pipeline/mvp1_tracking.yaml \
  --input /path/to/clip.mp4

python scripts/validate_run_outputs.py --run-dir /path/to/runs/<run_id>
```

### MVP-2 (spatial analytics)

```bash
python scripts/run_pipeline.py \
  --config configs/pipeline/mvp2_spatial_analytics.yaml \
  --input /path/to/clip.mp4

python scripts/validate_mvp2_outputs.py --run-dir /path/to/runs/<run_id>

# Kesilen koşuyu devam ettir
python scripts/run_pipeline.py \
  --config configs/pipeline/mvp2_spatial_analytics.yaml \
  --input /path/to/clip.mp4 \
  --resume-run-dir /path/to/runs/<run_id>

# Belirli stage’den yeniden hesapla
python scripts/run_pipeline.py \
  --config configs/pipeline/mvp2_spatial_analytics.yaml \
  --input /path/to/clip.mp4 \
  --resume-run-dir /path/to/runs/<run_id> \
  --rerun-from calibration
```

### Opta-benzeri aksiyon pipeline

```bash
python scripts/run_pipeline.py \
  --config configs/pipeline/opta_analytics.yaml \
  --input /path/to/clip.mp4
```

Kısa klipler yalnızca altyapı/smoke kanıtıdır; gerçek doğruluk için etiketli
pas / touch / identity ground-truth gerekir
(`configs/evaluation/short_clip_gt_template/`).

### Full-match (chunk orchestration)

```bash
python scripts/prepare_full_match.py --config configs/full_match/single_camera.yaml ...
python scripts/run_full_match.py --config configs/full_match/existing_pipeline_adapter.yaml ...
python scripts/resume_full_match.py ...
python scripts/validate_full_match_run.py --run-dir ...
```

Panel UI:

```bash
python apps/full_match_panel.py
```

### Testler

```bash
pytest
# veya odaklanmış
pytest tests/test_canonical_schemas.py tests/test_stage_resume.py
```

---

## Repo yapısı

```text
FutbolAnaliz/
├── apps/                      # Panel ve uygulama girişleri
├── configs/                   # Pipeline, full-match, events, quality, evaluation
│   ├── pipeline/              # mvp1 / mvp2 / opta YAML
│   ├── full_match/            # tek / çift / dört kamera + adapter
│   ├── evaluation/            # GT şablonları (CSV/JSON; frame dump’lar git dışı)
│   └── system/paths.yaml      # Yerel path sözleşmesi
├── docs/                      # Kurulum, readiness, SoccerNet audit raporları
├── scripts/                   # CLI: run/validate/bootstrap/evaluate/...
├── src/football_analytics/    # Ana Python paketi
│   ├── analytics/             # Metrik, possession, calibration helpers
│   ├── contracts/             # Canonical PyArrow şemaları
│   ├── evaluation/            # Ball / identity / action metrikleri
│   ├── events/                # Olay tespiti altyapısı
│   ├── full_match/            # Chunk scheduler, resume, adapter
│   ├── integrations/          # SoccerNet / dış repo adapter’ları
│   ├── multicamera/           # Sync, fusion, global identity
│   ├── opta/                  # Opta-benzeri aksiyon katmanı
│   ├── orchestration/         # PipelineRunner
│   ├── roles/                 # Rol sınıflandırma
│   ├── stages/                # Stage implementasyonları
│   ├── video/                 # Streaming I/O
│   └── visualization/         # Overlay / render
├── tests/                     # Unit + regression
├── third_party/               # License texts + manifest (kaynak kopyası yok)
├── patches/                   # Upstream’e uygulanan küçük patch notları
├── artifacts/                 # Küçük smoke/rapor JSON’ları (ağır binary git dışı)
├── model_registry.yaml        # Model checksum + kaynak kayıtları
├── external_repos.lock.yaml   # SoccerNet / 3P commit kilitleri
├── THIRD_PARTY_NOTICES.md     # Lisans / dağıtım politikası
├── PROJECT_CONTEXT.md         # Derin proje bağlamı (mimari + kararlar)
└── README.md                  # Bu dosya
```

---

## Canonical çıktılar (MVP-2)

| Artifact | Anlamı |
|---|---|
| `shot_segments.parquet` | Broadcas shot / scene-cut sınıfları |
| `track_identities.parquet` | Takım / rol kimlikleri |
| `camera_motion.parquet` | Kamera hareketi |
| `calibration.parquet` | Homografi / saha kalibrasyonu |
| `game_state.parquet` | Oyun durumu sinyalleri |
| `ball_state.parquet` | Top durumu |
| `possession_timeline.parquet` | Possession timeline |
| `track_quality.parquet` | Track kalite kapıları |
| `player_metrics.parquet` | Oyuncu hız / mesafe vb. |
| `team_metrics.parquet` | Takım özet metrikleri |

Ortak alanlar: `schema_version`, `run_id`, `match_id`, `frame_id`,
`timestamp_ms`, `source_method`, `confidence`, `valid`. Bilinmeyen değerler
nullable bırakılır; uydurma doldurma yapılmaz.

---

## Dış bağımlılıklar ve lisans

Ağır SoccerNet / kalibrasyon / reid bileşenleri **repo içine kopyalanmaz**.
Commit kilitleri `external_repos.lock.yaml`, lisans özeti
`THIRD_PARTY_NOTICES.md` içindedir.

| Bileşen | Rol | Not |
|---|---|---|
| Ultralytics YOLO | Detection | AGPL / commercial; ağırlıklar `~/models` |
| ByteTrack | Tracking | Ultralytics tracker |
| sn-reid (OSNet) | Re-ID özellikleri | Dış repo + model |
| PnLCalib | Otomatik kalibrasyon | GPL; **out-of-process** worker |
| sn-gamestate | GameState baseline | GPL; izole env |
| sn-trackeval | Evaluation adapter | MIT |

Bootstrap (makinede SoccerNet clone’ları varsa):

```bash
python scripts/bootstrap_external_repos.py
# veya
bash scripts/bootstrap_external_repos.sh
```

Manuel indirme / NDA adımları: `docs/setup/manual_actions_required.md`

---

## Önemli dürüstlük kuralları

- Kalibrasyon / kimlik / top kanıtı yetersizse metrikler **null** veya `valid=false`
- Jersey / rol / event uydurulmaz; güven eşiğinin altı `unresolved` kalır
- Kısa clip smoke ≠ yayın doğruluğu; accuracy için GT etiketleri şart
- GPL kod `src/football_analytics` içine gömülmez; izole process / clean-room adapter

Bilinen sınırlar: `docs/setup/mvp2_known_limitations.md`

---

## Dokümantasyon haritası

| Belge | İçerik |
|---|---|
| [PROJECT_CONTEXT.md](./PROJECT_CONTEXT.md) | Mimari, kararlar, modül sözlüğü, veri akışı |
| [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md) | Lisans / dağıtım |
| [docs/setup/mvp2_spatial_analytics_report.md](./docs/setup/mvp2_spatial_analytics_report.md) | MVP-2 uygulama raporu |
| [docs/setup/full_match_readiness/](./docs/setup/full_match_readiness/) | Full-match readiness matrisi |
| [docs/setup/reid_resolution_report.md](./docs/setup/reid_resolution_report.md) | ReID çözümü (coverage, stitch, SOLVED) |
| [docs/setup/soccernet_repo_tests/](./docs/setup/soccernet_repo_tests/) | SoccerNet repo audit / remediation |

---

## Lisans

Bu projenin kendi kaynak kodu **MIT** (`LICENSE`). Üçüncü parti modeller,
SoccerNet veri setleri ve GPL bileşenler kendi lisanslarına tabidir; yeniden
dağıtımdan önce `THIRD_PARTY_NOTICES.md` ve upstream şartlarını okuyun.

---

## Katkı / geliştirme notları

- Python format: Black / Ruff, line-length **100**
- Yeni stage eklerken: `contracts/schemas.py` + stage + test + YAML stage listesi
- Mutlak path’leri commit etmeden önce local `paths.yaml` / model path’lerini
  gözden geçirin; CI/başka makine için env veya relative override tercih edin
- Büyük frame dump, video, `.pt` / `.pth` ağırlıkları **bilerek git dışıdır**
  (`.gitignore`); checksum’lar `model_registry.yaml` içinde tutulur
