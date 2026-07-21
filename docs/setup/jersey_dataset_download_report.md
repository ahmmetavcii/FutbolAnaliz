# Jersey-2023 Dataset Download & Verification Report

Generated: 2026-07-18T15:03:14.760422+00:00
Overall: **PASS**

## Download
- Status: DOWNLOAD_COMPLETE
- Started: 2026-07-18T01:39:28.325006+03:00
- Finished: 2026-07-18T03:48:34.399401+03:00
- Splits: train, test
- Target: `/mnt/c/football_data/datasets/SoccerNet/jersey-2023`
- Disk usage (du -sh): `9.2G	/mnt/c/football_data/datasets/SoccerNet/jersey-2023`
- Filesystem (df -h):

```
Filesystem      Size  Used Avail Use% Mounted on
C:\             464G  285G  180G  62% /mnt/c
```

## Split statistics

| Split | Tracklet dirs | GT entries | Total images | Known-number tracklets | -1 tracklets | GT JSON |
|---|---:|---:|---:|---:|---:|---|
| train | 1427 | 1427 | 733001 | 1024 | 403 | `/mnt/c/football_data/datasets/SoccerNet/jersey-2023/train/train_gt.json` |
| test | 1211 | 1211 | 564547 | 856 | 355 | `/mnt/c/football_data/datasets/SoccerNet/jersey-2023/test/test_gt.json` |

## Selected tracklet
- Split: train
- player_id: 0
- Ground-truth jersey number: 10
- Image count: 578
- Source folder: `/mnt/c/football_data/datasets/SoccerNet/jersey-2023/train/images/0`
- Images opened OK: 578 / 578
- Corrupt images: 0

## Preview MP4
- FPS target: 10
- Frames written: 578
- Canvas: 256x256
- Primary: `/mnt/c/football_data/videos/test_clips/jersey_tracklet_preview.mp4`
- Staging: `/home/ahmet/workspace/staging/jersey_tracklet_preview.mp4`

| Location | bytes | OpenCV frames | OpenCV fps | ffprobe frames | ffprobe fps | >=20 | fps~10 | PASS |
|---|---:|---:|---:|---:|---:|:--:|:--:|:--:|
| primary | 2686064 | 578 | 10.0 | 578 | 10.0 | True | True | True |
| staging | 2686064 | 578 | 10.0 | 578 | 10.0 | True | True | True |

## Contact sheet
- Path: `/mnt/c/football_data/videos/test_clips/jersey_tracklet_contact_sheet.jpg`
- Frames placed: 12 (first 12)
- Bytes: 129610
- Reopen OK: True shape=[660, 640, 3]

## Errors / manual steps
- None
