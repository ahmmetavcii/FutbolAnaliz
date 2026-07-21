# Manual Actions Required

1. **sn-nvs CUDA Toolkit** — **deferred / not required for MVP**. Status stays `BLOCKED_BUILD`. Do **not** install `cuda-toolkit` unless you explicitly want sn-nvs later; installing a host CUDA toolkit can risk the stable `ai-dev` PyTorch cu128 stack. If you later choose to proceed in an isolated fashion, use a separate decision and env — not as part of normal MVP work.

2. **SoccerNet NDA / password** for broadcast videos, tracking videos, and MVFoul dataset:
   https://www.soccer-net.org/data

3. **Google Drive model weights**
   - sn-calibration: https://drive.google.com/file/d/1dbN7LdMV03BR1Eda8n7iKNIyYp9r07sM/view?usp=sharing
   - sn-teamspotting checkpoint: https://drive.google.com/drive/folders/16IqSkctIGp76ZYKKvJvMB_ggHcQsessM?usp=sharing

4. **sn-gamestate Zenodo / pretrained weights** when preparing the first GameState video smoke test.

5. **Optional WSL memory**: current WSL sees ~7.6 GiB RAM; to expose closer to 16 GB create/edit `%UserProfile%\.wslconfig` then `wsl --shutdown`.
