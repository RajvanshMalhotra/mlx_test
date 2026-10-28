# mlx_test: can a Mac and NVIDIA laptops train one model together in MLX?

**Why this exists.** We want a Mac (Apple chip) and two Windows laptops (NVIDIA) to fine-tune one 3B model together. MLX, Apple's ML framework, now also has an NVIDIA (CUDA) backend. If MLX runs on all three laptops, we can use **one framework everywhere** and skip building an MLX ↔ PyTorch translator. These tests settle that in about a day.

| Test | Question it answers | Who runs it |
|---|---|---|
| 0 · `env_check.py` | Does MLX install and see the GPU? | Every laptop |
| 1 · `test1_train.py` | Can this laptop fine-tune a 4-bit 3B model with LoRA? How fast, how much memory? | Every laptop |
| 3 · `test3_ring.py` | Can the laptops add up arrays together over Wi-Fi with MLX (the step training needs)? | All laptops at the same time |
| 4 · `test4_drift.py` + `compare.py` | Given identical inputs, do Mac (Metal) and NVIDIA (CUDA) compute the same loss and gradients? | Every laptop, then compare |
| `net_bench.py` | Plain network speed, no MLX. Run it if Test 3 fails, to tell a network problem from an MLX problem. | Two laptops |

Every test saves its result to `results/<your-computer-name>/`, **including failures**: a crash or out-of-memory error is a result too.

---

## 1. Set up

### Mac (Apple Silicon)
```bash
git clone https://github.com/RajvanshMalhotra/mlx_test.git && cd mlx_test
python3 -m venv .venv && source .venv/bin/activate      # Python 3.10–3.12
pip install -r requirements.txt
```

### Windows laptop with NVIDIA: try native Windows first
1. Update the NVIDIA driver to the latest version (GeForce Experience or nvidia.com).
2. In PowerShell:
```powershell
git clone https://github.com/RajvanshMalhotra/mlx_test.git; cd mlx_test
py -3.11 -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt "mlx[cuda12]"
python env_check.py
```
3. If the install fails, or `default_device` doesn't say `gpu`, try `pip install mlx-cuda-12` once more. If it still fails, **write down the error** (it's a result) and use WSL2 below.

### Windows laptop with NVIDIA: WSL2 (Linux inside Windows)
MLX's CUDA backend is officially supported on Linux.
1. In an admin PowerShell: `wsl --install -d Ubuntu-24.04`, then restart and open "Ubuntu".
2. The Windows NVIDIA driver already gives WSL2 the GPU. Check with `nvidia-smi` inside Ubuntu.
3. Inside Ubuntu:
```bash
sudo apt update && sudo apt install -y python3-venv git
git clone https://github.com/RajvanshMalhotra/mlx_test.git && cd mlx_test
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt "mlx[cuda12]"
```
4. **For Test 3 only:** WSL2 must accept connections from the other laptops. Create `C:\Users\<you>\.wslconfig` containing
```
[wsl2]
networkingMode=mirrored
```
then run `wsl --shutdown` and reopen Ubuntu. (Mirrored mode needs Windows 11 22H2 or newer.)

### Firewall (Windows, for Tests 3 and net_bench)
In an admin PowerShell:
```powershell
New-NetFirewallRule -DisplayName "mlx_test" -Direction Inbound -Protocol TCP -LocalPort 5000-5001 -Action Allow
```
On the Mac, click "Allow" if macOS asks whether Python may accept incoming connections.

---

## 2. Run the tests

**Each laptop, on its own:**
```bash
python env_check.py
python test1_train.py
python test4_drift.py
```
- The first run downloads the 3B model (~1.8 GB) from Hugging Face.
- If Test 1 runs out of memory, that's a result. Then also run a smaller model:
  `python test1_train.py --model mlx-community/Qwen2.5-1.5B-Instruct-4bit`
- **While Test 1 runs on the Mac**, watch Activity Monitor → Memory → Memory Pressure, and note whether it turns yellow or red.
- To check the scripts quickly first, use the tiny model: `--model mlx-community/Qwen2.5-0.5B-Instruct-4bit`

**All laptops together (Test 3):**
1. On the same network, find each laptop's IP: Mac `ipconfig getifaddr en0` · Windows `ipconfig` · WSL `hostname -I`.
   College Wi-Fi often blocks laptop-to-laptop traffic. If so, use a phone hotspot.
2. Make the same `hosts.json` on every laptop, listing IPs in rank order (Mac = rank 0):
   `python make_hostfile.py 192.168.1.10 192.168.1.20 192.168.1.30`
3. Start all of them within about 30 seconds of each other, each with its own rank:
   - Mac / Linux / WSL: `MLX_RANK=0 MLX_HOSTFILE=hosts.json python test3_ring.py`
   - Windows PowerShell: `$env:MLX_RANK=1; $env:MLX_HOSTFILE="hosts.json"; python test3_ring.py`
4. If it hangs or fails, run `net_bench.py` between two laptops (`server` on one, `client <ip>` on the other) to see whether the network itself works.

**Compare Metal vs CUDA (Test 4):** once results from two laptops are in the repo:
```bash
python compare.py <mac-folder-name> <nvidia-folder-name>
```

---

## PyTorch route (NEW, 2026-10-28): the plan is MLX on the Mac, PyTorch + CUDA on NVIDIA

We decided: **the Mac keeps MLX, the Windows laptops use PyTorch with CUDA.** So on the Windows laptops, run these tests (no MLX, no WSL needed). They measure the numbers our simulator and demo currently only *assume*:

| Test | Measures | Replaces this assumption |
|---|---|---|
| 5 · `torch_env_check.py` | GPU name, **dedicated** VRAM, CUDA works, gloo (network) available | — |
| 6 · `torch_train.py` | QLoRA training speed (tokens/s) and peak GPU memory, same settings as the Mac's Test 1 | "NVIDIA = 2.92 × the Mac" |
| 7 · `gloo_test.py` | Time to average 20 MB (one training step's gradients) across laptops over Wi-Fi | "all-reduce over 5 MB/s Wi-Fi" |
| `net_bench.py` | Raw Wi-Fi speed and delay between two laptops | "Wi-Fi 5 MB/s, 5 ms" |

### Set up (each Windows laptop, PowerShell)
```powershell
git clone https://github.com/RajvanshMalhotra/mlx_test.git; cd mlx_test
py -3.11 -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126   # RTX 50-series: use cu130 instead
pip install -r requirements-torch.txt
python torch_env_check.py
```
`cuda_available` must say `True`. If it says `False`, update the NVIDIA driver and reinstall torch with the line above.

### Run (each Windows laptop, on its own)
```powershell
python torch_train.py                                   # 1.5B model, same size the Mac measured
python torch_train.py --model Qwen/Qwen2.5-3B-Instruct  # 3B, if it fits
```
The first run downloads the model (about 3 GB for 1.5B). An out-of-memory error is a result too, so leave it in.

### Run (all three laptops together, same Wi-Fi or a phone hotspot)
1. Firewall, once, in an **admin** PowerShell:
   `New-NetFirewallRule -DisplayName "gloo_test" -Direction Inbound -Protocol TCP -LocalPort 1024-65535 -Program (Get-Command python).Source -Action Allow`
2. Find the Mac's IP on the Mac: `ipconfig getifaddr en0`.
3. Within about 60 seconds of each other:
   - Mac: `python gloo_test.py --rank 0 --world 3 --master <Mac IP>`
   - Windows 1: `python gloo_test.py --rank 1 --world 3 --master <Mac IP>`
   - Windows 2: `python gloo_test.py --rank 2 --world 3 --master <Mac IP>`
4. If it hangs at "joining": check that the firewall rule exists, and try adding `--iface Wi-Fi` on Windows and `--iface en0` on the Mac.
5. Also run `net_bench.py` between the Mac and one Windows laptop (`server` on the Mac, `client <Mac IP>` on Windows).

Then share the results as described in section 3 (`results-<your-name>` branch).

---

## 3. Share your results
```bash
git checkout -b results-<your-name>
git add results/
git commit -m "results from <your laptop>"
git push -u origin results-<your-name>
```

## 4. What the results decide

| Outcome | Decision |
|---|---|
| Tests 1 and 3 pass on the NVIDIA laptops (native or WSL2) | **MLX everywhere.** No translator needed; build on MLX. |
| Test 1 fails on NVIDIA (no install, missing kernels, too slow) | **Build the MLX ↔ PyTorch translator.** The failure is our evidence for why it's needed. |
| Test 1 passes but Test 3 fails while `net_bench` works | MLX's network layer doesn't connect Metal and CUDA: we write our own exchange over the network, still with MLX on every laptop. |
| Test 4: relative diffs > 1e-2 or cosine < 0.999 | Metal and CUDA disagree more than rounding noise; investigate before training together. |

## Settings used (identical on every laptop)
- Model: `mlx-community/Qwen2.5-3B-Instruct-4bit` (not gated; override with `--model`)
- LoRA: rank 8, scale 20, dropout 0, last 16 blocks, all linear layers (`model_setup.py`)
- Batch 1 × 256 tokens of fixed pseudo-random ids (seed 0), Adam lr 1e-4, 10 steps
