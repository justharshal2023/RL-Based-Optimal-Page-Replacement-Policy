# Page Replacement Simulator — FIFO · LRU · DQN

A Streamlit app that simulates and compares three page replacement policies:
- **FIFO** — First In, First Out
- **LRU** — Least Recently Used
- **DQN** — Deep Q-Network optimal replacement policy (trained via RL)

---

## Run Locally

```bash
# 1. Clone / unzip the project
cd page_replacement_sim

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch
streamlit run app.py
```

The app opens at **http://localhost:8501** automatically.

---

## Deploy Online (Streamlit Community Cloud) — Free

1. Push this folder to a **public GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app** → select your repo → set **Main file path** to `app.py`
4. Click **Deploy** — your app gets a public URL in ~1 minute

No server, no cost.

---

## Project Structure

```
page_replacement_sim/
├── app.py            ← Streamlit UI + charts
├── algorithms.py     ← FIFO, LRU, DQNAgent (pure NumPy)
├── requirements.txt  ← dependencies
└── README.md
```

---

## DQN Architecture

```
State  : one-hot vector (size = page_range) of pages in memory
Action : frame slot to evict  (0 … num_frames−1)
Reward : +1 hit / −1 fault

Network:
  Input(page_range) → Linear → ReLU → Linear → Q(action)

Training:
  Double DQN  +  Experience Replay  +  Target Network (copied every 10 ep)
  ε-greedy exploration with exponential decay
```

All implemented in **pure NumPy** — no PyTorch / TensorFlow needed.
