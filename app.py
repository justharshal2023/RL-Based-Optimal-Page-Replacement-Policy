import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from collections import deque
import random
import time

from algorithms import fifo, lru, DQNAgent

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Page Replacement Simulator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"], .stMarkdown, .stText, h1, h2, h3, p, label {
    font-family: 'Syne', sans-serif !important;
}
code, .stCode, pre, .monospace {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #090909 !important;
    border-right: 1px solid #1c1c1c;
}
section[data-testid="stSidebar"] .stMarkdown h2 {
    font-size: 0.65rem !important;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #444 !important;
    font-weight: 700;
    border-bottom: 1px solid #1a1a1a;
    padding-bottom: 0.4rem;
    margin-top: 1.2rem !important;
}

/* Main area */
.block-container {
    padding-top: 2rem !important;
    max-width: 1400px;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: #0c0c0c;
    border: 1px solid #1e1e1e;
    border-radius: 10px;
    padding: 1rem 1.2rem !important;
}
div[data-testid="metric-container"] label {
    font-size: 0.62rem !important;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #555 !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
}

/* Expander */
.streamlit-expanderHeader {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em;
    color: #888 !important;
}

/* Buttons */
.stButton > button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em;
    border-radius: 8px !important;
    border: 1px solid #333 !important;
    background: #111 !important;
    color: #eee !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #1e1e1e !important;
    border-color: #555 !important;
    color: #fff !important;
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #6366f1, #a78bfa) !important;
    border-radius: 4px !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    background: #0a0a0a;
    border-bottom: 1px solid #1a1a1a;
    padding: 0 0.5rem;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #555 !important;
    padding: 0.6rem 1.2rem !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #c4b5fd !important;
    border-bottom-color: #6366f1 !important;
}

/* Dataframe */
.stDataFrame { border-radius: 8px; overflow: hidden; }

/* Divider */
hr { border: none; border-top: 1px solid #1a1a1a; margin: 1.5rem 0; }

.big-title {
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1.1;
    color: #f0f0f0;
}
.big-title span { color: #6366f1; }
.subtitle-text {
    color: #555;
    font-size: 0.95rem;
    font-weight: 400;
    margin-top: 0.3rem;
}
.section-label {
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #444;
    margin-bottom: 0.6rem;
}
.algo-chip {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 4px;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Colour palette ─────────────────────────────────────────────────────────────
COLORS = {
    "FIFO": "#60a5fa",      # blue
    "LRU":  "#34d399",      # green
    "DQN":  "#a78bfa",      # violet
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def random_ref_string(length=20, page_range=8):
    return " ".join(str(random.randint(0, page_range - 1)) for _ in range(length))


def parse_ref_string(s):
    tokens = s.replace(",", " ").split()
    pages = [int(t) for t in tokens if t.isdigit()]
    return pages


def build_trace_df(log, num_frames):
    rows = []
    for step in log:
        frames = step["frames"] + [""] * (num_frames - len(step["frames"]))
        rows.append(
            {
                "Step": len(rows) + 1,
                "Page": step["page"],
                **{f"Frame {i}": (frames[i] if i < len(step["frames"]) else "") for i in range(num_frames)},
                "Result": "HIT ✓" if step["hit"] else "FAULT ✗",
            }
        )
    return pd.DataFrame(rows)


def hex_to_rgba(hex_color, alpha=1.0):
    """Convert '#rrggbb' to 'rgba(r,g,b,a)'."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def make_fault_bar(results):
    algos = list(results.keys())
    faults = [results[a]["faults"] for a in algos]
    hits   = [results[a]["hits"] for a in algos]
    colors = [COLORS[a] for a in algos]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Faults",
        x=algos, y=faults,
        marker_color=colors,
        marker_line_color="rgba(0,0,0,0)",
        text=faults, textposition="outside",
        textfont=dict(family="JetBrains Mono", size=13, color="#ccc"),
    ))
    fig.add_trace(go.Bar(
        name="Hits",
        x=algos, y=hits,
        marker_color=[hex_to_rgba(c, 0.27) for c in colors],
        marker_line_color="rgba(0,0,0,0)",
        text=hits, textposition="outside",
        textfont=dict(family="JetBrains Mono", size=13, color="#555"),
    ))
    fig.update_layout(
        barmode="group",
        plot_bgcolor="#080808",
        paper_bgcolor="#080808",
        font=dict(family="Syne", color="#888", size=12),
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=13, color="#aaa")),
        yaxis=dict(showgrid=True, gridcolor="#111", zeroline=False, title="Count"),
        legend=dict(
            orientation="h", y=1.1, x=0.5, xanchor="center",
            bgcolor="rgba(0,0,0,0)", font=dict(size=11, color="#777"),
        ),
        margin=dict(t=40, b=20, l=20, r=20),
        height=300,
    )
    return fig


def make_hit_rate_gauge(results):
    fig = go.Figure()
    for i, (algo, res) in enumerate(results.items()):
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=round(res["hit_rate"] * 100, 1),
            number=dict(suffix="%", font=dict(family="JetBrains Mono", size=28, color=COLORS[algo])),
            title=dict(text=algo, font=dict(family="Syne", size=12, color="#666")),
            gauge=dict(
                axis=dict(range=[0, 100], tickfont=dict(size=9, color="#444")),
                bar=dict(color=COLORS[algo], thickness=0.3),
                bgcolor="#111",
                borderwidth=0,
                steps=[dict(range=[0, 100], color="#0d0d0d")],
                threshold=dict(line=dict(color="#333", width=2), thickness=0.75, value=50),
            ),
            domain=dict(x=[i / 3, (i + 1) / 3], y=[0, 1]),
        ))
    fig.update_layout(
        paper_bgcolor="#080808",
        plot_bgcolor="#080808",
        margin=dict(t=30, b=10, l=20, r=20),
        height=220,
    )
    return fig


def make_reward_curve(rewards):
    window = max(1, len(rewards) // 30)
    smoothed = pd.Series(rewards).rolling(window, min_periods=1).mean().tolist()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=rewards, mode="lines",
        line=dict(color="#2a2a4a", width=1),
        name="Raw", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        y=smoothed, mode="lines",
        line=dict(color="#a78bfa", width=2),
        name=f"Smoothed (w={window})", showlegend=True,
    ))
    fig.update_layout(
        plot_bgcolor="#080808",
        paper_bgcolor="#080808",
        font=dict(family="Syne", color="#666", size=11),
        xaxis=dict(showgrid=False, zeroline=False, title="Episode"),
        yaxis=dict(showgrid=True, gridcolor="#111", zeroline=False, title="Total Reward"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#777")),
        margin=dict(t=10, b=20, l=20, r=20),
        height=240,
    )
    return fig


def make_trace_heatmap(log, num_frames, algo_name, color):
    steps = len(log)
    grid = np.full((num_frames, steps), -1.0)
    hit_mask = []

    for t, entry in enumerate(log):
        for f, p in enumerate(entry["frames"]):
            grid[f, t] = p
        hit_mask.append(1 if entry["hit"] else 0)

    fig = go.Figure()

    # Frame contents
    fig.add_trace(go.Heatmap(
        z=grid,
        colorscale=[[0, "#0a0a0a"], [1, hex_to_rgba(color, 0.33)]],
        showscale=False,
        xgap=2, ygap=2,
        zmin=-1, zmax=max(1, int(np.max(grid))),
    ))

    # Annotate page numbers
    for f in range(num_frames):
        for t in range(steps):
            v = grid[f, t]
            if v >= 0:
                fig.add_annotation(
                    x=t, y=f,
                    text=str(int(v)),
                    showarrow=False,
                    font=dict(family="JetBrains Mono", size=10,
                               color="#ccc" if log[t]["hit"] else "#f87171"),
                )

    # Hit/fault bar on top (row -1 position via shapes)
    for t, h in enumerate(hit_mask):
        fig.add_shape(
            type="rect",
            x0=t - 0.45, x1=t + 0.45,
            y0=num_frames - 0.1, y1=num_frames + 0.4,
            fillcolor="rgba(74,222,128,0.2)" if h else "rgba(248,113,113,0.2)",
            line_width=0,
        )

    fig.update_layout(
        plot_bgcolor="#080808",
        paper_bgcolor="#080808",
        font=dict(family="Syne", color="#666", size=10),
        xaxis=dict(
            title="Step", showgrid=False, zeroline=False,
            tickvals=list(range(steps)),
            ticktext=[str(entry["page"]) for entry in log],
            tickfont=dict(family="JetBrains Mono", size=10),
        ),
        yaxis=dict(
            title="Frame", showgrid=False, zeroline=False,
            tickvals=list(range(num_frames)),
            ticktext=[f"F{i}" for i in range(num_frames)],
        ),
        margin=dict(t=10, b=40, l=40, r=10),
        height=max(180, num_frames * 55 + 80),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-size:1.1rem;font-weight:800;letter-spacing:-0.01em;"
        "color:#c4b5fd;margin-bottom:0.2rem'>⚡ PAGE REPLACEMENT</div>"
        "<div style='font-size:0.65rem;color:#333;letter-spacing:0.12em;"
        "text-transform:uppercase;margin-bottom:1.5rem'>OS Simulator · DQN Edition</div>",
        unsafe_allow_html=True,
    )

    st.markdown("## Reference String")
    use_random = st.toggle("Random string", value=False)

    page_range = st.slider("Page range (0 … N-1)", 4, 16, 8)

    if use_random:
        ref_len = st.slider("Length", 10, 100, 100)
        if st.button("Regenerate"):
            st.session_state["ref_str"] = random_ref_string(ref_len, page_range)
        if "ref_str" not in st.session_state:
            st.session_state["ref_str"] = random_ref_string(ref_len, page_range)
        ref_input = st.text_area(
            "Generated string (editable)", value=st.session_state["ref_str"], height=120
        )
    else:
        default_ref = (
            "7 0 1 2 0 3 0 4 2 3 0 3 2 1 2 0 1 7 0 1 "
            "3 4 5 2 0 1 3 4 0 2 5 1 3 0 4 2 1 5 3 0 "
            "2 4 1 3 5 0 2 4 3 1 0 5 2 4 3 1 0 2 5 4 "
            "1 3 0 2 4 5 1 3 2 0 4 5 3 1 2 0 4 3 5 1 "
            "0 2 4 3 1 5 0 2 3 4 1 0 5 2 3 4 1 0"
        )
        ref_input = st.text_area(
            "Enter pages (space / comma separated)",
            value=default_ref,
            height=120,
        )

    st.markdown("## Memory")
    num_frames = st.slider("Number of frames", 1, 8, 3)

    st.markdown("## DQN Hyperparameters")
    episodes = st.slider("Training episodes", 50, 2000, 600, step=50)
    hidden_dim = st.select_slider("Hidden layer size", [32, 64, 128, 256], value=128)
    lr = st.select_slider(
        "Learning rate", [0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001
    )
    gamma = st.slider("Discount gamma", 0.5, 0.99, 0.95, step=0.01)
    epsilon_decay = st.slider("Epsilon decay per episode", 0.90, 0.999, 0.97, step=0.001)
    batch_size = st.select_slider("Batch size", [16, 32, 64, 128], value=64)
    lookahead = st.slider("Lookahead steps", 0, 10, 5)

    st.markdown("## Improvements")
    use_per            = st.toggle("Prioritised Replay (PER)",    value=True)
    use_shaped_reward  = st.toggle("Shaped Reward (Belady hint)", value=True)
    use_random_strings = st.toggle("Generalise (random strings)", value=True)
    locality = st.slider("Locality bias", 0.0, 1.0, 0.7, step=0.05)

    run_btn = st.button("Run Simulation", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='big-title'>Page Replacement <span>Simulator</span></div>"
    "<div class='subtitle-text'>FIFO · LRU · Deep Q-Network Optimal Policy</div>",
    unsafe_allow_html=True,
)
st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Main logic
# ─────────────────────────────────────────────────────────────────────────────
if run_btn or "results" in st.session_state:

    if run_btn:
        ref_string = parse_ref_string(ref_input)
        if len(ref_string) < 2:
            st.error("Please enter at least 2 pages.")
            st.stop()

        # ── Run classical algos ──
        res_fifo = fifo(ref_string, num_frames)
        res_lru  = lru(ref_string, num_frames)

        # ── Train DQN ──
        agent = DQNAgent(
            num_frames         = num_frames,
            page_range         = page_range,
            hidden_dim         = hidden_dim,
            lr                 = lr,
            gamma              = gamma,
            epsilon_decay      = epsilon_decay,
            batch_size         = batch_size,
            lookahead          = lookahead,
            use_per            = use_per,
            use_shaped_reward  = use_shaped_reward,
            use_random_strings = use_random_strings,
            locality           = locality,
        )

        st.markdown("**Training DQN agent…**")
        prog_bar   = st.progress(0)
        status_txt = st.empty()
        reward_placeholder = st.empty()
        reward_history = []

        def progress_cb(ep, total, reward, epsilon):
            reward_history.append(reward)
            prog_bar.progress(ep / total)
            status_txt.markdown(
                f"<span style='font-family:JetBrains Mono;font-size:0.8rem;color:#666'>"
                f"Episode {ep}/{total} · reward {reward:.1f} · epsilon {epsilon:.3f}"
                f"</span>",
                unsafe_allow_html=True,
            )

        rewards = agent.fit(ref_string, num_episodes=episodes, progress_cb=progress_cb)
        prog_bar.empty()
        status_txt.empty()
        reward_placeholder.empty()

        res_dqn = agent.run_inference(ref_string)

        st.session_state["results"]     = {
            "FIFO": res_fifo,
            "LRU":  res_lru,
            "DQN":  res_dqn,
        }
        st.session_state["rewards"]     = rewards
        st.session_state["td_errors"]   = agent.td_errors_log
        st.session_state["ref_string"]  = ref_string
        st.session_state["num_frames"]  = num_frames
        st.session_state["improvements"] = {
            "Slot-aware + Lookahead state": True,
            "Dueling network (V + A)":      True,
            "Prioritised Replay (PER)":     use_per,
            "Shaped Reward (Belady hint)":  use_shaped_reward,
            "Random string generalisation": use_random_strings,
        }

    results      = st.session_state["results"]
    rewards      = st.session_state["rewards"]
    td_errors    = st.session_state.get("td_errors", [])
    ref_string   = st.session_state["ref_string"]
    num_frames   = st.session_state["num_frames"]
    improvements = st.session_state.get("improvements", {})
    total        = len(ref_string)

    # ── KPI Row ───────────────────────────────────────────────────────────────
    st.markdown("<div class='section-label'>Summary Metrics</div>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    cols = [c1, c2, c3]
    for col, (algo, res) in zip(cols, results.items()):
        col.metric(f"{algo} Faults", res["faults"],
                   delta=f"{res['hit_rate']*100:.1f}% hit rate",
                   delta_color="normal")

    c4.metric("Ref String Length", total)
    c5.metric("Memory Frames",     num_frames)
    c6.metric("Page Range",        f"0–{page_range-1}")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊  Comparison", "🔬  Step Traces", "🧠  DQN Training", "📋  Data"]
    )

    # ── Tab 1: Comparison ─────────────────────────────────────────────────────
    with tab1:
        col_left, col_right = st.columns([1, 1], gap="large")

        with col_left:
            st.markdown("<div class='section-label'>Faults vs Hits</div>", unsafe_allow_html=True)
            st.plotly_chart(make_fault_bar(results), use_container_width=True)

        with col_right:
            st.markdown("<div class='section-label'>Hit Rate</div>", unsafe_allow_html=True)
            st.plotly_chart(make_hit_rate_gauge(results), use_container_width=True)

        st.markdown("<div class='section-label'>Reference String</div>", unsafe_allow_html=True)
        ref_display = " · ".join(
            f"<code style='color:#a78bfa;font-size:0.85rem'>{p}</code>"
            for p in ref_string
        )
        st.markdown(
            f"<div style='background:#0b0b0b;border:1px solid #1a1a1a;border-radius:8px;"
            f"padding:0.8rem 1rem;line-height:2;'>{ref_display}</div>",
            unsafe_allow_html=True,
        )

        # Summary table
        st.markdown("<br><div class='section-label'>Summary Table</div>", unsafe_allow_html=True)
        summary = pd.DataFrame([
            {
                "Algorithm": algo,
                "Faults": res["faults"],
                "Hits": res["hits"],
                "Hit Rate": f"{res['hit_rate']*100:.1f}%",
                "Fault Rate": f"{(1-res['hit_rate'])*100:.1f}%",
            }
            for algo, res in results.items()
        ])
        st.dataframe(summary, use_container_width=True, hide_index=True)

    # ── Tab 2: Step Traces ────────────────────────────────────────────────────
    with tab2:
        for algo, res in results.items():
            color = COLORS[algo]
            with st.expander(
                f"{'🔵' if algo=='FIFO' else '🟢' if algo=='LRU' else '🟣'}  {algo}  ·  "
                f"{res['faults']} faults · {res['hit_rate']*100:.1f}% hit rate",
                expanded=(algo == "DQN"),
            ):
                st.plotly_chart(
                    make_trace_heatmap(res["log"], num_frames, algo, color),
                    use_container_width=True,
                )
                df = build_trace_df(res["log"], num_frames)

                def color_result(val):
                    if "HIT" in str(val):
                        return "color: #4ade80; font-weight:600"
                    if "FAULT" in str(val):
                        return "color: #f87171; font-weight:600"
                    return ""

                st.dataframe(
                    df.style.applymap(color_result, subset=["Result"]),
                    use_container_width=True,
                    hide_index=True,
                )

    # ── Tab 3: DQN Training ───────────────────────────────────────────────────
    with tab3:
        col_l, col_r = st.columns([2, 1], gap="large")
        with col_l:
            st.markdown("<div class='section-label'>Training Reward Curve</div>", unsafe_allow_html=True)
            st.plotly_chart(make_reward_curve(rewards), use_container_width=True)

            if td_errors:
                st.markdown("<div class='section-label'>TD Error (Mean Squared)</div>", unsafe_allow_html=True)
                window = max(1, len(td_errors) // 20)
                smoothed_td = pd.Series(td_errors).rolling(window, min_periods=1).mean().tolist()
                fig_td = go.Figure()
                fig_td.add_trace(go.Scatter(
                    y=td_errors, mode="lines",
                    line=dict(color="#2a2a3a", width=1), name="Raw", showlegend=True,
                ))
                fig_td.add_trace(go.Scatter(
                    y=smoothed_td, mode="lines",
                    line=dict(color="#f59e0b", width=2), name=f"Smoothed (w={window})", showlegend=True,
                ))
                fig_td.update_layout(
                    plot_bgcolor="#080808", paper_bgcolor="#080808",
                    font=dict(family="Syne", color="#666", size=11),
                    xaxis=dict(showgrid=False, zeroline=False, title="Episode"),
                    yaxis=dict(showgrid=True, gridcolor="#111", zeroline=False, title="TD Error"),
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#777")),
                    margin=dict(t=10, b=20, l=20, r=20), height=200,
                )
                st.plotly_chart(fig_td, use_container_width=True)

        with col_r:
            st.markdown("<div class='section-label'>Training Stats</div>", unsafe_allow_html=True)
            last50 = rewards[-50:] if len(rewards) >= 50 else rewards
            st.metric("Avg reward (last 50 ep)", f"{np.mean(last50):.2f}")
            st.metric("Best episode reward",     f"{max(rewards):.0f}")
            st.metric("Worst episode reward",    f"{min(rewards):.0f}")
            st.metric("Total training episodes", len(rewards))
            if td_errors:
                st.metric("Final TD error",      f"{td_errors[-1]:.4f}")

            st.markdown("<br><div class='section-label'>Active Improvements</div>", unsafe_allow_html=True)
            for name, active in improvements.items():
                icon  = "✓" if active else "✗"
                color = "#4ade80" if active else "#555"
                st.markdown(
                    f"<div style='font-size:0.78rem;color:{color};"
                    f"margin-bottom:4px;font-family:JetBrains Mono'>"
                    f"{icon}  {name}</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>How the Improvements Work</div>", unsafe_allow_html=True)
        st.markdown(
            """
**1. Slot-aware + Lookahead state** — State vector encodes each frame slot separately
(preserving which page is in which slot) plus the next N pages in the reference string.
This gives the agent enough information to approximate Belady's Optimal.

**2. Dueling network (V + A)** — Hidden layer splits into a Value stream V(s) and
Advantage stream A(s,a). Q = V + A - mean(A). V learns how good the memory state is;
A learns which specific frame is worst to keep. More stable than a single Q head.

**3. Prioritised Experience Replay** — Transitions with larger TD-error are replayed
more often. The agent spends more time learning from its worst mistakes. IS weights
correct the sampling bias.

**4. Shaped Reward** — Instead of flat -1 per fault, the reward reflects how far in
the future the evicted page will be needed again (0 = perfect Belady eviction,
-1 = evicted a page needed immediately). Denser signal, faster learning.

**5. Random String Generalisation** — Each training episode uses a freshly generated
reference string with locality bias. The agent learns a transferable policy rather
than memorising one sequence.
"""
        )

    # ── Tab 4: Raw Data ───────────────────────────────────────────────────────
    with tab4:
        algo_choice = st.selectbox("Select algorithm", list(results.keys()))
        df_full = build_trace_df(results[algo_choice]["log"], num_frames)
        st.dataframe(df_full, use_container_width=True, hide_index=True)
        csv = df_full.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇  Download CSV",
            data=csv,
            file_name=f"{algo_choice}_trace.csv",
            mime="text/csv",
        )

else:
    st.markdown(
        "<div style='text-align:center;padding:4rem 0;color:#333;"
        "font-size:0.9rem;letter-spacing:0.1em;text-transform:uppercase'>"
        "Configure parameters in the sidebar and click <strong style='color:#555'>▶ Run Simulation</strong>"
        "</div>",
        unsafe_allow_html=True,
    )
