"""
Page Replacement Algorithms
  - FIFO
  - LRU
  - Improved DQN Agent with:
      1. Slot-aware + lookahead state representation
      2. Training on randomly generated reference strings (generalisation)
      3. Prioritised Experience Replay (PER)
      4. Dueling Network Architecture (V + A streams)
      5. Shaped reward (approximating Belady distance)
"""

import numpy as np
import random
from collections import deque


# ─────────────────────────────────────────────
#  FIFO
# ─────────────────────────────────────────────
def fifo(reference_string, num_frames):
    frames = []
    queue = deque()
    faults = 0
    log = []

    for page in reference_string:
        hit = page in frames
        if not hit:
            faults += 1
            if len(frames) < num_frames:
                frames.append(page)
                queue.append(page)
            else:
                evicted = queue.popleft()
                frames[frames.index(evicted)] = page
                queue.append(page)
        log.append({"page": page, "frames": list(frames), "hit": hit})

    hits = len(reference_string) - faults
    return {
        "faults": faults,
        "hits": hits,
        "hit_rate": hits / len(reference_string),
        "log": log,
    }


# ─────────────────────────────────────────────
#  LRU
# ─────────────────────────────────────────────
def lru(reference_string, num_frames):
    frames = []
    last_used = {}
    faults = 0
    log = []

    for t, page in enumerate(reference_string):
        hit = page in frames
        if not hit:
            faults += 1
            if len(frames) < num_frames:
                frames.append(page)
            else:
                lru_page = min(frames, key=lambda p: last_used.get(p, -1))
                frames[frames.index(lru_page)] = page
        last_used[page] = t
        log.append({"page": page, "frames": list(frames), "hit": hit})

    hits = len(reference_string) - faults
    return {
        "faults": faults,
        "hits": hits,
        "hit_rate": hits / len(reference_string),
        "log": log,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Improvement 1 — Slot-aware + Lookahead State Encoding
# ─────────────────────────────────────────────────────────────────────────────
def encode_state(frames, num_frames, page_range, ref_string, current_idx, lookahead=5):
    """
    State vector = [slot one-hots] + [lookahead one-hots]

    Slot one-hots  : num_frames x page_range floats
                     Each slot gets its own one-hot so the agent knows
                     exactly which page occupies which frame position.

    Lookahead      : lookahead x page_range floats
                     One-hot for each of the next `lookahead` pages in the
                     reference string so the agent can approximate Belady.
    """
    parts = []

    for slot in range(num_frames):
        vec = np.zeros(page_range, dtype=np.float32)
        if slot < len(frames):
            p = frames[slot]
            if 0 <= p < page_range:
                vec[p] = 1.0
        parts.append(vec)

    for i in range(lookahead):
        vec = np.zeros(page_range, dtype=np.float32)
        idx = current_idx + i + 1
        if idx < len(ref_string):
            p = ref_string[idx]
            if 0 <= p < page_range:
                vec[p] = 1.0
        parts.append(vec)

    return np.concatenate(parts)


def get_state_dim(num_frames, page_range, lookahead=5):
    return (num_frames + lookahead) * page_range


# ─────────────────────────────────────────────────────────────────────────────
#  Improvement 2 — Random Reference String Generator
# ─────────────────────────────────────────────────────────────────────────────
def generate_random_ref(length, page_range, locality=0.7):
    """
    Generate a reference string with temporal locality bias.
    locality = probability of staying near the current page.
    """
    pages = list(range(page_range))
    ref = []
    current = random.choice(pages)

    for _ in range(length):
        if random.random() < locality:
            delta = random.choice([-2, -1, 0, 1, 2])
            current = (current + delta) % page_range
        else:
            current = random.choice(pages)
        ref.append(current)

    return ref


# ─────────────────────────────────────────────────────────────────────────────
#  Improvement 3 — Prioritised Experience Replay (PER)
# ─────────────────────────────────────────────────────────────────────────────
class PrioritisedReplayBuffer:
    """
    Transitions with higher TD-error are sampled more often.
    Importance-sampling weights correct the resulting bias.
    alpha : how strongly to prioritise (0=uniform, 1=full priority)
    beta  : IS correction strength, annealed from beta_start to 1.0
    """

    def __init__(self, capacity=4000, alpha=0.6, beta_start=0.4, beta_steps=1000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_steps = beta_steps
        self.beta_step = 0

        self.buf = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def push(self, transition, td_error=1.0):
        priority = (abs(td_error) + 1e-5) ** self.alpha
        if self.size < self.capacity:
            self.buf.append(transition)
        else:
            self.buf[self.pos] = transition
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        probs = self.priorities[: self.size]
        probs = probs / probs.sum()
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)

        self.beta_step += 1
        frac = min(1.0, self.beta_step / self.beta_steps)
        self.beta = self.beta_start + frac * (1.0 - self.beta_start)

        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        batch = [self.buf[i] for i in indices]
        s, a, r, s_, d = zip(*batch)
        return (
            np.array(s, dtype=np.float32),
            np.array(a, dtype=np.int32),
            np.array(r, dtype=np.float32),
            np.array(s_, dtype=np.float32),
            np.array(d, dtype=np.float32),
            weights.astype(np.float32),
            indices,
        )

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = (abs(err) + 1e-5) ** self.alpha

    def __len__(self):
        return self.size


class _UniformReplayBuffer:
    """Fallback when PER is disabled."""

    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, transition, td_error=None):
        self.buf.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s_, d = zip(*batch)
        n = len(batch)
        return (
            np.array(s, dtype=np.float32),
            np.array(a, dtype=np.int32),
            np.array(r, dtype=np.float32),
            np.array(s_, dtype=np.float32),
            np.array(d, dtype=np.float32),
            np.ones(n, dtype=np.float32),
            list(range(n)),
        )

    def update_priorities(self, indices, td_errors):
        pass

    def __len__(self):
        return len(self.buf)


# ─────────────────────────────────────────────────────────────────────────────
#  Improvement 4 — Dueling Network Architecture
# ─────────────────────────────────────────────────────────────────────────────
class DuelingDQNNetwork:
    """
    Shared hidden layer splits into:
      V(s)    - scalar value of being in this state
      A(s,a)  - advantage of each action

    Q(s,a) = V(s) + A(s,a) - mean_a[A(s,a)]

    The mean subtraction ensures identifiability: V and A
    can't just shift each other arbitrarily.
    """

    def __init__(self, input_dim, hidden_dim, action_dim, lr=0.001):
        self.lr = lr
        self.action_dim = action_dim

        self.W_shared = np.random.randn(input_dim, hidden_dim) * np.sqrt(
            2.0 / input_dim
        )
        self.b_shared = np.zeros(hidden_dim)

        self.W_val = np.random.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)
        self.b_val = np.zeros(1)

        self.W_adv = np.random.randn(hidden_dim, action_dim) * np.sqrt(2.0 / hidden_dim)
        self.b_adv = np.zeros(action_dim)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_d(self, x):
        return (x > 0).astype(np.float32)

    def forward(self, x):
        self.x = x
        self.z1 = x @ self.W_shared + self.b_shared
        self.h = self.relu(self.z1)
        self.val = self.h @ self.W_val + self.b_val  # (batch, 1)
        self.adv = self.h @ self.W_adv + self.b_adv  # (batch, actions)
        q = self.val + self.adv - self.adv.mean(axis=1, keepdims=True)
        return q

    def backward(self, grad_q, weights=None):
        batch = grad_q.shape[0]
        if weights is None:
            weights = np.ones(batch, dtype=np.float32)
        w = weights[:, None]

        n = self.action_dim
        grad_val = (grad_q * w).sum(axis=1, keepdims=True)
        grad_adv = (
            grad_q * w * (1 - 1 / n) - (grad_q * w).sum(axis=1, keepdims=True) / n
        )

        dW_val = self.h.T @ grad_val
        db_val = grad_val.sum(axis=0)
        dh_val = grad_val @ self.W_val.T

        dW_adv = self.h.T @ grad_adv
        db_adv = grad_adv.sum(axis=0)
        dh_adv = grad_adv @ self.W_adv.T

        dh = dh_val + dh_adv
        dz1 = dh * self.relu_d(self.z1)
        dW_sh = self.x.T @ dz1
        db_sh = dz1.sum(axis=0)

        self.W_shared -= self.lr * dW_sh / batch
        self.b_shared -= self.lr * db_sh / batch
        self.W_val -= self.lr * dW_val / batch
        self.b_val -= self.lr * db_val / batch
        self.W_adv -= self.lr * dW_adv / batch
        self.b_adv -= self.lr * db_adv / batch

    def copy_weights_from(self, other):
        self.W_shared = other.W_shared.copy()
        self.b_shared = other.b_shared.copy()
        self.W_val = other.W_val.copy()
        self.b_val = other.b_val.copy()
        self.W_adv = other.W_adv.copy()
        self.b_adv = other.b_adv.copy()


# ─────────────────────────────────────────────────────────────────────────────
#  Improvement 5 — Shaped Reward
# ─────────────────────────────────────────────────────────────────────────────
def shaped_reward(evicted_page, ref_string, current_idx):
    """
    Reward the quality of the eviction choice:
      page never used again  ->  0.0   (perfect, matches Belady)
      page used far away     ->  close to 0.0
      page used very soon    ->  close to -1.0
    Range is always [-1, 0], so it's compatible with hit reward of +1.
    """
    future = ref_string[current_idx + 1 :]
    if evicted_page not in future:
        return 0.0
    dist = future.index(evicted_page)
    return (dist / max(len(future), 1)) - 1.0


# ─────────────────────────────────────────────────────────────────────────────
#  Improved DQN Agent
# ─────────────────────────────────────────────────────────────────────────────
class DQNAgent:
    def __init__(
        self,
        num_frames,
        page_range,
        hidden_dim=128,
        lr=0.001,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.97,
        batch_size=64,
        target_update=10,
        buffer_size=4000,
        lookahead=5,
        use_per=True,
        use_shaped_reward=True,
        use_random_strings=True,
        locality=0.7,
    ):
        self.num_frames = num_frames
        self.page_range = page_range
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.lookahead = lookahead
        self.use_per = use_per
        self.use_shaped_reward = use_shaped_reward
        self.use_random_strings = use_random_strings
        self.locality = locality

        sdim = get_state_dim(num_frames, page_range, lookahead)
        action_dim = num_frames

        self.online_net = DuelingDQNNetwork(sdim, hidden_dim, action_dim, lr)
        self.target_net = DuelingDQNNetwork(sdim, hidden_dim, action_dim, lr)
        self.target_net.copy_weights_from(self.online_net)

        self.replay = (
            PrioritisedReplayBuffer(buffer_size)
            if use_per
            else _UniformReplayBuffer(buffer_size)
        )

        self.step_count = 0
        self.episode_rewards = []
        self.td_errors_log = []

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_frames - 1)
        q = self.online_net.forward(state.reshape(1, -1))[0]
        return int(np.argmax(q))

    def _train_step(self):
        if len(self.replay) < self.batch_size:
            return None

        s, a, r, s_, d, is_weights, indices = self.replay.sample(self.batch_size)

        q_curr = self.online_net.forward(s)
        q_next_online = self.online_net.forward(s_)
        q_next_target = self.target_net.forward(s_)
        best_a = np.argmax(q_next_online, axis=1)
        q_next = q_next_target[np.arange(self.batch_size), best_a]

        targets = q_curr.copy()
        td_errors = np.zeros(self.batch_size, dtype=np.float32)

        for i in range(self.batch_size):
            td_target = r[i] + self.gamma * q_next[i] * (1 - d[i])
            td_errors[i] = td_target - q_curr[i, a[i]]
            targets[i, a[i]] = td_target

        if self.use_per:
            self.replay.update_priorities(indices, td_errors)

        loss_grad = 2 * (q_curr - targets)
        self.online_net.backward(loss_grad, weights=is_weights)

        return float(np.mean(td_errors**2))

    def fit(self, reference_string, num_episodes=300, progress_cb=None):
        ref_len = len(reference_string)

        for ep in range(num_episodes):
            if self.use_random_strings:
                ref = generate_random_ref(ref_len, self.page_range, self.locality)
            else:
                ref = reference_string

            frames = []
            total_reward = 0.0
            ep_td_errors = []

            for t, page in enumerate(ref):
                state = encode_state(
                    frames, self.num_frames, self.page_range, ref, t, self.lookahead
                )
                hit = page in frames

                if hit:
                    next_state = encode_state(
                        frames, self.num_frames, self.page_range, ref, t, self.lookahead
                    )
                    self.replay.push((state, 0, 1.0, next_state, False), td_error=0.0)
                    total_reward += 1.0
                    continue

                if len(frames) < self.num_frames:
                    frames.append(page)
                    next_state = encode_state(
                        frames, self.num_frames, self.page_range, ref, t, self.lookahead
                    )
                    self.replay.push((state, 0, -1.0, next_state, False), td_error=1.0)
                    total_reward -= 1.0
                else:
                    action = self.select_action(state)
                    evicted = frames[action]

                    reward = (
                        shaped_reward(evicted, ref, t)
                        if self.use_shaped_reward
                        else -1.0
                    )

                    frames[action] = page
                    next_state = encode_state(
                        frames, self.num_frames, self.page_range, ref, t, self.lookahead
                    )

                    q_now = self.online_net.forward(state.reshape(1, -1))[0, action]
                    q_next = self.target_net.forward(next_state.reshape(1, -1))[0].max()
                    td_err = abs(reward + self.gamma * q_next - q_now)

                    self.replay.push(
                        (state, action, reward, next_state, False), td_error=td_err
                    )
                    total_reward += reward

                loss = self._train_step()
                if loss is not None:
                    ep_td_errors.append(loss)
                self.step_count += 1

            self.episode_rewards.append(total_reward)
            self.td_errors_log.append(
                float(np.mean(ep_td_errors)) if ep_td_errors else 0.0
            )
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            if ep % self.target_update == 0:
                self.target_net.copy_weights_from(self.online_net)

            if progress_cb:
                progress_cb(ep + 1, num_episodes, total_reward, self.epsilon)

        return self.episode_rewards

    def run_inference(self, reference_string):
        frames = []
        faults = 0
        log = []

        for t, page in enumerate(reference_string):
            state = encode_state(
                frames,
                self.num_frames,
                self.page_range,
                reference_string,
                t,
                self.lookahead,
            )
            hit = page in frames

            if not hit:
                faults += 1
                if len(frames) < self.num_frames:
                    frames.append(page)
                else:
                    q = self.online_net.forward(state.reshape(1, -1))[0]
                    action = int(np.argmax(q))
                    frames[action] = page

            log.append({"page": page, "frames": list(frames), "hit": hit})

        hits = len(reference_string) - faults
        return {
            "faults": faults,
            "hits": hits,
            "hit_rate": hits / len(reference_string),
            "log": log,
        }
