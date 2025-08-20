import numpy as np
import pandas as pd
import os, sys, torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from filterpy.kalman import ExtendedKalmanFilter# as EKF
from filterpy.kalman import KalmanFilter
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset


TRAIN_SPLIT = 0.7


# Load CSV file
def load_data(csv_file, column_name):
    """Load CSV file and extract a specified column as a NumPy array."""
    df = pd.read_csv(csv_file)
    return df[column_name].values

# Define Kalman Filter model
def initialize_kalman_filter(Q, R):
    """Initialize a simple Kalman Filter model."""
    kf = KalmanFilter(dim_x=2, dim_z=1)  # 2 state variables, 1 measurement variable

    # State transition matrix
    kf.F = np.array([[1, 1], [0, 1]])  # Constant velocity model

    # Measurement function
    kf.H = np.array([[1, 0]])  # We only measure the first state (position)

    # Process noise covariance (Q)
    kf.Q = np.array([[Q, 0], [0, Q]])

    # Measurement noise covariance (R)
    kf.R = np.array([[R]])

    # Initial state estimate
    kf.x = np.array([[0], [0]])  # Initial position and velocity

    # Initial uncertainty
    kf.P = np.eye(2) * 1.0

    return kf


# Apply Kalman Filter for inference
def apply_kalman_filter(data, Q, R):
    """Run the Kalman filter on the full dataset using learned Q and R."""
    kf = initialize_kalman_filter(Q, R)
    estimates = []

    for z in data:
        kf.predict()
        kf.update(z)
        estimates.append(ekf.x[0, 0])  # Store filtered position estimate

    return np.array(estimates)



# Apply Kalman Filter for inference
def apply_ekf_filter(data, Q, R, dt=1):
    """Run the Kalman filter on the full dataset using learned Q and R."""
    #kf = initialize_kalman_filter(Q, R)
    ekf = initialize_ekf(Q, R, dt)  # Initialize with learned parameters
    estimates = []

    for z in data:
        #ekf.predict()
        #ekf.update(z, HJacobian=H_jacobian)
        ekf.F = F_jacobian(ekf.x, dt)
        ekf.H = H_jacobian(ekf.x)
        ekf.predict_update(z, H_jacobian, h_x)  # EKF update
        estimates.append(ekf.x[0, 0])  # Store filtered position estimate

    return np.array(estimates)

# Define LSTM model
class LSTMKalmanTuner(nn.Module):
    """LSTM model to predict Q and R parameters for the Kalman Filter."""
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(LSTMKalmanTuner, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)  # Output Q and R

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # Take the last output of the LSTM
        Q, R = torch.exp(output[:, 0]), torch.exp(output[:, 1])  # Ensure positivity
        return Q, R


# Create sequences for training
def create_sequences(data, seq_length=10):
    """Convert time series data into sequences for training the LSTM."""
    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        targets.append(data[i+seq_length])  # Predict next step
    return np.array(sequences), np.array(targets)


def kalman_loss(Q_pred, R_pred, sequences, targets):
    """Compute the Kalman filter loss using predicted Q and R."""
    loss = torch.zeros(1, requires_grad=True, dtype=torch.float32)  # ✅ Ensure loss tracks gradients

    batch_size = sequences.shape[0]

    for i in range(batch_size):
        kf = initialize_kalman_filter(Q_pred[i].item(), R_pred[i].item())  # Initialize with learned parameters

        for t in range(sequences.shape[1]):  # Run through the sequence
            kf.predict()
            kf.update(sequences[i, t])

        # Compute squared error between Kalman Filter output and target
        kf.predict()  # One more prediction step
        error = (kf.x[0, 0] - targets[i]) ** 2  # Squared error
        loss = loss + error  # ✅ Accumulate loss while maintaining gradients

    return loss / batch_size  # ✅ Keep it inside computation graph


def train_lstm1(data, seq_length=10, epochs=100, batch_size=16, lr=0.001):
    """Train an LSTM model to learn Q and R parameters from time-series data."""

    # Prepare data
    sequences, targets = create_sequences(data, seq_length)
    sequences = torch.tensor(sequences, dtype=torch.float32).unsqueeze(-1)  # Add feature dimension
    targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)
    train_loader = DataLoader(TensorDataset(sequences, targets), batch_size=batch_size, shuffle=True)

    # Initialize model
    model = LSTMKalmanTuner()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for seq_batch, target_batch in train_loader:
            optimizer.zero_grad()
            Q_pred, R_pred = model(seq_batch)  # Get learned Q and R
            loss = kalman_loss(Q_pred, R_pred, seq_batch.squeeze(-1), target_batch.squeeze(-1))  # Compute loss
            loss.backward()  # ✅ Now it works because loss tracks gradients
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    return model



# Nonlinear state transition function
def f_x(x, dt):
    """Nonlinear state transition function (Example: Quadratic Acceleration)."""
    return np.array([[x[0, 0] + x[1, 0] * dt + 0.5 * x[2, 0] * dt**2],  # Position
                     [x[1, 0] + x[2, 0] * dt],  # Velocity
                     [x[2, 0]]])  # Acceleration (constant)

# Jacobian of f_x (State transition function)
def F_jacobian(x, dt):
    """Jacobian matrix for the nonlinear state transition function."""
    return np.array([[1, dt, 0.5 * dt**2],
                     [0, 1, dt],
                     [0, 0, 1]])

# Measurement function (Linear in this case)
def h_x(x):
    """Measurement function: Extracts position from the state."""
    return np.array([[x[0, 0]]])  # We only measure position

# Jacobian of h_x (Measurement function)
def H_jacobian(x):
    """Jacobian matrix for the measurement function."""
    return np.array([[1, 0, 0]])

# Define Extended Kalman Filter (EKF)
def initialize_ekf(Q, R, dt=1):
    """Initialize an Extended Kalman Filter (EKF)."""
    ekf = ExtendedKalmanFilter(dim_x=3, dim_z=1)  # 3 state variables (position, velocity, acceleration), 1 measurement

    # Initial state estimate
    ekf.x = np.array([[0], [0], [0]])  # Initial position, velocity, acceleration

    # Process noise covariance
    ekf.Q = np.eye(3) * Q  # Learned from LSTM

    # Measurement noise covariance
    ekf.R = np.array([[R]])  # Learned from LSTM

    # Initial state transition matrix (updated dynamically)
    ekf.F = F_jacobian(ekf.x, dt)

    # Initial measurement function (updated dynamically)
    ekf.H = H_jacobian(ekf.x)

    return ekf

# Compute EKF loss (Prediction error)
def ekf_loss(Q_pred, R_pred, sequences, targets, dt=1):
    """Compute the EKF loss using predicted Q and R."""
    loss = torch.zeros(1, requires_grad=True, dtype=torch.float32)  # ✅ Ensure loss tracks gradients

    batch_size = sequences.shape[0]

    for i in range(batch_size):
        ekf = initialize_ekf(Q_pred[i].item(), R_pred[i].item(), dt)  # Initialize with learned parameters

        for t in range(sequences.shape[1]):  # Run through the sequence
            ekf.F = F_jacobian(ekf.x, dt)
            ekf.H = H_jacobian(ekf.x)
            ekf.predict_update(sequences[i, t], H_jacobian, h_x)  # EKF update

        # Compute squared error between EKF output and target
        ekf.predict()  # One more prediction step
        error = (ekf.x[0, 0] - targets[i]) ** 2  # Squared error
        loss = loss + error  # ✅ Accumulate loss while maintaining gradients

    return loss / batch_size  # ✅ Keep it inside computation graph

# Train LSTM model
def train_lstm(data, seq_length=10, epochs=300, batch_size=16, lr=0.001):
    """Train an LSTM model to learn Q and R parameters from time-series data."""

    # Prepare data
    sequences, targets = create_sequences(data, seq_length)
    sequences = torch.tensor(sequences, dtype=torch.float32).unsqueeze(-1)  # Add feature dimension
    targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)
    train_loader = DataLoader(TensorDataset(sequences, targets), batch_size=batch_size, shuffle=True)

    # Initialize model
    model = LSTMKalmanTuner()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for seq_batch, target_batch in train_loader:
            optimizer.zero_grad()
            Q_pred, R_pred = model(seq_batch)  # Get learned Q and R
            loss = ekf_loss(Q_pred, R_pred, seq_batch.squeeze(-1), target_batch.squeeze(-1))  # Compute loss
            loss.backward()  # ✅ Now it works because loss tracks gradients
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    return model



# Apply trained LSTM to predict Q and R
def apply_lstm_kalman_filter(model, data, seq_length=10, skip_lstm=False):
    """Use trained LSTM model to predict Q and R for each time step and apply the Kalman Filter."""
    if skip_lstm:
        filtered_data = apply_ekf_filter(data[seq_length:], 1, 1) 
        return filtered_data, 1, 1
    sequences, _ = create_sequences(data, seq_length)
    sequences = torch.tensor(sequences, dtype=torch.float32).unsqueeze(-1)

    model.eval()
    with torch.no_grad():
        Q_pred, R_pred = model(sequences)

    # Convert to NumPy
    Q_values, R_values = Q_pred.numpy(), R_pred.numpy()

    print(f"Q = {Q_values.mean()}, R = {R_values.mean()}")
    # Apply Kalman filter with learned parameters
    filtered_data = apply_ekf_filter(data[seq_length:], Q_values.mean(), R_values.mean())
    
    return filtered_data, Q_values, R_values


def summary_stats(data, filtered_data, tag=""):
    err = np.subtract(data[-len(filtered_data)+1:], filtered_data[:-1])
    mae = np.square(err).mean() #np.abs(err).mean()
    var = np.square(err-mae).std()
    print(f"mae = {mae}, std = {var} {tag}")
    return mae, var



class ClearTrack:
    def __init__(self, seq_length=10, epochs=100, batch_size=16, lr=0.001, sliding_window_size=30):
        self.seq_length = seq_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model = None
        self.trained = False
        self.X = None
        self.y = None
        self.sliding_window_size = sliding_window_size
        self.filtered_data = None
        self.Q_values = None
        self.R_values = None

    def update(self, X, y):
        """Update the training data and re-train the LSTM filter."""
        if self.X is None:
            self.X = X
            self.y = y
        else:
            self.X = np.vstack((self.X, X))
            self.y = np.append(self.y, y)

        if len(self.y) > self.seq_length + 1:
            # Apply sliding window
            self.X = self.X[-self.sliding_window_size:]
            self.y = self.y[-self.sliding_window_size:]

            self.model = train_lstm(
                np.array(self.y),
                seq_length=self.seq_length,
                epochs=self.epochs,
                batch_size=self.batch_size,
                lr=self.lr
            )
            self.trained = True

    def predict(self, X):
        """Predict using the trained LSTM-based filter."""
        if not self.trained or self.y is None or len(self.y) <= self.seq_length:
            return np.zeros(X.shape[0]), np.ones(X.shape[0])  # mimic prior mean, std

        # Filter predictions for each row in X using the full sequence
        filtered, Qs, Rs = apply_lstm_kalman_filter(
            self.model,
            np.array(self.y),
            seq_length=self.seq_length
        )
        self.filtered_data = filtered
        self.Q_values = Qs
        self.R_values = Rs

        mean = np.array([filtered[-1]] * X.shape[0])
        std = np.array([np.sqrt(Qs[-1])] * X.shape[0]) if Qs else np.ones(X.shape[0])
        return mean, std

    def reset(self):
        self.model = None
        self.trained = False
        self.X = None
        self.y = None
        self.filtered_data = None
        self.Q_values = None
        self.R_values = None




def test_clear_track():
    global TRAIN_SPLIT
    f = os.path.join(sys.path[0],'..','data','twitter_trace.csv')
    f = sys.argv[sys.argv.index("-f")+1] if "-f" in sys.argv else f
    xcol = sys.argv[sys.argv.index("-x")+1] if "-x" in sys.argv else 'Tweets 09-May-2023'
    ycol = sys.argv[sys.argv.index("-y")+1] if "-y" in sys.argv else 'Tweets 09-May-2023'
    L = float(sys.argv[sys.argv.index("-l")+1]) if "-l" in sys.argv else 0.0001
    E = int(sys.argv[sys.argv.index("-e")+1]) if "-e" in sys.argv else 10
    print("test_clear_track().f, x, y, L, E =", f, xcol, ycol, L, E)
 
    # Load CSV file
    csv_file = f  # Change this to your actual file
    column_name = xcol  # Change this to your actual column name
    
    data = load_data(csv_file, column_name)
    TRAIN_SPLIT = 0.4

    # Train LSTM to learn Q and R using Kalman loss
    print("Training LSTM to learn Kalman Filter parameters...")
    print(f"data.len = {len(data)}, split = {int(TRAIN_SPLIT*len(data))}")
    lstm_model = train_lstm(data[0:int(TRAIN_SPLIT*len(data))], seq_length=10, epochs=E, batch_size=16, lr=L)

    # Apply LSTM-trained Kalman Filter
    print("Applying LSTM-based Kalman Filter...")
    filtered_data, Q_values, R_values = apply_lstm_kalman_filter(lstm_model, data[int(TRAIN_SPLIT*len(data)):], seq_length=10)
    #lstm_model = train_lstm(data[0:int(TRAIN_SPLIT*len(data))], seq_length=10, epochs=100, batch_size=16, lr=0.001)
    maes = [[], []]
    for i in range(10):
        print(f"computing maes = {maes}, split = {TRAIN_SPLIT}")
        maes[0].append(summary_stats(data, apply_lstm_kalman_filter(lstm_model, data, seq_length=10, skip_lstm=False)[0], tag="(LSTM)")[0])
        maes[1].append(summary_stats(data, apply_lstm_kalman_filter(lstm_model, data, seq_length=10, skip_lstm=True)[0], tag="(No LSTM)")[0])
    print("mse.delta,l = ", np.array(maes[0]).mean()-np.array(maes[1]).mean(),L)
    # Plot results
    mae_fig = plt.figure(figsize=(10, 5))
    plt.plot(np.subtract(np.array(maes[0]), np.array(maes[1])), label="LSTM MAE  - None-LSTM MAE")
    plt.title("LSTM MAE - Non-LSTM MAE")
    plt.xlabel("Training data split")
    plt.ylabel("CPU MAE Delta")
    mae_fig.savefig("ksurf_plus_lstm_vs_non_lstm.png")
    print("Saved figure to ksurf_plus_lstm_vs_non_lstm.png")
    #plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(data[int(TRAIN_SPLIT*len(data)):], label="Raw Data", alpha=0.5)
    plt.plot(np.arange(10+1, len(filtered_data)+10+1), filtered_data, label="LSTM-Kalman Output", linewidth=2)
    plt.xlabel("Time Step")
    plt.ylabel(column_name)
    plt.legend()
    plt.title("LSTM-Kalman Filter Performance on CPU Usage")
    #plt.show()


# --- NEW IMPORTS (safe to add near the top) ---
import json, time, math, datetime as dt
from typing import Dict, List, Optional, Tuple

try:
    import requests
except Exception:  # fallback if requests is unavailable
    requests = None
    import urllib.request
    import urllib.parse

# ----------------------------------------------------------------------
# NEW: Low-latency Prometheus HTTP client (query_range)
# ----------------------------------------------------------------------
def prometheus_query_range(
    base_url: str,
    promql: str,
    start: dt.datetime,
    end: dt.datetime,
    step: str = "5s",
    timeout: int = 20,
    bearer_token: Optional[str] = None,
    verify_tls: bool = True,
    extra_headers: Optional[Dict[str, str]] = None,
) -> List[Tuple[float, float]]:
    """
    Fetch a time series via Prometheus HTTP API /api/v1/query_range.

    Returns: list of (timestamp_seconds, value_float).
    """
    # Normalize URL
    base = base_url.rstrip("/")
    url = f"{base}/api/v1/query_range"

    params = dict(
        query=promql,
        start=str(int(start.timestamp())),
        end=str(int(end.timestamp())),
        step=step,
    )
    headers = {"Accept": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"

    def _parse_json(raw: bytes) -> List[Tuple[float, float]]:
        import json as _json
        payload = _json.loads(raw.decode("utf-8"))
        if payload.get("status") != "success":
            raise RuntimeError(f"Prometheus error: {payload.get('error', 'unknown')}")
        result = payload.get("data", {}).get("result", [])
        if not result:
            return []
        # Use first series (you can aggregate in PromQL to ensure single series)
        series = result[0]["values"]  # [ [ts, "value"], ... ]
        out = []
        for t, v in series:
            try:
                out.append((float(t), float(v)))
            except Exception:
                # skip NaN/Inf or parse errors
                pass
        return out

    if requests is not None:
        with requests.Session() as s:
            s.headers.update(headers)
            resp = s.get(url, params=params, timeout=timeout, verify=verify_tls)
            resp.raise_for_status()
            return _parse_json(resp.content)
    else:
        # urllib fallback
        qs = urllib.parse.urlencode(params)
        req = urllib.request.Request(f"{url}?{qs}", headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return _parse_json(r.read())


# ----------------------------------------------------------------------
# NEW: Build a dataframe from multiple queries and align on timestamps.
# ----------------------------------------------------------------------
def _align_series(
    series_map: Dict[str, List[Tuple[float, float]]]
) -> "pd.DataFrame":
    """
    series_map: name -> list[(ts, val)]  (ts in seconds)
    Returns a DataFrame with columns: ['ts', <names...>] aligned on ts.
    """
    frames = []
    for name, seq in series_map.items():
        if not seq:
            continue
        df = pd.DataFrame(seq, columns=["ts", name])
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["ts"])

    df = frames[0]
    for f in frames[1:]:
        df = pd.merge(df, f, on="ts", how="outer")
    df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    return df


# ----------------------------------------------------------------------
# NEW: Quick Q,R estimation from filtered signal (residual-based)
# ----------------------------------------------------------------------
def _estimate_qr_from_series(
    y: np.ndarray, filtered: np.ndarray
) -> Tuple[float, float]:
    """
    Heuristic estimators:
      - R_hat: variance of measurement residuals (y - filtered) on alignment
      - Q_hat: variance of increments of filtered state (diff)
    Safeguards non-negativity.
    """
    if len(filtered) == 0 or len(y) < 2:
        return 1.0, 1.0
    # Align: filtered is typically produced with some lookback; use tail match
    m = min(len(y), len(filtered))
    y_tail = y[-m:]
    f_tail = filtered[-m:]
    res = y_tail - f_tail
    R_hat = float(np.nan_to_num(np.var(res), nan=1.0, posinf=1.0, neginf=1.0))
    inc = np.diff(f_tail)
    Q_hat = float(np.nan_to_num(np.var(inc), nan=1.0, posinf=1.0, neginf=1.0))
    # Avoid zero
    R_hat = max(R_hat, 1e-9)
    Q_hat = max(Q_hat, 1e-9)
    return Q_hat, R_hat


# ----------------------------------------------------------------------
# NEW: Main ingestion + tuning entry point (standalone)
# ----------------------------------------------------------------------
def collect_prometheus_and_tune(
    base_url: str,
    x_queries: List[str],
    y_query: str,
    start: Optional[dt.datetime] = None,
    end: Optional[dt.datetime] = None,
    step: str = "5s",
    csv_path: str = "ksurf_metrics.csv",
    qr_path: str = "ksurf_qr.json",
    bearer_token: Optional[str] = None,
    verify_tls: bool = True,
    min_points: int = 300,
    seq_length: int = 10,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pull metrics from Prometheus, write/append CSV, and when enough data exists:
    train the LSTM to get Q,R; also compute residual-based Q,R; store matrices.

    Returns:
        X (shape [n, k]) and y (shape [n,]) ready for Ksurf.update(X, y).
    """
    # Time window defaults: last 1 hour
    now = dt.datetime.utcnow() if end is None else end
    if start is None:
        start = now - dt.timedelta(hours=1)

    # Fetch y
    y_series = prometheus_query_range(
        base_url, y_query, start, now, step,
        bearer_token=bearer_token, verify_tls=verify_tls
    )
    # Fetch X features
    series_map = {"y": y_series}
    for i, q in enumerate(x_queries):
        seq = prometheus_query_range(
            base_url, q, start, now, step,
            bearer_token=bearer_token, verify_tls=verify_tls
        )
        series_map[f"x{i}"] = seq

    # Align and clean
    df = _align_series(series_map)
    if df.empty:
        print("[prom] no data returned for the given window")
        return np.zeros((0, len(x_queries))), np.zeros((0,))

    # Drop rows with missing y, and forward-fill X if needed
    df = df.dropna(subset=["y"]).reset_index(drop=True)
    x_cols = [c for c in df.columns if c.startswith("x")]
    if x_cols:
        df[x_cols] = df[x_cols].fillna(method="ffill").fillna(method="bfill")

    # Convert from epoch seconds to pandas datetime for the CSV
    df["ts_iso"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    # Order columns for CSV and append
    csv_cols = ["ts", "ts_iso"] + x_cols + ["y"]
    if not os.path.exists(csv_path):
        df[csv_cols].to_csv(csv_path, index=False)
    else:
        df[csv_cols].to_csv(csv_path, index=False, mode="a", header=False)

    # Build X, y arrays for Ksurf API
    X = df[x_cols].values.astype(np.float32) if x_cols else np.zeros((len(df), 0), dtype=np.float32)
    y = df["y"].values.astype(np.float32)

    # If enough points, train the LSTM (uses functions already in this file)
    if len(y) >= max(min_points, seq_length + 2):
        # Train on all y for stability; you can also use a split if desired
        model = train_lstm(
            data=y, seq_length=seq_length,
            epochs=epochs, batch_size=batch_size, lr=lr
        )
        # Apply to get filtered and (Q,R) predictions
        filtered, Q_vals, R_vals = apply_lstm_kalman_filter(model, y, seq_length=seq_length)
        # LSTM-derived scalars (geomean/mean are both reasonable; use mean)
        Q_lstm = float(np.asarray(Q_vals).mean()) if np.size(Q_vals) else 1.0
        R_lstm = float(np.asarray(R_vals).mean()) if np.size(R_vals) else 1.0
        # Residual-based fallback/confirmation
        Q_resid, R_resid = _estimate_qr_from_series(y, filtered)

        # Matrix forms for EKF (3x3 for your initialize_ekf, 1x1 for R)
        Q_mat = (np.eye(3) * Q_lstm).tolist()
        R_mat = np.array([[R_lstm]], dtype=float).tolist()

        payload = dict(
            timestamp=dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            n_samples=int(len(y)),
            seq_length=int(seq_length),
            epochs=int(epochs),
            step=str(step),
            Q_scalar_lstm=Q_lstm,
            R_scalar_lstm=R_lstm,
            Q_scalar_residual=Q_resid,
            R_scalar_residual=R_resid,
            Q_matrix_ekf=Q_mat,
            R_matrix_ekf=R_mat,
            csv_path=os.path.abspath(csv_path),
        )
        with open(qr_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[prom] wrote Q/R to {os.path.abspath(qr_path)}")
    else:
        print(f"[prom] collected {len(y)} points (<{min_points}); skipping LSTM for now.")

    return X, y


# ----------------------------------------------------------------------
# OPTIONAL: Example CLI usage (keeps your existing main intact)
#   python clear_track.py --prom http://localhost:9090 \
#       --x 'rate(container_cpu_usage_seconds_total[30s])' \
#       --y 'container_memory_working_set_bytes' \
#       --window 3600 --step 5s
# ----------------------------------------------------------------------
if __name__ == "__main__" and "--prom" in sys.argv:
    # Parse minimal CLI args for quick testing
    def _arg(flag, dflt=None):
        return sys.argv[sys.argv.index(flag)+1] if flag in sys.argv else dflt

    base = _arg("--prom", "http://localhost:9090")
    yq = _arg("--y", "node_load1")
    xqs = [_arg("--x")] if "--x" in sys.argv else []
    window = int(_arg("--window", "3600"))
    step = _arg("--step", "5s")
    csv_out = _arg("--csv", "ksurf_metrics.csv")
    qr_out = _arg("--qr", "ksurf_qr.json")

    end = dt.datetime.utcnow()
    start = end - dt.timedelta(seconds=window)

    X, y = collect_prometheus_and_tune(
        base_url=base,
        x_queries=[q for q in xqs if q],
        y_query=yq,
        start=start,
        end=end,
        step=step,
        csv_path=csv_out,
        qr_path=qr_out,
        min_points=300,
        seq_length=10,
        epochs=200,
        batch_size=32,
        lr=1e-3,
    )
    print(f"[prom] collected shapes: X={X.shape}, y={y.shape}")



# Main function
if __name__ == "__main__":
    test_clear_track()
