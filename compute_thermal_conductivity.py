import numpy as np
import matplotlib.pyplot as plt
import csv


# -------------------------------
# USER SETTINGS (edit these directly)
# -------------------------------
FILE_PATH = "profile1.mp"          # Input LAMMPS chunk-averaged profile
OUTPUT_CSV = "temperature_gradients.csv"
STEADY_THRESHOLD = 2e6             # Timesteps >= this are considered steady state
EXCLUDE_BINS = 2                   # Number of bins to exclude at each edge

# Spatial coordinate scaling to meters (e.g., nm -> 1e-9, Angstrom -> 1e-10)
COORD_UNIT_M = 1.0

# Heat flux input (choose ONE path)
# Path 1: Provide q directly in W/m^2 (signed); or as magnitude with direction
HEAT_FLUX_Q = None                 # e.g., -1.2e9 for signed q (None means not provided)
Q_IS_MAGNITUDE = False             # If True, interpret HEAT_FLUX_Q as magnitude
FLUX_DIR = None                    # '+' or '-', used when magnitude is provided or when computing from power

# Path 2: Provide power and area to compute q = P/A
POWER_W = None                     # Total heating/cooling power in W (None if unused)
AREA_M2 = None                     # Cross-sectional area in m^2 (None if unused)

# Reporting/plot options
USE_PROFILE = "both"               # 'avg', 'last', or 'both'
SHOW_PLOTS = True                  # Set to False to disable plots


def read_profile(file_path: str):
    timesteps = []
    profiles = {}
    with open(file_path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue

        parts = line.split()
        timestep = int(parts[0])
        n_chunks = int(parts[1])

        timesteps.append(timestep)
        profiles[timestep] = []

        for j in range(n_chunks):
            chunk_line = lines[i + 1 + j].split()
            coord = float(chunk_line[1])
            v_temp = float(chunk_line[3])
            profiles[timestep].append((coord, v_temp))

        i += n_chunks + 1

    timesteps = np.array(timesteps)
    coords = np.array([c for c, _ in profiles[timesteps[0]]])
    n_bins = len(coords)

    T_matrix = np.zeros((len(timesteps), n_bins))
    for k, ts in enumerate(timesteps):
        T_matrix[k] = [T for _, T in profiles[ts]]

    return timesteps, coords, T_matrix


def select_steady_region(timesteps: np.ndarray, T_matrix: np.ndarray, steady_threshold: float):
    steady_mask = timesteps >= steady_threshold
    if not np.any(steady_mask):
        # Fallback: if no timesteps beyond threshold, use the last 1/3 of frames
        cutoff_index = int(2 * len(timesteps) / 3)
        steady_mask = np.zeros_like(timesteps, dtype=bool)
        steady_mask[cutoff_index:] = True
    T_steady = T_matrix[steady_mask]
    T_avg = np.mean(T_steady, axis=0)
    T_last = T_matrix[-1]
    return T_avg, T_last


def compute_gradient(T: np.ndarray, y: np.ndarray, exclude: int = 2):
    if 2 * exclude >= len(T):
        raise ValueError("exclude is too large relative to the number of bins")
    T_inner = T[exclude:-exclude]
    y_inner = y[exclude:-exclude]
    dTdy = np.zeros_like(T_inner)
    if len(T_inner) < 2:
        raise ValueError("Not enough inner bins to compute gradients")
    if len(T_inner) == 2:
        dTdy[:] = (T_inner[1] - T_inner[0]) / (y_inner[1] - y_inner[0])
    else:
        dTdy[1:-1] = (T_inner[2:] - T_inner[:-2]) / (y_inner[2:] - y_inner[:-2])
        dTdy[0] = (T_inner[1] - T_inner[0]) / (y_inner[1] - y_inner[0])
        dTdy[-1] = (T_inner[-1] - T_inner[-2]) / (y_inner[-1] - y_inner[-2])
    return dTdy, y_inner, T_inner


def determine_heat_flux() -> float:
    # If q is provided directly
    if HEAT_FLUX_Q is not None:
        if Q_IS_MAGNITUDE:
            if FLUX_DIR is None:
                resp = input("Heat flux direction along coord ('+' or '-') [required]: ").strip()
                if resp not in {"+", "-"}:
                    raise ValueError("Invalid flux direction input")
                direction = 1.0 if resp == "+" else -1.0
            else:
                direction = 1.0 if FLUX_DIR == "+" else -1.0
            return direction * float(HEAT_FLUX_Q)
        else:
            return float(HEAT_FLUX_Q)

    # Else compute from power and area
    power = POWER_W
    area = AREA_M2
    if power is None:
        resp = input("Total power in W (or leave blank if not using power): ").strip()
        power = float(resp) if resp else None
    if area is None and power is not None:
        resp = input("Cross-sectional area in m^2 (required when power is provided): ").strip()
        area = float(resp) if resp else None
    if power is not None and area is not None:
        q_mag = float(power) / float(area)
        if FLUX_DIR is None:
            resp = input("Heat flux direction along coord ('+' or '-') [required]: ").strip()
            if resp not in {"+", "-"}:
                raise ValueError("Invalid flux direction input")
            direction = 1.0 if resp == "+" else -1.0
        else:
            direction = 1.0 if FLUX_DIR == "+" else -1.0
        return direction * q_mag

    raise ValueError("Provide HEAT_FLUX_Q (optionally with Q_IS_MAGNITUDE/FLUX_DIR) or both POWER_W and AREA_M2.")


def main():
    # 1. Read input profile
    timesteps, coords, T_matrix = read_profile(FILE_PATH)

    # 2. Steady-state selection
    T_avg, T_last = select_steady_region(timesteps, T_matrix, STEADY_THRESHOLD)

    # 3. Gradients (inner bins)
    grad_avg, coords_inner, T_avg_inner = compute_gradient(T_avg, coords, EXCLUDE_BINS)
    grad_last, _, T_last_inner = compute_gradient(T_last, coords, EXCLUDE_BINS)

    # 4. Overall linear fits on inner bins
    coeffs_avg = np.polyfit(coords_inner, T_avg_inner, 1)
    coeffs_last = np.polyfit(coords_inner, T_last_inner, 1)
    overall_grad_avg = coeffs_avg[0]
    overall_grad_last = coeffs_last[0]

    # 5. Convert gradients to per-meter using coord unit scale
    scale_m_per_unit = float(COORD_UNIT_M)
    if scale_m_per_unit <= 0:
        raise ValueError("COORD_UNIT_M must be positive")
    grad_avg_per_m = grad_avg / scale_m_per_unit
    grad_last_per_m = grad_last / scale_m_per_unit
    overall_grad_avg_per_m = overall_grad_avg / scale_m_per_unit
    overall_grad_last_per_m = overall_grad_last / scale_m_per_unit

    # 6. Manual heat flux input
    q_w_per_m2 = determine_heat_flux()

    # 7. Thermal conductivity via Fourier's law: q = -k dT/dy
    with np.errstate(divide='ignore', invalid='ignore'):
        k_local_avg = -q_w_per_m2 / grad_avg_per_m
        k_local_last = -q_w_per_m2 / grad_last_per_m
        k_overall_avg = -q_w_per_m2 / overall_grad_avg_per_m
        k_overall_last = -q_w_per_m2 / overall_grad_last_per_m

    # 8. Print summary
    print(f"(A) Overall gradient (avg steady, per meter): {overall_grad_avg_per_m:.6g} K/m")
    print(f"(B) Overall gradient (last step, per meter): {overall_grad_last_per_m:.6g} K/m")
    if USE_PROFILE in {"avg", "both"}:
        print(f"(C) Thermal conductivity using avg steady profile: {k_overall_avg:.6g} W/m-K")
    if USE_PROFILE in {"last", "both"}:
        print(f"(D) Thermal conductivity using last timestep profile: {k_overall_last:.6g} W/m-K")

    # 9. Save per-bin results (inner bins only)
    header = [
        "coord_unit", "coord_m", "T_avg", "T_last", "grad_avg_K_per_m", "grad_last_K_per_m",
        "k_local_avg_W_per_mK", "k_local_last_W_per_mK"
    ]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for y_unit, Ta, Tl, ga, gl, ka, kl in zip(
                coords_inner,
                T_avg_inner,
                T_last_inner,
                grad_avg_per_m,
                grad_last_per_m,
                k_local_avg,
                k_local_last,
        ):
            writer.writerow([
                y_unit,
                y_unit * scale_m_per_unit,
                Ta,
                Tl,
                ga,
                gl,
                ka,
                kl,
            ])
    print(f"Per-bin results saved to {OUTPUT_CSV}")

    # 10. Plots (inner bins only)
    if SHOW_PLOTS:
        plt.figure(figsize=(6, 4))
        plt.plot(coords_inner, T_avg_inner, 'o-', label='Average (steady-state)')
        plt.plot(coords_inner, T_last_inner, 's--', label='Last timestep')
        plt.xlabel("Position (coord unit)")
        plt.ylabel("Temperature (K)")
        plt.title("Temperature Profiles (edges excluded)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6, 4))
        plt.plot(coords_inner * scale_m_per_unit, grad_avg_per_m, 'o-', label='dT/dy (avg steady)')
        plt.plot(coords_inner * scale_m_per_unit, grad_last_per_m, 's--', label='dT/dy (last)')
        plt.xlabel("Position (m)")
        plt.ylabel("Temperature Gradient (K/m)")
        plt.title("Local Temperature Gradients (edges excluded)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6, 4))
        plt.plot(coords_inner * scale_m_per_unit, k_local_avg, 'o-', label='k local (avg steady)')
        plt.plot(coords_inner * scale_m_per_unit, k_local_last, 's--', label='k local (last)')
        plt.xlabel("Position (m)")
        plt.ylabel("Thermal Conductivity (W/m-K)")
        plt.title("Local Thermal Conductivity (edges excluded)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

