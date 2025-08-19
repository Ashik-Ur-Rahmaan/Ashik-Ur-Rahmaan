import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute temperature gradients and thermal conductivity from a LAMMPS "
            "chunk-averaged profile, with manual inputs for heat flux or power/area."
        )
    )

    # I/O
    parser.add_argument("--file", dest="file_path", type=str, default="profile1.mp",
                        help="Input LAMMPS chunk-averaged profile file path")
    parser.add_argument("--out-csv", dest="output_csv", type=str, default="temperature_gradients.csv",
                        help="Output CSV for per-bin results (gradients and k)")

    # Steady-state selection and bin trimming
    parser.add_argument("--steady-threshold", type=float, default=2e6,
                        help="Timesteps >= this are considered steady-state")
    parser.add_argument("--exclude-bins", type=int, default=2,
                        help="Number of bins to exclude at each edge")

    # Coordinate unit scaling
    parser.add_argument("--coord-unit-m", type=float, default=1.0,
                        help=(
                            "Meters per coordinate unit in the input profile (e.g., if coords are in nm, set 1e-9)"
                        ))

    # Heat flux / power input
    parser.add_argument("--heat-flux-q", type=float, default=None,
                        help="Heat flux q in W/m^2. Signed by default unless --q-is-magnitude is set")
    parser.add_argument("--q-is-magnitude", action="store_true",
                        help="Interpret --heat-flux-q as a magnitude; use --flux-dir for sign")
    parser.add_argument("--power-w", type=float, default=None,
                        help="Total heating/cooling power in Watts")
    parser.add_argument("--area-m2", type=float, default=None,
                        help="Cross-sectional area in m^2 (used with --power-w to compute flux)")
    parser.add_argument("--flux-dir", type=str, choices=["+", "-"], default=None,
                        help="Direction of heat flow along the coordinate axis: '+' or '-' (used if magnitude is provided)")

    # Which profile to use for reporting
    parser.add_argument("--use-profile", type=str, choices=["avg", "last", "both"], default="both",
                        help="Report conductivity using steady-state average, last timestep, or both")
    parser.add_argument("--no-plots", action="store_true", help="Disable plotting")

    return parser.parse_args()


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


def determine_heat_flux(args: argparse.Namespace) -> float:
    # If q is provided
    if args.heat_flux_q is not None:
        if args.q_is_magnitude:
            if args.flux_dir is None:
                # Interactive fallback
                resp = input("Heat flux direction along coord ('+' or '-') [required]: ").strip()
                if resp not in {"+", "-"}:
                    raise ValueError("Invalid flux direction input")
                direction = 1.0 if resp == "+" else -1.0
            else:
                direction = 1.0 if args.flux_dir == "+" else -1.0
            return direction * float(args.heat_flux_q)
        else:
            return float(args.heat_flux_q)

    # Else compute from power and area
    power = args.power_w
    area = args.area_m2
    if power is None:
        resp = input("Total power in W (or leave blank if not using power): ").strip()
        power = float(resp) if resp else None
    if area is None and power is not None:
        resp = input("Cross-sectional area in m^2 (required when power is provided): ").strip()
        area = float(resp) if resp else None
    if power is not None and area is not None:
        q_mag = float(power) / float(area)
        if args.flux_dir is None:
            resp = input("Heat flux direction along coord ('+' or '-') [required]: ").strip()
            if resp not in {"+", "-"}:
                raise ValueError("Invalid flux direction input")
            direction = 1.0 if resp == "+" else -1.0
        else:
            direction = 1.0 if args.flux_dir == "+" else -1.0
        return direction * q_mag

    raise ValueError("Provide --heat-flux-q (optionally with --q-is-magnitude/--flux-dir) or both --power-w and --area-m2.")


def main():
    args = parse_args()

    # 1. Read input profile
    timesteps, coords, T_matrix = read_profile(args.file_path)

    # 2. Steady-state selection
    T_avg, T_last = select_steady_region(timesteps, T_matrix, args.steady_threshold)

    # 3. Gradients (inner bins)
    grad_avg, coords_inner, T_avg_inner = compute_gradient(T_avg, coords, args.exclude_bins)
    grad_last, _, T_last_inner = compute_gradient(T_last, coords, args.exclude_bins)

    # 4. Overall linear fits on inner bins
    coeffs_avg = np.polyfit(coords_inner, T_avg_inner, 1)
    coeffs_last = np.polyfit(coords_inner, T_last_inner, 1)
    overall_grad_avg = coeffs_avg[0]
    overall_grad_last = coeffs_last[0]

    # 5. Convert gradients to per-meter using coord unit scale
    scale_m_per_unit = float(args.coord_unit_m)
    if scale_m_per_unit <= 0:
        raise ValueError("--coord-unit-m must be positive")
    grad_avg_per_m = grad_avg / scale_m_per_unit
    grad_last_per_m = grad_last / scale_m_per_unit
    overall_grad_avg_per_m = overall_grad_avg / scale_m_per_unit
    overall_grad_last_per_m = overall_grad_last / scale_m_per_unit

    # 6. Manual heat flux input
    try:
        q_w_per_m2 = determine_heat_flux(args)
    except Exception as e:
        print(f"Error determining heat flux: {e}")
        sys.exit(1)

    # 7. Thermal conductivity via Fourier's law: q = -k dT/dy
    # Handle potential divide-by-zero gracefully
    with np.errstate(divide='ignore', invalid='ignore'):
        k_local_avg = -q_w_per_m2 / grad_avg_per_m
        k_local_last = -q_w_per_m2 / grad_last_per_m
        k_overall_avg = -q_w_per_m2 / overall_grad_avg_per_m
        k_overall_last = -q_w_per_m2 / overall_grad_last_per_m

    # 8. Print summary
    print(f"(A) Overall gradient (avg steady, per meter): {overall_grad_avg_per_m:.6g} K/m")
    print(f"(B) Overall gradient (last step, per meter): {overall_grad_last_per_m:.6g} K/m")
    if args.use_profile in {"avg", "both"}:
        print(f"(C) Thermal conductivity using avg steady profile: {k_overall_avg:.6g} W/m-K")
    if args.use_profile in {"last", "both"}:
        print(f"(D) Thermal conductivity using last timestep profile: {k_overall_last:.6g} W/m-K")

    # 9. Save per-bin results (inner bins only)
    header = [
        "coord_unit", "coord_m", "T_avg", "T_last", "grad_avg_K_per_m", "grad_last_K_per_m",
        "k_local_avg_W_per_mK", "k_local_last_W_per_mK"
    ]
    with open(args.output_csv, "w", newline="") as f:
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
    print(f"Per-bin results saved to {args.output_csv}")

    # 10. Plots (inner bins only)
    if not args.no_plots:
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

