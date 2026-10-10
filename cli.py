import click
import numpy as np
import matplotlib.pyplot as plt
from solver import AdvectionSolver, profile_step, profile_gaussian, profile_parabola


@click.group()
def cli():
    pass


@cli.command()
@click.option("--output", default="stability_test.png")
def test_stability(output: str):
    solver = AdvectionSolver(L=20.0, N=200, u=1.0)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    schemes = ["upwind", "ftcs", "lax", "lax_wendroff"]
    titles = ["Upwind", "FTCS", "Lax", "Lax-Wendroff"]

    for idx, (scheme, title) in enumerate(zip(schemes, titles)):
        ax = axes[idx // 2, idx % 2]
        
        s_stable, _ = solver.solve(scheme, profile_gaussian, Co=0.8, t_final=3.0)
        s_unstable, _ = solver.solve(scheme, profile_gaussian, Co=1.2, t_final=3.0)
        s_exact = solver.exact_solution(profile_gaussian, t=3.0)

        ax.plot(solver.x, s_exact, "k--", label="Exact", alpha=0.7)
        if not np.any(np.isnan(s_stable)) and np.max(np.abs(s_stable)) < 5.0:
            ax.plot(solver.x, s_stable, "b-", label="Co = 0.8")
        else:
            ax.text(0.5, 0.5, "Unstable", transform=ax.transAxes, color="red")

        if not np.any(np.isnan(s_unstable)) and np.max(np.abs(s_unstable)) < 5.0:
            ax.plot(solver.x, s_unstable, "r-", label="Co = 1.2")
        else:
            ax.plot(solver.x, np.clip(s_unstable, -2.0, 3.0), "r:", label="Co = 1.2 (Unstable)")

        ax.set_title(title, fontsize=12)
        ax.set_ylim(-0.5, 1.8)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=8)

    plt.suptitle("CFL Stability Analysis", fontsize=14)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    click.echo(f"Saved: {output}")


@cli.command()
@click.option("--output-prefix", default="comparison")
def compare_schemes(output_prefix: str):
    solver = AdvectionSolver(L=20.0, N=200, u=1.0)
    
    profiles = {
        "step": (profile_step, "Step"),
        "gaussian": (profile_gaussian, "Gaussian"),
        "parabola": (profile_parabola, "Parabola")
    }
    times = [2.0, 6.0, 12.0]
    Co = 0.75

    schemes = [
        ("upwind", "Upwind", "tab:blue"),
        ("lax", "Lax", "tab:orange"),
        ("lax_wendroff", "Lax-Wendroff", "tab:green")
    ]

    for p_name, (p_fn, p_label) in profiles.items():
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
        for t_idx, t in enumerate(times):
            ax = axes[t_idx]
            s_exact = solver.exact_solution(p_fn, t)
            ax.plot(solver.x, s_exact, "k--", lw=2, label="Exact")

            for s_key, s_title, s_color in schemes:
                s_num, _ = solver.solve(s_key, p_fn, Co=Co, t_final=t)
                ax.plot(solver.x, s_num, color=s_color, label=s_title)

            ax.set_title(f"t = {t} s", fontsize=12)
            ax.set_xlabel("x")
            if t_idx == 0:
                ax.set_ylabel("s(x, t)")
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(fontsize=8)

        plt.suptitle(f"Scheme Comparison (Co = {Co}) - {p_label}", fontsize=14)
        plt.tight_layout()
        fname = f"{output_prefix}_{p_name}.png"
        plt.savefig(fname, dpi=300)
        click.echo(f"Saved: {fname}")


@cli.command()
@click.option("--output", default="courant_effect.png")
def study_courant(output: str):
    solver = AdvectionSolver(L=20.0, N=200, u=1.0)
    t_target = 8.0
    co_values = [0.1, 0.5, 0.8, 1.0]

    plt.figure(figsize=(10, 6))
    s_exact = solver.exact_solution(profile_gaussian, t_target)
    plt.plot(solver.x, s_exact, "k--", lw=2.5, label="Exact")

    colors = ["purple", "red", "blue", "green"]
    for Co, col in zip(co_values, colors):
        s_num, _ = solver.solve("upwind", profile_gaussian, Co=Co, t_final=t_target)
        l2_err = np.sqrt(np.mean((s_num - s_exact) ** 2))
        plt.plot(solver.x, s_num, color=col, lw=1.8, label=f"Co = {Co} (L2 = {l2_err:.4f})")

    plt.title("Courant Number Effect on Upwind Scheme (t = 8.0 s)", fontsize=13)
    plt.xlabel("x")
    plt.ylabel("s(x, t)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    click.echo(f"Saved: {output}")


if __name__ == "__main__":
    cli()