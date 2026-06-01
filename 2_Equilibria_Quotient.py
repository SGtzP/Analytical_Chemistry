# -*- coding: utf-8 -*-
"""
Created on Sat May 30 16:28:25 2026

@author: sgtzp
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

def reaction_quotient(conc, reactants, products, coeffs):
    """
    Calculate Q or K expression from concentrations.
    """

    numerator = 1.0
    denominator = 1.0

    for i in products:
        numerator *= conc[i] ** coeffs[i]

    for i in reactants:
        denominator *= conc[i] ** coeffs[i]

    if denominator == 0:
        return np.inf

    return numerator / denominator


def equilibrium_concentrations(
    initial_conc,
    reactants,
    products,
    coeffs,
    Kc
):
    """
    Solve for equilibrium concentrations using reaction extent x.
    """

    n = len(initial_conc)

    # Sign convention
    signs = np.zeros(n)

    for i in reactants:
        signs[i] = -1

    for i in products:
        signs[i] = +1

    def concentrations(x):
        return initial_conc + signs * coeffs * x

    def f(x):
        c = concentrations(x)

        if np.any(c < 0):
            return np.nan

        return reaction_quotient(
            c,
            reactants,
            products,
            coeffs
        ) - Kc

    # Determine physically allowable x range
    x_min = -1e6
    x_max = 1e6

    for i in reactants:
        x_max = min(
            x_max,
            initial_conc[i] / coeffs[i]
        )

    for i in products:
        x_min = max(
            x_min,
            -initial_conc[i] / coeffs[i]
        )

    # Scan for sign changes
    xs = np.linspace(x_min, x_max, 5000)

    root = None

    for a, b in zip(xs[:-1], xs[1:]):
        try:
            fa = f(a)
            fb = f(b)

            if np.isnan(fa) or np.isnan(fb):
                continue

            if fa * fb < 0:
                root = brentq(f, a, b)
                break

        except:
            continue

    if root is None:
        raise RuntimeError(
            "Could not locate a physical equilibrium root."
        )

    eq_conc = concentrations(root)

    return root, eq_conc


def main():

    print("=" * 60)
    print("GENERAL EQUILIBRIUM SOLVER")
    print("=" * 60)

    n = int(input("\nNumber of species: "))

    names = []
    initial_conc = np.zeros(n)
    coeffs = np.zeros(n)

    print("\nSpecies names")

    for i in range(n):
        names.append(
            input(f"Species {i+1} name: ")
        )

    print("\nInitial concentrations (M)")

    for i in range(n):
        initial_conc[i] = float(
            input(f"{names[i]} [M] = ")
        )

    print(
        "\nReactant indices "
        "(comma-separated, e.g. 1,2)"
    )

    reactants = [
        int(x.strip()) - 1
        for x in input("Reactants: ").split(",")
    ]

    print(
        "\nProduct indices "
        "(comma-separated, e.g. 3,4)"
    )

    products = [
        int(x.strip()) - 1
        for x in input("Products: ").split(",")
    ]

    print("\nStoichiometric coefficients")

    for i in range(n):
        coeffs[i] = float(
            input(f"{names[i]} coefficient: ")
        )

    Kc = float(
        input("\nEquilibrium constant Kc = ")
    )

    # Initial quotient
    Q0 = reaction_quotient(
        initial_conc,
        reactants,
        products,
        coeffs
    )

    print("\n" + "=" * 60)
    print("INITIAL ANALYSIS")
    print("=" * 60)

    print(f"Initial Q = {Q0:.6g}")
    print(f"Kc        = {Kc:.6g}")

    if np.isclose(Q0, Kc, rtol=1e-8):
        print("\nSystem already at equilibrium.")
        return

    elif Q0 < Kc:
        print(
            "\nQ < Kc  → reaction proceeds FORWARD"
        )

    else:
        print(
            "\nQ > Kc  → reaction proceeds REVERSE"
        )

    # Solve equilibrium
    extent, eq_conc = equilibrium_concentrations(
        initial_conc,
        reactants,
        products,
        coeffs,
        Kc
    )

    Qeq = reaction_quotient(
        eq_conc,
        reactants,
        products,
        coeffs
    )

    print("\n" + "=" * 60)
    print("EQUILIBRIUM RESULTS")
    print("=" * 60)

    print(f"Reaction extent x = {extent:.8f}\n")

    print(
        f"{'Species':<15}"
        f"{'Initial(M)':>15}"
        f"{'Equilibrium(M)':>20}"
    )

    print("-" * 50)

    for i in range(n):
        print(
            f"{names[i]:<15}"
            f"{initial_conc[i]:>15.6f}"
            f"{eq_conc[i]:>20.6f}"
        )

    print("\nVerification")

    print(f"Qeq = {Qeq:.8g}")
    print(f"Kc  = {Kc:.8g}")
    print(
        f"Relative error = "
        f"{abs(Qeq-Kc)/Kc:.3e}"
    )


    # -----------------------------
    # Automatic colors
    # -----------------------------
    cmap = plt.cm.Set3
    colors = {
        names[i]: cmap(i / max(len(names)-1, 1))
        for i in range(len(names))
    }
    
    # -----------------------------
    # Build plotting dictionaries
    # -----------------------------
    before = {
        names[i]: [initial_conc[i]]
        for i in range(len(names))
    }
    
    equil = {
        names[i]: [eq_conc[i]]
        for i in range(len(names))
    }
    
    mixtures = ["System"]
    
    # -----------------------------
    # Plot function
    # -----------------------------
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        sharey=True
    )
    
    def stacked_bar(ax, data, title):
    
        x = np.arange(len(mixtures))
        bottom = np.zeros(len(mixtures))
    
        for sp in names:
    
            vals = np.array(data[sp])
    
            ax.bar(
                x,
                vals,
                bottom=bottom,
                color=colors[sp],
                edgecolor='black',
                label=sp
            )
    
            for i, v in enumerate(vals):
    
                if v > 1e-8:
    
                    ax.text(
                        i,
                        bottom[i] + v/2,
                        f"{v:.4f}",
                        ha='center',
                        va='center',
                        fontsize=9,
                        fontweight='bold'
                    )
    
            bottom += vals
    
        ax.set_xticks(x)
        ax.set_xticklabels(mixtures)
    
        ax.set_ylabel("Concentration (M)")
        ax.set_title(title)
    
        ax.grid(
            axis='y',
            linestyle='--',
            alpha=0.3
        )
    
    # -----------------------------
    # Initial
    # -----------------------------
    stacked_bar(
        axes[0],
        before,
        "Before Reaction"
    )
    
    # -----------------------------
    # Equilibrium
    # -----------------------------
    stacked_bar(
        axes[1],
        equil,
        "At Equilibrium"
    )
    
    # -----------------------------
    # Shared legend
    # -----------------------------
    handles, labels = axes[0].get_legend_handles_labels()
    
    fig.legend(
        handles,
        labels,
        loc='upper center',
        ncol=min(len(names), 5)
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.show()
if __name__ == "__main__":
    main()

