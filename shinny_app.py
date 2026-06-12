from shiny import App, ui, render, reactive
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq


# --------------------------------------------------
# Chemistry Functions
# --------------------------------------------------

def reaction_quotient(conc, reactants, products, coeffs):

    numerator = 1.0
    denominator = 1.0

    for i in products:
        numerator *= conc[i] ** coeffs[i]

    for i in reactants:
        denominator *= conc[i] ** coeffs[i]

    if denominator == 0:
        return np.inf

    return numerator / denominator


def equilibrium_concentrations(initial_conc, reactants, products, coeffs, Kc):

    n = len(initial_conc)

    signs = np.zeros(n)

    for i in reactants:
        signs[i] = -1

    for i in products:
        signs[i] = 1

    def concentrations(x):
        return initial_conc + signs * coeffs * x

    def f(x):

        c = concentrations(x)

        if np.any(c < 0):
            return np.nan

        return (
            reaction_quotient(
                c,
                reactants,
                products,
                coeffs
            )
            - Kc
        )

    x_min = -1e6
    x_max = 1e6

    for i in reactants:
        x_max = min(x_max, initial_conc[i] / coeffs[i])

    for i in products:
        x_min = max(x_min, -initial_conc[i] / coeffs[i])

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

        except Exception:
            continue

    if root is None:
        raise RuntimeError(
            "Could not locate a physical equilibrium root."
        )

    return root, concentrations(root)


# --------------------------------------------------
# UI
# --------------------------------------------------

app_ui = ui.page_fluid(

    ui.h2("General Equilibrium Solver"),

    ui.layout_sidebar(

        ui.sidebar(

            ui.input_text(
                "names",
                "Species Names",
                "A,B,C,D"
            ),

            ui.input_text(
                "initial",
                "Initial Concentrations",
                "1,1,0,0"
            ),

            ui.input_text(
                "coeffs",
                "Stoichiometric Coefficients",
                "1,1,1,1"
            ),

            ui.input_text(
                "reactants",
                "Reactant Indices (1-based)",
                "1,2"
            ),

            ui.input_text(
                "products",
                "Product Indices (1-based)",
                "3,4"
            ),

            ui.input_numeric(
                "kc",
                "Kc",
                10.0
            ),

            ui.input_action_button(
                "solve",
                "Solve"
            )

        ),

        ui.output_text_verbatim("results"),

        ui.output_plot("equilibrium_plot")
    )
)


# --------------------------------------------------
# SERVER
# --------------------------------------------------

def server(input, output, session):

    @reactive.calc
    @reactive.event(input.solve)
    def solve_system():

        names = [
            x.strip()
            for x in input.names().split(",")
        ]

        initial_conc = np.array(
            [float(x) for x in input.initial().split(",")]
        )

        coeffs = np.array(
            [float(x) for x in input.coeffs().split(",")]
        )

        reactants = [
            int(x.strip()) - 1
            for x in input.reactants().split(",")
        ]

        products = [
            int(x.strip()) - 1
            for x in input.products().split(",")
        ]

        Kc = float(input.kc())

        Q0 = reaction_quotient(
            initial_conc,
            reactants,
            products,
            coeffs
        )

        extent, eq_conc = equilibrium_concentrations(
            initial_conc,
            reactants,
            products,
            coeffs,
            Kc
        )

        return {
            "names": names,
            "initial": initial_conc,
            "equilibrium": eq_conc,
            "extent": extent,
            "Q0": Q0
        }

    @output
    @render.text
    def results():

        try:

            res = solve_system()

            txt = []

            txt.append(
                f"Reaction extent x = {res['extent']:.6f}"
            )

            txt.append("")

            for n, c0, ceq in zip(
                res["names"],
                res["initial"],
                res["equilibrium"]
            ):

                txt.append(
                    f"{n}: "
                    f"Initial={c0:.4f} M, "
                    f"Equilibrium={ceq:.4f} M"
                )

            return "\n".join(txt)

        except Exception as e:
            return str(e)

    @output
    @render.plot
    def equilibrium_plot():

        res = solve_system()

        names = res["names"]

        initial = res["initial"]

        equilibrium = res["equilibrium"]

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(10, 5),
            sharey=True
        )

        axes[0].bar(names, initial)
        axes[0].set_title("Initial")

        axes[1].bar(names, equilibrium)
        axes[1].set_title("Equilibrium")

        return fig


app = App(app_ui, server)