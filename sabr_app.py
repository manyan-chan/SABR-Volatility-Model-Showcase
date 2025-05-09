from typing import Union  # Any for NDArray generic shape

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go  # For Figure type hint
import streamlit as st
from numpy.typing import NDArray  # Modern way to type NumPy arrays


# --- SABR Model Implementation (Hagan et al. 2002 approximation) ---
def sabr_implied_vol(
    F_forward: float,
    K_strikes: Union[float, NDArray[np.float64]],  # Specify dtype
    T_maturity: float,
    alpha_sabr: float,
    beta_sabr: float,
    rho_sabr: float,
    nu_sabr: float,
) -> NDArray[np.float64]:  # Specify dtype
    """
    Calculates SABR implied Black-Scholes volatility.
    Hagan, P. S., Kumar, D., Lesniewski, A. S., & Woodward, D. E. (2002).
    Managing smile risk. Wilmott Magazine, 84-108.

    Vectorized to handle K_strikes as a numpy array.
    """
    K: NDArray[np.float64] = np.asarray(K_strikes, dtype=np.float64)

    implied_vol: NDArray[np.float64] = np.full(K.shape, np.nan, dtype=np.float64)

    atm_mask: NDArray[np.bool_] = np.isclose(F_forward, K)  # isclose returns bool array

    if np.any(atm_mask):
        F_atm: float = F_forward
        term1_atm: float = alpha_sabr / (F_atm ** (1 - beta_sabr))
        term2_atm_factor: float = (
            ((1 - beta_sabr) ** 2 / 24)
            * (alpha_sabr**2 / (F_atm ** (2 - 2 * beta_sabr)))
            + (
                rho_sabr
                * beta_sabr
                * nu_sabr
                * alpha_sabr
                / (4 * (F_atm ** (1 - beta_sabr)))
            )
            + ((2 - 3 * rho_sabr**2) / 24) * nu_sabr**2
        )
        implied_vol[atm_mask] = term1_atm * (1 + term2_atm_factor * T_maturity)

    non_atm_mask: NDArray[np.bool_] = ~atm_mask
    if np.any(non_atm_mask):
        _F: float = F_forward
        _K_non_atm: NDArray[np.float64] = K[non_atm_mask]

        log_F_K: NDArray[np.float64] = np.log(_F / _K_non_atm)
        F_K_beta_half: NDArray[np.float64] = (_F * _K_non_atm) ** ((1 - beta_sabr) / 2)

        zeta: NDArray[np.float64] = (nu_sabr / alpha_sabr) * F_K_beta_half * log_F_K

        zeta_over_x_zeta: NDArray[np.float64] = np.full_like(zeta, np.nan)

        small_zeta_mask_local: NDArray[np.bool_] = np.abs(zeta) < 1e-7
        zeta_over_x_zeta[small_zeta_mask_local] = 1.0

        domain_for_x_calc: NDArray[np.bool_] = ~small_zeta_mask_local

        if np.any(domain_for_x_calc):
            zeta_sub: NDArray[np.float64] = zeta[domain_for_x_calc]
            sqrt_term_sub: NDArray[np.float64] = np.sqrt(
                1 - 2 * rho_sabr * zeta_sub + zeta_sub**2
            )

            log_arg_numerator_sub: NDArray[np.float64] = (
                sqrt_term_sub + zeta_sub - rho_sabr
            )

            valid_log_arg_mask_sub: NDArray[np.bool_] = log_arg_numerator_sub > 1e-12

            x_zeta_sub: NDArray[np.float64] = np.full_like(zeta_sub, np.nan)

            if np.any(valid_log_arg_mask_sub):
                x_zeta_sub[valid_log_arg_mask_sub] = np.log(
                    log_arg_numerator_sub[valid_log_arg_mask_sub] / (1 - rho_sabr)
                )

            usable_x_zeta_mask_sub: NDArray[np.bool_] = ~np.isnan(x_zeta_sub) & (
                np.abs(x_zeta_sub) > 1e-9
            )

            division_results_sub: NDArray[np.float64] = np.full_like(zeta_sub, np.nan)

            if np.any(usable_x_zeta_mask_sub):
                division_results_sub[usable_x_zeta_mask_sub] = (
                    zeta_sub[usable_x_zeta_mask_sub]
                    / x_zeta_sub[usable_x_zeta_mask_sub]
                )

            zeta_over_x_zeta[domain_for_x_calc] = division_results_sub

        term1_num: float = alpha_sabr

        term1_den_expansion: NDArray[np.float64] = (
            1
            + (((1 - beta_sabr) ** 2 / 24) * log_F_K**2)
            + (((1 - beta_sabr) ** 4 / 1920) * log_F_K**4)
        )
        term1_den: NDArray[np.float64] = F_K_beta_half * term1_den_expansion

        term1: NDArray[np.float64] = np.full_like(zeta_over_x_zeta, np.nan)
        safe_den_mask: NDArray[np.bool_] = np.abs(term1_den) > 1e-12

        valid_for_term1_calc_mask: NDArray[np.bool_] = safe_den_mask & ~np.isnan(
            zeta_over_x_zeta
        )
        if np.any(valid_for_term1_calc_mask):
            term1[valid_for_term1_calc_mask] = (
                term1_num / term1_den[valid_for_term1_calc_mask]
            ) * zeta_over_x_zeta[valid_for_term1_calc_mask]

        term2_factor: NDArray[np.float64] = (
            ((1 - beta_sabr) ** 2 / 24)
            * (alpha_sabr**2 / ((_F * _K_non_atm) ** (1 - beta_sabr)))
            + (rho_sabr * beta_sabr * nu_sabr * alpha_sabr / (4 * F_K_beta_half))
            + ((2 - 3 * rho_sabr**2) / 24) * nu_sabr**2
        )

        implied_vol_non_atm: NDArray[np.float64] = term1 * (
            1 + term2_factor * T_maturity
        )
        implied_vol[non_atm_mask] = implied_vol_non_atm

    return implied_vol


# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("SABR Volatility Model Showcase")
st.markdown("""
This app demonstrates the SABR (Stochastic Alpha, Beta, Rho) volatility model,
using Hagan et al.'s (2002) analytical approximation for implied Black-Scholes volatility.
Adjust the parameters in the sidebar to see how they affect the volatility smile.
""")

# --- Sidebar for Inputs ---
st.sidebar.header("SABR Model Parameters")

F_default: float = 100.0
T_default: float = 1.0

F_param: float = st.sidebar.number_input(
    "Forward Price (F)", min_value=0.01, value=F_default, step=1.0, format="%.2f"
)
T_param: float = st.sidebar.number_input(
    "Time to Maturity (T, years)",
    min_value=0.01,
    value=T_default,
    step=0.1,
    format="%.2f",
)

alpha_param_default: float = 0.20
beta_param_default: float = 0.5
rho_param_default: float = -0.3
nu_param_default: float = 0.40

alpha_param: float = st.sidebar.slider(
    "Alpha (α) - ATM Vol Level",
    min_value=0.01,
    max_value=1.0,
    value=alpha_param_default,
    step=0.01,
    format="%.2f",
)
beta_param: float = st.sidebar.slider(
    "Beta (β) - Skew/Backbone Shape",
    min_value=0.0,
    max_value=1.0,
    value=beta_param_default,
    step=0.01,
    format="%.2f",
)
rho_param: float = st.sidebar.slider(
    "Rho (ρ) - Correlation (Skew)",
    min_value=-0.99,
    max_value=0.99,
    value=rho_param_default,
    step=0.01,
    format="%.2f",
)
nu_param: float = st.sidebar.slider(
    "Nu (ν) - Vol of Vol (Smile Convexity)",
    min_value=0.0,
    max_value=2.0,
    value=nu_param_default,
    step=0.01,
    format="%.2f",
)

st.sidebar.markdown("---")
st.sidebar.header("Smile Plot Settings")
moneyness_min: float = st.sidebar.slider("Min Moneyness (K/F)", 0.5, 0.95, 0.7, 0.01)
moneyness_max: float = st.sidebar.slider("Max Moneyness (K/F)", 1.05, 1.5, 1.3, 0.01)
num_strikes: int = st.sidebar.slider("Number of Strikes", 10, 200, 50, 10)

# --- Calculations and Plotting ---
strikes_arr: NDArray[np.float64] = np.linspace(
    F_param * moneyness_min, F_param * moneyness_max, num_strikes, dtype=np.float64
)

implied_vols_arr: NDArray[np.float64] = sabr_implied_vol(
    F_forward=F_param,
    K_strikes=strikes_arr,
    T_maturity=T_param,
    alpha_sabr=alpha_param,
    beta_sabr=beta_param,
    rho_sabr=rho_param,
    nu_sabr=nu_param,
)

df_smile: pd.DataFrame = pd.DataFrame(
    {"Strike (K)": strikes_arr, "Implied Volatility": implied_vols_arr * 100}
)
df_smile.dropna(inplace=True)

fig: go.Figure = px.line(df_smile, x="Strike (K)", y="Implied Volatility", markers=True)
fig.update_layout(
    title=f"SABR Implied Volatility Smile (F={F_param:.2f}, T={T_param:.2f}y)",
    xaxis_title="Strike Price (K)",
    yaxis_title="Implied Volatility (%)",
    yaxis_tickformat=".2f",
)
fig.add_vline(
    x=F_param,
    line_dash="dash",
    line_color="red",
    annotation_text="Forward (F)",
    annotation_position="top right",
)

st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Current Parameters:")
    st.write(f"- Forward (F): {F_param:.2f}")
    st.write(f"- Time to Maturity (T): {T_param:.2f} years")
    st.write(f"- Alpha (α): {alpha_param:.3f}")
    st.write(f"- Beta (β): {beta_param:.3f}")
    st.write(f"- Rho (ρ): {rho_param:.3f}")
    st.write(f"- Nu (ν): {nu_param:.3f}")

with col2:
    st.subheader("SABR Model Interpretation:")
    # Changed from f-string to regular string to fix Ruff F541
    st.markdown("""
    - **α (Alpha)**: Roughly the at-the-money (ATM) volatility. Higher α shifts the smile upwards.
    - **β (Beta)**: Determines the backbone shape of the smile (how vol changes with forward rate).
        - β=1: Lognormal-like (Black-Scholes).
        - β=0: Normal-like (Bachelier).
        - 0 < β < 1: Most common, e.g., for equities/FX.
    - **ρ (Rho)**: Correlation between the forward rate and its volatility. Primarily controls the *skew* of the smile.
        - Negative ρ (common for equities): Smile slopes downwards (higher vol for OTM puts).
        - Positive ρ: Smile slopes upwards.
    - **ν (Nu)**: Volatility of volatility ("vol-of-vol"). Primarily controls the *convexity* or "smileyness" of the curve. Higher ν means a more pronounced U-shape.
    - **T (Time to Maturity)**: Affects the overall level and shape. Generally, longer maturities lead to flatter smiles, but the SABR formula components scale with T.
    """)

st.markdown("---")
st.subheader("SABR Implied Volatility Formula (Hagan et al. 2002 Approximation)")
st.latex(r"""
I(K) = \frac{\alpha}
           { (FK)^{\frac{1-\beta}{2}} \left( 1 + \frac{(1-\beta)^2}{24}\log^2\frac{F}{K} + \frac{(1-\beta)^4}{1920}\log^4\frac{F}{K} + \dots \right) } \times
           \frac{z}{x(z)} \times
           \left( 1 + \left[ \frac{(1-\beta)^2}{24}\frac{\alpha^2}{(FK)^{1-\beta}} + \frac{1}{4}\frac{\rho\beta\nu\alpha}{(FK)^{\frac{1-\beta}{2}}} + \frac{2-3\rho^2}{24}\nu^2 \right] T \right)
""")
st.markdown("where:")
st.latex(r"""
z = \frac{\nu}{\alpha} (FK)^{\frac{1-\beta}{2}} \log\frac{F}{K}
""")
st.latex(r"""
x(z) = \log \left( \frac{\sqrt{1-2\rho z+z^2} + z - \rho}{1-\rho} \right)
""")
st.caption(
    "Note: The formula shown is a common approximation. The implementation handles the ATM case (F=K) separately for stability, where z/x(z) approaches 1. Small numerical instabilities can still occur for extreme parameter values or strikes very far from the forward, potentially resulting in NaN values for volatility at those points."
)
