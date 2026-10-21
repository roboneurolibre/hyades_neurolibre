"""
ancova_assumptions.py

Helper functions to test the statistical assumptions required for an
ANCOVA analysis, using R (via rpy2) under the hood:

    1. ancova_homogeneity_tests          -> homogeneity of regression slopes
    2. ancova_normality_tests            -> normality of residuals (Shapiro-Wilk)
    3. ancova_variance_homogeneity_tests -> homogeneity of variances (Levene's test)
    4. ancova_assumption_tests           -> wrapper that runs all three and
                                             returns a combined long-format summary

Requires R packages: car (and base stats, already included in R)
Requires Python packages: pandas, rpy2
"""

import functools

import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter


def with_rpy2_converter(func):
    """Run func with rpy2's default converter active.

    rpy2 keeps its converter in a ContextVar set at import time. Some ipykernel versions (e.g. 7.0.x,
    used by the NeuroLibre Binder image) run each cell in a fresh context, so the converter is gone in
    later cells and every rpy2 call fails with "Conversion rules for `rpy2.robjects` appear to be missing".
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with localconverter(ro.default_converter):
            return func(*args, **kwargs)
    return wrapper


@with_rpy2_converter
def ancova_homogeneity_tests(df, metrics, structures, covar='age', group_col='group', structure_col='structure',
                              factor_covars=None):
    """
    covar : str
        Right-hand-side covariate formula fragment, e.g. 'age' or 'age + sex'.
    factor_covars : list[str] or None
        Names of covariates (from `covar`) that should be treated as categorical
        (e.g. a numerically-coded sex column like 0/1). String/object columns are
        auto-detected as factors by R, so this is only needed for numeric-coded
        categorical covariates.
    """
    # Check homogeneity of regression slopes assumption using R
    results = []
    factor_covars = factor_covars or []

    alpha = 0.05

    for structure in structures:
        df_str = df[df[structure_col] == structure]
        if df_str.empty:
            continue
        for metric in metrics:
            if metric not in df_str.columns or df_str[metric].dropna().empty:
                continue
            try:
                # Convert to R dataframe
                with localconverter(ro.default_converter + pandas2ri.converter):
                    rdf = ro.conversion.py2rpy(df_str)

                r_env = ro.Environment()
                r_env['rdf'] = rdf

                # Build factor-conversion lines for any numerically-coded categorical covariates
                factor_lines = "\n".join(f"rdf${fc} <- as.factor(rdf${fc})" for fc in factor_covars)

                # R code to run
                # Wrapping covar in parentheses ensures `group` interacts with EVERY
                # covariate term (e.g. covar="age + sex" -> group*(age + sex) expands to
                # group + age + sex + group:age + group:sex). Without the parentheses,
                # R's formula precedence (`*` binds tighter than `+`) would silently drop
                # the interaction for every covariate after the first one.
                r_code = f"""
                    rdf${group_col} <- as.factor(rdf${group_col})
                    {factor_lines}

                    model <- lm({metric} ~ {group_col}*({covar}), data=rdf)

                    anova_table <- car::Anova(model, type="II")
                    anova_df <- as.data.frame(anova_table)
                    anova_df$term <- rownames(anova_df)
                """

                # Run the model
                r_expr = ro.r.parse(text=r_code)
                ro.r.eval(r_expr, envir=r_env)

                anova_df_r = r_env['anova_df']

                # Pull back results R to pandas dataframe
                with localconverter(ro.default_converter + pandas2ri.converter):
                    anova_df = ro.conversion.rpy2py(anova_df_r)

                anova_df['metric'] = metric
                anova_df['structure'] = structure
                results.append(anova_df)

            except Exception as e:
                print(f"Error for {structure} {metric}: {e}")
                continue

    if results:
        results = pd.concat(results, ignore_index=True)
        results = results.rename(columns={'term': 'beta_term'})
        results["flag_slopes_violated"] = results["Pr(>F)"] < alpha
    else:
        results = pd.DataFrame()

    n_groups = results[['structure', 'metric']].drop_duplicates().shape[0] if not results.empty else 0
    print(f"Total results collected: {len(results)} rows across {n_groups} structure/metric combinations")
    return results


@with_rpy2_converter
def ancova_normality_tests(df, metrics, structures, covar='age', group_col='group', structure_col='structure',
                            factor_covars=None):
    # Check normality of residuals assumption (Shapiro-Wilk) using R
    results = []
    factor_covars = factor_covars or []

    alpha = 0.05

    for structure in structures:
        df_str = df[df[structure_col] == structure]
        if df_str.empty:
            continue
        for metric in metrics:
            if metric not in df_str.columns or df_str[metric].dropna().empty:
                continue
            try:
                # Convert to R dataframe
                with localconverter(ro.default_converter + pandas2ri.converter):
                    rdf = ro.conversion.py2rpy(df_str)

                r_env = ro.Environment()
                r_env['rdf'] = rdf

                # Build factor-conversion lines for any numerically-coded categorical covariates
                factor_lines = "\n".join(f"rdf${fc} <- as.factor(rdf${fc})" for fc in factor_covars)

                # R code to run
                r_code = f"""
                    rdf${group_col} <- as.factor(rdf${group_col})
                    {factor_lines}

                    model <- lm({metric} ~ {group_col} + {covar}, data=rdf)
                    resid_vals <- residuals(model)

                    shapiro_result <- shapiro.test(resid_vals)
                    shapiro_df <- data.frame(
                        statistic = as.numeric(shapiro_result$statistic),
                        `Pr(>F)` = shapiro_result$p.value,
                        check.names = FALSE
                    )
                    shapiro_df$term <- "shapiro_wilk"
                """

                # Run the model
                r_expr = ro.r.parse(text=r_code)
                ro.r.eval(r_expr, envir=r_env)

                shapiro_df_r = r_env['shapiro_df']

                # Pull back results R to pandas dataframe
                with localconverter(ro.default_converter + pandas2ri.converter):
                    shapiro_df = ro.conversion.rpy2py(shapiro_df_r)

                shapiro_df['metric'] = metric
                shapiro_df['structure'] = structure
                results.append(shapiro_df)

            except Exception as e:
                print(f"Error for {structure} {metric}: {e}")
                continue

    if results:
        results = pd.concat(results, ignore_index=True)
        results = results.rename(columns={'term': 'test_term'})
        results["flag_normality_violated"] = results["Pr(>F)"] < alpha
    else:
        results = pd.DataFrame()

    print(f"Total results collected: {len(results)}")
    return results


@with_rpy2_converter
def ancova_variance_homogeneity_tests(df, metrics, structures, group_col='group', structure_col='structure'):
    # Check homogeneity of variances assumption (Levene's test) using R
    results = []

    alpha = 0.05

    for structure in structures:
        df_str = df[df[structure_col] == structure]
        if df_str.empty:
            continue
        for metric in metrics:
            if metric not in df_str.columns or df_str[metric].dropna().empty:
                continue
            try:
                # Convert to R dataframe
                with localconverter(ro.default_converter + pandas2ri.converter):
                    rdf = ro.conversion.py2rpy(df_str)

                r_env = ro.Environment()
                r_env['rdf'] = rdf

                # R code to run
                r_code = f"""
                    rdf${group_col} <- as.factor(rdf${group_col})

                    levene_result <- car::leveneTest({metric} ~ {group_col}, data=rdf)
                    levene_df <- as.data.frame(levene_result)
                    levene_df$term <- rownames(levene_df)
                """

                # Run the model
                r_expr = ro.r.parse(text=r_code)
                ro.r.eval(r_expr, envir=r_env)

                levene_df_r = r_env['levene_df']

                # Pull back results R to pandas dataframe
                with localconverter(ro.default_converter + pandas2ri.converter):
                    levene_df = ro.conversion.rpy2py(levene_df_r)

                levene_df['metric'] = metric
                levene_df['structure'] = structure
                results.append(levene_df)

            except Exception as e:
                print(f"Error for {structure} {metric}: {e}")
                continue

    if results:
        results = pd.concat(results, ignore_index=True)
        results = results.rename(columns={'term': 'test_term'})
        # leveneTest output includes a trailing blank row ("") -- keep only the actual test row
        results = results[results['test_term'] != ""]
        results["flag_variance_violated"] = results["Pr(>F)"] < alpha
    else:
        results = pd.DataFrame()

    print(f"Total results collected: {len(results)}")
    return results


def ancova_assumption_tests(df, metrics, structures, covar='age', group_col='group', structure_col='structure',
                             factor_covars=None):
    """
    Wrapper that runs all ANCOVA assumption tests and returns a combined summary.

    Runs:
      1. Homogeneity of regression slopes (ancova_homogeneity_tests)
      2. Normality of residuals (ancova_normality_tests)
      3. Homogeneity of variances (ancova_variance_homogeneity_tests)

    Parameters
    ----------
    covar : str
        Right-hand-side covariate formula fragment, e.g. 'age' or 'age + sex'.
        Every covariate here gets its own group:<covariate> interaction tested
        for homogeneity of slopes.
    factor_covars : list[str] or None
        Names of covariates that should be treated as categorical (needed only
        for numerically-coded categorical covariates, e.g. sex coded as 0/1).

    Returns:
      summary : long-format dataframe with columns
                structure, metric, test_name, term, p_value, flag_violated
      raw     : dict with the raw per-test output dataframes
                {'slopes': ..., 'normality': ..., 'variance': ...}
    """

    raw_results = {}

    # --- 1. Slopes ---
    slopes_df = ancova_homogeneity_tests(
        df, metrics, structures, covar=covar, group_col=group_col, structure_col=structure_col,
        factor_covars=factor_covars
    )
    raw_results['slopes'] = slopes_df

    # --- 2. Normality ---
    normality_df = ancova_normality_tests(
        df, metrics, structures, covar=covar, group_col=group_col, structure_col=structure_col,
        factor_covars=factor_covars
    )
    raw_results['normality'] = normality_df

    # --- 3. Variance homogeneity ---
    variance_df = ancova_variance_homogeneity_tests(
        df, metrics, structures, group_col=group_col, structure_col=structure_col
    )
    raw_results['variance'] = variance_df

    # --- Build a unified long-format summary ---
    summary_rows = []

    if not slopes_df.empty:
        # Keep only the interaction term row(s), i.e. group:<covariate>, one per covariate
        # in `covar` (covar may be a single term like "age" or additive like "age + sex").
        # These are what actually test the homogeneity-of-slopes assumption -- main effect
        # rows (group, age, sex, ...) aren't part of this assumption check.
        covar_terms = [c.strip() for c in covar.split('+') if c.strip()]
        interaction_terms = [f"{group_col}:{c}" for c in covar_terms]
        sub = slopes_df[slopes_df['beta_term'].isin(interaction_terms)].copy()
        sub['test_name'] = 'homogeneity_of_slopes'
        sub['term'] = sub['beta_term']
        sub['p_value'] = sub['Pr(>F)']
        sub['flag_violated'] = sub['flag_slopes_violated']
        summary_rows.append(sub[['structure', 'metric', 'test_name', 'term', 'p_value', 'flag_violated']])

    if not normality_df.empty:
        sub = normality_df.copy()
        sub['test_name'] = 'normality_of_residuals'
        sub['term'] = sub['test_term']
        sub['p_value'] = sub['Pr(>F)']
        sub['flag_violated'] = sub['flag_normality_violated']
        summary_rows.append(sub[['structure', 'metric', 'test_name', 'term', 'p_value', 'flag_violated']])

    if not variance_df.empty:
        sub = variance_df.copy()
        sub['test_name'] = 'homogeneity_of_variance'
        sub['term'] = sub['test_term']
        sub['p_value'] = sub['Pr(>F)']
        sub['flag_violated'] = sub['flag_variance_violated']
        summary_rows.append(sub[['structure', 'metric', 'test_name', 'term', 'p_value', 'flag_violated']])

    if summary_rows:
        summary = pd.concat(summary_rows, ignore_index=True)
    else:
        summary = pd.DataFrame(columns=['structure', 'metric', 'test_name', 'term', 'p_value', 'flag_violated'])

    n_violated = summary['flag_violated'].sum() if not summary.empty else 0
    print(f"Total assumption checks run: {len(summary)} | Violations flagged: {n_violated}")

    return summary, raw_results


@with_rpy2_converter
def plot_normality_diagnostics(df, normality_df, covar='age', group_col='group', structure_col='structure',
                                factor_covars=None, only_violations=True, max_plots=12, n_cols=3):
    """
    Visually inspect residual normality for structure/metric combinations.

    For each case, refits the additive ANCOVA model (metric ~ group + covar) in R,
    pulls the residuals back to Python, and plots a histogram + Q-Q plot side by side.

    Parameters
    ----------
    df : pd.DataFrame
        Original long-format dataframe (same one passed to ancova_normality_tests).
    normality_df : pd.DataFrame
        Output of ancova_normality_tests (or the 'normality' entry from raw_results
        returned by ancova_assumption_tests). Used to pick which cases to plot.
    only_violations : bool
        If True (default), only plot cases flagged as violated. If False, plot all
        cases in normality_df (useful to sanity-check the whole batch).
    max_plots : int
        Safety cap on how many structure/metric combinations to render.
    n_cols : int
        Number of (histogram, Q-Q) plot pairs per row.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats as sstats

    factor_covars = factor_covars or []

    cases = normality_df.copy()
    if only_violations:
        cases = cases[cases['flag_normality_violated']]

    cases = cases[['structure', 'metric']].drop_duplicates()

    if cases.empty:
        print("No cases to plot (no violations found, or normality_df is empty).")
        return None

    if len(cases) > max_plots:
        print(f"{len(cases)} cases found, plotting first {max_plots} (increase max_plots to see more).")
        cases = cases.head(max_plots)

    n_cases = len(cases)
    n_rows = int(np.ceil(n_cases / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.atleast_2d(axes)

    for i, (_, row) in enumerate(cases.iterrows()):
        structure, metric = row['structure'], row['metric']
        df_str = df[df[structure_col] == structure]

        try:
            with localconverter(ro.default_converter + pandas2ri.converter):
                rdf = ro.conversion.py2rpy(df_str)

            r_env = ro.Environment()
            r_env['rdf'] = rdf

            factor_lines = "\n".join(f"rdf${fc} <- as.factor(rdf${fc})" for fc in factor_covars)

            r_code = f"""
                rdf${group_col} <- as.factor(rdf${group_col})
                {factor_lines}

                model <- lm({metric} ~ {group_col} + {covar}, data=rdf)
                resid_vals <- residuals(model)
            """
            r_expr = ro.r.parse(text=r_code)
            ro.r.eval(r_expr, envir=r_env)

            resid_vals = list(r_env['resid_vals'])

        except Exception as e:
            print(f"Error for {structure} {metric}: {e}")
            continue

        r_pos = (i // n_cols)
        c_pos = (i % n_cols) * 2

        # Histogram
        ax_hist = axes[r_pos, c_pos]
        ax_hist.hist(resid_vals, bins=20, color='steelblue', edgecolor='white')
        ax_hist.set_title(f"{structure} | {metric}\nresidual histogram", fontsize=9)
        ax_hist.set_xlabel("residual")

        # Q-Q plot
        ax_qq = axes[r_pos, c_pos + 1]
        sstats.probplot(resid_vals, dist="norm", plot=ax_qq)
        ax_qq.set_title(f"{structure} | {metric}\nQ-Q plot", fontsize=9)

        # Annotate with the Shapiro-Wilk p-value already computed
        p_val = normality_df.loc[
            (normality_df['structure'] == structure) & (normality_df['metric'] == metric), 'Pr(>F)'
        ]
        if not p_val.empty:
            ax_hist.text(
                0.02, 0.95, f"Shapiro p={p_val.values[0]:.4f}",
                transform=ax_hist.transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
            )

    # Hide any unused axes
    total_axes = n_rows * n_cols * 2
    for j in range(n_cases * 2, total_axes):
        r_pos = j // (n_cols * 2)
        c_pos = j % (n_cols * 2)
        axes[r_pos, c_pos].axis('off')

    plt.tight_layout()
    return fig
