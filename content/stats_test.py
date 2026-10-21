import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

from ancova_assumptions import with_rpy2_converter
from helpers import display_table  # noqa: F401  (re-exported so notebooks can call st.display_table)

def get_violated_terms(df_ancova_assumptions, structure, metric):
    """Return list of interaction terms to retain, based on homogeneity_of_slopes only."""
    sub = df_ancova_assumptions.query(
        "structure == @structure and metric == @metric "
        "and test_name == 'homogeneity_of_slopes' and flag_violated == True"
    )
    return sub['term'].tolist()  # e.g. ['group:age'], ['group:sex'], [], or both

@with_rpy2_converter
def ancova_tests(df, metrics, structures, df_ancova_assumptions, group_col = 'group', structure_col = 'structure', covar = 'age + sex'):
    # Run ANCOVA for each structure and metric, controlling for age and sex, and return the results in a dataframe
    results = []
    
    for structure in structures:    
        df_str = df[df[structure_col] == structure]
        if df_str.empty:
            continue
        for metric in metrics:
            if metric not in df_str.columns or df_str[metric].dropna().empty:
                continue

            # Check for violated assumptions and determine interaction terms to retain
            violated_terms = get_violated_terms(df_ancova_assumptions, structure, metric)
            interaction_str = ''.join(f" + {t}" for t in violated_terms)

            try:
                # Convert to R dataframe
                with localconverter(ro.default_converter + pandas2ri.converter):
                    rdf = ro.conversion.py2rpy(df_str)

                r_env = ro.Environment()
                r_env['rdf'] = rdf

                # Run code to R
                r_code = f"""
                    rdf${group_col} <- as.factor(rdf${group_col})
                    rdf$age_c <- rdf$age - mean(rdf$age, na.rm = TRUE)

                    model <- lm({metric} ~ {group_col} + age_c + sex{interaction_str.replace('age', 'age_c')}, data=rdf)

                    anova_table <- car::Anova(model, type="II")
                    anova_df <- as.data.frame(anova_table)
                    anova_df$term <- rownames(anova_df)

                    eta <- effectsize::eta_squared(model, partial = TRUE, ci = 0.95)
                    eta_df <- as.data.frame(eta)

                    omega <- effectsize::omega_squared(model, partial = TRUE, ci = 0.95)
                    omega_df <- as.data.frame(omega)

                    anova_df <- merge(anova_df, eta_df[, c("Parameter", "Eta2_partial", "CI_low", "CI_high")],
                        by.x = "term", by.y = "Parameter", all.x = TRUE)
                    anova_df <- merge(anova_df, omega_df[, c("Parameter", "Omega2_partial", "CI_low", "CI_high")],
                        by.x = "term", by.y = "Parameter", all.x = TRUE)
                """

                r_expr = ro.r.parse(text=r_code)
                ro.r.eval(r_expr, envir=r_env)

                anova_df_r = r_env['anova_df']
                with localconverter(ro.default_converter + pandas2ri.converter):
                    anova_df = ro.conversion.rpy2py(anova_df_r)

                anova_df['metric'] = metric
                anova_df['structure'] = structure
                anova_df['violated_terms'] = ','.join(violated_terms)
                results.append(anova_df)

            except Exception as e:
                print(f"Error for {structure} {metric}: {e}")
                continue

    if results:
        results = pd.concat(results, ignore_index=True)
        results = results.rename(columns={'term': 'beta_term'})

    else:
        results = pd.DataFrame()
    
    
    print(f"Total results collected: {len(results)}")
    return results

def post_ancova_correctMultipleComparison(df_postAncova, mc_method='fdr_by', mc_alpha=0.05, group_col = 'group', structure_col = 'structure'):
    """
    Run multiple-comparison correction on ANCOVA results for the group variable.
    Returns only the group variable ANCOVA results with adjusted p-values.
    """
    # GEt df of residuals to merge with group variable results later (to have Df_residual for effect size reporting))
    df_residual = df_postAncova.query("beta_term == 'Residuals'")[['metric',structure_col,'Df']].rename(columns={'Df':'Df_residual'})


    # Group variable ANCOVA results
    df_postAncova_group = df_postAncova.query("beta_term == @group_col").copy()

    # apply statsmodels multiple-comparison correction
    pvals = pd.to_numeric(df_postAncova_group['Pr(>F)'], errors='coerce').values
    try:
        _, pvals_adj, _, _ = multipletests(pvals, alpha=mc_alpha, method=mc_method)
        df_postAncova_group['pvals_adj'] = pvals_adj

        # Merge Df residuals
        df_postAncova_group = df_postAncova_group.merge(df_residual, on=['metric',structure_col], how='left')

        # Changing datatypes to save to int
        df_postAncova_group = df_postAncova_group.astype({'Df': int, 'Df_residual': int})

        # Adding a star level column based on the adjusted p-values
        df_postAncova_group['level'] = df_postAncova_group['pvals_adj'].apply(lambda x: '**' if x < 0.01 else '*' if x < 0.05 else np.nan)
    except Exception:
        # fallback: return NaNs if correction fails
        df_postAncova_group['pvals_adj'] = np.nan

    return df_postAncova_group


@with_rpy2_converter
def posthoc_eem_tests(df, df_postAncova, df_ancova_assumptions, mc_alpha=0.05,
                       structure_col='structure', group_col='group', covar='age + sex'):
    group_col_eem = 'group_eem' # eem needs a different name to avoid confusion with original group column in df

    df_postAncova_significant = df_postAncova.query(
        "pvals_adj < @mc_alpha and beta_term == @group_col"
    )[["structure", "metric"]]

    structures = df_postAncova_significant["structure"].to_list()
    metrics = df_postAncova_significant["metric"].to_list()

    results = []

    for structure, metric in zip(structures, metrics):

        df_str = df[df[structure_col] == structure].copy()
        df_str = df_str.rename(columns={group_col: group_col_eem}) # rename columns so EEM works

        if df_str.empty or metric not in df_str.columns or df_str[metric].dropna().empty:
            continue

        # Same lookup as ancova_tests, just renamed for clarity here
        violated_terms = get_violated_terms(df_ancova_assumptions, structure, metric)
        # terms come back as "group:age" / "group:sex" -> swap to match group_col_eem + centered age
        interaction_str = ''.join(
            f" + {group_col_eem}:age_c" if 'age' in t else f" + {group_col_eem}:sex"
            for t in violated_terms
        )

        try:
            df_r_input = df_str[[metric, group_col_eem] + covar.split(' + ')].dropna()

            with localconverter(ro.default_converter + pandas2ri.converter):
                rdf = ro.conversion.py2rpy(df_r_input)

            r_env = ro.Environment()
            r_env['rdf'] = rdf

            r_code = f"""
            library(marginaleffects)

            rdf${group_col_eem} <- as.factor(rdf${group_col_eem})
            rdf$age_c <- rdf$age - mean(rdf$age, na.rm = TRUE)

            model <- lm(`{metric}` ~ {group_col_eem} + age_c + sex{interaction_str}, data = rdf)

            cmp <- comparisons(
                model,
                variables = list({group_col_eem} = "pairwise"),
                newdata = "mean",
                adjust = "holm",
                df = df.residual(model) # forces t-distribution with correct dof
            )
            
            cmp_df <- as.data.frame(cmp)
            # Residual SD for partial d
            sigma <- sigma(model)
            cmp_df$sigma <- sigma

            # Primary effect size: partial Cohen's d 
            cmp_df$partial_d     <- cmp_df$estimate   / sigma
            cmp_df$partial_d_low <- cmp_df$conf.low   / sigma
            cmp_df$partial_d_high <- cmp_df$conf.high / sigma

            # Hedges' g correction factor
            correction <- 1 - (3 / (4 * df.residual(model) - 1))

            cmp_df$hedges_g      <- cmp_df$partial_d      * correction
            cmp_df$hedges_g_low  <- cmp_df$partial_d_low  * correction
            cmp_df$hedges_g_high <- cmp_df$partial_d_high * correction
            """

            r_expr = ro.r.parse(text=r_code)
            ro.r.eval(r_expr, envir=r_env)

            res_r = r_env['cmp_df']
            with localconverter(ro.default_converter + pandas2ri.converter):
                comparisons_df = ro.conversion.rpy2py(res_r)

            comparisons_df['metric'] = metric
            comparisons_df[structure_col] = structure
            comparisons_df['violated_terms'] = ','.join(violated_terms)

            # Getting rid of unnecesary columns in dataframe to avoid confusion
            drop_cols = [c for c in [metric, "term", "rowid", "s.value",
                         "predicted", "predicted_lo", "predicted_hi", "group_eem"]
                         if c in comparisons_df.columns]
            comparisons_df = comparisons_df.drop(columns=drop_cols)

            results.append(comparisons_df)

        except Exception as e:
            print(e)
            continue

    if not results:
        print("No result")
        return pd.DataFrame()

    res_df = pd.concat(results, ignore_index=True)

    # rename p_vals to holm adj to avoid confusion
    res_df = res_df.rename(columns={
        'p.value':   'p_holm',
        'conf.low':  'CI_low_holm',
        'conf.high': 'CI_high_holm',
        'statistic': 't_value'
    })

    res_df['level'] = res_df['p_holm'].apply(lambda x: '**' if x < 0.01 else '*' if x < 0.05 else np.nan)

    return res_df



# Ancova test for hippocampal subfield analysis.
@with_rpy2_converter
def ancova_tests_subfield(df, metrics, subfields,  covar=None, group_col='group', subfield_col = 'subfield'):
    """
    ANCOVA for subfield level analyses. Controls by covar
    """ 
    results = []
    
    for subfield in subfields:    
        df_str = df[df[subfield_col] == subfield]
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
                
                # Run code to R
                r_code = f"""
                    rdf${group_col} <- as.factor(rdf${group_col})
                     
                    model <- lm({metric} ~ {group_col} + {covar}, data=rdf) 
                     
                    anova_table <- car::Anova(model, type="II")
                    anova_df <- as.data.frame(anova_table)
                    anova_df$term <- rownames(anova_df)
                    
                    eta <- effectsize::eta_squared(model, partial = TRUE, ci = 0.95)
                    eta_df <- as.data.frame(eta)

                    omega <- effectsize::omega_squared(model, partial = TRUE, ci = 0.95)
                    omega_df <- as.data.frame(omega)

                    # eta_df has a column called "Parameter" matching the term names
                    anova_df <- merge(anova_df, eta_df[, c("Parameter", "Eta2_partial", "CI_low", "CI_high")],
                        by.x = "term", by.y = "Parameter",
                        all.x = TRUE)  # keeps Residuals row with NA for eta

                    anova_df <- merge(anova_df, omega_df[, c("Parameter", "Omega2_partial", "CI_low", "CI_high")],
                        by.x = "term", by.y = "Parameter",
                        all.x = TRUE)  # keeps Residuals row with NA for eta
                """
                
                # Run the model
                r_expr = ro.r.parse(text=r_code)
                ro.r.eval(r_expr, envir=r_env)

                anova_df_r = r_env['anova_df']
                
                # Pull back results R to pandas dataframe
                with localconverter(ro.default_converter + pandas2ri.converter):
                    anova_df = ro.conversion.rpy2py(anova_df_r)

                anova_df['metric'] = metric
                anova_df['subfield'] = subfield
                results.append(anova_df)
                
            except Exception as e:
                print(f"Error for {subfield} {metric}: {e}")
                continue
            
    if results:
        results = pd.concat(results, ignore_index=True)
        results = results.rename(columns={'term': 'beta_term'})

    else:
        results = pd.DataFrame()
    
    
    print(f"Total results collected: {len(results)}")  # Each metric-subfield pair should yield multiple rows (group, covars, residuals)
    return results

def format_table(df, two_decimals=(), p_values=()):
    """Return a display copy of a results table without significance stars: display metric names, statistics and effect sizes
    with 2 decimals, p-values with 4 decimals (2 significant figures in scientific notation below 0.0001)."""
    out = df.drop(columns=["level"], errors="ignore")  # exact p-values only, no significance stars
    if "metric" in out.columns:
        out["metric"] = out["metric"].map(metric_name)
    for col in two_decimals:
        out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    for col in p_values:
        out[col] = out[col].map(lambda p: "" if pd.isna(p) else (f"{p:.1e}" if p < 1e-4 else f"{p:.4f}"))
    return out.rename(columns=DISPLAY_COLUMNS)

# Display names and units for metrics in figures and tables (data columns keep their original names)
METRIC_NAMES = {
    "uFA": "µFA",
    "normalized_volume": "Normalized volume",
    "volume": "Normalized volume",
    "mahalanobis_distance": "MhD",
    "mahalanobis_distance_log": "log MhD",
    "thickness": "Thickness",
    "curvature": "Curvature",
    "gyrification": "Gyrification",
}
DISPLAY_COLUMNS = {
    "structure": "ROI",
    "subfield": "Subfield",
    "metric": "Metric",
    "contrast": "Contrast",
    "test_name": "Assumption",
    "term": "Test",
    "p_value": "p",
}
METRIC_UNITS = {
    "MD": "µm²/ms",
    "MTsat": "p.u.",
    "thickness": "mm",
    "volume": "% of TIV",
    "normalized_volume": "% of TIV",
}

def metric_name(metric):
    """Display name of a metric, e.g. 'uFA' -> 'µFA'."""
    return METRIC_NAMES.get(metric, metric)

def metric_axis_label(metric):
    """Axis label of a metric with its units, e.g. 'MD' -> 'MD (µm²/ms)'."""
    unit = METRIC_UNITS.get(metric)
    return f"{metric_name(metric)} ({unit})" if unit else metric_name(metric)
