"""MSA (Measurement System Analysis) Calculator Service.

Implements ANOVA-based Gauge R&R calculations using scipy and numpy.
This is the single source of truth for all MSA statistical calculations.
"""

import math
from typing import Literal

import numpy as np
from scipy import stats

from app.models.msa import MSAInput, MSAResult, VarianceComponents

# Maximum ndc value when GRR is effectively zero (perfect measurement system)
NDC_PERFECT_SYSTEM = 999


class MSACalculator:
    """Calculator for Gauge R&R analysis using ANOVA method.

    Implements the AIAG MSA Manual methodology for analyzing measurement
    system variation and determining system acceptability.
    """

    def calculate(self, data: MSAInput) -> MSAResult:
        """Perform complete Gauge R&R analysis.

        Args:
            data: MSAInput with parts, operators, trials, and measurements

        Returns:
            MSAResult with all Gauge R&R metrics and classification
        """
        # Convert measurements to numpy array for efficient computation
        # Shape: (parts, operators, trials)
        measurements = np.array(data.measurements)

        n_parts = len(data.parts)
        n_operators = len(data.operators)
        n_trials = data.trials

        # Calculate variance components using ANOVA
        variance_components = self._calculate_variance_components(
            measurements, n_parts, n_operators, n_trials
        )

        # Calculate standard deviations
        sigma_repeatability = math.sqrt(variance_components["repeatability"])
        sigma_reproducibility = math.sqrt(variance_components["reproducibility"])
        sigma_part = math.sqrt(variance_components["part"])
        sigma_grr = math.sqrt(
            variance_components["repeatability"] + variance_components["reproducibility"]
        )
        sigma_total = math.sqrt(variance_components["total"])

        # Calculate percentages (of total standard deviation)
        if sigma_total > 0:
            grr_percent = 100.0 * sigma_grr / sigma_total
            repeatability_percent = 100.0 * sigma_repeatability / sigma_total
            reproducibility_percent = 100.0 * sigma_reproducibility / sigma_total
            part_variation_percent = 100.0 * sigma_part / sigma_total
        else:
            # Edge case: no variation at all
            grr_percent = 0.0
            repeatability_percent = 0.0
            reproducibility_percent = 0.0
            part_variation_percent = 100.0

        # Calculate number of distinct categories (ndc)
        ndc = self._calculate_ndc(sigma_part, sigma_grr)

        # Determine category based on %GRR
        category = self._classify_grr(grr_percent)

        return MSAResult(
            grr_percent=round(grr_percent, 2),
            repeatability_percent=round(repeatability_percent, 2),
            reproducibility_percent=round(reproducibility_percent, 2),
            part_variation_percent=round(part_variation_percent, 2),
            ndc=ndc,
            category=category,
            components=VarianceComponents(
                equipment_variation=round(variance_components["repeatability"], 6),
                operator_variation=round(variance_components["reproducibility"], 6),
                part_variation=round(variance_components["part"], 6),
                total_variation=round(variance_components["total"], 6),
            ),
        )

    def _calculate_variance_components(
        self,
        measurements: np.ndarray,
        n_parts: int,
        n_operators: int,
        n_trials: int,
    ) -> dict[str, float]:
        """Calculate variance components using ANOVA decomposition.

        Args:
            measurements: 3D array (parts × operators × trials)
            n_parts: Number of parts
            n_operators: Number of operators
            n_trials: Number of trials per operator per part

        Returns:
            Dictionary with variance components:
            - repeatability: Equipment variation (within)
            - reproducibility: Operator variation (between operators)
            - part: Part-to-part variation
            - total: Total variation
        """
        grand_mean = measurements.mean()

        # Calculate means at different levels
        part_means = measurements.mean(axis=(1, 2))  # Mean across operators and trials
        operator_means = measurements.mean(axis=(0, 2))  # Mean across parts and trials
        cell_means = measurements.mean(axis=2)  # Mean for each part-operator combination

        # Calculate Sum of Squares
        # SS_part: Variation between parts
        ss_part = n_operators * n_trials * np.sum((part_means - grand_mean) ** 2)

        # SS_operator: Variation between operators
        ss_operator = n_parts * n_trials * np.sum((operator_means - grand_mean) ** 2)

        # SS_interaction: Part × Operator interaction
        # Calculated as: sum of (cell_mean - part_mean - operator_mean + grand_mean)²
        expected_cell_means = (
            part_means[:, np.newaxis]
            + operator_means[np.newaxis, :]
            - grand_mean
        )
        ss_interaction = n_trials * np.sum((cell_means - expected_cell_means) ** 2)

        # SS_repeatability (within / error): Variation within cells
        ss_within = np.sum((measurements - cell_means[:, :, np.newaxis]) ** 2)

        # SS_total
        ss_total = np.sum((measurements - grand_mean) ** 2)

        # Degrees of freedom
        df_part = n_parts - 1
        df_operator = n_operators - 1
        df_interaction = df_part * df_operator
        df_within = n_parts * n_operators * (n_trials - 1)

        # Mean Squares
        ms_part = ss_part / df_part if df_part > 0 else 0
        ms_operator = ss_operator / df_operator if df_operator > 0 else 0
        ms_interaction = ss_interaction / df_interaction if df_interaction > 0 else 0
        ms_within = ss_within / df_within if df_within > 0 else 0

        # Calculate variance components from Mean Squares
        # σ²_repeatability = MS_within (MSE)
        var_repeatability = ms_within

        # Handle single operator case
        if n_operators == 1:
            var_reproducibility = 0.0
            var_interaction = 0.0
            # σ²_part = (MS_part - MS_within) / (n_trials)
            var_part = max(0, (ms_part - ms_within) / n_trials)
        else:
            # Check if interaction is significant (F-test at α=0.25 for pooling decision)
            # Following AIAG MSA Manual approach
            if df_interaction > 0 and df_within > 0 and ms_within > 0:
                f_interaction = ms_interaction / ms_within
                p_value = 1 - stats.f.cdf(f_interaction, df_interaction, df_within)
                interaction_significant = p_value < 0.25
            else:
                interaction_significant = False

            if interaction_significant:
                # Interaction is significant - include it
                # σ²_interaction = (MS_interaction - MS_within) / n_trials
                var_interaction = max(0, (ms_interaction - ms_within) / n_trials)

                # σ²_reproducibility = (MS_operator - MS_interaction) / (n_parts × n_trials)
                var_reproducibility = max(
                    0, (ms_operator - ms_interaction) / (n_parts * n_trials)
                )

                # σ²_part = (MS_part - MS_interaction) / (n_operators × n_trials)
                var_part = max(
                    0, (ms_part - ms_interaction) / (n_operators * n_trials)
                )

                # Add interaction to reproducibility (operator-related variation)
                var_reproducibility = var_reproducibility + var_interaction
            else:
                # Interaction is NOT significant - pool with error
                var_interaction = 0.0

                # σ²_reproducibility = (MS_operator - MS_within) / (n_parts × n_trials)
                var_reproducibility = max(
                    0, (ms_operator - ms_within) / (n_parts * n_trials)
                )

                # σ²_part = (MS_part - MS_within) / (n_operators × n_trials)
                var_part = max(
                    0, (ms_part - ms_within) / (n_operators * n_trials)
                )

        # Total variance = part + repeatability + reproducibility
        var_total = var_part + var_repeatability + var_reproducibility

        return {
            "repeatability": var_repeatability,
            "reproducibility": var_reproducibility,
            "part": var_part,
            "total": var_total,
        }

    def _calculate_ndc(self, sigma_part: float, sigma_grr: float) -> int:
        """Calculate Number of Distinct Categories (ndc).

        ndc = floor(1.41 × (σ_part / σ_grr))

        Args:
            sigma_part: Standard deviation of part variation
            sigma_grr: Standard deviation of Gauge R&R

        Returns:
            Number of distinct categories (minimum 0)
        """
        if sigma_grr <= 0:
            # Perfect measurement system - very high discrimination
            if sigma_part > 0:
                return NDC_PERFECT_SYSTEM
            else:
                return 1  # No variation at all

        ndc = int(1.41 * sigma_part / sigma_grr)
        return max(0, ndc)

    def _classify_grr(self, grr_percent: float) -> Literal["excellent", "marginal", "unacceptable"]:
        """Classify measurement system based on %GRR.

        Per AIAG MSA Manual:
        - < 10%: Excellent - measurement system acceptable
        - 10% - 30%: Marginal - may be acceptable depending on application
        - > 30%: Unacceptable - measurement system needs improvement

        Args:
            grr_percent: Gauge R&R as percentage of total variation

        Returns:
            Category classification
        """
        if grr_percent < 10.0:
            return "excellent"
        elif grr_percent <= 30.0:
            return "marginal"
        else:
            return "unacceptable"
