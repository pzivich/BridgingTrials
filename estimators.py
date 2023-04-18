import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


class ClassicalEstimator:
    """Classical or standard estimator for combining randomized clinical trials together. This ignores any potential
    differences between the source populations of the two trials.

    Parameters
    ----------
    data : DataFrame
        Pandas DataFrame object containing all variables of interest
    treatment : str
        Column name of the treatment variable. Currently only binary is supported
    time : str
        Column name of the time variable
    delta : str
        Column name of the event indicator variable. Currently only binary is supported
    sample : str
        Column name of the location variable, only binary is currently supported
    """
    def __init__(self, data, treatment, time, delta, sample):
        # Extracting inputs
        self.treatment = treatment       # Treatment column name
        self.time = time                 # Follow-up time column name
        self.delta = delta               # Event indicator column name
        self.sample = sample             # Study or sample identifier column name

        # Extracting sorted data set for ease of processing
        self.data = data.copy().sort_values(by=[self.sample, self.treatment, self.time]).reset_index(drop=True)

        # Getting all integer times from zero to maximum follow-up time
        max_time = np.max(self.data[self.time])           # Maximum event time
        self.event_times = list(range(0, max_time+1, 1))  # All unique event times together

    def estimate(self):
        """Estimate the risk difference using the empirical distribution function. This approach assumes no right
        censoring.

        Returns
        -------
        DataFrame : estimated risk difference over time
        """
        ref_pop = 1 - self.data[self.sample]      # Referent study population
        target_pop = self.data[self.sample]       # Target study population

        # Calculate the risk using the empirical distribution function
        #   This assumes no censoring, so it is only a simple illustration
        risk_t0_p0 = self._edf_(in_group=ref_pop*(self.data[self.treatment] == 0))
        risk_t1_p0 = self._edf_(in_group=ref_pop*(self.data[self.treatment] == 1))
        risk_t1_p1 = self._edf_(in_group=target_pop*(self.data[self.treatment] == 1))
        risk_t2_p1 = self._edf_(in_group=target_pop*(self.data[self.treatment] == 2))

        # Calculate the risk difference and return the calculated value
        risk_difference = (risk_t2_p1 - risk_t1_p1) + (risk_t1_p0 - risk_t0_p0)

        # Processing into a data set
        results = pd.DataFrame()
        results['t'] = self.event_times
        results['RD'] = risk_difference
        return results

    def _edf_(self, in_group):
        """Internal function to compute the empirical distribution function for a given group.

        Parameters
        ----------
        in_group : ndarray
            Indicator if unit is in the group or not

        Returns
        -------
        ndarray : estimated risk over all unique time points
        """
        risk_over_t = []
        delta = np.asarray(self.data[self.delta])
        n_group = np.sum(in_group)
        for t in self.event_times:
            event_indicator = np.where(self.data[self.time] <= t, 1, 0)
            numerator = np.sum(in_group * event_indicator * delta)

            risk_at_t = numerator / n_group
            risk_over_t.append(risk_at_t)

        return np.asarray(risk_over_t)


class BridgeEstimator:
    def __init__(self, data, treatment, time, delta, sample):
        # Extracting inputs
        self.treatment = treatment       # Treatment column name
        self.time = time                 # Follow-up time column name
        self.delta = delta               # Event indicator column name
        self.sample = sample             # Study or sample identifier column name

        # Extracting sorted data set for ease of processing
        self.data = data.copy().sort_values(by=[self.sample, self.treatment, self.time]).reset_index(drop=True)

        # Getting all integer times from zero to maximum follow-up time
        max_time = np.max(self.data[self.time])           # Maximum event time
        self.event_times = list(range(0, max_time+1, 1))  # All unique event times together

        # Storage for later procedures
        self._nuisance_sampling_ = []
        self._fit_location_ = False
        self.weight = None

    def sampling_model(self, model, bound=0.01):
        r"""Sampling model, which predicts the probability of :math:`S=1` given the baseline covariates. The sampling
        model consists of the following

        .. math::
            pi_S(V_i) = \Pr(S=1 | V; \beta)

        Parameters
        ----------
        model : str
            Variables to predict the location differences via the patsy format. For example, 'var1 + var2 + var3'
        bound : float, list, optional
            Value between 0,1 to truncate predicted probabilities. Helps to avoid near positivity violations.
            Specifying this argument can improve finite sample performance for random positivity violations. However,
            truncating weights leads to additional confounding. Default is False, meaning no truncation of
            predicted probabilities occurs. Providing a single float assumes symmetric trunctation, where values below
            or above the threshold are set to the threshold value. Alternatively a list of floats can be provided for
            asymmetric trunctation, with the first value being the lower bound and the second being the upper bound
        """
        family = sm.families.Binomial()
        loc_fm = smf.glm(self.sample + " ~ " + model,
                         self.data,
                         family=family).fit()

        # Calculating sampling weight of bridging estimator
        pr_s = loc_fm.predict(self.data)
        pr_s = np.clip(pr_s, bound, 1-bound)

        # Calculating the re-weighted sample size for S=0, hat{n}_0
        target_pop = self.data[self.sample]
        self.weight = np.where(target_pop, 1, pr_s / (1-pr_s))

        # Marker to indicate model has been specified and fully fit
        self._fit_location_ = True

    def estimate(self):
        """Estimate the risk difference using a simplified version of a bridged treatment comparison estimator using a
        weighted empirical distribution function. This approach assumes no right censoring.

        Returns
        -------
        DataFrame : estimated risk difference over time
        """
        ref_pop = 1 - self.data[self.sample]      # Referent study population
        target_pop = self.data[self.sample]   # Target study population

        # Calculate the risk using the empirical distribution function
        #   This assumes no censoring, so it is only a simple illustration
        risk_t0_p0 = self._edf_(in_group=ref_pop*(self.data[self.treatment] == 0))
        risk_t1_p0 = self._edf_(in_group=ref_pop*(self.data[self.treatment] == 1))
        risk_t1_p1 = self._edf_(in_group=target_pop*(self.data[self.treatment] == 1))
        risk_t2_p1 = self._edf_(in_group=target_pop*(self.data[self.treatment] == 2))

        # Calculate the risk difference and return the calculated value
        risk_difference = (risk_t2_p1 - risk_t1_p1) + (risk_t1_p0 - risk_t0_p0)

        # Processing into a data set
        results = pd.DataFrame()
        results['t'] = self.event_times
        results['RD'] = risk_difference
        return results

    def _edf_(self, in_group):
        """Internal function to compute the weighted empirical distribution function for a given group.

        Parameters
        ----------
        in_group : ndarray
            Indicator if unit is in the group or not.
        weight : ndarray
            Weight assigned to units.

        Returns
        -------
        ndarray : estimated risk over all unique time points
        """
        risk_over_t = []
        weight = self.weight
        n_group = np.sum(in_group * weight)
        delta = self.data[self.delta]
        for t in self.event_times:
            event_indicator = np.where(self.data[self.time] <= t, 1, 0)
            numerator = np.sum(in_group * event_indicator * delta * weight)

            risk_at_t = numerator / n_group
            risk_over_t.append(risk_at_t)
        return np.asarray(risk_over_t)
