"""Multi-variable linear regression based on statsmodels
and Ordinary Least Squares (ols)."""

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import statsmodels.formula.api as fm
from patsy import LookupFactor, ModelDesc, Term  # pylint: disable=no-name-in-module
from statsmodels.sandbox.regression.predstd import wls_prediction_std

from openenergyid.enums import Granularity

from .helpers import resample_input_data


class ValidationParameters(BaseModel):
    """Parameters for validation of a multivariable linear regression model."""

    rsquared: float = Field(
        0.75, ge=0, le=1, description="Minimum acceptable value for the adjusted R-squared"
    )
    f_pvalue: float = Field(
        0.05, ge=0, le=1, description="Maximum acceptable value for the F-statistic"
    )
    pvalues: float = Field(
        0.05, ge=0, le=1, description="Maximum acceptable value for the p-values of the t-statistic"
    )


class MultiVariableLinearRegression:
    """Multi-variable linear regression.

    Based on statsmodels and Ordinary Least Squares (ols).

    Pass a dataframe with the variable to be modelled y (dependent variable)
    and the possible independent variables x.
    Specify as string the name of the dependent variable, and optionally pass a list with names of
    independent variables to try
    (by default all other columns will be tried as independent variables).

    The analysis is based on a forward-selection approach: starting from a simple model,
    the model is iteratively refined and verified until no statistical relevant improvements
    can be obtained.
    Each model in the iteration loop is stored in the attribute self.list_of_fits.
    The selected model is self.fit (=pointer to the last element of self.list_of_fits).

    The dataframe can contain daily, weekly, monthly, yearly ... values.  Each row is an instance.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        y: str,
        p_max: float = 0.05,
        list_of_x: list[str] = None,
        confint: float = 0.95,
        cross_validation: bool = False,
        allow_negative_predictions: bool = False,
        validation_params: ValidationParameters = None,
        granularity: Granularity = None,
    ):
        """Parameters
        ----------
        data : TimeSeries
            Datetimeindex and both independent variables (x) and dependent variable (y) as columns
        y : str
            Name of the dependent (endogeneous) variable to model
        p_max : float (default=0.05)
            Acceptable p-value of the t-statistic for estimated parameters
        list_of_x : list of str (default=None)
            If None (default), try to build a model with all columns in the dataframe
            If a list with column names is given, only try these columns as independent variables
        confint : float, default=0.95
            Two-sided confidence interval for predictions.
        cross_validation : bool, default=False
            If True, compute the model based on cross-validation (leave one out)
            Only possible if the df has less than 15 entries.
            Note: this will take much longer computation times!
        allow_negative_predictions : bool, default=False
            If True, allow predictions to be negative.
            For gas consumption or PV production, this is not physical
            so allow_negative_predictions should be False
        validation_params : ValidationParameters, default=None
            Parameters to validate the model.
        """
        self.data = data.copy()
        if y not in self.data.columns:
            raise AssertionError(
                f"The dependent variable {y} is not a column in the dataframe",
            )
        self.y = y

        self.p_max = p_max
        self.list_of_x = list_of_x or [x for x in self.data.columns if x != self.y]
        self.confint = confint
        self.cross_validation = cross_validation
        self.allow_negative_predictions = allow_negative_predictions
        self.validation_params = validation_params or ValidationParameters()
        self.granularity = granularity
        self._fit = None
        self._list_of_fits = []
        self.list_of_cverrors = []

    @property
    def fit(self) -> fm.ols:
        """Fits a model to the data.

        Returns
        -------
            The fitted model.

        Raises
        ------
            UnboundLocalError: If `do_analysis()` has not been run before calling `fit()`.
        """
        if self._fit is None:
            raise UnboundLocalError(
                'Run "do_analysis()" first to fit a model to the data.',
            )
        else:
            return self._fit

    @property
    def list_of_fits(self) -> list[fm.ols]:
        """Returns the list of fits generated by the model.

        Raises
        ------
            UnboundLocalError: If the model has not been fitted yet.

        Returns
        -------
            list: The list of fits generated by the model.
        """
        if not self._list_of_fits:
            raise UnboundLocalError(
                'Run "do_analysis()" first to fit a model to the data.',
            )
        else:
            return self._list_of_fits

    def do_analysis(self):
        """Find the best model (fit) and create self.list_of_fits and self.fit"""
        if self.cross_validation:
            return self._do_analysis_cross_validation()
        else:
            return self._do_analysis_no_cross_validation()

    def _do_analysis_no_cross_validation(self):
        """Find the best model (fit) and create self.list_of_fits and self.fit"""
        # first model is just the mean
        response_term = [Term([LookupFactor(self.y)])]
        model_terms = [Term([])]  # empty term is the intercept
        all_model_terms_dict = {x: Term([LookupFactor(x)]) for x in self.list_of_x}
        # ...then add another term for each candidate
        # model_terms += [Term([LookupFactor(c)]) for c in candidates]
        model_desc = ModelDesc(response_term, model_terms)
        self._list_of_fits.append(fm.ols(model_desc, data=self.data).fit())
        # try to improve the model until no improvements can be found

        while all_model_terms_dict:
            # try each x and overwrite the best_fit if we find a better one
            # the first best_fit is the one from the previous round
            ref_fit = self._list_of_fits[-1]
            best_fit = self._list_of_fits[-1]
            best_bic = best_fit.bic
            for x, term in all_model_terms_dict.items():
                # make new_fit, compare with best found so far
                model_desc = ModelDesc(
                    response_term,
                    ref_fit.model.formula.rhs_termlist + [term],
                )
                fit = fm.ols(model_desc, data=self.data).fit()
                if fit.bic < best_bic:
                    best_bic = fit.bic
                    best_fit = fit
                    best_x = x
            # Sometimes, the obtained fit may be better, but contains unsignificant parameters.
            # Correct the fit by removing the unsignificant parameters and estimate again
            best_fit = self._prune(best_fit, p_max=self.p_max)

            # if best_fit does not contain more variables than ref fit, exit
            if len(best_fit.model.formula.rhs_termlist) == len(
                ref_fit.model.formula.rhs_termlist,
            ):
                break
            else:
                self._list_of_fits.append(best_fit)
                all_model_terms_dict.pop(best_x)
        self._fit = self._list_of_fits[-1]

    def _do_analysis_cross_validation(self):
        """Find the best model (fit) based on cross-valiation (leave one out)"""
        assert (
            len(self.data) < 15
        ), "Cross-validation is not implemented if your sample contains more than 15 datapoints"

        # initialization: first model is the mean, but compute cv correctly.
        errors = []
        response_term = [Term([LookupFactor(self.y)])]
        model_terms = [Term([])]  # empty term is the intercept
        model_desc = ModelDesc(response_term, model_terms)
        for i in self.data.index:
            # make new_fit, compute cross-validation and store error
            df_ = self.data.drop(i, axis=0)
            fit = fm.ols(model_desc, data=df_).fit()
            cross_prediction = self._predict(fit=fit, data=self.data.loc[[i], :])
            errors.append(cross_prediction["predicted"] - cross_prediction[self.y])

        self._list_of_fits = [fm.ols(model_desc, data=self.data).fit()]
        self.list_of_cverrors = [np.mean(np.abs(np.array(errors)))]

        # try to improve the model until no improvements can be found
        all_model_terms_dict = {x: Term([LookupFactor(x)]) for x in self.list_of_x}
        while all_model_terms_dict:
            # import pdb;pdb.set_trace()
            # try each x in all_exog and overwrite if we find a better one
            # at the end of iteration (and not earlier), save the best of the iteration
            better_model_found = False
            best = dict(fit=self._list_of_fits[-1], cverror=self.list_of_cverrors[-1])
            for x, term in all_model_terms_dict.items():
                model_desc = ModelDesc(
                    response_term,
                    self._list_of_fits[-1].model.formula.rhs_termlist + [term],
                )
                # cross_validation, currently only implemented for monthly data
                # compute the mean error for a given formula based on leave-one-out.
                errors = []
                for i in self.data.index:
                    # make new_fit, compute cross-validation and store error
                    df_ = self.data.drop(i, axis=0)
                    fit = fm.ols(model_desc, data=df_).fit()
                    cross_prediction = self._predict(
                        fit=fit,
                        data=self.data.loc[[i], :],
                    )
                    errors.append(
                        cross_prediction["predicted"] - cross_prediction[self.y],
                    )
                cverror = np.mean(np.abs(np.array(errors)))
                # compare the model with the current fit
                if cverror < best["cverror"]:
                    # better model, keep it
                    # first, reidentify using all the datapoints
                    best["fit"] = fm.ols(model_desc, data=self.data).fit()
                    best["cverror"] = cverror
                    better_model_found = True
                    best_x = x

            if better_model_found:
                self._list_of_fits.append(best["fit"])
                self.list_of_cverrors.append(best["cverror"])

            else:
                # if we did not find a better model, exit
                break

            # next iteration with the found exog removed
            all_model_terms_dict.pop(best_x)

        self._fit = self._list_of_fits[-1]

    def _prune(self, fit: fm.ols, p_max: float) -> fm.ols:
        """If the fit contains statistically insignificant parameters, remove them.
        Returns a pruned fit where all parameters have p-values of the t-statistic below p_max

        Parameters
        ----------
        fit: fm.ols fit object
            Can contain insignificant parameters
        p_max : float
            Maximum allowed probability of the t-statistic

        Returns
        -------
        fit: fm.ols fit object
            Won't contain any insignificant parameters

        """

        def remove_from_model_desc(x: str, model_desc: ModelDesc) -> ModelDesc:
            """Return a model_desc without x"""
            rhs_termlist = []
            for t in model_desc.rhs_termlist:
                if not t.factors:
                    # intercept, add anyway
                    rhs_termlist.append(t)
                elif x != t.factors[0]._varname:  # pylint: disable=protected-access
                    # this is not the term with x
                    rhs_termlist.append(t)

            md = ModelDesc(model_desc.lhs_termlist, rhs_termlist)
            return md

        corrected_model_desc = ModelDesc(
            fit.model.formula.lhs_termlist[:],
            fit.model.formula.rhs_termlist[:],
        )
        pars_to_prune = fit.pvalues.where(fit.pvalues > p_max).dropna().index.tolist()
        try:
            pars_to_prune.remove("Intercept")
        except KeyError:
            pass
        while pars_to_prune:
            corrected_model_desc = remove_from_model_desc(
                pars_to_prune[0],
                corrected_model_desc,
            )
            fit = fm.ols(corrected_model_desc, data=self.data).fit()
            pars_to_prune = fit.pvalues.where(fit.pvalues > p_max).dropna().index.tolist()
            try:
                pars_to_prune.remove("Intercept")
            except KeyError:
                pass
        return fit

    @staticmethod
    def find_best_rsquared(list_of_fits: list[fm.ols]) -> fm.ols:
        """Return the best fit, based on rsquared"""
        res = sorted(list_of_fits, key=lambda x: x.rsquared)
        return res[-1]

    @staticmethod
    def find_best_akaike(list_of_fits: list[fm.ols]) -> fm.ols:
        """Return the best fit, based on Akaike information criterion"""
        res = sorted(list_of_fits, key=lambda x: x.aic)
        return res[0]

    @staticmethod
    def find_best_bic(list_of_fits: list[fm.ols]) -> fm.ols:
        """Return the best fit, based on Akaike information criterion"""
        res = sorted(list_of_fits, key=lambda x: x.bic)
        return res[0]

    def _predict(self, fit: fm.ols, data: pd.DataFrame) -> pd.DataFrame:
        """Return a df with predictions and confidence interval

        Notes
        -----
        The df will contain the following columns:
        - 'predicted': the model output
        - 'interval_u', 'interval_l': upper and lower confidence bounds.
        The result will depend on the following attributes of self:
        confint : float (default=0.95)
            Confidence level for two-sided hypothesis
        allow_negative_predictions : bool (default=True)
            If False, correct negative predictions to zero
            (typically for energy consumption predictions)

        Parameters
        ----------
        fit : Statsmodels fit
        data : pandas DataFrame or None (default)
            If None, use self.data

        Returns
        -------
        result : pandas DataFrame
            Copy of df with additional columns 'predicted', 'interval_u' and 'interval_l'
        """
        # Add model results to data as column 'predictions'
        result = data.copy()
        if "Intercept" in fit.model.exog_names:
            result["Intercept"] = 1.0
        result["predicted"] = fit.predict(result)
        if not self.allow_negative_predictions:
            result.loc[result["predicted"] < 0, "predicted"] = 0

        _prstd, interval_l, interval_u = wls_prediction_std(
            fit,
            result[fit.model.exog_names],
            alpha=1 - self.confint,
        )
        result["interval_l"] = interval_l
        result["interval_u"] = interval_u

        if "Intercept" in result:
            result.drop(labels=["Intercept"], axis=1, inplace=True)

        return result

    def add_prediction(self):
        """Add predictions and confidence interval to self.df
        self.df will contain the following columns:
        - 'predicted': the model output
        - 'interval_u', 'interval_l': upper and lower confidence bounds.

        Parameters
        ----------
        None, but the result depends on the following attributes of self:
        confint : float (default=0.95)
            Confidence level for two-sided hypothesis
        allow_negative_predictions : bool (default=True)
            If False, correct negative predictions to zero
            (typically for energy consumption predictions)

        Returns
        -------
        Nothing, adds columns to self.df
        """
        self.data = self._predict(fit=self.fit, data=self.data)

    @property
    def is_valid(self) -> bool:
        """Checks if the model is valid.

        Returns
        -------
            bool: True if the model is valid, False otherwise.
        """
        if self.fit.rsquared_adj < self.validation_params.rsquared:
            return False

        if self.fit.f_pvalue > self.validation_params.f_pvalue:
            return False

        param_keys = self.fit.pvalues.keys().tolist()
        param_keys.remove("Intercept")
        for k in param_keys:
            if self.fit.pvalues[k] > self.validation_params.pvalues:
                return False

        return True


def find_best_mvlr(
    data: pd.DataFrame,
    y: str,
    granularities: list[Granularity],
    **kwargs,
) -> MultiVariableLinearRegression:
    """Cycle through multiple granularities and return the best model."""
    for granularity in granularities:
        data = resample_input_data(data=data, granularity=granularity)
        mvlr = MultiVariableLinearRegression(data=data, y=y, granularity=granularity, **kwargs)
        mvlr.do_analysis()
        if mvlr.is_valid:
            return mvlr
    raise ValueError("No valid model found.")
