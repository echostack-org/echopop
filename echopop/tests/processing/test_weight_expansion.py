import numpy as np
import pandas as pd
import pytest
import xarray as xr

from echopop.survey.proportions import weight_proportions


def weight(values):
    return xr.DataArray(values, dims=['stratum', 'sex'],
                        coords={'stratum': [1, 2], 'sex': ['female', 'male']})


@pytest.mark.parametrize('reverse', [False, True])
def test_expanded_catch_denominator_by_stratum(reverse):
    aged = weight([[2., 5.], [8., 12.]])
    observed = weight([[1., 4.], [4., 6.]])
    unaged = weight([[100., 200.], [60., 40.]])
    observed_unaged = weight([[50., 200.], [30., 20.]])
    if reverse:
        observed_unaged = observed_unaged.sel(stratum=[2, 1])
    catch = pd.DataFrame({'stratum': [2, 1, 1], 'weight': [50., 40., 60.]})
    original = catch.copy(deep=True)
    result = weight_proportions(
        xr.Dataset({'aged': aged, 'unaged': unaged, 'aged_observed': observed,
                    'unaged_observed': observed_unaged}), catch, 'stratum', net_selectivity=True,
    )['proportion']
    np.testing.assert_allclose(result.sum('sex'), [7 / 121, 20 / 100])
    pd.testing.assert_frame_equal(catch, original)


def test_expansion_requires_complete_inputs():
    aged = weight([[1., 2.], [1., 2.]])
    with pytest.raises(ValueError, match='missing required variables'):
        weight_proportions(xr.Dataset({'aged': aged, 'unaged': aged,
                                      'aged_observed': aged}), pd.DataFrame(), 'stratum', net_selectivity=True)


@pytest.mark.parametrize('no_remainder', [False, True])
def test_missing_unaged_ratio(no_remainder):
    aged = weight([[1., 2.], [1., 2.]])
    catch = pd.DataFrame({'stratum': [1, 2], 'weight': [3., 3. if no_remainder else 10.]})
    data = xr.Dataset({'aged': aged, 'unaged': aged * 0,
                       'aged_observed': aged, 'unaged_observed': aged * 0})
    if no_remainder:
        result = weight_proportions(data, catch, 'stratum', net_selectivity=True)
        np.testing.assert_allclose(result['proportion'].sum('sex'), [1., 1.])
    else:
        with pytest.raises(ValueError, match='finite positive'):
            weight_proportions(data, catch, 'stratum', net_selectivity=True)


@pytest.mark.parametrize("observed_keys", [[], ["aged_observed"], ["aged_observed", "unaged_observed"]])
def test_selectivity_flag_controls_denominator(observed_keys):
    aged = weight([[2., 5.], [8., 12.]])
    data = xr.Dataset({"aged": aged, "unaged": aged * 2})
    for key in observed_keys:
        data[key] = aged / 2
    catch = pd.DataFrame({"stratum": [1, 2], "weight": [100., 50.]})
    result = weight_proportions(data, catch, "stratum", net_selectivity=False)
    np.testing.assert_allclose(result["proportion"].sum("sex"), [7 / 100, 20 / 50])


def test_selectivity_requires_observed_arrays():
    aged = weight([[1., 2.], [1., 2.]])
    with pytest.raises(ValueError, match="missing required variables"):
        weight_proportions(xr.Dataset({"aged": aged, "unaged": aged}),
                           pd.DataFrame(), "stratum", net_selectivity=True)


@pytest.mark.parametrize("net_selectivity", [False, True])
@pytest.mark.parametrize("missing", ["aged", "unaged", "aged_observed", "unaged_observed"])
def test_required_weight_keys(net_selectivity, missing):
    aged = weight([[1., 2.], [1., 2.]])
    data = xr.Dataset({key: aged for key in
                       ["aged", "unaged", "aged_observed", "unaged_observed"]
                       if key != missing})
    catch = pd.DataFrame({"stratum": [1, 2], "weight": [10., 10.]})
    if net_selectivity or missing in {"aged", "unaged"}:
        with pytest.raises(ValueError, match=missing):
            weight_proportions(data, catch, "stratum", net_selectivity=net_selectivity)
    else:
        result = weight_proportions(data, catch, "stratum", net_selectivity=False)
        np.testing.assert_allclose(result["proportion"].sum("sex"), [0.3, 0.3])


@pytest.mark.parametrize("remaining", [0., 0.819, 0.999, 1., 1.001])
@pytest.mark.parametrize("valid_ratio", [False, True])
def test_small_catch_remainder_ratio_fallback(remaining, valid_ratio):
    observed_aged = weight([[1., 2.], [1., 2.]])
    expanded_aged = observed_aged * 2
    unaged = observed_aged if valid_ratio else observed_aged * 0
    data = xr.Dataset({"aged": expanded_aged, "aged_observed": observed_aged,
                       "unaged": unaged * 2, "unaged_observed": unaged})
    catch = pd.DataFrame({"stratum": [1, 2], "weight": [3. + remaining] * 2})
    if not valid_ratio and remaining >= 1.:
        with pytest.raises(ValueError, match="finite positive"):
            weight_proportions(data, catch, "stratum", net_selectivity=True)
    else:
        result = weight_proportions(data, catch, "stratum", net_selectivity=True)
        ratio = 2. if valid_ratio else 1.
        np.testing.assert_allclose(result["proportion"].sum("sex"),
                                   [6. / (6. + remaining * ratio)] * 2)
