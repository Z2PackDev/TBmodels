import pytest
import numpy as np

import tbmodels


@pytest.mark.parametrize('num_models', range(1, 4))
def test_join_models(sample, num_models):
    model = tbmodels.io.load(sample('InAs_nosym.hdf5'))
    model_list = [model] * num_models
    joined_model = tbmodels.Model.join_models(*model_list)

    for k in [[0., 0., 0.], [0.1231, 0.236, 0.84512]]:
        assert np.allclose(
            sorted(list(model.eigenval(k)) * num_models),
            joined_model.eigenval(k)
        )
