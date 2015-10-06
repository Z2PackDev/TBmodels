#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    05.10.2015 16:54:52 CEST
# File:    supercell.py

from common import *

class SupercellTestCase(BuildDirTestCase):
    def createH(self, t1, t2):

        builder = tbmodels.Builder()

        # create the two atoms
        builder.add_atom([1], [0, 0, 0], 1)
        builder.add_atom([-1], [0.5, 0.5, 0], 0)

        # add hopping between different atoms
        builder.add_hopping(((0, 0), (1, 0)),
                           tbmodels.helpers.combine([0, -1], [0, -1], 0),
                           t1,
                           phase=[1, -1j, 1j, -1])

        # add hopping between neighbouring orbitals of the same type
        builder.add_hopping(((0, 0), (0, 0)),
                           tbmodels.helpers.neighbours([0, 1],
                                                        forward_only=True),
                           t2,
                           phase=[1])
        builder.add_hopping(((1, 0), (1, 0)),
                           tbmodels.helpers.neighbours([0, 1],
                                                        forward_only=True),
                           -t2,
                           phase=[1])
        self.model = builder.create()

    def test_periodic_zero(self):
        self.createH(0., 0.)

        supercell_model = self.model.supercell([1, 2, 1])
        res = in_place_replace(supercell_model.hamilton([0.1, 0.2, 0.7]))
        res_uc = in_place_replace(supercell_model.uc)

        self.assertFullAlmostEqual(res, supercell_model.hamilton([0.1, 0.2, 0.7]))
        self.assertFullAlmostEqual(res_uc, supercell_model.uc)

    def test_periodic(self):
        self.createH(0.2, 0.3)

        supercell_model = self.model.supercell([1, 2, 3])
        res = in_place_replace(supercell_model.hamilton([0.1, 0.2, 0.7]))
        res_uc = in_place_replace(supercell_model.uc)

        self.assertFullAlmostEqual(res, supercell_model.hamilton([0.1, 0.2, 0.7]))
        self.assertFullAlmostEqual(res_uc, supercell_model.uc)

    def test_non_periodic_1(self):
        self.createH(0.2, 0.3)

        supercell_model = self.model.supercell([1, 2, 3], periodic=[True, True, False])
        res = in_place_replace(supercell_model.hamilton([0.1, 0.2, 0.7]))
        res_uc = in_place_replace(supercell_model.uc)

        self.assertFullAlmostEqual(res, supercell_model.hamilton([0.1, 0.2, 0.7]))
        self.assertFullAlmostEqual(res_uc, supercell_model.uc)

    def test_non_periodic_2(self):
        self.createH(0.2, 0.3)

        supercell_model = self.model.supercell([1, 2, 3], periodic=[True, False, True])
        res = in_place_replace(supercell_model.hamilton([0.1, 0.2, 0.7]))
        res_uc = in_place_replace(supercell_model.uc)

        self.assertFullAlmostEqual(res, supercell_model.hamilton([0.1, 0.2, 0.7]))
        self.assertFullAlmostEqual(res_uc, supercell_model.uc)

    def test_non_periodic_3(self):
        self.createH(0.2, 0.3)

        supercell_model = self.model.supercell([1, 2, 3], periodic=[False, True, True])
        res = in_place_replace(supercell_model.hamilton([0.1, 0.2, 0.7]))
        res_uc = in_place_replace(supercell_model.uc)

        self.assertFullAlmostEqual(res, supercell_model.hamilton([0.1, 0.2, 0.7]))
        self.assertFullAlmostEqual(res_uc, supercell_model.uc)

    def test_non_periodic_4(self):
        self.createH(0.9, 0.3)

        supercell_model = self.model.supercell([1, 2, 3], periodic=[False, False, True])
        res = in_place_replace(supercell_model.hamilton([0.1, 0.2, 0.7]))
        res_uc = in_place_replace(supercell_model.uc)

        self.assertFullAlmostEqual(res, supercell_model.hamilton([0.1, 0.2, 0.7]))
        self.assertFullAlmostEqual(res_uc, supercell_model.uc)

    def test_non_periodic_5(self):
        self.createH(0.9, 0.3)

        supercell_model = self.model.supercell([1, 2, 3], periodic=[False, True, False])
        res = in_place_replace(supercell_model.hamilton([0.1, 0.2, 0.7]))
        res_uc = in_place_replace(supercell_model.uc)

        self.assertFullAlmostEqual(res, supercell_model.hamilton([0.1, 0.2, 0.7]))
        self.assertFullAlmostEqual(res_uc, supercell_model.uc)

    def test_non_periodic_6(self):
        self.createH(0.9, 0.3)

        supercell_model = self.model.supercell([1, 2, 3], periodic=[True, False, False])
        res = in_place_replace(supercell_model.hamilton([0.1, 0.2, 0.7]))
        res_uc = in_place_replace(supercell_model.uc)

        self.assertFullAlmostEqual(res, supercell_model.hamilton([0.1, 0.2, 0.7]))
        self.assertFullAlmostEqual(res_uc, supercell_model.uc)

    def test_non_periodic_7(self):
        self.createH(0.1, -0.3)

        supercell_model = self.model.supercell([1, 2, 3], periodic=[False, False, False])
        res = in_place_replace(supercell_model.hamilton([0.1, 0.2, 0.7]))
        res_uc = in_place_replace(supercell_model.uc)

        self.assertFullAlmostEqual(res, supercell_model.hamilton([0.1, 0.2, 0.7]))
        self.assertFullAlmostEqual(res_uc, supercell_model.uc)

    def test_passivation_1(self):
        self.createH(0.1, -0.3)

        supercell_model = self.model.supercell([1, 2, 3], passivation = lambda x, y, z: ([1., 2.] if x else None))
        res = in_place_replace(supercell_model.hamilton([0.1, 0.2, 0.7]))
        res_uc = in_place_replace(supercell_model.uc)

        self.assertFullAlmostEqual(res, supercell_model.hamilton([0.1, 0.2, 0.7]))
        self.assertFullAlmostEqual(res_uc, supercell_model.uc)

    def test_passivation_2(self):
        self.createH(0., 0.)

        supercell_model = self.model.supercell([1, 2, 3], passivation = lambda x, y, z: ([1., -2.] if (y and z) else None))
        res = in_place_replace(supercell_model.hamilton([0.1, 0.2, 0.7]))
        res_uc = in_place_replace(supercell_model.uc)

        self.assertFullAlmostEqual(res, supercell_model.hamilton([0.1, 0.2, 0.7]))
        self.assertFullAlmostEqual(res_uc, supercell_model.uc)

if __name__ == "__main__":
    unittest.main()
