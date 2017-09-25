import tbmodels
import itertools

model = tbmodels.Model(
    on_site=[1, -1], dim=3, occ=1, pos=[[0., 0., 0.], [0.5, 0.5, 0.]]
)

t1, t2 = (0.1, 0.2)
for phase, R in zip([1, -1j, 1j, -1], itertools.product([0, -1], [0, -1],
                                                        [0])):
    model.add_hop(t1 * phase, 0, 1, R)

for R in ((r[0], r[1], 0) for r in itertools.permutations([0, 1])):
    model.add_hop(t2, 0, 0, R)
    model.add_hop(-t2, 1, 1, R)
