results/compare_bands.pdf: data/model_nosym.hdf5 data/reference_model.hdf5 results/model_symmetrized.hdf5
	python compare_bands.py

results/model_symmetrized.hdf5: data/model_nosym.hdf5 results/symmetries.hdf5
	tbmodels symmetrize -i $< -s $(word 2,$^) -o $@

results/symmetries.hdf5: generate_symmetries.py data/model_nosym.hdf5
	mkdir -p results
	python generate_symmetries.py

clean:
	rm -r results
