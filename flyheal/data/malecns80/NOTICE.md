# MaleCNS v1.0: subset of 80 neurons

Data creators: FlyEM (HHMI Janelia), University of Cambridge (Department of Zoology), MRC Laboratory of Molecular Biology, and Google Research. [Original MaleCNS dataset and download page](https://male-cns.janelia.org/download/).

Data license: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). No endorsement of FlyHeal by the data creators is implied. This notice applies to the dataset assets in this directory.

Previous derivation: the visual-to-descending subset distributed by [cobanov/flyjump](https://github.com/cobanov/flyjump), pinned to commit `e34c6614e7d13a1018585f705cdeb59b9f38291d`, `public/data/connectome/graph.json`. Its author selected 80 cells, retained all internal directed edges, assigned graph indices and artificial input/output roles, and supplied simplified transmitter signs. FlyHeal retains that graph byte for byte and adds independently verified source URLs, checksums, a validation report, and this attribution. No upstream application code is included by this data extraction.

Original body IDs, cell types, soma coordinates, predicted consensus neurotransmitter labels, and measured directed contact counts are preserved. Independent checks against the three official v1.0 tables confirmed all 80 neurons and the complete induced subgraph: 1,296 directed edges and 26,029 synaptic contacts. `validation.json` records the scope and outcomes; `manifest.json` records exact URLs and SHA-256 values.

Input injection, motor readout, assumed transmitter signs, normalized weights, and simulated rate dynamics are engineered modeling choices. The subset was selected from visual pathways for a Dino experiment. Its reuse for foraging does not establish an olfactory or food-seeking circuit, physiological accuracy, or whole-brain simulation. Neurotransmitter labels are dataset predictions, not independently measured chemistry.
