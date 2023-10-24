include("../folder_path.jl")
import Pkg
Pkg.activate(project_folder)

using JLD2, BSON, PhageTree, BioSeqInt, Flux, ValueHistories, LinearAlgebra
include(project_folder*"phagetree_train/utils.jl")

#loading data a creating a small dataset
file1 = load("$(project_folder)my_data/data_experiment1.jld2")
sequences_experiment1 = file1["sequences_experiment1"]
counts_experiment1 = file1["counts_experiment1"]

file2 = load("$(project_folder)my_data/data_experiment2.jld2")
sequences_experiment2 = file2["sequences_experiment2"]
counts_experiment2 = file2["counts_experiment2"]

sequences_experiment1_1hot = string2int(sequences_experiment1)
plasmid_experiment1 = PhageTree.Experiment(sequences_experiment1_1hot, counts_experiment1[:,1], "plasmid_exp1")
virus_experiment1 = PhageTree.Experiment(sequences_experiment1_1hot, counts_experiment1[:,2], plasmid_experiment1, "virus_exp1")
sequences_experiment2_1hot = string2int(sequences_experiment2)
plasmid_experiment2 = PhageTree.Experiment(sequences_experiment2_1hot, counts_experiment2[:,1], "plasmid_exp2")
virus_experiment2 = PhageTree.Experiment(sequences_experiment2_1hot, counts_experiment2[:,2], plasmid_experiment2, "virus_exp2")
data_experiment = PhageTree.Data(plasmid_experiment1, plasmid_experiment2)


minibatch = 256
epochs = 1:200
model = Model(
    (PhageTree.ZeroEnergy(), PhageTree.DeepEnergy(create_model_energy())),
    zeros(2,2),
    zeros(2),
    [false false; true true],
    [true true; false false]
)

history = learn!(model, data_experiment; 
        epochs = epochs, reg = () -> l2reg(model, 0.01, (2,5,10,12,15)), batchsize=minibatch)

BSON.@save project_folder*"phagetree_models/train_experiment2_b256.bson" model history minibatch epochs


