include("../folder_path.jl")
import Pkg
Pkg.activate(project_folder)

using JLD2, BSON, Flux, BioSeqInt, Statistics, Random, ValueHistories, ProgressMeter, LinearAlgebra, MLDataPattern
include(project_folder*"cnn_train/cnn_model.jl")
include(project_folder*"cnn_train/train_utils.jl")

#loading data a creating a small dataset
file1 = load("$(project_folder)my_data/data_experiment1.jld2")
file2 = load("$(project_folder)my_data/data_experiment2.jld2")
sequences_experiment1 = file1["sequences_experiment1"]
sequences_experiment2 = file2["sequences_experiment2"]
counts_experiment1 = file1["counts_experiment1"]
counts_experiment2 = file2["counts_experiment2"]
sequences_experiment1, counts_experiment1 = filter_counts(sequences_experiment1, counts_experiment1)
sequences_experiment2, counts_experiment2 = filter_counts(sequences_experiment2, counts_experiment2)

θthreshold1 = -1.4792505511322604
θthreshold2 = -0.1828115693971919

θexperiment1 = [log(counts_experiment1[m,2] / counts_experiment1[m,1]) for m in axes(counts_experiment1, 1)]
labels_experiment1 = Int.(θexperiment1 .>= θthreshold1) .+ 1 |> x -> Flux.OneHotArray(Flux.batch(x), 2) |> Array{Float32, 2}
θexperiment2 = [log(counts_experiment2[m,2] / counts_experiment2[m,1]) for m in axes(counts_experiment2, 1)]
labels_experiment2 = Int.(θexperiment2 .>= θthreshold2) .+ 1 |> x -> Flux.OneHotArray(Flux.batch(x), 2) |> Array{Float32, 2}
sequences_experiment1_1hot=string2int(sequences_experiment1)
sequences_experiment2_1hot=string2int(sequences_experiment2)

Xtrain = cat(sequences_experiment1_1hot, sequences_experiment2_1hot, dims=4)
Ytrain = cat(labels_experiment1, labels_experiment2, dims=2)

model = create_model()
minibatch = 2048
nsamples = size(sequences_experiment1_1hot, 4)
n_mini_batches = (nsamples ÷ minibatch) + (nsamples % minibatch > 0 ? 1 : 0)
epochs = 1:200
p = Progress(length(epochs)*n_mini_batches)


history = train_model_regularization!(model, Xtrain, Ytrain, p; 
                        regularization_intensity = 0.01, epochs=epochs, minibatch_size=minibatch)

println("learning completed, now saving the model...")

BSON.@save project_folder*"cnn_models/train_experiment2_thrfit.bson" model history minibatch epochs
