include("../folder_path.jl")
import Pkg
Pkg.activate(project_folder)

using JLD2, BSON, Flux, BioSeqInt, Statistics, Random, ValueHistories, ProgressMeter, LinearAlgebra
include(project_folder*"cnn_train/cnn_model.jl")
include(project_folder*"cnn_train/train_utils.jl")

#loading data a creating a small dataset
file = load("$(project_folder)my_data/data_experiment1.jld2")
sequences_experiment1 = file["sequences_experiment1"]
counts_experiment1 = file["counts_experiment1"]
sequences_experiment1, counts_experiment1 = filter_counts(sequences_experiment1, counts_experiment1)

θthreshold1 = -1.0

θexperiment1 = [log(counts_experiment1[m,2] / counts_experiment1[m,1]) for m in axes(counts_experiment1, 1)]
labels_experiment1 = Int.(θexperiment1 .>= θthreshold1) .+ 1 |> x -> Flux.OneHotArray(Flux.batch(x), 2) |> Array{Float32, 2}
sequences_experiment1_1hot=string2int(sequences_experiment1)

mb_sizes = [32, 64, 128, 256, 512]
models = [create_model() for _ in eachindex(mb_sizes)]
histories = [MVHistory() for _ in eachindex(mb_sizes)]

nsamples = size(sequences_experiment1_1hot, 4)
n_mini_batches = [(nsamples ÷ m) + (nsamples % m > 0 ? 1 : 0) for m in mb_sizes]
epochs = 1:20
p = Progress(length(epochs)*sum(n_mini_batches))

Threads.@threads for i in eachindex(mb_sizes)

    train_model_regularization!(models[i], sequences_experiment1_1hot, labels_experiment1, p; 
                            regularization_intensity = 0.01, epochs=epochs, minibatch_size=mb_sizes[i], history=histories[i])
end

BSON.@save project_folder*"cnn_models/finetune_experiment1.bson" models histories