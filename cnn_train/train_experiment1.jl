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

θthreshold1 = -1.4792505511322604

θexperiment1 = [log(counts_experiment1[m,2] / counts_experiment1[m,1]) for m in axes(counts_experiment1, 1)]
labels_experiment1 = Int.(θexperiment1 .>= θthreshold1) .+ 1 |> x -> Flux.OneHotArray(Flux.batch(x), 2) |> Array{Float32, 2}
sequences_experiment1_1hot=string2int(sequences_experiment1)

model = create_model()

minibatch = 128
nsamples = size(sequences_experiment1_1hot, 4)
n_mini_batches = (nsamples ÷ minibatch) + (nsamples % minibatch > 0 ? 1 : 0)
epochs = 1:200
p = Progress(length(epochs)*n_mini_batches)

history = train_model_regularization!(model, sequences_experiment1_1hot, labels_experiment1, p; 
                        regularization_intensity = 0.01, epochs=epochs, minibatch_size=minibatch)


BSON.@save project_folder*"cnn_models/train_experiment1_thrfit.bson" model history minibatch epochs