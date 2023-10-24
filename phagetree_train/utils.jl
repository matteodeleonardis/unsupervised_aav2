onehot(v; A=21) = Flux.OneHotArray(Flux.batch(v), A) |> Array{Float32, 3}


function string2int(sequences; A=21, L=57)
    int_seqs = aa2int.(sequences)
    int_seqs = map(x->vcat(x, fill(21, L-length(x))), int_seqs)
    reshape(onehot(int_seqs, A=A), A, L, :)
end


function create_model_energy()
        
    model = Chain(
        x-> reshape(x, size(x,1), size(x,2), 1, size(x,3)), #adds the channel dimension
        Conv((20,7), 1 => 12, relu, pad=SamePad()),
        BatchNorm(12),
        MaxPool((20,2), stride=(1,2)),
        Conv((1,7), 12 => 24, relu, pad=SamePad()),
        BatchNorm(24),
        MaxPool((1,2),stride=(1,2)),
        Flux.flatten,
        x -> reshape(x, size(x,1), 1, :),
        Dense(672 => 128, relu),
        BatchNorm(1),
        Dense(128 => 64, relu),
        BatchNorm(1),
        x -> reshape(x, size(x,1), :),
        Dense(64 => 2, identity),
        @views x -> x[2,:] .- x[1,:]
    )

    return model
end


l2reg(mod, α, layers) = α*( sum([sum(abs2, mod.states[2].m.layers[i].weight) + sum(abs2, mod.states[2].m.layers[i].bias) for i in layers] ))


