struct Inflate_Mat end
struct Squeeze_Mat end

function (a::Inflate_Mat)(x)
    return reshape(x, size(x,1), 1, :)
end

function (a::Squeeze_Mat)(x)
    return  reshape(x, size(x,1), :)
end

function create_model()

    model = Chain(
        Conv((20,7), 1 => 12, relu, pad=SamePad()),
        BatchNorm(12),
        MaxPool((20,2), stride=(1,2)),
        Conv((1,7), 12 => 24, relu, pad=SamePad()),
        BatchNorm(24),
        MaxPool((1,2),stride=(1,2)),
        Flux.flatten,
        Inflate_Mat(),
        Dense(672 => 128, relu),
        BatchNorm(1),
        Dense(128 => 64, relu),
        BatchNorm(1),
        Squeeze_Mat(),
        Dense(64 => 2, identity)
    )

    return model
end