onehot(v; A=21) = Flux.OneHotArray(Flux.batch(v), A) |> Array{Float32, 3}


function string2int(sequences; A=21, L=57)
    int_seqs = aa2int.(sequences)
    int_seqs = map(x->vcat(x, fill(21, L-length(x))), int_seqs)
    reshape(onehot(int_seqs, A=A), A, L, 1, :)
end    


function train_model_early_stopping!(model, Xtrain, Ytrain, Xvalidation, Yvalidation, progressbar; patience, opt=ADAM(), epochs=1:1, history=MVHistory(),
    minibatch_size)

    es = Flux.early_stopping(()->Flux.logitcrossentropy(model(Xvalidation), Yvalidation), patience, init_score=Inf)

    ps = Flux.params(model)

    for e in epochs
        stop = false
        for (x, y) in RandomBatches((Xtrain, Ytrain), size=minibatch_size)
            gs = gradient(ps) do
                loss = Flux.logitcrossentropy(model(x), y)
                return loss
            end
            Flux.update!(opt, ps, gs)
            
            stop = es()
            if stop
                break
            end
            next!(progressbar)
        end

        push!(history, :training_loss, Flux.logitcrossentropy(model(Xtrain), Ytrain))
        push!(history, :validation_loss, Flux.logitcrossentropy(model(Xvalidation), Yvalidation))
        
        if stop
            break
        end
    end

    return history
end


function train_replicas_early_stopping(Nrep, Xtrain, Ytrain, Xvalidation, Yvalidation; patience, opt=ADAM, epochs=1:1,
    minibatch_size)

    n_mini_batches = size(Xtrain, 4)÷minibatch_size + (size(Xtrain, 4) % minibatch_size > 0 ? 1 : 0)
    p = Progress(length(epochs)*n_mini_batches*Nrep)

    models = [create_model() for _ in 1:Nrep]
    histories = [MVHistory() for _ in 1:Nrep]

    Threads.@threads for k in eachindex(models)

        train_model_early_stopping!(models[k], Xtrain, Ytrain, Xvalidation, Yvalidation, p;
            patience=patience, opt=opt(), epochs=epochs, history=histories[k], minibatch_size=minibatch_size)
    end

    return models, histories
end


l2_reg(regularization_intensity, x) = regularization_intensity * sum(sum(y -> norm(y)^2, x))


function train_model_regularization!(model, Xtrain, Ytrain, progressbar=nothing; regularization_intensity = 0.0, opt=ADAM(), epochs=1:1, history=MVHistory(), minibatch_size)

    
    ps = Flux.params(model)

    for e in epochs
        rand_shuffle = randperm(size(Xtrain, 4))
        minibatches = Flux.chunk(Xtrain[:,:,:,rand_shuffle], size=minibatch_size)
        ymb = Flux.chunk(Ytrain[:,rand_shuffle], size=minibatch_size)
        for (i, mb) in pairs(minibatches)
            gs = gradient(ps) do
                loss = Flux.logitcrossentropy(model(mb), ymb[i])
                return loss + l2_reg(regularization_intensity, ps)
            end
            Flux.update!(opt, ps, gs)
            
            if !(isnothing(progressbar))
                next!(progressbar)
            end
        end

        push!(history, :training_loss, Flux.logitcrossentropy(model(Xtrain), Ytrain))
    end

    return history
end


function train_replicas_regularization(Nrep, Xtrain, Ytrain; regularization_intensity = 0.0, opt=ADAM, epochs=1:1, minibatch_size)

    n_mini_batches = size(Xtrain, 4)÷minibatch_size + (size(Xtrain, 4) % minibatch_size > 0 ? 1 : 0)
    p = Progress(length(epochs)*n_mini_batches*Nrep)


    models = [create_model() for _ in 1:Nrep]
    histories = [MVHistory() for _ in 1:Nrep]

    Threads.@threads for k in eachindex(models)

        train_model_regularization!(models[k], Xtrain, Ytrain, p;
            regularization_intensity = regularization_intensity, epochs=epochs, opt=opt(), history=histories[k], minibatch_size=minibatch_size)
    end

    return models, histories
end


function filter_counts(sequences, counts; min_counts=100)
    idx_sel = findall(@view(counts[:,1]) .>= min_counts)
    filtered_sequences = sequences[idx_sel]
    filtered_counts = counts[idx_sel, :]
    return filtered_sequences, filtered_counts
end