function load_data_phagetree(path, tag)
    
    file = JLD2.load(path)
    seqs = file["sequences_$tag"]
    cnts = file["counts_$tag"]
    seqs_1hot=string2int(seqs, channeldim=false)
    plasmid = PhageTree.Experiment(seqs_1hot, cnts[:,1], "plasmid_$tag")
    virus =PhageTree.Experiment(seqs_1hot, cnts[:,2], plasmid, "virus_$tag")
    return PhageTree.Data(plasmid)
end


function load_experiment_phagetree(path, tag)
    file = JLD2.load(path)
    seqs = file["sequences_$tag"]
    cnts = file["counts_$tag"]
    seqs_1hot=string2int(seqs, channeldim=false)
    plasmid = PhageTree.Experiment(seqs_1hot, cnts[:,1], "plasmid_$tag")
    virus = PhageTree.Experiment(seqs_1hot, cnts[:,2], plasmid, "virus_$tag")
    return plasmid,  virus
end

function load_data_cnn(path, threshold, tag)
    
    file = JLD2.load(path)
    seqs = file["sequences_$tag"]
    cnts = file["counts_$tag"]
    seqs, cnts = filter_counts(seqs, cnts)
    
    θ = [log(cnts[m,2] / cnts[m,1]) for m in axes(cnts, 1)]
    labels = Int.(θ .>= threshold) .+ 1 |> x -> Flux.OneHotArray(Flux.batch(x), 2) |> Array{Float32, 2}
    seqs_1hot=string2int(seqs, channeldim=true)
    
    return (seqs_1hot, cnts, labels)
end


function onehot(v; A=21) 
    return Flux.OneHotArray(Flux.batch(v), A) |> Array{Float32, 3}
end


function string2int(sequences; A=21, L=57, channeldim=false)
    int_seqs = aa2int.(sequences)
    int_seqs = map(x->vcat(x, fill(21, L-length(x))), int_seqs)
    if channeldim
        return reshape(onehot(int_seqs, A=A), A, L, 1, :)
    else
        return reshape(onehot(int_seqs, A=A), A, L, :)
    end
end  


function onehot2string(seqs1hot; channel=false, padding=false)
    nsamples = channel ? size(seqs1hot, 4) : size(seqs1hot, 3)
    seqsint = Vector{Vector{Int}}(undef, nsamples)
    if channel
        seqsint .= [findfirst.(seqs1hot[:, i, 1, s].>0.0 for i in axes(seqs1hot,2)) for s in axes(seqs1hot, 4)]
    else 
        seqsint .= [findfirst.(seqs1hot[:, i, s].>0.0 for i in axes(seqs1hot,2)) for s in axes(seqs1hot, 3)]
    end
    if !padding
        return map(x->replace(String(int2aa.(x)), '-'=>""), seqsint)
    end
    return seqsint
end


function filter_counts(sequences, counts; min_counts=100)
    idx_sel = findall(@view(counts[:,1]) .>= min_counts)
    filtered_sequences = sequences[idx_sel]
    filtered_counts = counts[idx_sel, :]
    return filtered_sequences, filtered_counts
end


function cnn_average_score(models, X)

    return mean(m(X) |> x -> x[2,:].-x[1,:] for m in models)
end


function cor_sp(x, y)
    if min(length(x), length(y)) == 0
        return NaN
    end
    idx = intersect(findall(isfinite.(x)), findall(isfinite.(y)))
    return cor(x[idx], y[idx])
end


function roc(ls, θ, θthreshold)
    labels = Int.(θ .>= θthreshold)
    order_idx = sortperm(ls)
    true_pos = zeros(length(order_idx))
    false_pos = zeros(length(order_idx))
    n_pos = sum(labels .== 1)
    n_neg = sum(labels .== 0)
    for k in eachindex(order_idx)
        true_pos[k] = sum(labels[order_idx[k:end]] .== 1)/n_pos
        false_pos[k] = sum(labels[order_idx[k:end]] .== 0)/n_neg
    end

    return true_pos, false_pos
end


function optimal_threshold(ls, true_pos, false_pos)

    gscore = sqrt.(true_pos .* (1.0 .- false_pos))
    opt = argmax(gscore)

    return sort(ls)[opt]
end


function compute_accuracy(pred, thr_pred, proxy, thr_proxy)
    
    @assert length(pred) == length(proxy)
    return (sum((pred .>= thr_pred) .&& (proxy .>= thr_proxy)) +
        sum((pred .< thr_pred) .&& (proxy .< thr_proxy)) ) / length(pred)
end


function auc(x,y)
    bins = [x[i+1]-x[i] for i in 1:length(x)-1]
    return sort([abs(dot(bins, y[1:end-1])), abs(dot(bins, y[2:end]))])
end


function confusion_matrix(ls, ls_thr, theta, theta_thr)
    print("TP: ", sum(( ls .>= ls_thr ).*( theta .>= theta_thr)), 
        "\t FN: ", sum(( ls .< ls_thr ).*( theta .>= theta_thr)), "\n",
        "FP: ", sum(( ls .>= ls_thr ).*( theta .< theta_thr)),
        "\t TN: ", sum(( ls .< ls_thr ).*( theta .< theta_thr)))
    return (tpr = sum(( ls .>= ls_thr ).*( theta .>= theta_thr))/sum(theta .>= theta_thr),
        fpr = sum(( ls .>= ls_thr ).*( theta .< theta_thr))/sum(theta .< theta_thr))
end


function eval_gmm(gmm, x)
    g1 = Normal(gmm.μ[1,1], gmm.Σ[1,1])
    g2 = Normal(gmm.μ[2,1], gmm.Σ[2,1])
    g = MixtureModel([g1, g2], gmm.w)
    return pdf.(g, x)
end


function find_valley(gmm, resolution::Int)
    xmin = gmm.μ[1,1]
    xmax = gmm.μ[2,1]
    r = LinRange(xmin, xmax, resolution)
    y = eval_gmm(gmm, r)
    xvalley = argmin(y)
    return r[xvalley]
end


function binomial_threshold(data::Vector; show_plot=true, resolution)
    
    gmm = GMM(2,1)
    gmm.μ .+= randn(2,1)
    gmm.Σ .*= 1e-1
    
    em!(gmm, reshape(data, :, 1); nIter=20)
    threshold = find_valley(gmm, resolution)
    
    if show_plot
        x=LinRange(minimum(data), maximum(data), 50)
        hist(data, bins=100, density=true)
        plot(x, eval_gmm(gmm, x), color="red")
        axvline(threshold, color="purple")
    end
    
    return threshold
end