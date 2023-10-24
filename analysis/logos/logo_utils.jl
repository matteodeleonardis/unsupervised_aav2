function frequency_matrix(sequences::Vector)
    w = zeros(21, length(sequences[1]))
    for s in sequences
        for i in eachindex(s)
            w[s[i], i] += 1.0
        end
    end
    w ./= sum(w, dims=1)
    return w
end
    
        