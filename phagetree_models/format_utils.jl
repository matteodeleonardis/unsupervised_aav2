function convert_phagetree_model(model)

    new_model = Model(length(model.ζ))

    #convolutional layers
    new_model.states[2].m.layers[2].bias .= model.states[2].m.layers[2].bias
    new_model.states[2].m.layers[2].weight .= model.states[2].m.layers[2].weight

    new_model.states[2].m.layers[3].β .= model.states[2].m.layers[3].β
    new_model.states[2].m.layers[3].γ .= model.states[2].m.layers[3].γ
    new_model.states[2].m.layers[3].μ  .= model.states[2].m.layers[3].μ
    new_model.states[2].m.layers[3].σ² .= model.states[2].m.layers[3].σ²

    new_model.states[2].m.layers[5].bias .= model.states[2].m.layers[5].bias
    new_model.states[2].m.layers[5].weight .= model.states[2].m.layers[5].weight

    new_model.states[2].m.layers[6].β .= model.states[2].m.layers[6].β
    new_model.states[2].m.layers[6].γ .= model.states[2].m.layers[6].γ
    new_model.states[2].m.layers[6].μ  .= model.states[2].m.layers[6].μ
    new_model.states[2].m.layers[6].σ² .= model.states[2].m.layers[6].σ²

    #dense layers 10,12,15
    new_model.states[2].m.layers[10].bias .= model.states[2].m.layers[10].bias
    new_model.states[2].m.layers[10].weight .= model.states[2].m.layers[10].weight

    new_model.states[2].m.layers[11].β .= model.states[2].m.layers[11].β
    new_model.states[2].m.layers[11].γ .= model.states[2].m.layers[11].γ
    new_model.states[2].m.layers[11].μ  .= model.states[2].m.layers[11].μ
    new_model.states[2].m.layers[11].σ² .= model.states[2].m.layers[11].σ²

    new_model.states[2].m.layers[12].bias .= model.states[2].m.layers[12].bias
    new_model.states[2].m.layers[12].weight .= model.states[2].m.layers[12].weight

    new_model.states[2].m.layers[13].β .= model.states[2].m.layers[13].β
    new_model.states[2].m.layers[13].γ .= model.states[2].m.layers[13].γ
    new_model.states[2].m.layers[13].μ  .= model.states[2].m.layers[13].μ
    new_model.states[2].m.layers[13].σ² .= model.states[2].m.layers[13].σ²

    new_model.states[2].m.layers[15].bias .= model.states[2].m.layers[15].bias
    new_model.states[2].m.layers[15].weight .= model.states[2].m.layers[15].weight

    #model pars
    new_model.μ .= model.μ
    new_model.ζ .= model.ζ
    new_model.select .= model.select
    new_model.washed .= model.washed


    return new_model
end
    

