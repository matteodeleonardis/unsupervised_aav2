function convert_cnn_model(model)

    new_model = create_model()

    new_model.layers[1].weight .= model.layers[1].weight
    new_model.layers[1].bias .= model.layers[1].bias

    new_model.layers[2].β .= model.layers[2].β
    new_model.layers[2].γ .= model.layers[2].γ
    new_model.layers[2].μ  .= model.layers[2].μ
    new_model.layers[2].σ² .= model.layers[2].σ²

    new_model.layers[4].weight .= model.layers[4].weight
    new_model.layers[4].bias .= model.layers[4].bias

    new_model.layers[5].β .= model.layers[5].β
    new_model.layers[5].γ .= model.layers[5].γ
    new_model.layers[5].μ  .= model.layers[5].μ
    new_model.layers[5].σ² .= model.layers[5].σ²

    #9,11,14
    new_model.layers[9].weight .= model.layers[9].weight
    new_model.layers[9].bias .= model.layers[9].bias

    new_model.layers[10].β .= model.layers[10].β
    new_model.layers[10].γ .= model.layers[10].γ
    new_model.layers[10].μ  .= model.layers[10].μ
    new_model.layers[10].σ² .= model.layers[10].σ²

    new_model.layers[11].weight .= model.layers[11].weight
    new_model.layers[11].bias .= model.layers[11].bias

    new_model.layers[12].β .= model.layers[12].β
    new_model.layers[12].γ .= model.layers[12].γ
    new_model.layers[12].μ  .= model.layers[12].μ
    new_model.layers[12].σ² .= model.layers[12].σ²

    new_model.layers[14].weight .= model.layers[14].weight
    new_model.layers[14].bias .= model.layers[14].bias

    return new_model
end
