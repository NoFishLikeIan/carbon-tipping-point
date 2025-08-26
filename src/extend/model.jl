function Model.d(T, m, model::TippingModel)
    Model.d(T, m, model.damages, model.hogg, model.feedback)
end
function Model.d(T, m, model::LinearModel)
    Model.d(T, m, model.damages, model.hogg)
end
