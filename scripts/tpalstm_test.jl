using Pkg
Pkg.activate("..")

using FluxArchitectures
using Plots
using Flux


# Load some sample data
@info "Loading data"
poollength = 10
horizon = 15

include("datasets.jl")
X_train, y_train, X_test, y_test = load_dataset("BasicMotions");

ts = X_train[1,:,:]'

ranges = [  Interval{Closed, Closed}(0.1, 0.4),
            Interval{Open, Closed}(0.4, 0.7),
            Interval{Open, Closed}(0.7,1.0)]

X_ = trim_timeseries(X_train, y_train, ranges)

# convert to array of arrays
data = [X_[i,:,:] for i in 1:size(X_,1)]

# strip NaN's
data = map(x->permutedims(x[:,1:(any(isnan.(x)) ? findfirst(y->isnan(y), x[1,:]) - 1 : size(x,2))]), data)

function prepare_data(data, poollength, horizon; normalise=true)
    datalength = size(data,1) - poollength + 1
    extendedlength = datalength + poollength - 1
    extendedlength > size(data, 1) && throw(ArgumentError("datalength $(datalength) larger than available data $(size(data, 1) - poollength + 1)"))
    (normalise == true) && (data = Flux.normalise(data, dims=1))
    features = similar(data, size(data, 2), poollength, 1, datalength)
    for i in 1:datalength
      features[:,:,:,i] .= permutedims(data[i:(i + poollength - 1) ,:])
    end
    labels = circshift(data[1:datalength,1], -horizon)
    return features, labels
end

dataset = []
for timeseries in data
    input, target = prepare_data(timeseries,poollength,horizon, normalise=false)
    push!(dataset,(input, target))
end


# Define the network architecture
@info "Creating model and loss"
inputsize = 6
hiddensize = 30
layers = 2
filternum = 32
filtersize = 1

# Define the neural net
model = TPALSTM(inputsize, hiddensize, poollength, layers, filternum, filtersize) |> gpu

# MSE loss
function loss(x, y)
    Flux.ChainRulesCore.ignore_derivatives() do
        Flux.reset!(model)
    end
    return Flux.mse(model(x), permutedims(y))
end

# Callback for plotting the training
# cb = function ()
#     Flux.reset!(model)
#     pred = model(input) |> permutedims |> cpu
#     Flux.reset!(model)
#     p1 = plot(pred, label="Predict")
#     p1 = plot!(cpu(target), label="Data", title="Loss $(loss(input, target))")
#     display(plot(p1))
# end

# Training loop
@info "Start loss" loss = loss(dataset[1]...)
@info "Starting training"
i = 1
for (input, target) in dataset
    @info "Training dataset $i"
    Flux.train!(loss, Flux.params(model), Iterators.repeated((input, target), 50), Adam(0.02))#, cb=cb)
    @info "Dataset final loss" loss = loss(dataset[i]...)
    global i += 1
end
@info "Finished"
@info "Final loss" loss = loss(dataset[1]...)

ts = X_test[1,:,:]'
input, target = FluxArchitectures.prepare_data(ts, poollength, 10, horizon, normalise=false)
@info "Test data loss" loss = loss(input, target)
