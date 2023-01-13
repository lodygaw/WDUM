using Pkg
Pkg.activate("..")

using Flux

# import data
include("datasets.jl")
X_train, y_train = load_dataset("BasicMotions", "train");

ranges = [  Interval{Closed, Closed}(0.1, 0.4),
            Interval{Open, Closed}(0.4, 0.7),
            Interval{Open, Closed}(0.7,1.0)]

X_ = trim_timeseries(X_train, y_train, ranges)

# convert to array of arrays
data = [X_[i,:,:] for i in 1:size(X_,1)]

# strip NaN's
data = map(x->x[:,1:(any(isnan.(x)) ? findfirst(y->isnan(y), x[1,:]) - 1 : size(x,2))], data)

# data = Flux.normalise(data[1],dims=1)


# Define the sequence length
# sequence_length = 1

# train_data = []
# for i in 1:length(data)
#     for j in 1:(size(data[i], 2) - sequence_length)
#         x = data[i][:, j:(j+sequence_length-1)]
#         y = data[i][:, (j+1):(j+sequence_length)]
#         push!(train_data, (x, y))
#     end
# end


input = Flux.unstack(data[1][:,1:end-1], 2)
target = Flux.unstack(data[1][:,2:end], 2)

input = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
target = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

hidden_size = 10
n_features = 6

model = Chain(
        LSTM(n_features, hidden_size),
        LSTM(hidden_size, hidden_size),
        Dense(hidden_size, n_features)
    )

function loss(x, y)
    Flux.reset!(model)
    Flux.mse(model(x), y)
end

test_data = data[:, 1:end-1]
predictions = model(test_data)

# Train the model
for _ in 1:100
    # shuffle!(train_data)
    for (xt, yt) in zip(input, target) # Loop over each time step
        Flux.train!(loss, Flux.params(model), [(xt,yt)], Adam())
    end
    # Flux.train!(loss, Flux.params(model), train_data, Adam(0.02))
    # Test the model
    # predictions = model(data[:, 1:end-1])

    # println(loss(predictions,data[:, 2:end]))
end



