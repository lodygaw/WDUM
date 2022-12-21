using StatsBase: mean

function single_timeseries_forecast(X::AbstractArray{T,2}, target_length::Int, window::Int, offset::Int) where {T<:AbstractFloat}
	@assert window > offset "Window has to be larger than offset!"

	n_columns  = size(X)[1]
	n_timepoints = findfirst(x->isnan(x), X[1,:]) - 1

	@debug "Dimension: $(n_columns), current length: $(n_timepoints), target length: $(target_length)"

	# prepare cells
	cells = []
	for i in 1:(n_timepoints-window)
		j = i + window - 1
		push!(cells, X[:,i:j])
	end

	if length(cells) == 0
		push!(cells,X[:,1:n_timepoints])
	end


	while n_timepoints < target_length
		n_forecast = window-offset

		a = n_timepoints + 1
		b = n_timepoints + n_forecast 

		if b > target_length
			b = target_length
			n_forecast = b - a + 1
		end

		@debug "Forecasting $(n_forecast) timepoints"
		
		# get original data for comparison
		org = X[:, (n_timepoints-offset+1):n_timepoints]

		min_error = Inf
		index = -1
		for (i,cell) in enumerate(cells)
			error = MAE(cell[:,1:offset], org)
			if error < min_error
				min_error = error
				index = i
			end
		end

		@debug "Minimal error: $(min_error) achieved with cell $(index)"

		forecast = cells[index][:,(end - n_forecast+1):end]

		X[:, a:b] = forecast

		n_timepoints = b
	end
	return X[:, 1:target_length]
end

@inline function MAE(a,b)
	return sum(abs.(a.-b))/prod(size(a))
end

abstract type Mean end
abstract type Original end

function forecast(type::Type{Original}, X::AbstractArray{T,3}, target_length, window, offset) where {T<:AbstractFloat}
	n_instances, n_columns, _ = size(X)

	_X = zeros(T, n_instances, n_columns, target_length)

	for i in 1:n_instances
		_X[i,:,:] = single_timeseries_forecast(X[i, :, :], target_length, window, offset)
	end

	return _X
end

function forecast(type::Type{Mean}, X::AbstractArray{T,3}, target_length) where {T<:AbstractFloat}
	n_instances, n_columns, _ = size(X)

	_X = zeros(T, n_instances, n_columns, target_length)


	for i in 1:n_instances
		n_timepoints = findfirst(x->isnan(x), X[i,1,:]) - 1
		for j in 1:n_columns
			_X[i, j, 1:target_length] = X[i, j, 1:target_length]
			_X[i, j, (n_timepoints+1):end] .= mean(X[i, j, 1:n_timepoints])
		end
	end

	return _X
end
