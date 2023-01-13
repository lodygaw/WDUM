using StatsBase: mean
using Random: shuffle
import Base.Threads: @threads

function single_timeseries_forecast(X::AbstractArray{T,2}, target_length::Int, window::Int, overlap::Int) where {T<:AbstractFloat}
	@assert window > overlap "Window has to be larger than overlap!"

	n_columns  = size(X)[1]
	n_timepoints = any(isnan, X) ? (findfirst(isnan, X[1,:]) - 1) : size(X,2)

	@assert n_timepoints - window >= 0 "Window is larger than timeseries length!"
	
	@debug "Dimension: $(n_columns), current length: $(n_timepoints), target length: $(target_length)"

	# prepare cells
	cells = []
	for i in 1:(n_timepoints - window + 1)
		j = i + window - 1
		push!(cells, X[:,i:j])
	end


	while n_timepoints < target_length
		n_forecast = window - overlap

		a = n_timepoints + 1
		b = n_timepoints + n_forecast 

		if b > target_length
			b = target_length
			n_forecast = b - a + 1
		end

		@debug "Forecasting $(n_forecast) timepoints"
		
		# get original data for comparison
		org = X[:, (n_timepoints-overlap+1):n_timepoints]

		min_error = Inf
		index = -1
		for (i,cell) in enumerate(cells)
			error = MAE(cell[:,1:overlap], org)
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

function find_best_offset(target, source, overlap)

	min_error = Inf
	offset = 0

	for i in 1:(size(source, 2) - overlap - 2)

		b₁ = size(target, 2)
		a₁ = b₁ - overlap + 1

		a₂ = i
		b₂ = a₂ + overlap - 1
		

		error = MAE(view(target, :, a₁:b₁), view(source,:, a₂:b₂))

		if error < min_error
			min_error = error
			offset = i
		end
	end

	return min_error, offset
end


function attempt_forecast(i::Int, target, data, overlap, threshold)
	flag = false
	
	for j in shuffle(1:size(data, 1)) # check other timeseries in random order

		if i == j continue end # do not forecast data with itself 

		source = data[j]

		error, offset = find_best_offset(target, source, overlap)

		if error < threshold
			a = overlap + offset + 1
			b = size(source, 2)

			@debug "Forecasted timeseries nr $i with timeseries nr $j, overlap = $overlap, source offset = $offset, current length = $(size(target,2)), forecast length = $(size(source[:,a:b],2))"

			target = hcat(target,source[:,a:b])
			flag = true
			break
		end
	end
	return target, flag
end

function perform_best_possible_forecast(i, target, data, overlap)
	
	min_error = Inf
	best = (0, 0)

	for j in 1:size(data, 1)

		if i == j continue end # do not forecast data with itself 
		source = data[j]

		error, offset = find_best_offset(target, source, overlap)

		if error < min_error
			min_error = error
			best = offset, j 
		end
	end

	offset, index = best

	source = data[index]

	# TODO: check if this should not be one of the super-parameters of forecast function 
	# if threshold could not be met allow to forecast only one timestep
	min_forecast = 1

	a₁ = overlap + offset + 1
	b₁ = a₁ + min_forecast 				# might need a check if b₁ is not bigger than length 

	target = hcat(target, source[:,a₁:b₁])
	@debug "Could not reach error threshold, performed best possible forecast of timseries nr $i with timeseries nr $index, error = $min_error, current length = $(size(target,2))"

	return target, true
end

abstract type Mean end
abstract type NaiveSingle end
abstract type NaiveMultiple end

function forecast(type::Type{NaiveSingle}, X::AbstractArray{T,3}, target_length::Int, window::Int, overlap::Int) where {T<:AbstractFloat}
	@info """Forecasting timeseries using "NaiveSingle" method.
	Parameters:
		$target_length - target length
		$window - window
		$overlap - overlap
	"""

	n_instances, n_columns, _ = size(X)

	_X = zeros(T, n_instances, n_columns, target_length)

	for i in 1:n_instances
		_X[i,:,:] = single_timeseries_forecast(X[i, :, :], target_length, window, overlap)
	end

	return _X
end

function forecast(type::Type{Mean}, X::AbstractArray{T,3}, target_length) where {T<:AbstractFloat}
	@info """Forecasting timeseries using mean value padding method.
	Parameters:
		$target_length - target length
	"""

	n_instances, n_columns, _ = size(X)

	_X = zeros(T, n_instances, n_columns, target_length)


	for i in 1:n_instances
		n_timepoints = any(isnan, X[i,1,:]) ? (findfirst(isnan, X[i,1,:]) - 1) : size(X,3)
		for j in 1:n_columns
			_X[i, j, 1:target_length] = X[i, j, 1:target_length]
			_X[i, j, (n_timepoints+1):end] .= mean(X[i, j, 1:n_timepoints])
		end
	end

	return _X
end

function forecast(type::Type{NaiveMultiple}, X::AbstractArray{T,3}, target_length::Int, max_overlap::Int, min_overlap::Int, threshold::T) where {T<:AbstractFloat}
	@info """Forecasting timeseries using "NaiveMultiple" method.
	Parameters:
		$target_length - target length
		$max_overlap - max overlap
		$min_overlap - min overlap
		$threshold - error threshold
	"""

	n_instances, n_columns, _ = size(X)

	# allocate output array
	_X = zeros(T, n_instances, n_columns, target_length)

	# convert input to array of arrays
	data = [X[i,:,:] for i in 1:size(X,1)]

	# strip NaN's
	data = map(x->x[:,1:(any(isnan.(x)) ? findfirst(y->isnan(y), x[1,:]) - 1 : size(x,2))], data)

	@threads for i in 1:n_instances
		target = Array(data[i])
		target_current_length = size(target, 2)

		# if timeseries is longer or equal to target length just copy it to output array
		if target_current_length >= target_length
			_X[i,:,:] = target[:,1:target_length]
			continue			
		end	

		@assert min_overlap <= target_current_length "Minimal overlap is bigger than length of timeseries nr $i"

		start_overlap = min(max_overlap, target_current_length)
		overlap = start_overlap

		@debug "========\nStart forecasting timeseries nr $i, length = $(target_current_length), start overlap = $(overlap)\n========"

		while true

			if overlap > min_overlap
				target, flag = attempt_forecast(i, target, data, overlap, threshold)				
			else
				target, flag = perform_best_possible_forecast(i, target, data, overlap)
			end

			if flag
				if (size(target, 2) >= target_length)
					@debug "Finished forecasting timeseries nr $i"
					_X[i,:,:] = target[:,1:target_length]
					break
				else
					@debug "Resetting overlap to start value"
					overlap = start_overlap
				end
			else
				@debug "Could not forecast with overlap = $overlap and current error threshold\nReducing overlap..."
				overlap = max(ceil(Int, overlap/2), min_overlap)
			end
		end
	end

	return _X
end
