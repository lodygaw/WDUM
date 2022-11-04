#=
ROCKET.
RandOm Convolutional KErnel Transform
@article{dempster_etal_2019,
  author  = {Dempster, Angus and Petitjean, Francois and Webb,
  Geoffrey I},
  title   = {ROCKET: Exceptionally fast and accurate time series
  classification using random convolutional kernels},
  year    = {2019},
  journal = {arXiv:1910.13051}
}
=#
import Random: seed!
using StaticArrays
using Distributions: Uniform, Normal
using LinearAlgebra: dot
using StatsBase: sample, mean, std

mutable struct Rocket
	num_kernels :: Int64
	normalize :: Bool
	seed :: Union{Int64, Nothing}
	kernels :: Union{Tuple, Nothing} 
	n_columns :: Int64
end

Rocket(num_kernels=10_000, normalize=true, seed=nothing) = Rocket(num_kernels, normalize, seed, nothing, 0)

# size(X) = (n_instances, n_dimensions, series_length)
function fit!(r::Rocket, X::Array{Float64, 3})
	_, r.n_columns, n_timepoints = size(X)
	r.kernels = generate_kernels(n_timepoints, r.num_kernels, r.n_columns, r.seed);
end

function transform!(r::Rocket, X::Array{Float64, 3})
	if r.normalize
		# numpy is row-major while julia column-major - it has to be fixed here probably (or while reading data)
		X .= (X .- mean(X, dims=3)) ./ (std(X, dims=3) .+ 1e-8)
	end

	t = apply_kernels(X, r.kernels)
	return t
end

function generate_kernels(n_timepoints::Int64, num_kernels::Int64, n_columns::Int64, seed::Union{Int64,Nothing})
	if !isnothing(seed)
		seed!(seed)
	end

	candidate_lengths = [7, 9, 11]
	lengths = sample(candidate_lengths, num_kernels)

	num_channel_indices = map(x->floor(Int, 2^rand(Uniform(0, log2(min(n_columns, x) + 1)))), lengths)

	channel_indices = zeros(Int, sum(num_channel_indices))
	weights 			= zeros(dot(lengths, num_channel_indices))
	biases 				= zeros(num_kernels)
	dilations 		= zeros(num_kernels)
	paddings 			= zeros(num_kernels)

	a₁ = 1 		# for weights
	a₂ = 1 		# for channel indices

	for i in 1:num_kernels

		_length = lengths[i]
		_num_channel_indices = num_channel_indices[i]

		_weights = rand(Normal(), _num_channel_indices * _length)

		b₁ = a₁ + (_num_channel_indices * _length)
		b₂ = a₂ + _num_channel_indices

		a₃ = 1 		# for weights (per channel)

		for _ in 1:_num_channel_indices
			b₃ = a₃ + _length
			_weights[a₃:b₃-1] .-= mean(_weights[a₃:b₃-1])
			a₃ = b₃
		end

		weights[a₁:b₁-1] .= _weights

		channel_indices[a₂:b₂-1] .= sample(collect(1:10), _num_channel_indices, replace=false)

		biases[i] = rand(Uniform(-1,1))

		dilations[i] = dilation = floor(Int, 2^rand(Uniform(0, log2((n_timepoints-1)/(_length-1)))))

		paddings[i] = rand(Bool) == true ? floor(Int, ((_length - 1) * dilation) / 2) : 0 

		a₁ = b₁
		a₂ = b₂
	end

	return weights, lengths, biases, dilations, paddings, num_channel_indices, channel_indices
end

function apply_kernels(X::Array{Float64, 3}, kernels::Tuple)
	weights, lengths, biases, dilations, paddings, num_channel_indices, channel_indices = kernels

	n_instances, n_columns, _ = size(X)
	num_kernels = length(lengths)

	_X = zeros(n_instances, num_kernels*2)

	for i in 1:n_instances
		a₁ = 1 			# for weights		
		a₂ = 1 			# for channel_indices
		a₃ = 1 			# for features

		for j in 1:num_kernels
			b₁ = a₁ + num_channel_indices[j] * lengths[j]
			b₂ = a₂ + num_channel_indices[j]
			b₃ = a₃	+ 2

			@assert num_channel_indices != 1 "Univariate case not implemented!"

			_weights = reshape(weights[a₁:b₁-1],(num_channel_indices[j], lengths[j]))

			_X[i, a₃:b₃-1] .= apply_kernel_multivariate(
				X[i,:,:],
				_weights,
				lengths[j],
				biases[j],
				dilations[j],
				paddings[j],
				num_channel_indices[j],
				channel_indices[a₂:b₂-1]
				)

			a₁ = b₁
			a₂ = b₂
			a₃ = b₃	
		end
	end
end

function apply_kernel_multivariate(X, weights, length, bias, dilation, padding, num_channel_indices, channel_indices)
	n_columns, n_timepoints = size(X)

	output_length = (n_timepoints + (2 * padding)) - ((length - 1) * dilation)

	_ppv = 0
	_max = -Inf

	endl = (n_timepoints + padding) - ((length - 1) * dilation)

	for i = -padding:(endl-1)
		_sum = bias
		index = i

		for j=1:length
			if (index > -1) && (index < n_timepoints)
				for k=1:num_channel_indices
					_sum += weights[k,j] * X[channel_indices[k], index]
				end
			end
			index = index + dilation
			
			_sum > _max ? _max = _sum : nothing
			_sum > 0    ? _ppv += 1   : nothing
		end
	end
	return (_ppv/output_length), _max
end	

