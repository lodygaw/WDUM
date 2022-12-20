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
using Distributions: Uniform, Normal
using LinearAlgebra: dot
using StatsBase: sample, mean, std
import Base.Threads: @threads

struct RocketKernel{FLOAT}
	weights 	:: Matrix{FLOAT}
	length 		:: UInt32
	bias 		:: FLOAT
	dilation 	:: Int32
	padding 	:: Int32
	n_channels 	:: UInt32
	channels 	:: Vector{UInt32}
end

mutable struct Rocket{FLOAT, NKERNELS, NORMALIZE}
	kernels 	:: Vector{RocketKernel{FLOAT}} 
	n_columns 	:: UInt32
	n_timepts 	:: UInt32
end

Rocket(num_kernels=10_000, normalize=true, precision=Float32) = Rocket{precision, num_kernels, normalize}(RocketKernel{precision}[], 0, 0)

function fit!(r::Rocket{FLOAT, NKERNELS, NORMALIZE}, X::Array{FLOAT, 3}, seed=nothing) where {FLOAT, NKERNELS, NORMALIZE}
	if !isnothing(seed)
		seed!(seed)
	end

	_, r.n_columns, r.n_timepts = size(X)
	generate_kernels!(r)
end

function transform!(r::Rocket{FLOAT,NKERNELS,NORMALIZE}, X::Array{FLOAT, 3}) where {FLOAT,NKERNELS,NORMALIZE}
	if NORMALIZE
		X = (X .- mean(X, dims=3)) ./ (std(X, dims=3) .+ convert(FLOAT,1e-8))
	end

	transform = apply_kernels(X, r.kernels)
	return transform 
end

function generate_kernels!(r::Rocket{FLOAT,NKERNELS,NORMALIZE}) where {FLOAT,NKERNELS,NORMALIZE}

	candidate_lengths = [7, 9, 11]

	for i in 1:NKERNELS
		_length 	= sample(candidate_lengths)
		_n_channels = floor(Int, 2^rand(Uniform(0, log2(min(r.n_columns, _length) + 1))))
		_weights 	= Matrix{FLOAT}(rand(Normal(0,1), _n_channels, _length))

		for i in 1:_n_channels
			_weights[i,:] .-= mean(view(_weights, i, :))
		end

		_channels 	= sample(collect(1:r.n_columns), _n_channels, replace=false)
		_bias 		= rand(Uniform(-1,1))
		_dilation 	= floor(Int, 2^rand(Uniform(0, log2((r.n_timepts-1)/(_length-1)))))
		_padding 	= rand(Bool) == true ? floor(Int, ((_length - 1) * _dilation) / 2) : 0 

		push!(r.kernels, RocketKernel{FLOAT}(_weights, _length, _bias, _dilation, _padding, _n_channels, _channels))
	end
	return nothing
end

function apply_kernels(X::Array{FLOAT, 3}, kernels::Vector{RocketKernel{FLOAT}}) where {FLOAT}

	n_instances, n_columns, _ = size(X)
	num_kernels = length(kernels)

	_X = zeros(FLOAT, n_instances, num_kernels*2)

	N = length(kernels)

	@inbounds for i in 1:n_instances
		@threads for j in 1:N
			_X[i, (2j-1):2j] .= apply_kernel_multivariate(view(X, i, :, :), kernels[j])
		end
	end
	return _X
end

function apply_kernel_multivariate(X, rk::RocketKernel{FLOAT}) where {FLOAT}
	weights, length, bias, dilation, padding, n_channels, channels = rk.weights, rk.length, rk.bias, rk.dilation, rk.padding, rk.n_channels, rk.channels

	n_columns, n_timepts = size(X)

	output_length = (n_timepts + (2 * padding)) - ((length - 1) * dilation)

	_ppv = convert(FLOAT,0)
	_max = convert(FLOAT,-Inf)

	stop = (n_timepts + padding) - ((length - 1) * dilation) - 1

	for i = -padding:(stop)
		_sum = bias
		index = i

		for j=1:length
			if (index > -1) && (index < n_timepts)
				for k=1:n_channels
					_sum += weights[k,j] * X[channels[k], index+1]
				end
			end
			index = index + dilation
		end	
		_sum > _max ? _max = _sum : nothing
		_sum > 0    ? _ppv += 1   : nothing
	end
	return _ppv/output_length, _max
end	

function apply_kernel_univariate(X, rk::RocketKernel{FLOAT}) where {FLOAT}
	weights, length, bias, dilation, padding = rk.weights, rk.length, rk.bias, rk.dilation, rk.padding

	n_timepts, = size(X)

	output_length = (n_timepts + (2 * padding)) - ((length - 1) * dilation)

	_ppv = convert(FLOAT,0)
	_max = convert(FLOAT,-Inf)

	stop = (n_timepts + padding) - ((length - 1) * dilation) - 1

	for i = -padding:(stop)
		_sum = bias
		index = i

		for j=1:length
			if (index > -1) && (index < n_timepts)
				_sum += weights[j] * X[index+1]
			end
			index = index + dilation
		end	
		_sum > _max ? _max = _sum : nothing
		_sum > 0    ? _ppv += 1   : nothing
		
	end
	return _ppv/output_length, _max
end	