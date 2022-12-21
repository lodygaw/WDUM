import DataFrames, ARFFFiles
using Random: shuffle!, shuffle
using StatsBase: sample
using Plots

abstract type Bound end
abstract type Closed <: Bound end
abstract type Open <: Bound end

struct Interval{T, L<:Bound, R<:Bound}
    left :: T
    right :: T
end

Interval{L,R}(a::T, b::T) where {L,R,T} = Interval{T,L,R}(a,b)

function sample_end_index(n_timepoints::Int, interval::Interval{T,L,R}) where {T,R,L}

    mod, left = Tuple{Float64, Int}(modf(interval.left * n_timepoints))
    if !((mod == 0.0) && (L == Closed)) left += 1 end  

    mod, right = Tuple{Float64, Int}(modf(interval.right * n_timepoints))
    if (mod == 0.0) && (R == Open) right -= 1 end  

    end_index = sample(collect(left:right))

    return end_index
end


function load_dataset(name::String, split="", dataset_directory="./datasets")
    if split == ""
        X_train, y_train = load_dataset(name, "train", dataset_directory)
        X_test, y_test = load_dataset(name, "test", dataset_directory)
        return X_train, y_train, X_test, y_test
    end
    
    @assert split in ["train", "test"] """You must specify "train" or "test" as a split parameter!"""
    @info "Loading dataset: $(name), scope: $(split)"
    
    r = Regex("$(name)Dimension[0-9]*_$(uppercase(split)).arff")
    filenames = filter(x->match(r,x)!==nothing, readdir("$(dataset_directory)/$(name)/"))
    
    X = permutedims(cat((Array{Float32}(ARFFFiles.load(DataFrames.DataFrame, "$(dataset_directory)/$(name)/$(filename)")[:,1:end-1]) for filename in filenames)..., dims=3), (1,3,2))
    y = Array{String}(ARFFFiles.load(DataFrames.DataFrame, "$(dataset_directory)/$(name)/$(first(filenames))")[:,end])
    
    n_instances, n_columns, n_timepoints = size(X)
    @info "Dataset size: \n\tInstances: $(n_instances) \n\tColumns: $(n_columns) \n\tTimepoints: $(n_timepoints)"
    return X,y
end


function split_array_into_chunks(v::Vector, chunks::Int)
    len = length(v)
    mod = len%chunks
    n_chunks = [(len-mod)÷chunks for _ in 1:chunks]
    n_chunks[1:mod] .+= 1 

    #shuffle vector to randomize 
    shuffle!(v)

    result = []
    a = 1
    b = 1
    for (i, n_chunk) in enumerate(n_chunks)
        b = a + n_chunk - 1
        push!(result, v[a:b])
        a = b + 1
    end
    return result
end

# trimming timeseries with stratification over classes:
# ranges - Tuple of Interval structs
# data is distributed evenly between the intervals
# trimming is being done by substituting trimmed data with NaN
function trim_timeseries(X::Array{T,3}, y::Array{String,1}, ranges::Vector) where {T<:AbstractFloat}
    n_ranges = length(ranges)
    classes = unique(y)

    n_timeseries, n_columns, n_timepoints = size(X)

    _X = similar(X)
    _X .= X

    for class in classes
        chunks = shuffle(split_array_into_chunks(findall(x->x==class, y), n_ranges))
        @debug """Trimming timeseries from "$(class)"" class ($(length(findall(x->x==class, y))) elements) \n Dividing into $(n_ranges) ranges consisting of $(length.(chunks)) elements"""
        for i in 1:n_ranges
            for j in chunks[i]
                end_index = sample_end_index(n_timepoints, ranges[i]) 
                _X[j, :, end_index+1:end] .= NaN
            end
        end
    end
    return _X
end


function show_stratification(X,y)
    for class in unique(y)
        println("Class $(class) stratification ($(length(findall(x->x==class, y))) elements):")
        println("  Range 0.1<=x<=0.4: \t$(count(x->(0.1<=x<=0.4),[count(x->!isnan(x), X[i,1,:])/size(X)[3] for i in findall(x->x==class,y)]))/$(length(findall(x->x==class,y)))")
        println("  Range 0.4<x<=0.7: \t$(count(x->(0.4<x<=0.7),[count(x->!isnan(x), X[i,1,:])/size(X)[3] for i in findall(x->x==class,y)]))/$(length(findall(x->x==class,y)))")
        println("  Range 0.7<x<=1.0: \t$(count(x->(0.7<x<=1.0),[count(x->!isnan(x), X[i,1,:])/size(X)[3] for i in findall(x->x==class,y)]))/$(length(findall(x->x==class,y)))")
        println("")
    end
end

function plot_timeseries(X::Array{T,2}) where {T<:AbstractFloat}
    n_instances, n_timepoints = size(X)
    y = [X[i,:] for i in 1:n_instances]
    x = collect(1:n_timepoints)

    plot(x, y, layout=(n_instances, 1), legend=false)
end