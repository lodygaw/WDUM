module WDUM
export Datasets, Forecasts, Rocket
	module Datasets
		export Interval, Closed, Open
		export load_dataset, download_dataset, plot_timeseries, trim_timeseries, show_stratification
		include("datasets.jl")
	end
	module Forecasts
		export Mean, NaiveSingle, NaiveMultiple
		export forecast
		include("forecast.jl")
	end
	module ROCKET
		export Rocket, fit!, transform!
		include("rocket.jl")
	end
end 