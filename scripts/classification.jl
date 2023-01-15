using Pkg
Pkg.activate("..")

using WDUM.Datasets
using WDUM.Forecasts
using WDUM.ROCKET

using ScikitLearn: @sk_import

@sk_import linear_model : RidgeClassifierCV
@sk_import preprocessing : StandardScaler
@sk_import pipeline : make_pipeline

X_train, y_train, X_test, y_test = load_dataset("Cricket");

ranges = [Interval{Closed, Closed}(0.1, 0.4),
          Interval{Open, Closed}(0.4, 0.7),
          Interval{Open, Closed}(0.7,1.0)]

X_train = trim_timeseries(X_train, y_train, ranges)
X_test = trim_timeseries(X_test, y_test, ranges)

# show_stratification(X_train, y_train)

target_length = 1000

X_train_forecasted_org = forecast(NaiveSingle, X_train, target_length, 4, 3)
X_test_forecasted_org = forecast(NaiveSingle, X_test, target_length, 4, 3)

X_train_forecasted_mult = forecast(NaiveMultiple, X_train, target_length, 100, 20, 0.4f0)
X_test_forecasted_mult = forecast(NaiveMultiple, X_test, target_length, 100, 20, 0.4f0)

X_train_forecasted_mean = forecast(Mean, X_train, target_length)
X_test_forecasted_mean = forecast(Mean, X_test, target_length)

rocket_org = Rocket()
rocket_mean = Rocket()
rocket_mult = Rocket()

fit!(rocket_org, X_train_forecasted_org);
fit!(rocket_mean, X_train_forecasted_mean);
fit!(rocket_mult, X_train_forecasted_mean);

X_train_transform_org = transform!(rocket_org, X_train_forecasted_org)
X_train_transform_mean = transform!(rocket_mean, X_train_forecasted_mean)
X_train_transform_mult = transform!(rocket_mult, X_train_forecasted_mult)

X_test_transform_org = transform!(rocket_org, X_test_forecasted_org)
X_test_transform_mean = transform!(rocket_mean, X_test_forecasted_mean)
X_test_transform_mult = transform!(rocket_mult, X_test_forecasted_mult)

alphas = [1.00000000e-03, 4.64158883e-03, 2.15443469e-02, 1.00000000e-01,
       4.64158883e-01, 2.15443469e+00, 1.00000000e+01, 4.64158883e+01,
       2.15443469e+02, 1.00000000e+03]

ridge_org = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=alphas))
ridge_mean = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=alphas))
ridge_mult = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=alphas))

ridge_org.fit(X_train_transform_org, y_train)
ridge_mean.fit(X_train_transform_mean, y_train)
ridge_mult.fit(X_train_transform_mult, y_train)

score_org = ridge_org.score(X_test_transform_org, y_test)
score_mean = ridge_mean.score(X_test_transform_mean, y_test)
score_mult = ridge_mult.score(X_test_transform_mult, y_test)

@info """ Scores:
       $score_mean -> mean padding
       $score_org -> NaiveSingle
       $score_mult -> NaiveMultiple
"""