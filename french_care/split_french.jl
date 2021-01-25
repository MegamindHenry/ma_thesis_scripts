using JudiLing # our package
using CSV # read csv files into dataframes
using BenchmarkTools
using Random

JudiLing.train_val_split(
    joinpath(@__DIR__, "data", "french.csv"),
    joinpath(@__DIR__, "data"),
    ["Lexeme", "Person", "Number", "Gender", "Tense", "Aspect", "Class", "Mood"],
    data_prefix="french",
    split_max_ratio=0.05,
    n_grams_target_col=:Syllables,
    n_grams_tokenized=true,
    n_grams_sep_token="-",
    grams=2,
    n_grams_keep_sep=true,
    start_end_token="#",
    random_seed=314,
    verbose=true
  )
