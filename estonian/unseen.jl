using JudiLing # our package
using CSV # read csv files into dataframes
using BenchmarkTools

# load latin file
estonian_train = CSV.DataFrame!(CSV.File(joinpath(@__DIR__, "data", "estonian_train.csv")))
estonian_val = CSV.DataFrame!(CSV.File(joinpath(@__DIR__, "data", "estonian_val.csv")))

cue_obj_train = JudiLing.make_cue_matrix(
  estonian_train,
  # estonian_val,
  grams=3,
  target_col=:Word,
  tokenized=false,
  keep_sep=false
  )

cues = cue_obj_train.f2i

words = estonian_val.Word

for w in words
  t = split(w, "")
  ngs = JudiLing.make_ngrams(t, 3, false)
  for ng in ngs
    if haskey(cues, ng)
      # println(1)
    else
      println(2)
    end
  end
end