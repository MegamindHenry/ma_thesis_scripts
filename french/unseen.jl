using JudiLing # our package
using CSV # read csv files into dataframes
using Random
using BenchmarkTools

french = CSV.DataFrame!(CSV.File(joinpath(@__DIR__, "data", "french.csv")))
# shuffle data
rng = MersenneTwister(314)
french = french[shuffle(rng, 1:size(french, 1)),:]
# french = french[1:2500,:]
# sr = 0.2
# n = size(french, 1)
# tv = convert(Int64, floor(n*sr))
tv = 1000
french_train = french[1:end-tv,:]
french_val = french[end-tv+1:end,:]

cue_obj_train = JudiLing.make_cue_matrix(
  french_train,
  # estonian_val,
  grams=2,
  target_col=:Syllables,
  tokenized=true,
  sep_token="-",
  keep_sep=true
  )

cues = cue_obj_train.f2i

words = french_val.Syllables


total = 0
for w in words
  t = split(w, "-")
  ngs = JudiLing.make_ngrams(t, 2, true, "-")
  for ng in ngs
    # println(ng)
    if haskey(cues, ng)
      # println(1)
    else
      global total += 1
    end
  end
end

# 113
@show total
