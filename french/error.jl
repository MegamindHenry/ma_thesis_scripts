using DataFrames
using JudiLing # our package
using CSV # read csv files into dataframes
using Random
using BenchmarkTools

df_learn = CSV.DataFrame!(CSV.File(joinpath(@__DIR__, "data", "french_learn_res_val.csv")))
# df_error = df_learn[(df_learn.isbest .== true) .& (df_learn.iscorrect .== false),:]

# df_271 = df_learn[(df_learn.utterance .== 271),:]
# df_319 = df_learn[(df_learn.utterance .== 319),:]


# 1288

# df_correct = df_learn[(df_learn.iscorrect .== true),:]
# @show size(df_correct) # 1115

# df_correct = df_learn[(df_learn.iscorrect .== true) .& (df_learn.isbest .== true),:]
# @show size(df_correct) # 1032

df_pred = df_learn[(df_learn.isbest .== true),:]

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


total = 1000
noval = 0
noval_c = 0
correct = 0

for i in 1:1000
  w = words[i]
  t = split(w, "-")
  ngs = JudiLing.make_ngrams(t, 2, true, "-")
  is_noval = false
  for ng in ngs
    # println(ng)
    if haskey(cues, ng)
      # println(1)
    else
      is_noval = true
    end
  end

  if is_noval
    global noval += 1
    if df_pred[i,:iscorrect] == true
      global noval_c += 1
    end 
  end

  if df_pred[i,:iscorrect] == true
      global correct += 1
    end 
end

@show noval
@show noval_c
@show correct
