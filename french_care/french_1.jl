using JudiLing # our package
using CSV # read csv files into dataframes
using BenchmarkTools
using Random

# load latin file
french_train = CSV.DataFrame!(CSV.File(joinpath(@__DIR__, "data", "french_train.csv")))
french_val = CSV.DataFrame!(CSV.File(joinpath(@__DIR__, "data", "french_val.csv")))

# create C matrixes for training datasets
cue_obj_train, cue_obj_val = JudiLing.make_cue_matrix(
  french_train,
  french_val,
  grams=2,
  target_col=:Syllables,
  tokenized=true,
  sep_token="-",
  keep_sep=true
  )

# create C matrixes for training datasets
cue_obj_train1, cue_obj_val1 = JudiLing.make_combined_cue_matrix(
  french_train,
  french_val,
  grams=2,
  target_col=:Syllables,
  tokenized=true,
  sep_token="-",
  keep_sep=true
  )

# retrieve dim of C
# we set the S matrixes as the same dimensions
n_features = size(cue_obj_train.C, 2)
S_train, S_val = JudiLing.make_S_matrix(
  french_train,
  french_val,
  ["Lexeme"],
  ["Person", "Number", "Gender", "Tense", "Aspect", "Class", "Mood"],
  ncol=n_features,
  add_noise=true)

# we use cholesky function to calculate mapping G from S to C
G_train = JudiLing.make_transform_matrix(S_train, cue_obj_train.C)

# we calculate Chat matrixes by multiplying S and G 
Chat_train = S_train * G_train
Chat_val = S_val * G_train

@show JudiLing.eval_SC(Chat_train, cue_obj_train.C)
@show JudiLing.eval_SC(Chat_train, cue_obj_train.C, french_train, :Syllables)
@show JudiLing.eval_SC(Chat_val, cue_obj_val.C)
@show JudiLing.eval_SC(Chat_val, cue_obj_val.C, french_val, :Syllables)

# we calculate F as we did for G
F_train = JudiLing.make_transform_matrix(cue_obj_train.C, S_train)

# we calculate Shat as we did for Chat
Shat_train = cue_obj_train.C * F_train
Shat_val = cue_obj_val.C * F_train

@show JudiLing.eval_SC(Shat_train, S_train)
@show JudiLing.eval_SC(Shat_train, S_train, french_train, :Syllables)
@show JudiLing.eval_SC(Shat_val, S_val)
@show JudiLing.eval_SC(Shat_val, S_val, french_val, :Syllables)

# here we only use a adjacency matrix as we got it from training dataset
A_train = cue_obj_train1.A

# we calculate how many timestep we need for learn_paths and huo function
max_t = JudiLing.cal_max_timestep(french_train, french_val, :Syllables)