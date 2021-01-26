using JudiLing # our package
using CSV # read csv files into dataframes
using BenchmarkTools

# load latin file
latin = CSV.DataFrame!(CSV.File(joinpath(@__DIR__, "data", "latin.csv")))
# display(latin)

# create C matrixes for training datasets
cue_obj = JudiLing.make_cue_matrix(
  latin,
  grams=3,
  target_col=:Word,
  tokenized=false,
  keep_sep=false
  )

# retrieve dim of C
# we set the S matrixes as the same dimensions
n_features = size(cue_obj.C, 2)
S = JudiLing.make_S_matrix(
  latin,
  ["Lexeme"],
  ["Person","Number","Tense","Voice","Mood"],
  ncol=n_features,
  add_noise=true)

# we use cholesky function to calculate mapping G from S to C
G = JudiLing.make_transform_matrix(S, cue_obj.C)

# we calculate Chat matrixes by multiplying S and G 
Chat = S * G

@show JudiLing.eval_SC(Chat, cue_obj.C)
@show JudiLing.eval_SC(Chat, cue_obj.C, latin, :Word)

# we calculate F as we did for G
F = JudiLing.make_transform_matrix(cue_obj.C, S)

# we calculate Shat as we did for Chat
Shat = cue_obj.C * F

@show JudiLing.eval_SC(Shat, S)
@show JudiLing.eval_SC(Shat, S, latin, :Word)

# here we only use a adjacency matrix as we got it from training dataset
A = cue_obj.A

# we calculate how many timestep we need for learn_paths and huo function
max_t = JudiLing.cal_max_timestep(latin, :Word)

# we calculate learn_paths and build_paths function
res_learn, gpi_learn = JudiLing.learn_paths(
  latin,
  latin,
  cue_obj.C,
  S,
  F,
  Chat,
  A,
  cue_obj.i2f,
  cue_obj.f2i, # api changed in 0.3.1
  check_gold_path=true,
  gold_ind=cue_obj.gold_ind,
  Shat_val=Shat,
  max_t=max_t,
  max_can=10,
  grams=3,
  threshold=0.05, # 0.05345303464774335
  tokenized=false,
  keep_sep=false,
  target_col=:Word,
  verbose=true)

acc_learn = JudiLing.eval_acc(
  res_learn,
  cue_obj.gold_ind,
  verbose=false
)

println("Acc for learn: $acc_learn")

res_build = JudiLing.build_paths(
    latin,
    cue_obj.C,
    S,
    F,
    Chat,
    A,
    cue_obj.i2f,
    cue_obj.gold_ind,
    max_t=max_t,
    n_neighbors=3,
    verbose=true
    )

acc_build = JudiLing.eval_acc(
  res_build,
  cue_obj.gold_ind,
  verbose=false
)

println("Acc for build: $acc_build")

df_learn = JudiLing.write2df(
  res_learn,
  latin,
  cue_obj,
  cue_obj,
  grams=3,
  tokenized=false,
  sep_token=nothing,
  start_end_token="#",
  output_sep_token="",
  path_sep_token=":",
  target_col=:Word
  )

df_learn_gpi = JudiLing.write2df(
  gpi_learn,
  )

JudiLing.write2csv(
  res_learn,
  latin,
  cue_obj,
  cue_obj,
  "latin_learn_res.csv",
  grams=3,
  tokenized=false,
  sep_token=nothing,
  start_end_token="#",
  output_sep_token="",
  path_sep_token=":",
  target_col=:Word,
  root_dir=@__DIR__,
  output_dir="latin_out"
  )

JudiLing.write2csv(
  res_build,
  latin,
  cue_obj,
  cue_obj,
  "latin_build_res.csv",
  grams=3,
  tokenized=false,
  sep_token=nothing,
  start_end_token="#",
  output_sep_token="",
  path_sep_token=":",
  target_col=:Word,
  root_dir=@__DIR__,
  output_dir="latin_out"
  )

JudiLing.write2csv(
  gpi_learn,
  "latin_learn_gpi.csv",
  root_dir=@__DIR__,
  output_dir="latin_out"
  )

mkpath(joinpath(@__DIR__,"out"))
fio = open(joinpath(@__DIR__,"out", "acc.out"), "w")
write(fio, "learn: $acc_learn\n")
write(fio, "build: $acc_build\n")
close(fio)