using JudiLing # our package
using CSV # read csv files into dataframes
using BenchmarkTools

# load latin file
estonian_train = CSV.DataFrame!(CSV.File(joinpath(@__DIR__, "data", "estonian_train.csv")))
estonian_val = CSV.DataFrame!(CSV.File(joinpath(@__DIR__, "data", "estonian_val.csv")))
# display(latin)

# create C matrixes for training datasets
cue_obj_train, cue_obj_val = JudiLing.make_cue_matrix(
  estonian_train,
  estonian_val,
  grams=3,
  target_col=:Word,
  tokenized=false,
  keep_sep=false
  )

# retrieve dim of C
# we set the S matrixes as the same dimensions
n_features = size(cue_obj_train.C, 2)
S_train, S_val = JudiLing.make_S_matrix(
  estonian_train,
  estonian_val,
  ["Lexeme"],
  ["Case","Number"],
  ncol=n_features,
  add_noise=true)

# we use cholesky function to calculate mapping G from S to C
G_train = JudiLing.make_transform_matrix(S_train, cue_obj_train.C)

# we calculate Chat matrixes by multiplying S and G 
Chat_train = S_train * G_train
Chat_val = S_val * G_train

@show JudiLing.eval_SC(Chat_train, cue_obj_train.C)
@show JudiLing.eval_SC(Chat_train, cue_obj_train.C, estonian_train, :Word)
@show JudiLing.eval_SC(Chat_val, cue_obj_val.C)
@show JudiLing.eval_SC(Chat_val, cue_obj_val.C, estonian_val, :Word)

# we calculate F as we did for G
F_train = JudiLing.make_transform_matrix(cue_obj_train.C, S_train)

# we calculate Shat as we did for Chat
Shat_train = cue_obj_train.C * F_train
Shat_val = cue_obj_val.C * F_train

@show JudiLing.eval_SC(Shat_train, S_train)
@show JudiLing.eval_SC(Shat_train, S_train, estonian_train, :Word)
@show JudiLing.eval_SC(Shat_val, S_val)
@show JudiLing.eval_SC(Shat_val, S_val, estonian_val, :Word)

# here we only use a adjacency matrix as we got it from training dataset
A_train = cue_obj_train.A

A = JudiLing.make_combined_adjacency_matrix(
  estonian_train,
  estonian_val,
  grams=3,
  target_col=:Word,
  tokenized=false,
  keep_sep=false
  )

# we calculate how many timestep we need for learn_paths and huo function
max_t = JudiLing.cal_max_timestep(estonian_train, estonian_val, :Word)

# we calculate learn_paths and build_paths function
res_learn_train, gpi_learn_train = JudiLing.learn_paths(
  estonian_train,
  estonian_train,
  cue_obj_train.C,
  S_train,
  F_train,
  Chat_train,
  A_train,
  cue_obj_train.i2f,
  cue_obj_train.f2i, # api changed in 0.3.1
  check_gold_path=true,
  gold_ind=cue_obj_train.gold_ind,
  Shat_val=Shat_train,
  max_t=max_t,
  max_can=10,
  grams=3,
  threshold=0.05, # 0.05345303464774335
  tokenized=false,
  keep_sep=false,
  target_col=:Word,
  verbose=true)

acc_learn_train = JudiLing.eval_acc(
  res_learn_train,
  cue_obj_train.gold_ind,
  verbose=false
)

# we calculate learn_paths and build_paths function
res_learn_val, gpi_learn_val = JudiLing.learn_paths(
  estonian_train,
  estonian_val,
  cue_obj_train.C,
  S_val,
  F_train,
  Chat_val,
  A_train,
  cue_obj_train.i2f,
  cue_obj_train.f2i, # api changed in 0.3.1
  check_gold_path=true,
  gold_ind=cue_obj_val.gold_ind,
  Shat_val=Shat_val,
  max_t=max_t,
  max_can=10,
  grams=3,
  threshold=0.05,
  is_tolerant=true,
  tolerance=-0.1,
  max_tolerance=3,
  tokenized=false,
  keep_sep=false,
  target_col=:Word,
  verbose=true)

acc_learn_val = JudiLing.eval_acc(
  res_learn_val,
  cue_obj_val.gold_ind,
  verbose=false
)

JudiLing.write2csv(
  res_learn_train,
  estonian_train,
  cue_obj_train,
  cue_obj_train,
  "estonian_learn_res_train.csv",
  grams=3,
  tokenized=false,
  sep_token=nothing,
  start_end_token="#",
  output_sep_token="",
  path_sep_token=":",
  target_col=:Word,
  root_dir=@__DIR__,
  output_dir="out"
  )

JudiLing.write2csv(
  gpi_learn_train,
  "estonian_learn_gpi_train.csv",
  root_dir=@__DIR__,
  output_dir="out"
  )

JudiLing.write2csv(
  res_learn_train,
  estonian_train,
  cue_obj_train,
  cue_obj_train,
  "estonian_learn_res_train.csv",
  grams=3,
  tokenized=false,
  sep_token=nothing,
  start_end_token="#",
  output_sep_token="",
  path_sep_token=":",
  target_col=:Word,
  root_dir=@__DIR__,
  output_dir="out"
  )

JudiLing.write2csv(
  gpi_learn_train,
  "estonian_learn_gpi_train.csv",
  root_dir=@__DIR__,
  output_dir="out"
  )

JudiLing.write2csv(
  res_learn_val,
  estonian_val,
  cue_obj_train,
  cue_obj_val,
  "estonian_learn_res_val.csv",
  grams=3,
  tokenized=false,
  sep_token=nothing,
  start_end_token="#",
  output_sep_token="",
  path_sep_token=":",
  target_col=:Word,
  root_dir=@__DIR__,
  output_dir="out"
  )

JudiLing.write2csv(
  gpi_learn_val,
  "estonian_learn_gpi_val.csv",
  root_dir=@__DIR__,
  output_dir="out"
  )

res_build_train = JudiLing.build_paths(
    estonian_train,
    cue_obj_train.C,
    S_train,
    F_train,
    Chat_train,
    A_train,
    cue_obj_train.i2f,
    cue_obj_train.gold_ind,
    max_t=max_t,
    n_neighbors=3,
    verbose=true
    )

acc_build_train = JudiLing.eval_acc(
  res_build_train,
  cue_obj_train.gold_ind,
  verbose=false
)

res_build_val = JudiLing.build_paths(
    estonian_val,
    cue_obj_train.C,
    S_val,
    F_train,
    Chat_val,
    A_train,
    cue_obj_train.i2f,
    cue_obj_train.gold_ind,
    max_t=max_t,
    n_neighbors=20,
    verbose=true
    )

acc_build_val = JudiLing.eval_acc(
  res_build_val,
  cue_obj_val.gold_ind,
  verbose=false
)

JudiLing.write2csv(
  res_build_train,
  estonian_train,
  cue_obj_train,
  cue_obj_train,
  "estonian_build_res_train.csv",
  grams=3,
  tokenized=false,
  sep_token=nothing,
  start_end_token="#",
  output_sep_token="",
  path_sep_token=":",
  target_col=:Word,
  root_dir=@__DIR__,
  output_dir="out"
  )

JudiLing.write2csv(
  res_build_val,
  estonian_val,
  cue_obj_train,
  cue_obj_val,
  "estonian_build_res_val.csv",
  grams=3,
  tokenized=false,
  sep_token=nothing,
  start_end_token="#",
  output_sep_token="",
  path_sep_token=":",
  target_col=:Word,
  root_dir=@__DIR__,
  output_dir="out"
  )

mkpath(joinpath(@__DIR__,"out"))
fio = open(joinpath(@__DIR__,"out", "acc.out"), "w")
println(fio, "Acc for learn train: $acc_learn_train")
println(fio, "Acc for learn val: $acc_learn_val")
println(fio, "Acc for build train: $acc_build_train")
println(fio, "Acc for build val: $acc_build_val")
close(fio)