using JudiLing # our package
using CSV # read csv files into dataframes
using BenchmarkTools

function run_test()
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

  # @show JudiLing.eval_SC(Chat_train, cue_obj_train.C)
  # @show JudiLing.eval_SC(Chat_train, cue_obj_train.C, estonian_train, :Word)
  # @show JudiLing.eval_SC(Chat_val, cue_obj_val.C)
  # @show JudiLing.eval_SC(Chat_val, cue_obj_val.C, estonian_val, :Word)

  # we calculate F as we did for G
  F_train = JudiLing.make_transform_matrix(cue_obj_train.C, S_train)

  # we calculate Shat as we did for Chat
  Shat_train = cue_obj_train.C * F_train
  Shat_val = cue_obj_val.C * F_train

  # @show JudiLing.eval_SC(Shat_train, S_train)
  # @show JudiLing.eval_SC(Shat_train, S_train, estonian_train, :Word)
  # @show JudiLing.eval_SC(Shat_val, S_val)
  # @show JudiLing.eval_SC(Shat_val, S_val, estonian_val, :Word)

  # here we only use a adjacency matrix as we got it from training dataset
  A_train = cue_obj_train.A

  # we calculate how many timestep we need for learn_paths and huo function
  max_t = JudiLing.cal_max_timestep(estonian_train, estonian_val, :Word)

  # we calculate learn_paths and build_paths function
  res_learn_val = JudiLing.learn_paths(
    estonian_train,
    estonian_val,
    cue_obj_train.C,
    S_val,
    F_train,
    Chat_val,
    A_train,
    cue_obj_train.i2f,
    cue_obj_train.f2i, # api changed in 0.3.1
    check_gold_path=false,
    # gold_ind=cue_obj_train.gold_ind,
    # Shat_val=Shat_train,
    max_t=max_t,
    max_can=10,
    grams=3,
    threshold=0.05, # 0.05345303464774335
    is_tolerant=true,
    tolerance=-0.1,
    max_tolerance=3,
    tokenized=false,
    keep_sep=false,
    target_col=:Word,
    verbose=true)
end

b = @benchmarkable run_test() samples=1
r = run(b)
mkpath(joinpath(@__DIR__,"out"))
fio = open(joinpath(@__DIR__,"out", "time_build_val.out"), "w")
write(fio, "time: $(time(r)/1e9)s\n")
write(fio, "memory: $(memory(r)/1e9)GB\n")
close(fio)
;