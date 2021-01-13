using JudiLing # our package
using CSV # read csv files into dataframes
using BenchmarkTools
using Random

function run_test()
  # load latin file
  french = CSV.DataFrame!(CSV.File(joinpath(@__DIR__, "data", "french.csv")))
  # shuffle data
  rng = MersenneTwister(314)
  french = french[shuffle(rng, 1:size(french, 1)),:]
  french = french[1:2500,:]
  sr = 0.2
  n = size(french, 1)
  tv = convert(Int64, floor(n*sr))
  french_train = french[1:end-tv,:]
  french_val = french[end-tv+1:end,:]
  # display(latin)

  # create C matrixes for training datasets
  cue_obj_train, cue_obj_val = JudiLing.make_combined_cue_matrix(
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
  S_train, S_val = JudiLing.make_combined_S_matrix(
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
  A_train = cue_obj_train.A

  # we calculate how many timestep we need for learn_paths and huo function
  max_t = JudiLing.cal_max_timestep(french_train, french_val, :Syllables)

  # we calculate learn_paths and build_paths function
  res_learn_train = JudiLing.learn_paths(
    french_train,
    french_train,
    cue_obj_train.C,
    S_train,
    F_train,
    Chat_train,
    A_train,
    cue_obj_train.i2f,
    cue_obj_train.f2i, # api changed in 0.3.1
    check_gold_path=false,
    # gold_ind=cue_obj_train.gold_ind,
    # Shat_val=Shat_train,
    max_t=max_t,
    max_can=10,
    grams=2,
    threshold=0.5, # 0.05345303464774335
    tokenized=true,
    sep_token="-",
    keep_sep=true,
    target_col=:Syllables,
    verbose=true)
end

b = @benchmarkable run_test() samples=1
r = run(b)
mkpath(joinpath(@__DIR__,"out"))
fio = open(joinpath(@__DIR__,"out", "time_learn_train.out"), "w")
write(fio, "time: $(time(r)/1e9)s\n")
write(fio, "memory: $(memory(r)/1e9)GB\n")
close(fio)
;