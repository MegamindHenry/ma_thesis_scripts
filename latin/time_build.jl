using JudiLing # our package
using CSV # read csv files into dataframes
using BenchmarkTools

function run_test()
  latin = CSV.DataFrame!(CSV.File(joinpath(@__DIR__, "data", "latin.csv")))

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

  # we calculate F as we did for G
  F = JudiLing.make_transform_matrix(cue_obj.C, S)

  # we calculate Shat as we did for Chat
  Shat = cue_obj.C * F

  # here we only use a adjacency matrix as we got it from training dataset
  A = cue_obj.A

  # we calculate how many timestep we need for learn_paths and huo function
  max_t = JudiLing.cal_max_timestep(latin, :Word)

  # we calculate learn_paths and build_paths function
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
end

b = @benchmarkable run_test() samples=1
r = run(b)
mkpath(joinpath(@__DIR__,"out"))
fio = open(joinpath(@__DIR__,"out", "time_build.out"), "w")
write(fio, "time: $(time(r)/1e9)s\n")
write(fio, "memory: $(memory(r)/1e9)Gbytes\n")
close(fio)
;