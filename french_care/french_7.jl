JudiLing.write2csv(
  res_build_train,
  french_train,
  cue_obj_train,
  cue_obj_train,
  "french_build_res_train.csv",
  grams=2,
  tokenized=true,
  sep_token="-",
  start_end_token="#",
  output_sep_token="",
  path_sep_token=":",
  target_col=:Syllables,
  root_dir=@__DIR__,
  output_dir="out"
  )

JudiLing.write2csv(
  res_build_val,
  french_val,
  cue_obj_train,
  cue_obj_val,
  "french_build_res_val.csv",
  grams=2,
  tokenized=true,
  sep_token="-",
  start_end_token="#",
  output_sep_token="",
  path_sep_token=":",
  target_col=:Syllables,
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