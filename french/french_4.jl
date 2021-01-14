JudiLing.write2csv(
  res_learn_train,
  french_train,
  cue_obj_train,
  cue_obj_train,
  "french_learn_res_train.csv",
  grams=2,
  tokenized=false,
  sep_token=nothing,
  start_end_token="#",
  output_sep_token="",
  path_sep_token=":",
  target_col=:Syllables,
  root_dir=@__DIR__,
  output_dir="out"
  )

JudiLing.write2csv(
  gpi_learn_train,
  "french_learn_gpi_train.csv",
  root_dir=@__DIR__,
  output_dir="out"
  )

JudiLing.write2csv(
  res_learn_val,
  french_val,
  cue_obj_train,
  cue_obj_val,
  "french_learn_res_val.csv",
  grams=2,
  tokenized=false,
  sep_token=nothing,
  start_end_token="#",
  output_sep_token="",
  path_sep_token=":",
  target_col=:Syllables,
  root_dir=@__DIR__,
  output_dir="out"
  )

JudiLing.write2csv(
  gpi_learn_val,
  "french_learn_gpi_val.csv",
  root_dir=@__DIR__,
  output_dir="out"
  )