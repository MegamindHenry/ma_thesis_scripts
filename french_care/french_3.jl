# we calculate learn_paths and build_paths function
res_learn_val, gpi_learn_val = JudiLing.learn_paths(
  french_train,
  french_val,
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
  grams=2,
  threshold=0.1,
  is_tolerant=true,
  tolerance=-0.1,
  max_tolerance=3,
  tokenized=true,
  sep_token="-",
  keep_sep=true,
  target_col=:Syllables,
  sparse_ratio=0.05,
  verbose=true)

acc_learn_val = JudiLing.eval_acc(
  res_learn_val,
  cue_obj_val.gold_ind,
  verbose=false
)