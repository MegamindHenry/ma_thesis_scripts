# we calculate learn_paths and build_paths function
res_learn_train, gpi_learn_train = JudiLing.learn_paths(
  french_train,
  french_train,
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
  grams=2,
  threshold=0.1, # 0.05345303464774335
  tokenized=true,
  sep_token="-",
  keep_sep=true,
  target_col=:Syllables,
  sparse_ratio=0.05,
  verbose=true)

acc_learn_train = JudiLing.eval_acc(
  res_learn_train,
  cue_obj_train.gold_ind,
  verbose=false
)