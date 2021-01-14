res_build_train = JudiLing.build_paths(
    french_train,
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