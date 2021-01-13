df_error = df_learn[(df_learn.isbest .== true) .& (df_learn.iscorrect .== false),:]

