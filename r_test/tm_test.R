setwd("/Users/xuefeng/Documents/GitHub/r_test")
getwd()

# install.packages("ggplot2")
# install.packages("pryr")
# install.packages("devtools")
# devtools::install_github("hadley/lineprof")

# mem_change(x <- 1:1e6)

library(WpmWithLdl)
test = read.csv("estonian_test.csv", header=T, stringsAsFactors=F)
train = read.csv("estonian_train.csv", header=T, stringsAsFactors=F)
est = rbind(train, test)

run <- function(threshold1) {
  cue_obj = make_cue_matrix(formula=~Lexeme+Case+Number, data=est,
                            wordform="Word", grams=3) 

  S = make_S_matrix(formula=~Lexeme+Case+Number, data=est,
                    wordform="Word", grams=3)

  prod = learn_production(cue_obj=cue_obj, S=S)

  # by default, this function uses 4 cores
  prodAcc = accuracy_production(m=prod, data=est, wordform="Word", no_cores=4, threshold=threshold1)
}
library(pryr)
gc(reset=TRUE)
Sys.time()
run(0.1)
Sys.time()
gc()
run(0.01)
Sys.time()
run(0.001)
Sys.time()
run(0.0001)
Sys.time()
run(0.00001)
Sys.time()
