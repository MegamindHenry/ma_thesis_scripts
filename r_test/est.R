library(WpmWithLdl)
test = read.csv("./data/estonian_test.csv", header=T, stringsAsFactors=F)
train = read.csv("./data/estonian_train.csv", header=T, stringsAsFactors=F)
est = rbind(train, test)

Sys.time()
cue_obj = make_cue_matrix(formula=~Lexeme+Case+Number, data=est,
                          wordform="Word", grams=3) 
Sys.time()                  

Sys.time()
S = make_S_matrix(formula=~Lexeme+Case+Number, data=est,
                  wordform="Word", grams=3)
Sys.time()

Sys.time()
prod = learn_production(cue_obj=cue_obj, S=S)
Sys.time()

Sys.time()
# by default, this function uses 4 cores
prodAcc = accuracy_production(m=prod, data=est, wordform="Word", no_cores=4)
Sys.time()

