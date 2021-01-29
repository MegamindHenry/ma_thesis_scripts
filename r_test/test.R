setwd("/Users/xuefeng/Documents/GitHub/r_test")
getwd()

train = read.csv("latin_train.csv", header=T, stringsAsFactors=F)
val = read.csv("latin_val.csv", header=T, stringsAsFactors=F)

train$test = FALSE
val$test = TRUE
lat = rbind(train, val)

library(WpmWithLdl)
Sys.time()
latUS = test_unseen(formula = ~ Lexeme + Person + Number + Tense + Voice + Mood,
                    data = lat, grams = 3, wordform = "Word", testword = "test")
latUS$comprehension$acc
#[1] 0.9946524
latUS$production$acc
#[1] 0.1336898
Sys.time()

Sys.time()
latUS2 = test_unseen(formula = ~ Lexeme + Person + Number + Tense + Voice + Mood,
                     data = lat, grams = 3, wordform = "Word", testword = "test",
                     threshold = 0.01)
latUS2$comprehension$acc
#[1] 0.9946524
latUS2$production$acc
#[1] 0.2058824
Sys.time()

Sys.time()
latUS3 = test_unseen(formula = ~ Lexeme + Person + Number + Tense + Voice + Mood,
                     data = lat, grams = 3, wordform = "Word", testword = "test",
                     threshold = 0.001)
latUS3$comprehension$acc
#[1] 0.9946524
latUS3$production$acc
#[1] 0.2085561
Sys.time()

Sys.time()
latUS4 = test_unseen(formula = ~ Lexeme + Person + Number + Tense + Voice + Mood,
                     data = lat, grams = 3, wordform = "Word", testword = "test",
                     threshold = -1)
latUS4$comprehension$acc
#[1] 0.9946524
latUS4$production$acc
#[1] 0.671123
Sys.time()



