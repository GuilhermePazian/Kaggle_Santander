# pacotes utilizados

packages= c("tidyverse","data.table","reshape2","captioner","gridExtra","xtable","ggpubr", "e1071","caret","h2o",
            "ggfortify","psych","xtable", "quadprog","knitr","scales", "kableExtra","gbm",
            "mboost","randomForest","glmnet","gam", "kernlab", "mlr","reticulate","SuperLearner","h2o")

#mostra quais pacotes foram carregados
invisible(lapply(packages, require, character.only = TRUE))

#############################################
#path_arq = "coloque aqui o caminho no pc onde está o arquivo treino"
#path_arq2 = "coloque aqui o caminho no pc onde está o arquivo teste"
#############################################


kaggle_sat_train = read.csv(path_arq)
kaggle_sat_train$ID=NULL
#kaggle_sat_test = read.csv(path_arq2)
#dimensão dos dados
#dim(kaggle_sat_train)
# estrutura dos dados
#glimpse(kaggle_sat_train)

#kaggle_sat_train = data.frame(kaggle_sat_train %>% sapply(function(x) as.numeric(x)))

#tipo = data.frame(kaggle_sat_train %>% sapply(function(x) typeof(x)))

#table(tipo[,1])



#temos uma variável resposta continua. TEmos que prever 

#ggplot(kaggle_sat_train,aes(x=target, y = X1adc3dc1b))+geom_point()

#summary(kaggle_sat$X1adc3dc1b)

#corr = cor(kaggle_sat_train[3:4993])

#colSums(is.na(kaggle_sat_train))

ind = sample(3, nrow(kaggle_sat_train), replace=TRUE, prob = c(0.7,0.1,.2))

# dados de treino
train = kaggle_sat_train[ind==1,]

# dados para validação do modelo
validation = kaggle_sat_train[ind==2,]

# dados de teste
test = kaggle_sat_train[ind==3,]

h2o.init()

# Import a sample binary outcome train/test set into H2O

# Identify predictors and response
y <- "target"
x <- setdiff(names(train), y)


# Number of CV folds (to generate level-one data for stacking)
nfolds <- 10

# There are a few ways to assemble a list of models to stack toegether:
# 1. Train individual models and put them in a list
# 2. Train a grid of models
# 3. Train several grids of models
# Note: All base models must have the same cross-validation folds and
# the cross-validated predicted values must be kept.


# 1. Generate a 2-model ensemble (GBM + RF)

# Train & Cross-validate a GBM
#my_gbm <- h2o.gbm(x = x,
#                  y = y,
#                  training_frame = as.h2o(train),
#                  distribution = "gamma",
#                  ntrees = 10,
#                  max_depth = 3,
#                  min_rows = 2,
#                  learn_rate = 0.2,
#                  nfolds = nfolds,
#                  fold_assignment = "Modulo",
#                  keep_cross_validation_predictions = TRUE,
#                  seed = 1)

# Train & Cross-validate a RF
my_rf <- h2o.randomForest(x = x,
                          y = y,
                          training_frame = as.h2o(train),
                          validation_frame = as.h2o(validation),
                          ntrees = 500,
                          nfolds = nfolds,
                          fold_assignment = "Modulo",
                          keep_cross_validation_predictions = TRUE,
                          stopping_metric = "RMSLE",
                          seed = 1)

# Train a stacked ensemble using the GBM and RF above
ensemble <- h2o.stackedEnsemble(x = x,
                                y = y,
                                training_frame = as.h2o(train),
                                model_id = "my_ensemble_binomial",
                                base_models = list(my_gbm, my_rf))

# Eval ensemble performance on a test set
perf <- h2o.performance(ensemble, newdata = as.h2o(test))

# Compare to base learner performance on the test set
perf_gbm_test <- h2o.performance(my_gbm, newdata = as.h2o(test))
perf_rf_test <- h2o.performance(my_rf, newdata = as.h2o(test))
baselearner_best_auc_test <- max(h2o.auc(perf_gbm_test), h2o.auc(perf_rf_test))
ensemble_auc_test <- h2o.auc(perf)
print(sprintf("Best Base-learner Test AUC:  %s", baselearner_best_auc_test))
print(sprintf("Ensemble Test AUC:  %s", ensemble_auc_test))

# Generate predictions on a test set (if neccessary)
pred <- h2o.predict(ensemble, newdata = as.h2o(test))


# 2. Generate a random grid of models and stack them together

# GBM Hyperparamters
learn_rate_opt <- c(0.01, 0.03)
max_depth_opt <- c(3, 4, 5, 6, 9)
sample_rate_opt <- c(0.7, 0.8, 0.9, 1.0)
col_sample_rate_opt <- c(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
hyper_params <- list(learn_rate = learn_rate_opt,
                     max_depth = max_depth_opt,
                     sample_rate = sample_rate_opt,
                     col_sample_rate = col_sample_rate_opt)

search_criteria <- list(strategy = "RandomDiscrete",
                        max_models = 3,
                        seed = 1)

gbm_grid <- h2o.grid(algorithm = "gbm",
                     grid_id = "gbm_grid_gamma",
                     x = x,
                     y = y,
                     training_frame = as.h2o(train),
                     ntrees = 10,
                     seed = 1,
                     nfolds = nfolds,
                     fold_assignment = "Modulo",
                     keep_cross_validation_predictions = TRUE,
                     hyper_params = hyper_params,
                     search_criteria = search_criteria)

# Train a stacked ensemble using the GBM grid
ensemble <- h2o.stackedEnsemble(x = x,
                                y = y,
                                training_frame = as.h2o(train),
                                model_id = "ensemble_gbm_grid_gamma",
                                base_models = gbm_grid@model_ids)

# Eval ensemble performance on a test set
perf <- h2o.performance(ensemble, newdata = as.h2o(test))

# Compare to base learner performance on the test set
.getauc <- function(mm) h2o.auc(h2o.performance(h2o.getModel(mm), newdata = test))
baselearner_aucs <- sapply(gbm_grid@model_ids, .getauc)
baselearner_best_auc_test <- max(baselearner_aucs)
ensemble_auc_test <- h2o.auc(perf)
print(sprintf("Best Base-learner Test AUC:  %s", baselearner_best_auc_test))
print(sprintf("Ensemble Test AUC:  %s", ensemble_auc_test))

# Generate predictions on a test set (if neccessary)
pred <- h2o.predict(ensemble, newdata = test)



hyper_params <- list(
  activation=c("Rectifier","Tanh","Maxout","RectifierWithDropout","TanhWithDropout","MaxoutWithDropout"),
  hidden=list(c(20,20),c(50,50),c(30,30,30),c(25,25,25,25)),
  input_dropout_ratio=c(0,0.05),
  l1=seq(0,1e-4,1e-6),
  l2=seq(0,1e-4,1e-6)
)
hyper_params

## Stop once the top 5 models are within 1% of each other (i.e., the windowed average varies less than 1%)
search_criteria = list(strategy = "RandomDiscrete", max_runtime_secs = 360, max_models = 100, 
                          seed=1234567, stopping_rounds=5, stopping_tolerance=1e-2)

dl_random_grid <- h2o.grid(
  algorithm="deeplearning",
  grid_id = "dl_grid_random",
  training_frame=as.h2o(train),
  x=x, 
  y=y,
  seed = 1,
  nfolds = nfolds,
  fold_assignment = "Modulo",
  keep_cross_validation_predictions = TRUE,
  epochs=1,
  stopping_metric="logloss",
  stopping_tolerance=1e-2,        ## stop when logloss does not improve by >=1% for 2 scoring events
  stopping_rounds=2,
  score_validation_samples=10000, ## downsample validation set for faster scoring
  score_duty_cycle=0.025,         ## don't score more than 2.5% of the wall time
  max_w2=10,                      ## can help improve stability for Rectifier
  hyper_params = hyper_params,
  search_criteria = search_criteria
)                                
grid <- h2o.getGrid("dl_grid_random",sort_by="logloss",decreasing=FALSE)
grid

grid@summary_table[1,]
best_model <- h2o.getModel(grid@model_ids[[1]]) ## model with lowest logloss
best_model