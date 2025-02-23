#to eliminate %>% not found error
library(dplyr)

library(keras)
#eliminale the ImportError function: No module named 'keras'
# install_keras()

K <- keras::backend()
library(tensorflow)

set.seed(2) #for reproducibility


# these numbers are for the educational data
num_stats <- 13
num_skills <- 4

N <- 10000 # number of subjects
tr <- 10000 
# how many to train on - should eventually move to 85% train, 15% test
 
#we can use it later
#training_data = 8500
#test_data = 1500
batch_size <- 50
epochs <- 20

#modified Q-matrix for sports data

Q <- matrix(c(
  1, 0, 0, 0,
  1, 0, 1, 0,
  0, 0, 1, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
  0, 0, 1, 0,
  1, 0, 0, 1,
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 1, 0, 0,
  0, 1, 0, 0,
  0, 0, 0, 1), nrow=num_stats, ncol=num_skills, byrow=TRUE)
colnames(Q) <- paste("Dim",c(1:num_skills),sep="")
rownames(Q) <- paste("Item",c(1:num_stats),sep="")
Q = t(Q)


# build up neural network
input <- layer_input(shape = c(num_stats), name="input")

# hidden layers - architecture will need to be tuned
h <- layer_dense(input, 10, activation = 'tanh', name='hidden1')
# h <- layer_dense(h, 5, activation='sigmoid', name='hidden2')
# h <- layer_dense(h, 10, activation='sigmoid', name='hidden3')
# h <- layer_dense(h, 5, activation='sigmoid', name='hidden4')
z_mean <- layer_dense(h, num_skills, name='z_mean')
z_log_var <- layer_dense(h, num_skills, name='z_log_var')

# re-parameterization trick
sampling <- function(arg){
  z_mean <- arg[,1:(num_skills)]
  z_log_var <- arg[, (num_skills + 1):(2*num_skills)]
  eps <- k_random_normal(
    shape = c(k_shape(z_mean)[[1]]),
    mean=0, stddev=1
  )
  z_mean + k_exp(z_log_var/2)*eps
}

z <- layer_concatenate(list(z_mean, z_log_var), name="z_sample") %>%
  layer_lambda(sampling)

encoder <- keras_model(input, z_mean)
encoder_var <- keras_model(input, z_log_var)
summary(encoder)

# restrict connections in decoder
q_constraint <- function(w){
  target <- w * Q
  diff = w - target
  w <- w * k_cast(k_equal(diff,0), k_floatx()) # enforce Q-matrix connections
  w * k_cast(k_greater_equal(w, 0), k_floatx()) # require non-negative weights
}

out <- layer_dense(z, units=num_stats, activation='sigmoid',
                   # kernel_regularizer=regularize_Q,
                   kernel_constraint=q_constraint,
                   name='vae_out')

vae <- keras_model(input, out)

vae_loss <- function(input, output){
  cross_entropy_loss <- (num_stats/1.0)* loss_binary_crossentropy(input, output)
  mse <- (num_stats/1.0) * loss_mean_squared_error(input,output)
  kl_loss <- -0.5*k_mean(1+z_log_var - k_square(z_mean) - k_exp(z_log_var), axis=-1L)
  mse + kl_loss
}

vae %>% compile(optimizer = "SGD", loss = vae_loss, metrics= 'accuracy')
summary(vae)



#Loading data for soprts analytics
data_sports = read.csv("final_data.csv", sep=',', header=TRUE)
extra_data = read.csv("rawdata.csv", sep=',', header=TRUE)
ctct = extra_data$ctct_z
data_sports$cntct = ctct
#randomize the rows to split train/test since the original data was sorted
data_sports <- data_sports[sample(nrow(data_sports)),]
# need to pick out the features we want
Y <- select(data_sports, X1B, X2B, HR, R, RBI, BB, IBB, SO, SAC, GDP, SB, CS, BB.K)
Y[is.na(Y)] <- 0
Y$CS <- -Y$CS #I don't think this got changed, since this stat had all zeros for weights
data_train <- as.matrix(Y[1:7600,])
data_test <- as.matrix(Y[7601:8604,])


#########  Model 1 training ---------------------------------------------------
vae %>% fit(
  data_train, data_train, 
  shuffle = TRUE, 
  epochs = epochs, 
  batch_size = batch_size 
)

# Get skill predictions
skill_preds <- predict(encoder,data_test)

# Estimated weights 
W <-get_weights(vae)

plot(data_sports[7601:8604,]$cntct, skill_preds[,1], xlab='cntct', ylab='Contact')
# plot(data_sports[7601:8604,]$BABIP, skill_preds[,1], xlab='BABIP', ylab='Contact')
plot(data_sports[7601:8604,]$Spd, skill_preds[,2], xlab='SPD', ylab='Baserunning')
plot(data_sports[7601:8604,]$ISO, skill_preds[,3], xlab='ISO', ylab='Power')
plot(data_sports[7601:8604,]$OBP, skill_preds[,4], xlab='OBP', ylab='Pitch Intuition')

cor(data_sports[7601:8604,]$cntct, skill_preds[,1]) #0.2138401
cor(data_sports[7601:8604,]$Spd, skill_preds[,2]) #0.7952529
cor(data_sports[7601:8604,]$ISO, skill_preds[,3]) #0.9032751
cor(data_sports[7601:8604,]$OBP, skill_preds[,4]) #0.6840747


# These were useful in the educational setting, not sure if they will be in sports analytics
discr <- as.matrix(W[[7]])
diff <- as.matrix(W[[8]])
print(discr)
print(diff)

all_player_skills <- predict(encoder,as.matrix(Y))
all_ids = as.matrix(data_sports$Season_playerid)

# ids = c()
# j = 1
# for (s in 1:4){
#   highest_10_ind = order(-all_player_skills[,s])[1:10]
#   for (i in 1:10){
#     ids[j] = all_ids[highest_10_ind[i]]
#     j = j + 1
#   }
# }
# print(ids)
