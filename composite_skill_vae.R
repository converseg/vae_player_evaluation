library(keras)
K <- keras::backend()
library(tensorflow)

# This script is for a comprehensive player evaluation - one latent trait (skill), and no Q-matrix
num_stats <- 13
num_skills <- 1 #only one skill causes a problem

N <- 8604 # number of subjects
tr <- 7601 # how many to train on
batch_size <- 50
epochs <- 10

# build up neural network
input <- layer_input(shape = c(num_stats), name="input")

# hidden layers - architecture will need to be tuned
h <- layer_dense(input, 10, activation = 'sigmoid', name='hidden1')
h <- layer_dense(h, 5, activation='sigmoid', name='hidden2')
h <- layer_dense(h, 10, activation='sigmoid', name='hidden3')
h <- layer_dense(h, 5, activation='sigmoid', name='hidden4')
z_mean <- layer_dense(h, num_skills, name='z_mean')
z_log_var <- layer_dense(h, num_skills, name='z_log_var')

# re-parameterization trick
sampling <- function(arg){
  z_mean <- arg[,1:1]
  z_log_var <- arg[, 2:2]
  eps <- k_random_normal(
    shape = c(1), #had to fix this line
    mean=0, stddev=1
  )
  z_mean + k_exp(z_log_var/2)*eps
}

nonneg_constraint <- function(w){
  w * k_cast(k_greater_equal(w, 0), k_floatx()) # require non-negative weights
}

z <- layer_concatenate(list(z_mean, z_log_var), name="z_sample") %>%
  layer_lambda(sampling)

encoder <- keras_model(input, z_mean)
encoder_var <- keras_model(input, z_log_var)
summary(encoder)

out <- layer_dense(z, units=num_stats, activation='sigmoid',
                   kernel_constraint=nonneg_constraint, name='vae_out')

vae <- keras_model(input, out)

vae_loss <- function(input, output){
  cross_entropy_loss <- (num_stats/1.0)* loss_binary_crossentropy(input, output)
  kl_loss <- -0.5*(1+2*z_log_var - k_square(z_mean) - k_exp(2*z_log_var)) # I editted this, don't know if correct
  cross_entropy_loss + kl_loss
}

vae %>% compile(optimizer = "SGD", loss = vae_loss, metrics= 'accuracy')
summary(vae)


# LOAD DATA
set.seed(20) #for reproducibility

#Loading data for soprts analytics
data_sports = read.csv("final_data.csv", sep=',', header=TRUE)
#randomize the rows to split train/test since the original data was sorted
data_sports <- data_sports[sample(nrow(data_sports)),]
data_sports[is.na(data_sports)] <- 0
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
composite_skill <- predict(encoder,data_test)
hist(composite_skill)

plot(data_sports[7601:8604,]$WAR, composite_skill, xlab='WAR', ylab='Composite Skill')
plot(data_sports[7601:8604,]$wRC., composite_skill, xlab='WRC+', ylab='Composite Skill')
plot(data_sports[7601:8604,]$WPA, composite_skill, xlab='WPA', ylab='Composite Skill') # Lots of NA here

cor(data_sports[7601:8604,]$WAR, composite_skill) # 0.658261
cor(data_sports[7601:8604,]$wRC., composite_skill) # 0.7565359
cor(data_sports[7601:8604,]$WPA, composite_skill) # 0.6065109

# Estimated weights 
W <-get_weights(vae)
hist(composite_skill) # This looks Gaussian, but the scale is not N(0,1)

# These were useful in the educational setting, not sure if they will be in sports analytics
# also these indices will be different
discr <- as.matrix(W[[7]])
# diff <- as.matrix(W[[8]])
print(discr)
# print(diff)

all_player_skills <- predict(encoder,as.matrix(Y))
all_ids = as.matrix(data_sports$Season_playerid)
ids = c()
highest_10_ind = order(-all_player_skills)[1:10]
for (i in 1:10){
  ids[i] = all_ids[highest_10_ind[i]]
}
print(ids)
