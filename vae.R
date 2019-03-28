library(keras)
K <- keras::backend()
library(tensorflow)

# these numbers are for the educational data
num_stats <- 13
num_skills <- 4

N <- 10000 # number of subjects
tr <- 10000 # how many to train on - should eventually move to 85% train, 15% test
batch_size <- 50
epochs <- 10

Q <- matrix(c(
  1, 1, 0,
  0, 1, 0,
  1, 0, 1,
  0, 0, 1,
  0, 0, 1,
  0, 0, 1,
  1, 0, 1,
  0, 1, 0,
  0, 0, 1,
  1, 0, 0,
  1, 0, 1,
  1, 0, 1,
  1, 0, 0,
  1, 0, 0,
  0, 0, 1,
  1, 0, 1,
  0, 1, 1,
  0, 0, 1,
  0, 0, 1,
  1, 0, 1,
  1, 0, 1,
  0, 0, 1,
  0, 1, 0,
  0, 1, 0,
  1, 0, 0,
  0, 0, 1,
  1, 0, 0,
  0, 0, 1), nrow=num_stats, ncol=num_skills, byrow=TRUE)
colnames(Q) <- paste("Dim",c(1:num_skills),sep="")
rownames(Q) <- paste("Item",c(1:num_stats),sep="")
Q = t(Q)

# build up neural network
input <- layer_input(shape = c(num_stats), name="input")

# hidden layers - architecture will need to be tuned
h <- layer_dense(input, 10, activation = 'tanh', name='hidden')
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
  kl_loss <- -0.5*k_mean(1+z_log_var - k_square(z_mean) - k_exp(z_log_var), axis=-1L)
  cross_entropy_loss + kl_loss
}

vae %>% compile(optimizer = "SGD", loss = vae_loss, metrics= 'accuracy')
summary(vae)


# LOAD DATA
# Educational data - not in this repository
# r=1
# Y <- as.matrix(read.csv(file=paste("Yrep",r,".csv",sep=""), sep=";", header=FALSE))		#item response values
# Y <- array(data=Y, dim=c(N, num_stats))
# data_train <- Y[1:tr,]

# other data - don't have this yet
# Y <- as.matrix(read.csv(file='baseball_stats.csv'), sep=',', header=TRUE)
# data_train <- Y[1:tr,]

#########  Model 1 training ---------------------------------------------------
vae %>% fit(
  data_train, data_train, 
  shuffle = TRUE, 
  epochs = epochs, 
  batch_size = batch_size, 
)

# Get skill predictions
skill_preds <- predict(encoder,data_train)

# Estimated weights 
W <-get_weights(vae)

# These were useful in the educational setting, not sure if they will be in sports analytics
discr <- as.matrix(W[[7]])
diff <- as.matrix(W[[8]])
print(discr)
print(diff)

