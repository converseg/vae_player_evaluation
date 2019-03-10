library(keras)
K <- keras::backend()
library(tensorflow)

num_stats <- 28
num_skills <- 3

N <- 10000 #number of subjects
tr <- 10000 #how many to train on
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
# hidden layers can be toyed with
h <- layer_dense(input, 10, activation = 'tanh', name='hidden')
z_mean <- layer_dense(h, num_skills, name='z_mean')
z_log_var <- layer_dense(h, num_skills, name='z_log_var')

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

# don't need this
# regularize_Q <- function(weight_matrix){
#   target <- weight_matrix * Q
#   diff = weight_matrix - target
#   loss <- 100000 * k_mean(diff)
#   k_relu(loss)
# }

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

# Decoder - we never really access this - don't need it
# decoder_input <- layer_input(shape=num_skills, name='decoder_input')
# decoder_out <- layer_dense(decoder_input, num_stats, activation='sigmoid', name='decoder_output')
# decoder <- keras_model(decoder_input, out)
# summary(decoder)


vae_loss <- function(input, output){
  cross_entropy_loss <- (num_stats/1.0)* loss_binary_crossentropy(input, output)
  kl_loss <- -0.5*k_mean(1+z_log_var - k_square(z_mean) - k_exp(z_log_var), axis=-1L)
  cross_entropy_loss + kl_loss
}

vae %>% compile(optimizer = "SGD", loss = vae_loss, metrics= 'accuracy')
summary(vae)

# Load data
# Educational data - not in this repository
r=1
Y <- as.matrix(read.csv(file=paste("Yrep",r,".csv",sep=""), sep=";", header=FALSE))		#item response values
Y <- array(data=Y, dim=c(N, num_stats))
data_train <- Y[1:tr,]

# other data
# Y <- as.matrix(read.csv(file='baseball_stats.csv'), sep=',', header=TRUE)
# data_train <- Y[1:tr,]

#########  Model 1 training and save results ---------------------------------------------------
vae %>% fit(
  data_train, data_train, 
  shuffle = TRUE, 
  epochs = epochs, 
  batch_size = batch_size, 
  #  validation_data = list(data_test,data_test[,c(10,13,14,25,27,2,8,23,24,4,5,6,9,15,18,19,22,26,28,1,3,7,11,12,16,20,21,17)])
)

# Get skill predictions
skill_preds <- predict(encoder,data_train)

# Estimated weights 
W <-get_weights(vae)

discr <- as.matrix(W[[7]])
diff <- as.matrix(W[[8]])
print(discr)
print(diff)

# load in data to check results with
# a_values     <- as.matrix(read.csv("a_values.csv", sep=";", header=FALSE))	#discrimination parameters
# b_values     <- as.matrix(read.csv("b_values.csv", sep=";", header=FALSE))	#difficulty parameters
# theta_values <- as.matrix(read.csv("theta_values.csv", sep=";", header=FALSE))	#latent trait values


