mean = 5
n = 10
var = 5

#variance = (n*alpha*beta*( alpha + beta + n))/ ((alpha + beta)^2 * ( alpha + beta + 1))

# variance = n*p*(1-p) [1 + (n -1)*p]
#p = 1 / ( alpha + beta + 1)

# se beta = (n*alpha/mean) - alpha
# p = 1/((n*alpha/mean) -1)
 
#variance = n*(1/((n*alpha/mean) -1))*(1- 1/((n*alpha/mean) -1)) [1 + (n-1)*(1/((n*alpha/mean) -1))]
beta = (n*alpha/mean) - alpha

