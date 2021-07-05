# Load libraries
library(loadeR)
library(transformeR) 
library(downscaleR)
library(climate4R.value)
library(magrittr)
library(sp)
library(keras)
library(tensorflow)
library(downscaleR.keras)
library(loadeR.2nc)

##########################################################################

# Load data 
load('./data/x.rda')
load('./data/y.rda')

# Spatial subset for CNN compatibility
y <- subsetSpatial(y, lonLim = c(-164.75, -60.75),
		   latLim = c(11.75, 69.75),
	           outside = FALSE)

# Select the train years
years_train <- 1979:2002

x_train <- subsetGrid(x, years = years_train)
y_train <- subsetGrid(y, years = years_train)

# Model configuration
model_name <- 'MNN'
connections <- c('conv', 'conv')

# Generate xyT
ysub <- y_train; xsub <- x_train

ysub <- binaryGrid(gridArithmetics(ysub, 0.99, operator = '-'),
                   condition = 'GE',
                   threshold = 0,
                   partial = TRUE)

# Standardize
xyT <- prepareData.keras(scaleGrid(xsub, type = 'standardize'),
                         y = ysub,
                         first.connection = connections[1],
                         last.connection = connections[2],
                         channels = 'last')

# Save final object
elements_list <- list()

elements_list[['model_name']] <- model_name
elements_list[['connections']] <- connections
elements_list[['xyT']] <- xyT
elements_list[['x_train']] <- xyT$x.global
elements_list[['y_train']] <- xyT$y$Data
elements_list[['years_train']] <- years_train

save(elements_list, file = './data/elements_list.rda')
