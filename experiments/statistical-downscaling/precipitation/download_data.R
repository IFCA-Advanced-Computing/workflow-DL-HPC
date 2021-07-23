DATA_PATH <- './data/'

# Login into UDG http://meteo.unican.es/udg-tap/home
library(loadeR)
library(transformeR)
library(magrittr)

# Enter account info
loginUDG('***', '***')

# Location and temporal selection
longitude <- c(-165, -60)
latitude <- c(12, 70)
time <- 1979:2008

# Predictor's variables
variables <- c('z@500','z@700','z@850','z@1000', 
               'hus@500','hus@700','hus@850','hus@1000', 
               'ta@500','ta@700','ta@850','ta@1000', 
               'ua@500','ua@700','ua@850','ua@1000', 
               'va@500','va@700','va@850','va@1000')

# Download and save predictor (ERA-Interim)
x <- lapply(variables, function(x) {
  		loadGridData(dataset = 'ECMWF_ERA-Interim-ESD',
               	     var = x,
                     lonLim = longitude,
                     latLim = latitude, 
                     years = time)
      }) %>% makeMultiGrid()

save(x, file = paste0(DATA_PATH, 'x.rda'))

# Download and save predictand (EWEMBI)
y <- loadGridData(dataset = 'PIK_Obs-EWEMBI',
                  var = 'pr',
                  lonLim = longitude,
                  latLim = latitude, 
                  years = time)

save(y, file = paste0(DATA_PATH, 'y.rda'))
