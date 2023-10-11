# dependency imports
library(tidyverse)
library(here)
library(cowplot)
library(grid)
library(gridExtra)
library(ggh4x)
library(latex2exp)
library(rngtools)
library(egg)
library(backports)
library(khroma)

purrr::map(here::here("R", list.files(here::here("R"))), source)

# R options
options(future.rng.onMisuse = "ignore")
RNGkind(kind = "L'Ecuyer-CMRG")
