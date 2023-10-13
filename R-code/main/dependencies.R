# dependency imports
chooseCRANmirror(ind = 1)
is_pacman_installed <- "pacman" %in% rownames(installed.packages())
if (is_pacman_installed == FALSE) install.packages("pacman")

# load-install-cran
cran_packs <- c(
  "tidyverse", "here", "knitr", "egg", "latex2exp", "khroma"
)

pacman::p_load(cran_packs, update = FALSE, character.only = TRUE)


purrr::map(here::here("R", list.files(here::here("R"))), source)

