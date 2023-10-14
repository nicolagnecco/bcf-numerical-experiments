source("main/dependencies.R")

smooth_rainbow <- colour("smooth rainbow")

# Constant definitions
CSVFILE <- "../results/output_data/experiment_2-nullspace.csv"
FILENAME <-"../results/figures/column-space.pdf"


# Function definitions
cut_above <- function(x, thres){
  x[x > thres] <- NaN
  return(x)
}

# Null space experiment
dat <- read_csv(CSVFILE) %>% 
  group_by(eigengap, p, r, q, n) %>% 
  summarise(dist = mean(dist_null_space)) %>% 
  filter(q %in% c(5)) %>% 
  mutate(n = factor(n),
         q = factor(q),
         p = texify_column(p, "p"),
         r = texify_column(r, "r"))

gg <- ggplot(dat) +
  facet_grid(p ~ r,  labeller = label_parsed) +
  geom_line(aes(x = eigengap, y = dist, col = n), alpha = .75) +
  geom_point(aes(x = eigengap, y = dist, col = n), alpha = .75,
             shape = 21, fill = "white", size = 1, stroke = .75) +
  scale_color_manual(values = smooth_rainbow(12, range = c(0.2, 0.65))) +
  xlab(TeX("$\\tau = \\sqrt{Eigengap}$")) +
  ylab(TeX("||$\\Pi_{M_0} - \\Pi_{\\hat{M_0}}||_{F}^2$"))

gg

save_myplot(plt = gg, plt_nm = FILENAME, width = 1.5, height = 1.5)
