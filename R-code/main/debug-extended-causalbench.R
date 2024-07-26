source("main/dependencies.R")


# Import data
dat2 <- dat
dat <- read_csv("../results/_debug/df.csv")
genes <- colnames(dat)[1:6]


dat2plot <- dat %>% filter(env == "train" | Z == "ENSG00000231500")

ggplot(dat2plot %>% filter(algo=="BCF")) +
  geom_point(aes(x = !!sym(genes[2]), y = !!sym(genes[1]), col = Z), alpha = 0.5) +
  coord_equal()

ggplot(dat2plot) +
  facet_grid(env ~ algo) +
  geom_point(aes(x = !!sym(genes[2]), y = !!sym(genes[1]))) +
  geom_point(aes(x = !!sym(genes[2]), y = y_pred, col = algo)) +
  coord_cartesian(xlim = c(0, 5))


dat2plot %>% 
  group_by(Z) %>% 
  summarise(x1 = mean(ENSG00000231500),
            x2 = mean(ENSG00000145425))


ggplot(dat2plot %>% filter(algo=="BCF")) +
  geom_point(aes(x = !!sym(genes[2]), y = !!sym(genes[3]), col = Z), alpha = 0.5) +
  coord_equal()



