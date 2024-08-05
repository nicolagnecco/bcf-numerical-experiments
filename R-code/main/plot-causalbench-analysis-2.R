source("main/dependencies.R")

# Y, X, envs = X_1 s.t. do(X_1 = is weak)


# 1. try choosing points randomly (FALLBACK) !!! NO INJECTION OF CONFOUNDER
#
# 2: different SELECTION
# 2.a choose training envs s.t. may have causal effect on predictor
#    (check on ints that have causal effect (median(Y | envs)) -> (Matthieu?))
# want to select genes that are causally related.
# 2.b fix response, look for predictors that are very correlated
# for every i, j in {1, ..., how 622}
# i -> j: |median(Y_j|obs) - median(Y_j | do(Y_i))|
#
# M \in R^{622 x 622}: colmax ->
# {1, 2, 3}:
# i. take gene with most parents
#
#
# remove gene i (i.e., col and row i), if colmax(M_{., i}) = min(colmax(M_{., i})),
# maybe over rows remove gene with smallest causal effect.
#
# 3. pooling training envs (do not contains those knocked, possibly) into 1
# -> will keep one direction, but hopefully more useful.


res <- read_csv(
    "../results/causalbench-analysis-2/n_preds_3-confounders_False/20240802-140555/causalbench-res.csv"
)

# few shots random training obs
res <- read_csv(
    "../results/causalbench-analysis-2/n_preds_3-confounders_False/20240802-141446/causalbench-res.csv"
)


res <- read_csv(
    "../results/causalbench-analysis-2/n_preds_3-confounders_True/20240802-091537/causalbench-res.csv"
)

# few shots random training obs
res <- read_csv(
    "../results/causalbench-analysis-2/n_preds_3-confounders_True/20240802-142756/causalbench-res.csv"
)
# few shots random training obs more focused on small interv_strength
res <- read_csv(
    "../results/causalbench-analysis-2/n_preds_3-confounders_True/20240802-143239//causalbench-res.csv"
)
res <- read_csv(
  "../results/causalbench-analysis-2/n_preds_3-confounders_False/20240806-001205//causalbench-res.csv"
)


res <- res %>%
    mutate(algorithm = refactor_methods(algorithm, rev = TRUE))


res %>%
    select(response, predictors, training_envs, confounders, test_envs) %>%
    unique() %>%
    view()

ggplot(res) +
    geom_boxplot(aes(x = factor(interv_strength), y = mse, col = algorithm)) +
    scale_color_manual(values = my_colors, guide = guide_legend(reverse = TRUE))


# for each response, and intervention strength,
# take maximum over training environments, confounders, and, most importantly,
# over direction of perturbations
dat2plot <- res %>%
    group_by(algorithm, response, interv_strength) %>%
    summarise(mse = max(mse))

dat2plot_agg <- dat2plot %>%
    ungroup() %>%
    group_by(algorithm, interv_strength) %>%
    summarise(mse = mean(mse))

ggplot(data = dat2plot_agg %>% filter(TRUE)) +
    geom_line(mapping = aes(
        x = interv_strength, y = mse,
        col = algorithm, size = algorithm, linetype = algorithm
    )) +
    # geom_line(data=dat2plot, mapping=aes(x = interv_strength, y = mse, col = algorithm,
    # group = interaction(algorithm, response)),
    # alpha = .2) +
    geom_point(aes(x = interv_strength, y = mse, col = algorithm, shape = algorithm),
        fill = "white", size = 2, stroke = 0.75, alpha = 0.75
    ) +
    scale_color_manual(values = my_colors, guide = guide_legend(reverse = TRUE)) +
    scale_size_manual(values = my_sizes, guide = guide_legend(reverse = TRUE)) +
    scale_linetype_manual(values = my_linetypes, guide = guide_legend(reverse = TRUE)) +
    scale_shape_manual(values = my_shapes, guide = guide_legend(reverse = TRUE))
