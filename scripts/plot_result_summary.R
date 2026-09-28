library("tidyr")
library("ggplot2")
library("scales")

path <- "supplementary"
datasets <- list.files(path, pattern = "contrast")
patt <- "z_threshold_(.*).csv"
res_list <- lapply(datasets, FUN = function(ds_name) {
  path_ds <- file.path(path, ds_name)
  filenames <- list.files(path_ds, pattern = patt)
  dat_list <- lapply(filenames, FUN = function(filename) {
    z <- as.numeric(gsub(patt, "\\1", filename))
    pathname <- file.path(path_ds, filename)
    dat <- read.csv(pathname, check.names = FALSE)
    w <- which(!is.na(dat[["TDP (ARI)"]]))
    if (length(w)) {
      data.frame(dataset = ds_name, z = z, dat[w, ], check.names = FALSE)
    }
  })
  Reduce(rbind, dat_list)
})

res <- Reduce(rbind, res_list)
idxs <- grep("TDP", colnames(res))
mins <- matrixStats::rowMaxs(as.matrix(res[, idxs]))
res$trivial <- (mins == 0)

voxel_size <- 27
res[["Cluster Size (voxels)"]] <- res[["Cluster Size (mm3)"]] / voxel_size
x_lab <- sprintf("Cluster Size in #voxels (1 voxel = %s mm3)", voxel_size)

res2 <- pivot_longer(res, starts_with("TDP"),  
             names_to = "method", names_pattern = "TDP \\((.*)\\)", 
             values_to = "TDP") %>%
  dplyr::filter(method %in% c("ARI", "Notip", "pARI")) %>%
  dplyr::mutate(z = as.factor(z)) %>%
  dplyr::mutate(method = as.factor(method))

df_points <- subset(res2, z %in% c(3, 4, 5))

x_range <- range(df_points$`Cluster Size (voxels)`)
x_seq <- 10^seq(log10(x_range[1]), log10(x_range[2]), length.out = 500)

min_tdp_pari <- function(x, delta = 27) pmax(0, 1 - delta / x)
color_palette <- "Dark2"
pari_color <- scales::brewer_pal(palette = color_palette)(nlevels(df_points$method))[
  which(levels(df_points$method) == "pARI")
]

df_bound <- data.frame(
  `Cluster Size (voxels)` = x_seq,
  ymax = min_tdp_pari(x_seq),
  check.names = FALSE
)

p <- ggplot(df_points,
            aes(y = TDP, x = `Cluster Size (voxels)`, 
                color = method, group = z, shape = method)) +
  # Admissible zone for pARI
  geom_ribbon(
    data = df_bound,
    aes(x = `Cluster Size (voxels)`, ymin = 0, ymax = ymax),
    inherit.aes = FALSE,
    fill = pari_color,
    alpha = 0.15,
    colour = NA
  ) +
  geom_point(alpha = 0.7, size = 1) +
  geom_function(
    aes(linetype = "bound"),
    inherit.aes = FALSE,
    fun = min_tdp_pari,
    colour = "black",
    data = data.frame(x = 1, z = unique(df_points$z))  # minimal data.frame
  ) +
  scale_linetype_manual(
    name = NULL,
    values = c("bound" = "dotted"),
    labels = c("bound" = expression(bar(TDP)(S) == (1 - delta/abs(S))["+"]))
  ) +
  guides(
    color = guide_legend(order = 1),
    shape = guide_legend(order = 1),
    linetype = guide_legend(order = 2, override.aes = list(colour = "black"))) +
  scale_color_brewer(palette = color_palette) +
  xlab(x_lab) +
  ylab("TDP lower bound") +
  facet_grid(. ~ z, labeller = function(...) label_both(..., sep = " = ")) +
  scale_x_log10(labels = label_log()) +
  theme_bw()
p
ggsave(p, file = "figures/Figure-3_TDP-vs-cluster-size.png", 
       width = 8, height = 4)


# boxplot/violin plots requested for rebuttal
p <- ggplot(df_points,
            aes(x = method, y = TDP,
                color = method, fill = method)) +
  geom_violin(alpha = 0.3, trim = FALSE, bounds = c(0,1)) +
  scale_color_brewer(palette = "Dark2") +
  scale_fill_brewer(palette = "Dark2") +
  xlab("Method") +
  ylab("FDP upper bound") +
  facet_grid(. ~ z, labeller = function(...) label_both(..., sep = " = ")) +
  theme_bw() 
p
