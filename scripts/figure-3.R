library("tidyr")
library("ggplot2")
library("scales")

path <- "results"
datasets <- list.files(path, pattern = "task")
patt <- "1000_perm_z_threshold_(.*).csv"
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

res2 <- pivot_longer(res, starts_with("TDP"),  
             names_to = "method", names_pattern = "TDP \\((.*)\\)", 
             values_to = "TDP") %>%
  dplyr::mutate(z = as.factor(z))

df_points <- subset(res2, z %in% c(3, 4, 5))

p <- ggplot(df_points,
            aes(y = TDP, x = `Cluster Size (mm3)`, 
                color = method, group = z, shape = method)) +
  geom_point(alpha = 0.5, size = 1) +
  scale_color_brewer(palette = "Dark2") +
  ylab("TDP lower bound") +
  facet_grid(. ~ z, labeller = function(...) label_both(..., sep = " = ")) +
  scale_x_log10(labels = label_log()) + 
  theme_bw() 
p
ggsave(p, file = "figures/Notip-vs-pARI_TDP-vs-cluster-size_B=1000.png", 
       width = 6, height = 4)

