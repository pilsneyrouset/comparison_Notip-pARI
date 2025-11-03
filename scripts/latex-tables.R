library("dplyr")
library("knitr")
library("kableExtra")

path <- "results"
task <- "task36"

datasets <- list.files(path, pattern = task)
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
      data.frame(z = z, dat[w, ], check.names = FALSE)
    }
  })
  Reduce(rbind, dat_list)
})

res <- Reduce(rbind, res_list)
idxs <- grep("TDP", colnames(res))

mins <- matrixStats::rowMaxs(as.matrix(res[, idxs]))
mean(mins==0) # 70%

df <- res[mins > 0, ]
nms <- gsub("Cluster ", "", colnames(df))
colnames(df) <- nms

z_vals <- unique(df$z)
out_path <- "tables"
dir.create(out_path, showWarnings = FALSE)
for (z_val in z_vals) {
  filename <- sprintf("TDP-cluster-table_%s_z=%s.tex", task, z_val)
  pathname <- file.path(out_path, filename)
  dat_tab <- subset(df, z == z_val)[, -1]
  # tab <- knitr::kable(dat_tab, row.names = FALSE, format = "latex")
  
  # highlight largest TDP in each row
  highlighted <- dat_tab %>%
    rowwise() %>%
    mutate(
      across(
        c(`TDP (ARI)`, `TDP (Notip)`, `TDP (pARI)`),
        ~ ifelse(.x == max(c_across(c(`TDP (ARI)`, `TDP (Notip)`, `TDP (pARI)`))),
                 cell_spec(.x, "latex", bold = TRUE),
                 as.character(.x))
      )
    ) %>%
    ungroup()
  
  # Rename columns to avoid redundancy
  colnames(highlighted)[7:9] <- c("ARI", "Notip", "pARI")
  
  tab <- kable(highlighted, format = "latex", escape = FALSE, booktabs = TRUE,
        linesep = "") %>%
    add_header_above(
      c(" " = 6, "TDP lower bound" = 3)
    )
  write(tab, file = pathname)
}
