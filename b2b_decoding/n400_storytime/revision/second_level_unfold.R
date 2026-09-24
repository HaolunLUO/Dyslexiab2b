#!/usr/bin/env Rscript
# Second-level age/group model on per-child Unfold CP window means.
# Unweighted OLS. Pointwise Unfold SEs are not the SE of the window mean
# (autocorrelated EEG; Unfold warns they are too small), so they are not used as weights.

suppressPackageStartupMessages(library(jsonlite))

pool <- "/orcd/pool/005/haolun52"
root <- file.path(pool, "n400_storytime_v1", "unfold")
lmm <- file.path(pool, "n400_storytime_v1", "revision", "lmm")
groups <- read.csv(file.path(pool, "dyslexia_natualistics_listing", "b2b_decoding", "joint_v4", "cohort_groups.csv"))
groups <- groups[groups$include_primary == 1, c("participant", "group")]
ages_path <- file.path(pool, "n400_storytime_v1", "stimulus", "lmm_input.csv")
# Subject-level age from the trial table, same parser already applied there.
trials <- read.csv(ages_path, stringsAsFactors = FALSE)
age <- aggregate(age ~ subject, trials, function(x) x[[1]])
names(age)[1] <- "participant"

rows <- list()
for (d in list.dirs(root, recursive = FALSE, full.names = TRUE)) {
  id <- basename(d)
  done <- file.path(d, "fit_done.json")
  if (!file.exists(done)) next
  meta <- fromJSON(done)
  rows[[id]] <- data.frame(
    participant = id,
    window_sur_cp = meta$combined$window_sur_cp,
    window_sur_fz = meta$combined$window_sur_fz,
    window_sur_p1_cp = meta$combined$window_sur_p1_cp,
    stringsAsFactors = FALSE
  )
}
if (length(rows) == 0) {
  stop("no unfold fit_done.json files")
}
df <- do.call(rbind, rows)
df <- merge(df, groups, by = "participant", all.x = TRUE)
df <- merge(df, age, by = "participant", all.x = TRUE)
df$g1 <- ifelse(df$group == "TD", 2/3,
         ifelse(df$group == "dyslexia_normal_CAP", -1/3,
         ifelse(df$group == "dyslexia_atypical_CAP", -1/3, NA)))
df$g2 <- ifelse(df$group == "TD", 0,
         ifelse(df$group == "dyslexia_normal_CAP", 0.5,
         ifelse(df$group == "dyslexia_atypical_CAP", -0.5, NA)))
# Same subject-mean centre as the trial LMM (age-complete N = 57), not recomputed
# on whichever Unfold fits happen to exist.
age_line <- readLines(file.path(lmm, "age_center.txt"))
age_center <- as.numeric(strsplit(age_line[[1]], " ", fixed = TRUE)[[1]][[2]])
df$age_c <- df$age - age_center
out <- file.path(pool, "n400_storytime_v1", "revision", "unfold_second_level")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
write.csv(df, file.path(out, "child_windows.csv"), row.names = FALSE)

fit_one <- function(y, data) {
  data <- data[is.finite(data[[y]]) & is.finite(data$age_c) & is.finite(data$g1), ]
  m <- lm(reformulate(c("age_c", "g1", "g2"), response = y), data = data)
  cf <- as.data.frame(coef(summary(m)))
  cf$term <- rownames(cf)
  ci <- confint(m)
  cf$ci95_low <- ci[cf$term, 1]
  cf$ci95_high <- ci[cf$term, 2]
  cf$outcome <- y
  cf$n <- nrow(data)
  cf
}
coefs <- rbind(
  fit_one("window_sur_cp", df),
  fit_one("window_sur_fz", df),
  fit_one("window_sur_p1_cp", df)
)
write.csv(coefs, file.path(out, "second_level_coef.csv"), row.names = FALSE)
# Pointwise Unfold SEs are not the variance of the 350–550 ms mean
# (Unfold warns they ignore autocorrelation and are too small). No
# inverse-variance weighted second-level model is fit.
writeLines(
  "weighted_second_level_not_fit: window-mean variances were not estimated; pointwise Unfold SEs are not used as weights\n",
  file.path(out, "weights_note.txt")
)
pop <- data.frame(
  outcome = c("window_sur_cp", "window_sur_fz", "window_sur_p1_cp"),
  n = sapply(c("window_sur_cp", "window_sur_fz", "window_sur_p1_cp"), function(y) sum(is.finite(df[[y]]))),
  mean = sapply(c("window_sur_cp", "window_sur_fz", "window_sur_p1_cp"), function(y) mean(df[[y]], na.rm = TRUE)),
  se = sapply(c("window_sur_cp", "window_sur_fz", "window_sur_p1_cp"), function(y) {
    x <- df[[y]]
    x <- x[is.finite(x)]
    sd(x) / sqrt(length(x))
  })
)
write.csv(pop, file.path(out, "population_means.csv"), row.names = FALSE)
# Compare to the fixed-window LMM surprisal term if it exists.
core_coef <- file.path(lmm, "core_coef.csv")
if (file.exists(core_coef)) {
  cc <- read.csv(core_coef)
  sur <- cc[cc$term == "sur_z", ]
  write.csv(data.frame(
    unfold_mean_cp = pop$mean[pop$outcome == "window_sur_cp"],
    unfold_se_cp = pop$se[pop$outcome == "window_sur_cp"],
    lmm_sur_z = if (nrow(sur)) sur$Estimate else NA,
    lmm_ci_low = if (nrow(sur)) sur$ci95_low else NA,
    lmm_ci_high = if (nrow(sur)) sur$ci95_high else NA
  ), file.path(out, "unfold_vs_fixed_window.csv"), row.names = FALSE)
}
cat("second level n", nrow(df), "\n")
print(pop)
