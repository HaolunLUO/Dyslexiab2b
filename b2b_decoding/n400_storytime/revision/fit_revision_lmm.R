#!/usr/bin/env Rscript
# Storytime N400 revision LMMs. Formulas are frozen in
# docs/n400-storytime-revision-plan.md. This script does not prune terms.

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
})

if (!nzchar(Sys.getenv("SLURM_JOB_ID"))) {
  stop("Refusing to fit outside Slurm. Submit run_lmm.sbatch from a login node.")
}

args <- commandArgs(trailingOnly = TRUE)
in_csv <- if (length(args) >= 1) args[[1]] else
  "/orcd/pool/005/haolun52/n400_storytime_v1/stimulus/lmm_input.csv"
out_dir <- if (length(args) >= 2) args[[2]] else
  "/orcd/pool/005/haolun52/n400_storytime_v1/revision/lmm"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

ctrl_fast <- lmerControl(
  optimizer = "bobyqa",
  optCtrl = list(maxfun = 1000000L),
  calc.derivs = FALSE,
  check.conv.singular = .makeCC(action = "ignore", tol = 1e-4)
)
ctrl_check <- lmerControl(
  optimizer = "bobyqa",
  optCtrl = list(maxfun = 1000000L),
  calc.derivs = TRUE,
  check.conv.singular = .makeCC(action = "ignore", tol = 1e-4)
)

log_line <- function(...) {
  msg <- paste0(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " ", paste(..., collapse = " "))
  cat(msg, "\n")
  write(msg, file = file.path(out_dir, "fit_log.txt"), append = TRUE)
}

converged <- function(m) {
  if (!inherits(m, "merMod")) {
    return(list(ok = FALSE, reason = "not_merMod"))
  }
  if (isSingular(m, tol = 1e-4)) {
    return(list(ok = FALSE, reason = "singular"))
  }
  msgs <- m@optinfo$conv$lme4$messages
  if (!is.null(msgs) && length(msgs) > 0) {
    return(list(ok = FALSE, reason = paste(msgs, collapse = " | ")))
  }
  code <- m@optinfo$conv$opt
  if (is.numeric(code) && length(code) == 1 && is.finite(code) && code != 0) {
    return(list(ok = FALSE, reason = paste("opt_code", code)))
  }
  list(ok = TRUE, reason = "ok")
}

fit_lmer <- function(formula, data, reml = TRUE, check = FALSE) {
  control <- if (check) ctrl_check else ctrl_fast
  t0 <- proc.time()[["elapsed"]]
  m <- tryCatch(
    lmer(formula, data = data, REML = reml, control = control),
    error = function(e) e
  )
  elapsed <- proc.time()[["elapsed"]] - t0
  if (inherits(m, "error")) {
    return(list(model = NULL, elapsed = elapsed, ok = FALSE, reason = conditionMessage(m)))
  }
  st <- converged(m)
  list(model = m, elapsed = elapsed, ok = st$ok, reason = st$reason)
}

controls <- paste(
  "duration_z", "position_z", "section",
  "sur_m1_z", "sur_p1_z", "freq_m1_z", "freq_p1_z",
  "assoc_m1_z", "assoc_p1_z", "freq_z", "assoc_z",
  sep = " + "
)

item_rungs <- c(
  item_intercept = "(1 | item)",
  item_age = "(1 + age_c || item)",
  item_age_g1 = "(1 + age_c + g1 || item)",
  item_age_g1_g2 = "(1 + age_c + g1 + g2 || item)"
)
subject_re <- "(1 + sur_z || subject)"

coef_frame <- function(m) {
  sm <- as.data.frame(coef(summary(m)))
  sm$term <- rownames(sm)
  ci <- suppressMessages(confint(m, parm = "beta_", method = "Wald"))
  ci <- as.data.frame(ci)
  ci$term <- rownames(ci)
  names(ci)[1:2] <- c("ci95_low", "ci95_high")
  merge(sm, ci[, c("term", "ci95_low", "ci95_high")], by = "term", all.x = TRUE, sort = FALSE)
}

save_model <- function(m, tag) {
  saveRDS(m, file.path(out_dir, paste0(tag, ".rds")))
  write.csv(coef_frame(m), file.path(out_dir, paste0(tag, "_coef.csv")), row.names = FALSE)
  sink(file.path(out_dir, paste0(tag, "_summary.txt")))
  print(summary(m))
  cat("\n--- convergence ---\n")
  print(converged(m))
  vc <- as.data.frame(VarCorr(m))
  print(vc)
  sink()
  vc <- as.data.frame(VarCorr(m))
  write.csv(vc, file.path(out_dir, paste0(tag, "_varcorr.csv")), row.names = FALSE)
}

# Equal-weight contrasts. Code span of each contrast is 1.
add_contrasts <- function(data, age_center) {
  data$g1 <- NA_real_
  data$g2 <- NA_real_
  data$g1[data$group == "TD"] <- 2 / 3
  data$g2[data$group == "TD"] <- 0
  data$g1[data$group == "dyslexia_normal_CAP"] <- -1 / 3
  data$g2[data$group == "dyslexia_normal_CAP"] <- 0.5
  data$g1[data$group == "dyslexia_atypical_CAP"] <- -1 / 3
  data$g2[data$group == "dyslexia_atypical_CAP"] <- -0.5
  data$age_c <- data$age - age_center
  data
}

group_codes <- function(group) {
  if (group == "TD") return(c(g1 = 2 / 3, g2 = 0))
  if (group == "dyslexia_normal_CAP") return(c(g1 = -1 / 3, g2 = 0.5))
  if (group == "dyslexia_atypical_CAP") return(c(g1 = -1 / 3, g2 = -0.5))
  stop(group)
}

lincomb_row <- function(m, weights, label) {
  b <- fixef(m)
  missing <- setdiff(names(weights), names(b))
  if (length(missing)) stop(paste("missing terms", paste(missing, collapse = ",")))
  L <- matrix(0, nrow = 1, ncol = length(b))
  colnames(L) <- names(b)
  for (nm in names(weights)) L[1, nm] <- weights[[nm]]
  ct <- contest(m, L, joint = FALSE)
  data.frame(
    label = label,
    estimate = unname(ct[1, "Estimate"]),
    se = unname(ct[1, "Std. Error"]),
    df = unname(ct[1, "df"]),
    t = unname(ct[1, "t value"]),
    p = unname(ct[1, "Pr(>|t|)"]),
    ci95_low = unname(ct[1, "Estimate"] - qt(0.975, ct[1, "df"]) * ct[1, "Std. Error"]),
    ci95_high = unname(ct[1, "Estimate"] + qt(0.975, ct[1, "df"]) * ct[1, "Std. Error"]),
    stringsAsFactors = FALSE
  )
}

simple_slopes <- function(m) {
  ages <- c(m1 = -1, mean = 0, p1 = 1)
  groups <- c("TD", "dyslexia_normal_CAP", "dyslexia_atypical_CAP")
  rows <- list()
  for (g in groups) {
    code <- group_codes(g)
    for (an in names(ages)) {
      w <- list(
        sur_z = 1,
        `sur_z:g1` = unname(code["g1"]),
        `sur_z:g2` = unname(code["g2"]),
        `sur_z:age_c` = unname(ages[[an]])
      )
      rows[[paste(g, an)]] <- lincomb_row(
        m, w, sprintf("surprisal|%s|age_c=%g", g, ages[[an]])
      )
    }
  }
  do.call(rbind, rows)
}

term_p <- function(m, term) {
  sm <- coef(summary(m))
  if (!(term %in% rownames(sm))) stop(paste("missing", term))
  sm[term, "Pr(>|t|)"]
}

try_ladder <- function(fixed, data, tag, item_names) {
  chosen <- NULL
  notes <- list()
  for (nm in item_names) {
    fml <- as.formula(paste(fixed, "+", subject_re, "+", item_rungs[[nm]]))
    log_line(tag, "trying", nm)
    fit <- fit_lmer(fml, data, reml = TRUE, check = FALSE)
    log_line(tag, nm, "elapsed", round(fit$elapsed, 1), "ok", fit$ok, fit$reason)
    notes[[nm]] <- list(elapsed = fit$elapsed, ok = fit$ok, reason = fit$reason)
    if (!fit$ok) {
      # Keep going only while we have not yet got a converged simpler model
      # that a richer one failed to improve on. A failure stops the ladder.
      if (!is.null(fit$model)) {
        saveRDS(fit$model, file.path(out_dir, paste0(tag, "_", nm, "_FAILED.rds")))
      }
      break
    }
    chosen <- list(name = nm, model = fit$model, elapsed = fit$elapsed)
    save_model(fit$model, paste0(tag, "_", nm))
  }
  writeLines(
    jsonlite_or_dump(notes),
    file.path(out_dir, paste0(tag, "_ladder.txt"))
  )
  chosen
}

jsonlite_or_dump <- function(x) {
  if (requireNamespace("jsonlite", quietly = TRUE)) {
    return(jsonlite::toJSON(x, auto_unbox = TRUE, pretty = TRUE))
  }
  paste(capture.output(print(x)), collapse = "\n")
}

log_line("host", Sys.info()[["nodename"]], "job", Sys.getenv("SLURM_JOB_ID"))
log_line("reading", in_csv)
d <- read.csv(in_csv, stringsAsFactors = FALSE)
d <- d[d$keep == 1 & is.finite(d$dv_cp) & is.finite(d$dv_fz), ]
d$subject <- factor(d$subject)
d$item <- factor(d$item_id)
d$section <- factor(d$section)
d$group <- factor(d$group)
d$trial_uid <- factor(paste(d$subject, d$item_id, sep = "::"))

age_subjects <- tapply(d$age, d$subject, function(x) x[[1]])
age_subjects <- age_subjects[is.finite(age_subjects)]
age_center <- mean(age_subjects)
log_line("age_center_subject_mean", age_center, "n_age", length(age_subjects))
writeLines(
  c(
    paste("age_center", age_center),
    paste("n_age_subjects", length(age_subjects))
  ),
  file.path(out_dir, "age_center.txt")
)

d <- add_contrasts(d, age_center)
if (any(!is.finite(d$g1) | !is.finite(d$g2))) {
  stop("unassigned group contrast")
}

d_age <- droplevels(d[is.finite(d$age), ])
log_line(
  "age-complete trials", nrow(d_age),
  "subjects", nlevels(d_age$subject),
  "items", nlevels(d_age$item)
)

core_fixed <- paste(
  "dv_cp ~ sur_z * age_c + sur_z * g1 + sur_z * g2 +",
  controls
)
if (file.exists(file.path(out_dir, "core.rds"))) {
  log_line("core.rds exists, skipping ladder")
  core <- list(
    name = readLines(file.path(out_dir, "core_status.txt"), warn = FALSE)[[1]],
    model = readRDS(file.path(out_dir, "core.rds")),
    elapsed = NA_real_
  )
  core$name <- sub("^rung ", "", core$name)
  core_m <- core$model
  core_status <- "loaded"
} else {
core <- try_ladder(core_fixed, d_age, "core", names(item_rungs))
if (is.null(core)) {
  log_line("CORE_FLOOR_FAILED need MixedModels")
  writeLines("NEED_MIXEDMODELS\n", file.path(out_dir, "STATUS.txt"))
  quit(status = 2)
}

# Derivative check on the chosen rung only.
log_line("core derivative check", core$name)
fml_core <- as.formula(paste(core_fixed, "+", subject_re, "+", item_rungs[[core$name]]))
checked <- fit_lmer(fml_core, d_age, reml = TRUE, check = TRUE)
log_line("core check elapsed", round(checked$elapsed, 1), checked$ok, checked$reason)
core_m <- if (checked$ok) checked$model else core$model
core_status <- if (checked$ok) "converged_with_derivs" else paste("fit_ok_derivs_failed:", checked$reason)
save_model(core_m, "core")
writeLines(
  c(paste("rung", core$name), paste("status", core_status), paste("elapsed_first", core$elapsed)),
  file.path(out_dir, "core_status.txt")
)
}

ss <- simple_slopes(core_m)
write.csv(ss, file.path(out_dir, "core_simple_slopes.csv"), row.names = FALSE)

p_g1 <- term_p(core_m, "sur_z:g1")
p_g2 <- term_p(core_m, "sur_z:g2")
p_age <- term_p(core_m, "sur_z:age_c")
holm <- data.frame(
  term = c("sur_z:g1", "sur_z:g2"),
  role = c("prespecified_group", "prespecified_group"),
  p = c(p_g1, p_g2),
  stringsAsFactors = FALSE
)
holm$p_holm <- p.adjust(holm$p, method = "holm")
age_row <- data.frame(
  term = "sur_z:age_c", role = "own_test_not_in_holm", p = p_age, p_holm = NA_real_
)
write.csv(rbind(holm, age_row), file.path(out_dir, "core_holm.csv"), row.names = FALSE)

# Shrinkage from the subject slope.
re <- ranef(core_m, condVar = TRUE)$subject
slope_sd <- NA_real_
vc <- as.data.frame(VarCorr(core_m))
hit <- vc$grp == "subject" & vc$var1 == "sur_z" & (is.na(vc$var2) | vc$var2 == "")
if (any(hit)) slope_sd <- sqrt(vc$vcov[which(hit)[1]])
post <- attr(re, "postVar")
cond_se <- rep(NA_real_, nrow(re))
sur_ix <- match("sur_z", colnames(re))
if (!is.null(post) && !is.na(sur_ix)) {
  if (is.list(post) && is.array(post[[sur_ix]]) && length(dim(post[[sur_ix]])) == 3) {
    cond_se <- sqrt(as.numeric(post[[sur_ix]][1, 1, ]))
  } else if (is.array(post) && length(dim(post)) == 3) {
    cond_se <- sqrt(as.numeric(post[sur_ix, sur_ix, ]))
  }
}
shrink <- data.frame(
  subject = rownames(re),
  ranef_sur_z = re[, "sur_z"],
  cond_se = as.numeric(cond_se),
  stringsAsFactors = FALSE
)
# Child slope at that child's age and group (fixed interactions + RE).
b <- fixef(core_m)
sub_meta <- unique(d_age[, c("subject", "group", "g1", "g2", "age_c", "age")])
sub_meta$subject <- as.character(sub_meta$subject)
shrink <- merge(shrink, sub_meta, by = "subject", all.x = TRUE)
shrink$blup_surprisal <- b["sur_z"] +
  b["sur_z:g1"] * shrink$g1 +
  b["sur_z:g2"] * shrink$g2 +
  b["sur_z:age_c"] * shrink$age_c +
  shrink$ranef_sur_z
write.csv(shrink, file.path(out_dir, "core_subject_blups.csv"), row.names = FALSE)
shrink_sum <- data.frame(
  slope_sd = slope_sd,
  sd_ranef = sd(shrink$ranef_sur_z),
  sd_blup = sd(shrink$blup_surprisal),
  mean_cond_se = mean(shrink$cond_se, na.rm = TRUE),
  median_cond_se = median(shrink$cond_se, na.rm = TRUE)
)
write.csv(shrink_sum, file.path(out_dir, "core_shrinkage.csv"), row.names = FALSE)
log_line("shrinkage", paste(names(shrink_sum), unlist(shrink_sum), sep = "=", collapse = " "))

# Exploratory three-way. Same RE as the core rung. Not in the Holm family.
expl_fixed <- paste(
  "dv_cp ~ sur_z * age_c * g1 + sur_z * age_c * g2 +",
  controls
)
# sur_z * age_c * g1 expands g1:age_c and the three-way; g2 likewise.
# The two three-ways do not include g1:g2. That matches the frozen plan.
log_line("exploratory three-way")
if (file.exists(file.path(out_dir, "exploratory_threeway.rds")) ||
    file.exists(file.path(out_dir, "exploratory_threeway_FAILED.txt"))) {
  log_line("exploratory already saved, skipping")
} else {
expl <- fit_lmer(
  as.formula(paste(expl_fixed, "+", subject_re, "+", item_rungs[[core$name]])),
  d_age, reml = TRUE, check = FALSE
)
log_line("exploratory", expl$ok, expl$reason, "elapsed", round(expl$elapsed, 1))
if (!expl$ok) {
  log_line("exploratory fallback to core RE floor if the rung was richer")
  expl <- fit_lmer(
    as.formula(paste(expl_fixed, "+", subject_re, "+ (1 | item)")),
    d_age, reml = TRUE, check = FALSE
  )
  log_line("exploratory floor", expl$ok, expl$reason, "elapsed", round(expl$elapsed, 1))
}
if (expl$ok) {
  save_model(expl$model, "exploratory_threeway")
} else if (!is.null(expl$model)) {
  saveRDS(expl$model, file.path(out_dir, "exploratory_threeway_FAILED.rds"))
  writeLines(expl$reason, file.path(out_dir, "exploratory_threeway_FAILED.txt"))
}
}

# Sensitivity: all children, no age.
sens_fixed <- paste(
  "dv_cp ~ sur_z * g1 + sur_z * g2 +",
  controls
)
sens_items <- c("item_intercept", "item_age_g1", "item_age_g1_g2")
# age_c is absent. Map the frozen item ladder without age_c.
sens_rungs <- c(
  item_intercept = "(1 | item)",
  item_g1 = "(1 + g1 || item)",
  item_g1_g2 = "(1 + g1 + g2 || item)"
)
log_line("sensitivity no age, trials", nrow(d), "subjects", nlevels(droplevels(d$subject)))
if (file.exists(file.path(out_dir, "sensitivity_noage.rds")) ||
    file.exists(file.path(out_dir, "sensitivity_FAILED.txt"))) {
  log_line("sensitivity already saved, skipping")
} else {
sens_chosen <- NULL
for (nm in names(sens_rungs)) {
  log_line("sensitivity trying", nm)
  fit <- fit_lmer(
    as.formula(paste(sens_fixed, "+", subject_re, "+", sens_rungs[[nm]])),
    d, reml = TRUE, check = FALSE
  )
  log_line("sensitivity", nm, fit$ok, fit$reason, "elapsed", round(fit$elapsed, 1))
  if (!fit$ok) break
  sens_chosen <- list(name = nm, model = fit$model)
  save_model(fit$model, paste0("sensitivity_", nm))
}
if (!is.null(sens_chosen)) {
  save_model(sens_chosen$model, "sensitivity_noage")
  writeLines(sens_chosen$name, file.path(out_dir, "sensitivity_rung.txt"))
} else {
  writeLines("SENSITIVITY_FAILED\n", file.path(out_dir, "sensitivity_FAILED.txt"))
}
}

# Region model. CP = +1/2, Fz = -1/2.
log_line("region model")
if (file.exists(file.path(out_dir, "region.rds")) ||
    file.exists(file.path(out_dir, "region_FAILED.txt"))) {
  log_line("region already saved, skipping")
} else {
cp <- d_age
cp$dv <- cp$dv_cp
cp$region <- 0.5
fz <- d_age
fz$dv <- fz$dv_fz
fz$region <- -0.5
long <- rbind(cp, fz)
long$trial_uid <- factor(paste(long$subject, long$item_id, sep = "::"))
long$subject <- factor(long$subject)
long$item <- factor(long$item)
region_fixed <- paste(
  "dv ~ sur_z * region * g1 + sur_z * region * g2 + sur_z * age_c + region * age_c +",
  controls
)
region_fml <- as.formula(paste(
  region_fixed, "+", subject_re, "+ (1 | item) + (1 | trial_uid)"
))
region <- fit_lmer(region_fml, long, reml = TRUE, check = FALSE)
log_line("region with trial", region$ok, region$reason, "elapsed", round(region$elapsed, 1))
if (!region$ok) {
  region_fml <- as.formula(paste(region_fixed, "+", subject_re, "+ (1 | item)"))
  region <- fit_lmer(region_fml, long, reml = TRUE, check = FALSE)
  log_line("region without trial", region$ok, region$reason, "elapsed", round(region$elapsed, 1))
  if (region$ok) writeLines("dropped_trial_uid\n", file.path(out_dir, "region_re_note.txt"))
}
if (region$ok) {
  log_line("region derivative check")
  checked <- fit_lmer(region_fml, long, reml = TRUE, check = TRUE)
  log_line("region check", checked$ok, checked$reason, "elapsed", round(checked$elapsed, 1))
  if (checked$ok) {
    region <- checked
  } else {
    writeLines(
      paste("fit_ok_derivs_failed:", checked$reason),
      file.path(out_dir, "region_derivs_note.txt")
    )
  }
}
if (region$ok) {
  save_model(region$model, "region")
  cf <- coef_frame(region$model)
  key <- grepl("sur_z", cf$term) & (grepl("region", cf$term) | grepl(":g1", cf$term) | grepl(":g2", cf$term) | cf$term == "sur_z")
  write.csv(cf[key, ], file.path(out_dir, "region_key_terms.csv"), row.names = FALSE)
} else if (!is.null(region$model)) {
  saveRDS(region$model, file.path(out_dir, "region_FAILED.rds"))
  writeLines(region$reason, file.path(out_dir, "region_FAILED.txt"))
}
}

# Phase 2: section split-half. Section is constant inside each half, so it is dropped.
# Item slopes are attempted only when the core rung had them.
half_blup <- function(data, half_name) {
  sub <- droplevels(data[as.character(data$section) == half_name, ])
  # section has one level; drop it from the fixed controls.
  controls_half <- gsub(" \\+ section \\+", " + ", paste0(" ", controls, " "))
  controls_half <- trimws(gsub("section \\+ ", "", controls_half))
  controls_half <- trimws(gsub(" \\+ section", "", controls_half))
  fixed <- paste("dv_cp ~ sur_z * age_c + sur_z * g1 + sur_z * g2 +", controls_half)
  re_item <- if (!is.null(core) && core$name != "item_intercept") item_rungs[[core$name]] else "(1 | item)"
  log_line("splithalf", half_name, "trials", nrow(sub), "re_item", re_item)
  half_fml <- as.formula(paste(fixed, "+", subject_re, "+", re_item))
  fit <- fit_lmer(half_fml, sub, reml = TRUE, check = FALSE)
  log_line("splithalf", half_name, fit$ok, fit$reason, "elapsed", round(fit$elapsed, 1))
  if (!fit$ok && re_item != "(1 | item)") {
    half_fml <- as.formula(paste(fixed, "+", subject_re, "+ (1 | item)"))
    fit <- fit_lmer(half_fml, sub, reml = TRUE, check = FALSE)
    log_line("splithalf floor", half_name, fit$ok, fit$reason, "elapsed", round(fit$elapsed, 1))
  }
  if (!fit$ok) {
    writeLines(fit$reason, file.path(out_dir, paste0("splithalf_", half_name, "_FAILED.txt")))
    return(NULL)
  }
  log_line("splithalf derivative check", half_name)
  checked <- fit_lmer(half_fml, sub, reml = TRUE, check = TRUE)
  log_line("splithalf check", half_name, checked$ok, checked$reason, "elapsed", round(checked$elapsed, 1))
  if (checked$ok) fit <- checked
  save_model(fit$model, paste0("splithalf_", half_name))
  re <- ranef(fit$model)$subject
  b <- fixef(fit$model)
  meta <- unique(sub[, c("subject", "group", "g1", "g2", "age_c")])
  meta$subject <- as.character(meta$subject)
  out <- data.frame(subject = rownames(re), ranef_sur_z = re[, "sur_z"], stringsAsFactors = FALSE)
  out <- merge(out, meta, by = "subject", all.x = TRUE)
  # Interaction names follow formula order sur_z * age_c, sur_z * g1, sur_z * g2.
  out$blup <- unname(b["sur_z"]) +
    unname(b["sur_z:g1"]) * out$g1 +
    unname(b["sur_z:g2"]) * out$g2 +
    unname(b["sur_z:age_c"]) * out$age_c +
    out$ranef_sur_z
  out$half <- half_name
  out
}

if (!file.exists(file.path(out_dir, "splithalf_blups.csv"))) {
  b1 <- half_blup(d_age, "1")
  b2 <- half_blup(d_age, "2")
  if (!is.null(b1) && !is.null(b2)) {
    mrg <- merge(b1[, c("subject", "group", "blup")], b2[, c("subject", "blup")],
                 by = "subject", suffixes = c("_s1", "_s2"))
    write.csv(mrg, file.path(out_dir, "splithalf_blups.csv"), row.names = FALSE)
    r_all <- cor(mrg$blup_s1, mrg$blup_s2)
    sb <- if (is.finite(r_all) && r_all > -1) 2 * r_all / (1 + r_all) else NA_real_
    # One-way random ICC, absolute agreement, k = 2 halves.
    wide <- as.matrix(mrg[, c("blup_s1", "blup_s2")])
    n <- nrow(wide)
    k <- 2
    grand <- mean(wide)
    msb <- k * sum((rowMeans(wide) - grand)^2) / (n - 1)
    msw <- sum((wide - rowMeans(wide))^2) / (n * (k - 1))
    icc <- (msb - msw) / (msb + (k - 1) * msw)
    by_g <- lapply(split(mrg, mrg$group), function(g) {
      data.frame(
        group = g$group[[1]],
        n = nrow(g),
        pearson_r = cor(g$blup_s1, g$blup_s2),
        stringsAsFactors = FALSE
      )
    })
    rel <- data.frame(
      n = n,
      pearson_r = r_all,
      spearman_brown = sb,
      icc_oneway_absolute = icc,
      gate_threshold = 0.70,
      gate_open = is.finite(sb) && sb >= 0.70 && r_all >= 0,
      stringsAsFactors = FALSE
    )
    write.csv(rel, file.path(out_dir, "splithalf_reliability.csv"), row.names = FALSE)
    write.csv(do.call(rbind, by_g), file.path(out_dir, "splithalf_by_group.csv"), row.names = FALSE)
    log_line("reliability r", r_all, "SB", sb, "ICC", icc, "gate", rel$gate_open)
  } else {
    writeLines("SPLITHALF_FAILED\n", file.path(out_dir, "splithalf_FAILED.txt"))
  }
} else {
  log_line("splithalf_blups.csv exists, skipping")
}

writeLines("LME4_DONE\n", file.path(out_dir, "STATUS.txt"))
log_line("done")
