#!/usr/bin/env Rscript
# Phase 4 trial-level behaviour models. Runs only if the split-half gate is open.
# One model per outcome already reported. No new outcomes.

if (!nzchar(Sys.getenv("SLURM_JOB_ID"))) {
  stop("Refusing to fit behaviour models outside Slurm. Submit run_behaviour.sbatch.")
}

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
})

pool <- "/orcd/pool/005/haolun52"
lmm_dir <- file.path(pool, "n400_storytime_v1", "revision", "lmm")
beh_dir <- file.path(pool, "n400_storytime_v1", "revision", "behaviour")
dir.create(beh_dir, recursive = TRUE, showWarnings = FALSE)
script_dir <- Sys.getenv(
  "N400_REV_DIR",
  unset = file.path(pool, "_work_Dyslexiab2b_n400rev", "b2b_decoding", "n400_storytime", "revision")
)

log_line <- function(...) {
  msg <- paste0(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " ", paste(..., collapse = " "))
  cat(msg, "\n")
  write(msg, file = file.path(beh_dir, "fit_log.txt"), append = TRUE)
}

rel_path <- file.path(lmm_dir, "splithalf_reliability.csv")
if (!file.exists(rel_path)) {
  writeLines("NO_RELIABILITY_FILE\n", file.path(beh_dir, "behaviour_SKIPPED.txt"))
  stop("split-half reliability file is missing; gate cannot open")
}
rel <- read.csv(rel_path, stringsAsFactors = FALSE)
gate <- isTRUE(rel$gate_open[[1]]) || identical(tolower(as.character(rel$gate_open[[1]])), "true")
log_line(
  "gate_open", gate,
  "r", rel$pearson_r[[1]],
  "SB", rel$spearman_brown[[1]]
)
if (!gate) {
  writeLines(
    c(
      "GATE_CLOSED",
      "The by-child surprisal slope is not precise enough for trial-level behavioural individual-difference claims.",
      paste("pearson_r", rel$pearson_r[[1]]),
      paste("spearman_brown", rel$spearman_brown[[1]]),
      paste("icc", rel$icc_oneway_absolute[[1]])
    ),
    file.path(beh_dir, "behaviour_SKIPPED.txt")
  )
  log_line("gate closed; no behaviour models fit")
  quit(save = "no", status = 0)
}

raw_csv <- file.path(beh_dir, "raw_scores.csv")
if (!file.exists(raw_csv)) {
  py <- file.path(script_dir, "prepare_behaviour_scores.py")
  status <- system2("python3", py)
  if (status != 0 || !file.exists(raw_csv)) stop("score table was not written")
}

ctrl <- lmerControl(
  optimizer = "bobyqa",
  optCtrl = list(maxfun = 1000000L),
  calc.derivs = FALSE,
  check.conv.singular = .makeCC(action = "ignore", tol = 1e-4)
)

converged <- function(m) {
  if (!inherits(m, "merMod")) return(list(ok = FALSE, reason = "not_merMod"))
  if (isSingular(m, tol = 1e-4)) return(list(ok = FALSE, reason = "singular"))
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

holm_two <- function(p1, p2) {
  p <- c(p1, p2)
  ord <- order(p)
  adj <- numeric(2)
  running <- 0
  for (rank in seq_along(ord)) {
    i <- ord[[rank]]
    val <- min(1, max(running, (2 - rank + 1) * p[[i]]))
    adj[[i]] <- val
    running <- val
  }
  adj
}

zscore <- function(x) {
  m <- is.finite(x)
  if (sum(m) < 2) return(rep(NA_real_, length(x)))
  mu <- mean(x[m])
  sd <- stats::sd(x[m])  # sample SD, ddof = 1
  if (!is.finite(sd) || sd == 0) {
    out <- rep(NA_real_, length(x))
    out[m] <- 0
    return(out)
  }
  out <- rep(NA_real_, length(x))
  out[m] <- (x[m] - mu) / sd
  out
}

in_csv <- file.path(pool, "n400_storytime_v1", "stimulus", "lmm_input.csv")
d <- read.csv(in_csv, stringsAsFactors = FALSE)
d <- d[d$keep == 1 & is.finite(d$dv_cp) & is.finite(d$dv_fz), ]
age_line <- readLines(file.path(lmm_dir, "age_center.txt"))
age_center <- as.numeric(strsplit(age_line[[1]], " ", fixed = TRUE)[[1]][[2]])
d$g1 <- NA_real_
d$g2 <- NA_real_
d$g1[d$group == "TD"] <- 2 / 3
d$g2[d$group == "TD"] <- 0
d$g1[d$group == "dyslexia_normal_CAP"] <- -1 / 3
d$g2[d$group == "dyslexia_normal_CAP"] <- 0.5
d$g1[d$group == "dyslexia_atypical_CAP"] <- -1 / 3
d$g2[d$group == "dyslexia_atypical_CAP"] <- -0.5
d$age_c <- d$age - age_center
d <- d[is.finite(d$age), ]
d$subject <- as.character(d$subject)

scores <- read.csv(raw_csv, stringsAsFactors = FALSE)
scores$subject <- as.character(scores$subject)

# Composites on the age-complete EEG children who have each subtest, then
# the model z-scores the composite again on the children who enter.
age_ids <- unique(d$subject)
scores <- scores[scores$subject %in% age_ids, ]
pa_src <- c(pa_tone = "pa_tone", pa_phoneme = "pa_phoneme", pa_syllable = "pa_syllable")
ran_src <- c(ran_digits = "ran_digits", ran_mixed = "ran_mixed", ran_colors = "ran_colors", ran_object = "ran_object")
for (nm in c(names(pa_src), names(ran_src))) {
  if (!nm %in% names(scores)) scores[[nm]] <- NA_real_
  scores[[paste0(nm, "_z")]] <- zscore(scores[[nm]])
}
scores$pa_mean <- rowMeans(scores[, paste0(names(pa_src), "_z")], na.rm = TRUE)
scores$pa_mean[!is.finite(scores$pa_mean)] <- NA_real_
scores$ran_mean <- rowMeans(scores[, paste0(names(ran_src), "_z")], na.rm = TRUE)
scores$ran_mean[!is.finite(scores$ran_mean)] <- NA_real_

controls <- paste(
  "duration_z", "position_z", "section",
  "sur_m1_z", "sur_p1_z", "freq_m1_z", "freq_p1_z",
  "assoc_m1_z", "assoc_p1_z", "freq_z", "assoc_z",
  sep = " + "
)
# Match the Phase 1 rung. The subject surprisal slope stays if the item term is singular.
status_lines <- readLines(file.path(lmm_dir, "core_status.txt"))
core_rung <- sub("^rung ", "", status_lines[[1]])
re_by_rung <- c(
  item_intercept = "(1 | item)",
  item_age = "(1 + age_c || item)",
  item_age_g1 = "(1 + age_c + g1 || item)",
  item_age_g1_g2 = "(1 + age_c + g1 + g2 || item)"
)
if (!(core_rung %in% names(re_by_rung))) stop(paste("unknown core rung", core_rung))
re_rich <- paste("(1 + sur_z || subject) +", re_by_rung[[core_rung]])
re_floor <- "(1 + sur_z || subject) + (1 | item)"
log_line("core_rung", core_rung)

outcomes <- list(
  list(key = "literacy", col = "literacy", family = "primary", label = "识字量"),
  list(key = "pa", col = "pa_mean", family = "primary", label = "PA"),
  list(key = "gap", col = "gap", family = "secondary", label = "噪声间隙（识别门限）"),
  list(key = "dichotic", col = "dichotic", family = "secondary", label = "数字分听-自由回忆"),
  list(key = "ran", col = "ran_mean", family = "exploratory", label = "RAN_zmean"),
  list(key = "ran_object", col = "ran_object", family = "exploratory", label = "快速命名-物体")
)

coef_frame <- function(m) {
  sm <- as.data.frame(coef(summary(m)))
  sm$term <- rownames(sm)
  ci <- suppressMessages(confint(m, parm = "beta_", method = "Wald"))
  ci <- as.data.frame(ci)
  ci$term <- rownames(ci)
  names(ci)[1:2] <- c("ci95_low", "ci95_high")
  merge(sm, ci[, c("term", "ci95_low", "ci95_high")], by = "term", all.x = TRUE, sort = FALSE)
}

fit_one <- function(spec) {
  sc <- scores[, c("subject", spec$col)]
  names(sc)[2] <- "raw_outcome"
  # Z-score on the children who enter this model.
  enter <- unique(d$subject[d$subject %in% sc$subject[is.finite(sc$raw_outcome)]])
  sc <- sc[sc$subject %in% enter, ]
  sc$outcome_z <- zscore(sc$raw_outcome)
  sub <- merge(d, sc[, c("subject", "outcome_z")], by = "subject")
  sub <- sub[is.finite(sub$outcome_z), ]
  sub$subject <- factor(sub$subject)
  sub$item <- factor(sub$item_id)
  sub$section <- factor(sub$section)
  fixed <- paste(
    "dv_cp ~ sur_z * outcome_z + sur_z * age_c + g1 + g2 + sur_z:g1 + sur_z:g2 +",
    controls
  )
  log_line(spec$key, "subjects", nlevels(sub$subject), "trials", nrow(sub))
  m <- tryCatch(
    lmer(as.formula(paste(fixed, "+", re_rich)), data = sub, REML = TRUE, control = ctrl),
    error = function(e) e
  )
  rung <- core_rung
  if (inherits(m, "error") || !converged(m)$ok) {
    reason <- if (inherits(m, "error")) conditionMessage(m) else converged(m)$reason
    log_line(spec$key, "rich failed", reason, "; floor")
    m <- tryCatch(
      lmer(as.formula(paste(fixed, "+", re_floor)), data = sub, REML = TRUE, control = ctrl),
      error = function(e) e
    )
    rung <- "item_intercept"
  }
  if (inherits(m, "error") || !converged(m)$ok) {
    reason <- if (inherits(m, "error")) conditionMessage(m) else converged(m)$reason
    log_line(spec$key, "FAILED", reason)
    writeLines(reason, file.path(beh_dir, paste0(spec$key, "_FAILED.txt")))
    return(NULL)
  }
  st <- converged(m)
  cf <- coef_frame(m)
  cf$outcome <- spec$key
  cf$label <- spec$label
  cf$family <- spec$family
  cf$rung <- rung
  cf$n_subjects <- nlevels(sub$subject)
  cf$n_trials <- nrow(sub)
  write.csv(cf, file.path(beh_dir, paste0(spec$key, "_coef.csv")), row.names = FALSE)
  hit <- cf[cf$term == "sur_z:outcome_z", ]
  log_line(spec$key, "ok", st$reason, "rung", rung, "beta", hit$Estimate, "p", hit[["Pr(>|t|)"]])
  hit
}

rows <- lapply(outcomes, fit_one)
rows <- rows[!vapply(rows, is.null, logical(1))]
if (!length(rows)) stop("no behaviour model converged")
tab <- do.call(rbind, rows)
tab$p <- tab[["Pr(>|t|)"]]
tab$p_holm <- NA_real_
for (fam in c("primary", "secondary")) {
  ix <- which(tab$family == fam)
  if (length(ix) == 2) tab$p_holm[ix] <- holm_two(tab$p[ix[1]], tab$p[ix[2]])
}
# Search that produced the exploratory RAN terms in the original results:
# 23 Sheet1 test names, and object naming is 1 of 4 RAN subtests.
tab$search_note <- ifelse(
  tab$family == "exploratory",
  "Original extended search: 23 Sheet1 test names; object naming is 1 of 4 RAN subtests. RAN direction is unresolved.",
  ""
)
write.csv(tab, file.path(beh_dir, "behaviour_key_terms.csv"), row.names = FALSE)
log_line("behaviour done")
