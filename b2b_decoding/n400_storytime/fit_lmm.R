#!/usr/bin/env Rscript
# Nested LMMs for the Storytime N400. Primary DV is CP 350-550 ms.
# Forward LRT uses ML. Coefficients use REML + lmerTest Satterthwaite.

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
})

args <- commandArgs(trailingOnly = TRUE)
in_csv <- if (length(args) >= 1) args[[1]] else
  "/orcd/pool/005/haolun52/n400_storytime_v1/stimulus/lmm_input.csv"
out_dir <- if (length(args) >= 2) args[[2]] else
  "/orcd/pool/005/haolun52/n400_storytime_v1/stimulus/lmm"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

ctrl <- lmerControl(
  optimizer = "bobyqa",
  optCtrl = list(maxfun = 100000),
  calc.derivs = FALSE
)

d <- read.csv(in_csv, stringsAsFactors = FALSE)
d$subject <- factor(d$subject)
d$item <- factor(d$item_id)
d$section <- factor(d$section)
d$group <- factor(d$group)

base_terms <- c(
  "duration_z", "position_z", "section",
  "sur_m1_z", "sur_p1_z", "freq_m1_z", "freq_p1_z",
  "assoc_m1_z", "assoc_p1_z"
)
full_terms <- c(base_terms, "freq_z", "assoc_z", "sur_z")

re_ladder <- c(
  maximal = "(1 + sur_z + freq_z + assoc_z | subject) + (1 + sur_z | item)",
  no_item_slope = "(1 + sur_z + freq_z + assoc_z | subject) + (1 | item)",
  intercepts = "(1 | subject) + (1 | item)"
)

fit_one <- function(dv, terms, data, re, reml) {
  fixed <- paste(terms, collapse = " + ")
  fml <- as.formula(paste(dv, "~", fixed, "+", re))
  setTimeLimit(cpu = 480, elapsed = 600, transient = TRUE)
  on.exit(setTimeLimit(cpu = Inf, elapsed = Inf, transient = TRUE), add = TRUE)
  tryCatch(
    lmer(fml, data = data, REML = reml, control = ctrl),
    error = function(e) e
  )
}

ok_fit <- function(m) {
  inherits(m, "merMod") && !isSingular(m, tol = 1e-4)
}

choose_re <- function(dv, data) {
  notes <- list()
  for (nm in names(re_ladder)) {
    m <- fit_one(dv, full_terms, data, re_ladder[[nm]], reml = FALSE)
    if (inherits(m, "error")) {
      notes[[nm]] <- conditionMessage(m)
      next
    }
    singular <- isSingular(m, tol = 1e-4)
    notes[[nm]] <- if (singular) "singular" else "ok"
    if (!singular) {
      return(list(name = nm, re = re_ladder[[nm]], notes = notes))
    }
  }
  list(name = "intercepts_forced", re = re_ladder[["intercepts"]], notes = notes)
}

coef_frame <- function(m) {
  sm <- as.data.frame(coef(summary(m)))
  sm$term <- rownames(sm)
  ci <- suppressMessages(confint(m, parm = "beta_", method = "Wald"))
  ci <- as.data.frame(ci)
  ci$term <- rownames(ci)
  names(ci)[1:2] <- c("ci95_low", "ci95_high")
  out <- merge(sm, ci[, c("term", "ci95_low", "ci95_high")], by = "term", all.x = TRUE, sort = FALSE)
  out
}

write_fit <- function(m, path) {
  sink(path)
  print(summary(m))
  sink()
}

run_nest <- function(dv, data, re) {
  steps <- list(
    m0 = base_terms,
    m1 = c(base_terms, "freq_z"),
    m2 = c(base_terms, "freq_z", "assoc_z"),
    m3 = full_terms
  )
  models <- list()
  for (nm in names(steps)) {
    m <- fit_one(dv, steps[[nm]], data, re, reml = FALSE)
    if (!inherits(m, "merMod")) {
      stop(sprintf("nest %s %s failed: %s", dv, nm, conditionMessage(m)))
    }
    models[[nm]] <- m
  }
  lrt <- as.data.frame(anova(models$m0, models$m1, models$m2, models$m3))
  lrt$model <- rownames(lrt)
  m_reml <- fit_one(dv, full_terms, data, re, reml = TRUE)
  if (!inherits(m_reml, "merMod")) {
    stop(sprintf("REML full %s failed: %s", dv, conditionMessage(m_reml)))
  }
  list(models = models, lrt = lrt, reml = m_reml)
}

backward <- function(dv, data, re) {
  terms <- full_terms
  dropped <- character()
  repeat {
    m <- fit_one(dv, terms, data, re, reml = TRUE)
    if (!inherits(m, "merMod")) {
      return(list(terms = terms, dropped = dropped, model = NULL,
                  error = conditionMessage(m)))
    }
    sm <- coef(summary(m))
    optional <- setdiff(terms, "section")
    optional <- intersect(optional, rownames(sm))
    if (length(optional) == 0) {
      return(list(terms = terms, dropped = dropped, model = m, error = NA))
    }
    pvals <- sm[optional, "Pr(>|t|)"]
    if (!any(is.finite(pvals)) || max(pvals, na.rm = TRUE) <= 0.05) {
      return(list(terms = terms, dropped = dropped, model = m, error = NA))
    }
    drop <- names(which.max(pvals))
    dropped <- c(dropped, drop)
    terms <- setdiff(terms, drop)
    if (length(terms) == 0) {
      return(list(terms = terms, dropped = dropped, model = m, error = NA))
    }
  }
}

wald_row <- function(est, se, df, term) {
  tval <- est / se
  p <- 2 * pt(abs(tval), df, lower.tail = FALSE)
  crit <- qt(0.975, df)
  data.frame(
    term = term, estimate = est, se = se, df = df, t = tval, p = p,
    ci95_low = est - crit * se, ci95_high = est + crit * se
  )
}

simple_slopes <- function(m, main, inter, label) {
  b <- fixef(m)
  V <- as.matrix(vcov(m))
  sm <- coef(summary(m))
  df <- sm[inter, "df"]
  est0 <- unname(b[main])
  se0 <- sqrt(unname(V[main, main]))
  est1 <- unname(b[main] + b[inter])
  se1 <- sqrt(unname(V[main, main] + V[inter, inter] + 2 * V[main, inter]))
  rbind(
    wald_row(est0, se0, df, paste0(label, "_ref0")),
    wald_row(est1, se1, df, paste0(label, "_ref1"))
  )
}

cat("N rows", nrow(d), "subjects", nlevels(d$subject), "items", nlevels(d$item), "\n")

re_cp <- choose_re("dv_cp", d)
cat("CP random effects:", re_cp$name, "\n")
print(re_cp$notes)
nest_cp <- run_nest("dv_cp", d, re_cp$re)
write.csv(nest_cp$lrt, file.path(out_dir, "lrt_cp.csv"), row.names = FALSE)
write.csv(coef_frame(nest_cp$reml), file.path(out_dir, "coef_full_cp.csv"), row.names = FALSE)
write_fit(nest_cp$reml, file.path(out_dir, "summary_full_cp.txt"))
back_cp <- backward("dv_cp", d, re_cp$re)
writeLines(
  c(paste("re", re_cp$name), paste("terms", paste(back_cp$terms, collapse = " ")),
    paste("dropped", paste(back_cp$dropped, collapse = " "))),
  file.path(out_dir, "backward_cp.txt")
)
if (inherits(back_cp$model, "merMod")) {
  write.csv(coef_frame(back_cp$model), file.path(out_dir, "coef_best_cp.csv"), row.names = FALSE)
}

re_fz <- choose_re("dv_fz", d)
cat("Fz random effects:", re_fz$name, "\n")
nest_fz <- run_nest("dv_fz", d, re_fz$re)
write.csv(nest_fz$lrt, file.path(out_dir, "lrt_fz.csv"), row.names = FALSE)
write.csv(coef_frame(nest_fz$reml), file.path(out_dir, "coef_full_fz.csv"), row.names = FALSE)
write_fit(nest_fz$reml, file.path(out_dir, "summary_full_fz.txt"))
back_fz <- backward("dv_fz", d, re_fz$re)
writeLines(
  c(paste("re", re_fz$name), paste("terms", paste(back_fz$terms, collapse = " ")),
    paste("dropped", paste(back_fz$dropped, collapse = " "))),
  file.path(out_dir, "backward_fz.txt")
)
if (inherits(back_fz$model, "merMod")) {
  write.csv(coef_frame(back_fz$model), file.path(out_dir, "coef_best_fz.csv"), row.names = FALSE)
}

# Descriptive full model inside each group (CP and Fz), intercepts if the
# pooled RE does not converge in the subset.
group_levels <- list(
  TD = d$group == "TD",
  dyslexia = d$group != "TD",
  dyslexia_normal_CAP = d$group == "dyslexia_normal_CAP",
  dyslexia_atypical_CAP = d$group == "dyslexia_atypical_CAP"
)
g_rows <- list()
for (gname in names(group_levels)) {
  sub <- droplevels(d[group_levels[[gname]], ])
  for (dv in c("dv_cp", "dv_fz")) {
    re_use <- if (dv == "dv_cp") re_cp$re else re_fz$re
    re_name <- if (dv == "dv_cp") re_cp$name else re_fz$name
    m <- fit_one(dv, full_terms, sub, re_use, reml = TRUE)
    if (!ok_fit(m)) {
      m <- fit_one(dv, full_terms, sub, re_ladder[["intercepts"]], reml = TRUE)
      re_name <- "intercepts"
    }
    if (!inherits(m, "merMod")) {
      cat("group fit failed", gname, dv, conditionMessage(m), "\n")
      next
    }
    cf <- coef_frame(m)
    cf$group <- gname
    cf$dv <- dv
    cf$re <- re_name
    cf$n_trials <- nrow(sub)
    cf$n_subjects <- nlevels(sub$subject)
    g_rows[[paste(gname, dv)]] <- cf
  }
}
gdf <- do.call(rbind, g_rows)
write.csv(gdf, file.path(out_dir, "coef_by_group.csv"), row.names = FALSE)

# Moderation on the best CP fixed structure, forcing frequency and surprisal.
mod_terms <- unique(c(back_cp$terms, "freq_z", "sur_z"))
mod_terms <- mod_terms[mod_terms != "section"]
# section stays first-class
mod_terms <- unique(c("section", mod_terms))

center_age <- function(data) {
  data <- droplevels(data)
  ages <- tapply(data$age, data$subject, function(x) x[[1]])
  ages <- ages[is.finite(ages)]
  data$age_c <- data$age - mean(ages)
  data
}

fit_mod <- function(data, group_col, tag) {
  data <- center_age(data)
  data$g <- data[[group_col]]
  terms <- c(mod_terms, "freq_z", "assoc_z", "sur_z", "age_c", "g")
  # assoc_z only if it survived or was in the best model; always present in data.
  if (!("assoc_z" %in% back_cp$terms)) {
    terms <- setdiff(terms, "assoc_z")
  }
  terms <- unique(terms)
  fixed <- paste(terms, collapse = " + ")
  fml <- as.formula(paste(
    "dv_cp ~", fixed, "+ sur_z:g + freq_z:g +", re_cp$re
  ))
  m <- tryCatch(lmer(fml, data = data, REML = TRUE, control = ctrl), error = function(e) e)
  if (!ok_fit(m)) {
    fml <- as.formula(paste(
      "dv_cp ~", fixed, "+ sur_z:g + freq_z:g +", re_ladder[["intercepts"]]
    ))
    m <- tryCatch(lmer(fml, data = data, REML = TRUE, control = ctrl), error = function(e) e)
  }
  if (!inherits(m, "merMod")) {
    stop(sprintf("moderation %s failed: %s", tag, conditionMessage(m)))
  }
  cf <- coef_frame(m)
  cf$model <- tag
  write.csv(cf, file.path(out_dir, paste0("coef_mod_", tag, ".csv")), row.names = FALSE)
  write_fit(m, file.path(out_dir, paste0("summary_mod_", tag, ".txt")))
  list(model = m, n_trials = nrow(data), n_subjects = nlevels(droplevels(data$subject)),
       n_items = nlevels(droplevels(data$item)))
}

d_age <- d[is.finite(d$age), ]
d_age$group_dys <- as.integer(d_age$group != "TD")
dys <- d_age[d_age$group != "TD", ]
dys$group_acap <- as.integer(dys$group == "dyslexia_atypical_CAP")

mod_dys <- fit_mod(d_age, "group_dys", "td_vs_dyslexia")
mod_cap <- fit_mod(dys, "group_acap", "ncap_vs_acap")

p_of <- function(m, term) {
  sm <- coef(summary(m))
  if (!(term %in% rownames(sm))) {
    stop(paste("missing term", term))
  }
  sm[term, "Pr(>|t|)"]
}

tests <- data.frame(
  test = c(
    "surprisal_x_td_vs_dyslexia",
    "surprisal_x_ncap_vs_acap",
    "frequency_x_td_vs_dyslexia",
    "frequency_x_ncap_vs_acap"
  ),
  family_role = c("primary", "primary", "secondary", "secondary"),
  p_raw = c(
    p_of(mod_dys$model, "sur_z:g"),
    p_of(mod_cap$model, "sur_z:g"),
    p_of(mod_dys$model, "freq_z:g"),
    p_of(mod_cap$model, "freq_z:g")
  ),
  stringsAsFactors = FALSE
)
tests$p_holm <- p.adjust(tests$p_raw, method = "holm")
tests$p_holm_primary_surprisal_only <- NA_real_
tests$p_holm_primary_surprisal_only[1:2] <- p.adjust(tests$p_raw[1:2], method = "holm")
write.csv(tests, file.path(out_dir, "moderation_holm.csv"), row.names = FALSE)

ss <- rbind(
  cbind(contrast = "td_vs_dyslexia", simple_slopes(mod_dys$model, "sur_z", "sur_z:g", "surprisal")),
  cbind(contrast = "td_vs_dyslexia", simple_slopes(mod_dys$model, "freq_z", "freq_z:g", "frequency")),
  cbind(contrast = "ncap_vs_acap", simple_slopes(mod_cap$model, "sur_z", "sur_z:g", "surprisal")),
  cbind(contrast = "ncap_vs_acap", simple_slopes(mod_cap$model, "freq_z", "freq_z:g", "frequency"))
)
write.csv(ss, file.path(out_dir, "simple_slopes.csv"), row.names = FALSE)

meta <- data.frame(
  model = c("td_vs_dyslexia", "ncap_vs_acap"),
  n_trials = c(mod_dys$n_trials, mod_cap$n_trials),
  n_subjects = c(mod_dys$n_subjects, mod_cap$n_subjects),
  n_items = c(mod_dys$n_items, mod_cap$n_items)
)
write.csv(meta, file.path(out_dir, "moderation_n.csv"), row.names = FALSE)
writeLines(c(re_cp$name, re_fz$name), file.path(out_dir, "re_choice.txt"))
cat("LMM done\n")
print(tests)
