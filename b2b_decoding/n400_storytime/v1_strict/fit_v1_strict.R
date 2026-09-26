#!/usr/bin/env Rscript
# v1-strict Storytime N400 LMMs. Formulas are locked in
# docs/n400-v1-strict-spec.md. Refuses to run outside Slurm.

if (!nzchar(Sys.getenv("SLURM_JOB_ID"))) {
  stop("Refusing to fit outside Slurm. Submit run_lmm.sbatch.")
}

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
})

args <- commandArgs(trailingOnly = TRUE)
in_csv <- if (length(args) >= 1) args[[1]] else
  "/orcd/pool/005/haolun52/n400_storytime_v1/v1_strict/trials/lmm_input.csv"
out_dir <- if (length(args) >= 2) args[[2]] else
  "/orcd/pool/005/haolun52/n400_storytime_v1/v1_strict/models"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

ctrl <- lmerControl(
  optimizer = "bobyqa",
  optCtrl = list(maxfun = 1000000L),
  calc.derivs = FALSE,
  check.conv.singular = .makeCC(action = "ignore", tol = 1e-4)
)
RE <- "(1 | subject) + (1 | item)"
CONTROLS <- c(
  "conc_z", "duration_z", "pos_sent_z", "sent_story_z",
  "freq_m1_z", "freq_p1_z", "assoc_m1_z", "assoc_p1_z",
  "cloze_m1_z", "cloze_p1_z"
)
BACK_START <- c(CONTROLS, "freq_z", "assoc_z", "cloze_z")

log_line <- function(...) {
  msg <- paste0(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " ", paste(..., collapse = " "))
  cat(msg, "\n")
  write(msg, file = file.path(out_dir, "fit_log.txt"), append = TRUE)
}

write_table <- function(df, path) {
  old <- getOption("digits")
  options(digits = 15)
  on.exit(options(digits = old), add = TRUE)
  write.csv(df, path, row.names = FALSE)
}

stage_done <- function(name) file.exists(file.path(out_dir, paste0("stage_", name, ".done")))
stage_mark <- function(name) writeLines(format(Sys.time()), file.path(out_dir, paste0("stage_", name, ".done")))

conv_text <- function(m) {
  if (is.null(m) || !inherits(m, "merMod")) return(NA_character_)
  msgs <- m@optinfo$conv$lme4$messages
  code <- m@optinfo$conv$opt
  bits <- character()
  if (!is.null(msgs) && length(msgs) > 0) bits <- c(bits, paste(msgs, collapse = " | "))
  if (is.numeric(code) && length(code) == 1 && is.finite(code) && code != 0) {
    bits <- c(bits, paste("opt_code", code))
  }
  if (isSingular(m, tol = 1e-4)) bits <- c(bits, "singular")
  if (length(bits) == 0) "ok" else paste(bits, collapse = " | ")
}

formula_of <- function(dv, terms) {
  fixed <- if (length(terms) == 0) "1" else paste(terms, collapse = " + ")
  as.formula(paste(dv, "~", fixed, "+", RE), env = .GlobalEnv)
}

fit_lmer <- function(dv, terms, data, reml) {
  f <- formula_of(dv, terms)
  t0 <- proc.time()[["elapsed"]]
  m <- tryCatch(lmer(f, data = data, REML = reml, control = ctrl), error = function(e) e)
  elapsed <- proc.time()[["elapsed"]] - t0
  if (inherits(m, "error")) {
    return(list(model = NULL, elapsed = elapsed, ok = FALSE, reason = conditionMessage(m),
                formula = paste(deparse(f), collapse = " ")))
  }
  list(model = m, elapsed = elapsed, ok = TRUE, reason = conv_text(m),
       formula = paste(deparse(f), collapse = " "))
}

must_fit <- function(dv, terms, data, reml, label) {
  fit <- fit_lmer(dv, terms, data, reml)
  log_line(label, if (reml) "REML" else "ML", "elapsed", round(fit$elapsed, 1), fit$reason)
  if (!fit$ok) stop(label, " failed: ", fit$reason)
  fit
}

coef_frame <- function(m, label) {
  sm <- as.data.frame(coef(summary(m)))
  sm$term <- rownames(sm)
  rownames(sm) <- NULL
  sm$label <- label
  sm$conv <- conv_text(m)
  sm$singular <- isSingular(m, tol = 1e-4)
  if (all(c("df", "Std. Error", "Estimate") %in% names(sm))) {
    crit <- qt(0.975, sm$df)
    sm$ci95_low <- sm$Estimate - crit * sm[["Std. Error"]]
    sm$ci95_high <- sm$Estimate + crit * sm[["Std. Error"]]
  }
  sm
}

ml_row <- function(m, name, comparison, lrt = NULL) {
  ll <- logLik(m)
  data.frame(
    model = name,
    comparison = comparison,
    AIC = AIC(m),
    BIC = BIC(m),
    logLik = as.numeric(ll),
    npar = attr(ll, "df"),
    nobs = nobs(m),
    chisq = if (is.null(lrt)) NA_real_ else lrt$chisq,
    df = if (is.null(lrt)) NA_real_ else lrt$df,
    p = if (is.null(lrt)) NA_real_ else lrt$p,
    formula = paste(deparse(formula(m)), collapse = " "),
    conv = conv_text(m),
    stringsAsFactors = FALSE
  )
}

lrt_pair <- function(m0, m1) {
  tab <- as.data.frame(anova(m0, m1))
  last <- nrow(tab)
  list(
    chisq = tab$Chisq[last],
    df = tab$Df[last],
    p = tab[["Pr(>Chisq)"]][last]
  )
}

term_p <- function(m, terms) {
  sm <- coef(summary(m))
  rn <- rownames(sm)
  sapply(terms, function(term) {
    hits <- rn[rn == term]
    if (length(hits) == 0 && grepl(":", term, fixed = TRUE)) {
      bits <- strsplit(term, ":", fixed = TRUE)[[1]]
      alt <- paste(rev(bits), collapse = ":")
      hits <- rn[rn == alt]
    }
    if (length(hits) != 1) return(NA_real_)
    unname(sm[hits, "Pr(>|t|)"])
  })
}

lookup_coef <- function(m, term) {
  b <- fixef(m)
  if (term %in% names(b)) return(term)
  if (grepl(":", term, fixed = TRUE)) {
    alt <- paste(rev(strsplit(term, ":", fixed = TRUE)[[1]]), collapse = ":")
    if (alt %in% names(b)) return(alt)
  }
  NA_character_
}

screen_base <- function(data, dv, label) {
  base <- c("g01", CONTROLS)
  m0 <- must_fit(dv, base, data, FALSE, paste(label, "base ML"))
  rows <- list()
  kept <- character()
  for (term in CONTROLS) {
    alt_terms <- c(base, paste0("g01:", term))
    m1 <- must_fit(dv, alt_terms, data, FALSE, paste(label, "screen", term))
    lrt <- lrt_pair(m0$model, m1$model)
    keep <- is.finite(lrt$p) && lrt$p < 0.05
    if (keep) kept <- c(kept, paste0("g01:", term))
    rows[[term]] <- data.frame(
      roi = label, term = paste0("g01:", term), chisq = lrt$chisq, df = lrt$df,
      p = lrt$p, kept = keep, stringsAsFactors = FALSE
    )
  }
  final_terms <- c(base, kept)
  m_reml <- must_fit(dv, final_terms, data, TRUE, paste(label, "final base REML"))
  list(terms = final_terms, screen = do.call(rbind, rows), coef = coef_frame(m_reml$model, paste(label, "base")))
}

ladder <- function(data, dv, base_terms, label) {
  steps <- list(
    c(base_terms, "freq_z"),
    c(base_terms, "freq_z", "g01:freq_z"),
    c(base_terms, "freq_z", "g01:freq_z", "assoc_z"),
    c(base_terms, "freq_z", "g01:freq_z", "assoc_z", "g01:assoc_z"),
    c(base_terms, "freq_z", "g01:freq_z", "assoc_z", "g01:assoc_z", "cloze_z"),
    c(base_terms, "freq_z", "g01:freq_z", "assoc_z", "g01:assoc_z", "cloze_z", "g01:cloze_z")
  )
  names(steps) <- paste0("M", 1:6)
  models <- list()
  models$M0 <- must_fit(dv, base_terms, data, FALSE, paste(label, "M0"))$model
  rows <- list(ml_row(models$M0, "M0", "base"))
  coefs <- list()
  prev_name <- "M0"
  for (nm in names(steps)) {
    fit <- must_fit(dv, steps[[nm]], data, FALSE, paste(label, nm))
    models[[nm]] <- fit$model
    lrt <- lrt_pair(models[[prev_name]], fit$model)
    rows[[nm]] <- ml_row(fit$model, nm, paste("vs", prev_name), lrt)
    reml <- must_fit(dv, steps[[nm]], data, TRUE, paste(label, nm, "REML"))
    coefs[[nm]] <- coef_frame(reml$model, paste(label, nm))
    prev_name <- nm
  }
  m1_cloze_terms <- c(steps$M1, "cloze_z")
  m1c <- must_fit(dv, m1_cloze_terms, data, FALSE, paste(label, "M1+cloze"))
  lrt <- lrt_pair(models$M1, m1c$model)
  rows$M1c <- ml_row(m1c$model, "M1_plus_cloze", "vs M1", lrt)
  reml <- must_fit(dv, base_terms, data, TRUE, paste(label, "M0 REML"))
  coefs$M0 <- coef_frame(reml$model, paste(label, "M0"))
  list(
    table = do.call(rbind, rows),
    coef = do.call(rbind, coefs),
    bic = sapply(models, BIC),
    m6_terms = steps$M6
  )
}

backwards <- function(data, dv, label) {
  terms <- BACK_START
  steps <- list()
  repeat {
    m_reml <- must_fit(dv, terms, data, TRUE, paste(label, "back reml", length(terms)))
    if (length(terms) == 0) break
    pmap <- term_p(m_reml$model, terms)
    finite <- is.finite(pmap)
    eligible <- !finite | pmap >= 0.05
    if (!any(eligible)) {
      log_line(label, "stop: all terms p < 0.05")
      break
    }
    cand <- terms[eligible]
    pp <- pmap[eligible]
    pp[!is.finite(pp)] <- Inf
    idx <- match(cand, BACK_START)
    drop <- cand[order(pp, idx, decreasing = TRUE)[1]]
    reduced <- setdiff(terms, drop)
    m_full <- must_fit(dv, terms, data, FALSE, paste(label, "back full", drop))
    m_red <- must_fit(dv, reduced, data, FALSE, paste(label, "back reduced", drop))
    lrt <- lrt_pair(m_red$model, m_full$model)
    aic_full <- AIC(m_full$model)
    aic_red <- AIC(m_red$model)
    stop_aic <- aic_red > aic_full
    stop_lrt <- is.finite(lrt$p) && lrt$p < 0.1
    steps[[length(steps) + 1]] <- data.frame(
      label = label, drop = drop, p_satt = unname(pmap[drop]),
      aic_full = aic_full, aic_reduced = aic_red,
      chisq = lrt$chisq, df = lrt$df, p_lrt = lrt$p,
      accepted = !(stop_aic || stop_lrt),
      stringsAsFactors = FALSE
    )
    log_line(label, "candidate", drop, "p", pmap[drop], "aic", aic_full, "->", aic_red,
             "lrt_p", lrt$p, "accepted", !(stop_aic || stop_lrt))
    if (stop_aic || stop_lrt) break
    terms <- reduced
  }
  m_final <- must_fit(dv, terms, data, TRUE, paste(label, "back final"))
  list(terms = terms, steps = if (length(steps)) do.call(rbind, steps) else data.frame(),
       coef = coef_frame(m_final$model, label))
}

force_cloze <- function(terms) {
  if ("cloze_z" %in% terms) terms else c(terms, "cloze_z")
}

wald_combo <- function(m, terms_est) {
  b <- fixef(m)
  V <- as.matrix(vcov(m))
  present <- vapply(terms_est, lookup_coef, character(1), m = m)
  if (anyNA(present)) stop("missing coefficient: ", paste(terms_est, collapse = " "))
  est <- sum(b[present])
  se <- sqrt(sum(V[present, present]))
  z <- est / se
  data.frame(estimate = est, se = se, z = z, p = 2 * pnorm(-abs(z)))
}

moderator_block <- function(data, back_terms, moderators, family, label) {
  rows <- list()
  for (mod in names(moderators)) {
    col <- moderators[[mod]]
    use <- data[is.finite(data[[col]]), , drop = FALSE]
    n_subj <- length(unique(use$subject))
    if (n_subj < 8 || nrow(use) < 50) {
      rows[[mod]] <- data.frame(
        family = family, label = label, moderator = mod, term = paste0(mod, ":cloze"),
        chisq = NA_real_, df = NA_real_, p = NA_real_,
        estimate = NA_real_, se = NA_real_, df_satt = NA_real_, t = NA_real_, p_satt = NA_real_,
        n_subjects = n_subj, n_trials = nrow(use), cloze_forced = NA,
        aic_main = NA_real_, note = "too_few",
        stringsAsFactors = FALSE
      )
      next
    }
    base <- back_terms
    main_terms <- c(base, col)
    m_main <- must_fit("dv_cp", main_terms, use, FALSE, paste(label, mod, "main"))
    both <- force_cloze(base)
    forced <- !("cloze_z" %in% base)
    m0 <- must_fit("dv_cp", c(both, col), use, FALSE, paste(label, mod, "with cloze"))
    m1 <- must_fit("dv_cp", c(both, col, paste0(col, ":cloze_z")), use, FALSE, paste(label, mod, "x cloze"))
    lrt <- lrt_pair(m0$model, m1$model)
    reml <- must_fit("dv_cp", c(both, col, paste0(col, ":cloze_z")), use, TRUE, paste(label, mod, "x reml"))
    cf <- coef_frame(reml$model, paste(label, mod))
    inter_name <- lookup_coef(reml$model, paste0(col, ":cloze_z"))
    inter <- cf[cf$term == inter_name, ]
    rows[[paste0(mod, "_lrt")]] <- data.frame(
      family = family, label = label, moderator = mod, term = paste0(mod, ":cloze"),
      chisq = lrt$chisq, df = lrt$df, p = lrt$p,
      estimate = if (nrow(inter)) inter$Estimate else NA_real_,
      se = if (nrow(inter)) inter[["Std. Error"]] else NA_real_,
      df_satt = if (nrow(inter)) inter$df else NA_real_,
      t = if (nrow(inter)) inter[["t value"]] else NA_real_,
      p_satt = if (nrow(inter)) inter[["Pr(>|t|)"]] else NA_real_,
      n_subjects = n_subj, n_trials = nrow(use),
      cloze_forced = forced,
      aic_main = AIC(m_main$model),
      note = "x_cloze_lrt",
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, rows)
}

attach_z <- function(d, cols, newname) {
  u <- unique(d[, c("subject", cols), drop = FALSE])
  if (any(duplicated(u$subject))) stop("subject-level scores are not unique")
  zvec <- function(x) {
    ok <- is.finite(x)
    out <- rep(NA_real_, length(x))
    if (sum(ok) < 3) return(out)
    m <- mean(x[ok])
    s <- sqrt(mean((x[ok] - m)^2))
    if (!is.finite(s) || s <= 0) return(out)
    out[ok] <- (x[ok] - m) / s
    out
  }
  if (length(cols) == 1) {
    u[[newname]] <- zvec(u[[cols]])
  } else {
    ok <- rep(TRUE, nrow(u))
    for (col in cols) ok <- ok & is.finite(u[[col]])
    parts <- lapply(cols, function(col) {
      z <- rep(NA_real_, nrow(u))
      if (sum(ok) >= 3) z[ok] <- zvec(u[[col]][ok])
      z
    })
    comp <- rep(NA_real_, nrow(u))
    comp[ok] <- Reduce(`+`, lapply(parts, function(z) z[ok])) / length(cols)
    u[[newname]] <- zvec(comp)
  }
  merge(d, u[, c("subject", newname)], by = "subject", all.x = TRUE)
}

vif_table <- function(data, terms) {
  f <- as.formula(paste("~", paste(terms, collapse = " + ")))
  mm <- model.matrix(f, data)
  mm <- mm[, colnames(mm) != "(Intercept)", drop = FALSE]
  qr_mm <- qr(mm)
  if (qr_mm$rank < ncol(mm)) {
    mm <- mm[, qr_mm$pivot[seq_len(qr_mm$rank)], drop = FALSE]
  }
  rows <- lapply(seq_len(ncol(mm)), function(j) {
    y <- mm[, j]
    x <- cbind(1, mm[, -j, drop = FALSE])
    fit <- lm.fit(x, y)
    ss_tot <- sum((y - mean(y))^2)
    r2 <- if (ss_tot <= 0) NA_real_ else 1 - sum(fit$residuals^2) / ss_tot
    vif <- if (!is.finite(r2) || r2 >= 1) Inf else 1 / (1 - r2)
    data.frame(term = colnames(mm)[j], r2 = r2, vif = vif, stringsAsFactors = FALSE)
  })
  do.call(rbind, rows)
}

log_line("host", Sys.info()[["nodename"]], "job", Sys.getenv("SLURM_JOB_ID"),
         "R", paste(R.version$major, R.version$minor, sep = "."),
         "lme4", as.character(packageVersion("lme4")),
         "lmerTest", as.character(packageVersion("lmerTest")))

raw <- read.csv(in_csv, stringsAsFactors = FALSE)
raw$subject <- factor(raw$subject)
raw$item <- factor(raw$item)
log_line("rows", nrow(raw), "subjects", length(unique(raw$subject)), "items", length(unique(raw$item)))

# ---- location ----
if (!stage_done("location")) {
  long <- do.call(rbind, lapply(c(Fz = "dv_fz", Cz = "dv_cz", Pz = "dv_pz", Oz = "dv_oz"), function(col) {
    data.frame(
      subject = raw$subject, item = raw$item, dv = raw[[col]],
      freq_z = raw$freq_z, assoc_z = raw$assoc_z, cloze_z = raw$cloze_z,
      channel = names(c(Fz = "dv_fz", Cz = "dv_cz", Pz = "dv_pz", Oz = "dv_oz"))[
        match(col, c(Fz = "dv_fz", Cz = "dv_cz", Pz = "dv_pz", Oz = "dv_oz"))
      ],
      stringsAsFactors = FALSE
    )
  }))
  # The lapply names are the channel labels because the vector is named.
  long$channel <- factor(long$channel, levels = c("Pz", "Cz", "Oz", "Fz"))
  if (anyNA(long$channel)) stop("location channel labels failed")
  long <- long[is.finite(long$dv), ]
  full_terms <- c(
    "freq_z", "assoc_z", "cloze_z", "channel",
    "freq_z:channel", "assoc_z:channel", "cloze_z:channel"
  )
  m_full <- must_fit("dv", full_terms, long, FALSE, "location full")
  m_full_r <- must_fit("dv", full_terms, long, TRUE, "location full REML")
  drop_one <- function(which_int) {
    terms <- full_terms[full_terms != which_int]
    m0 <- must_fit("dv", terms, long, FALSE, paste("location drop", which_int))
    lrt <- lrt_pair(m0$model, m_full$model)
    data.frame(subset = "Fz_Cz_Pz_Oz", term = which_int, chisq = lrt$chisq, df = lrt$df, p = lrt$p)
  }
  lrt_rows <- rbind(
    drop_one("freq_z:channel"),
    drop_one("assoc_z:channel"),
    drop_one("cloze_z:channel")
  )
  sub <- long[long$channel != "Fz", ]
  sub$channel <- factor(sub$channel, levels = c("Pz", "Cz", "Oz"))
  m_sub <- must_fit("dv", full_terms, sub, FALSE, "location CP-only")
  drop_sub <- function(which_int) {
    terms <- full_terms[full_terms != which_int]
    m0 <- must_fit("dv", terms, sub, FALSE, paste("location CP drop", which_int))
    lrt <- lrt_pair(m0$model, m_sub$model)
    data.frame(subset = "Cz_Pz_Oz", term = which_int, chisq = lrt$chisq, df = lrt$df, p = lrt$p)
  }
  lrt_rows <- rbind(
    lrt_rows,
    drop_sub("freq_z:channel"),
    drop_sub("assoc_z:channel"),
    drop_sub("cloze_z:channel")
  )
  af <- as.data.frame(anova(m_full_r$model))
  af$term <- rownames(af)
  af$subset <- "Fz_Cz_Pz_Oz_REML_anova"
  write_table(lrt_rows, file.path(out_dir, "location_lrt.csv"))
  write_table(af, file.path(out_dir, "location_anova.csv"))
  write_table(coef_frame(m_full_r$model, "location"), file.path(out_dir, "location_coef.csv"))
  stage_mark("location")
}

# ---- CP base and ladder ----
if (!stage_done("ladder_cp")) {
  base <- screen_base(raw, "dv_cp", "CP")
  write_table(base$screen, file.path(out_dir, "base_screen_cp.csv"))
  write_table(base$coef, file.path(out_dir, "table1_base_cp.csv"))
  saveRDS(base$terms, file.path(out_dir, "base_terms_cp.rds"))
  lad <- ladder(raw, "dv_cp", base$terms, "CP")
  write_table(lad$table, file.path(out_dir, "table2_ladder_cp.csv"))
  write_table(lad$coef, file.path(out_dir, "ladder_cp_coef.csv"))
  saveRDS(lad$m6_terms, file.path(out_dir, "m6_terms_cp.rds"))
  saveRDS(lad$bic, file.path(out_dir, "ladder_cp_bic.rds"))
  stage_mark("ladder_cp")
}

# ---- backwards ----
if (!stage_done("backwards")) {
  pieces_s <- list()
  pieces_c <- list()
  subsets <- list(
    TD = raw[raw$g01 == 0, ],
    dyslexia = raw[raw$g01 == 1, ],
    nCAP = raw[!is.na(raw$cap01) & raw$cap01 == 0, ],
    aCAP = raw[!is.na(raw$cap01) & raw$cap01 == 1, ]
  )
  for (nm in names(subsets)) {
    bk <- backwards(subsets[[nm]], "dv_cp", nm)
    saveRDS(bk$terms, file.path(out_dir, paste0("back_terms_", nm, ".rds")))
    if (nrow(bk$steps)) pieces_s[[nm]] <- bk$steps
    pieces_c[[nm]] <- bk$coef
  }
  write_table(do.call(rbind, pieces_s), file.path(out_dir, "table3_backwards_steps.csv"))
  write_table(do.call(rbind, pieces_c), file.path(out_dir, "table4_backwards_coef.csv"))
  stage_mark("backwards")
}

# ---- posthoc ----
if (!stage_done("posthoc")) {
  m6 <- readRDS(file.path(out_dir, "m6_terms_cp.rds"))
  m6a <- c(m6, "cloze_z:freq_z")
  m6b <- c(m6a, "cloze_z:freq_z:g01")
  a0 <- must_fit("dv_cp", m6, raw, FALSE, "posthoc M6")
  a1 <- must_fit("dv_cp", m6a, raw, FALSE, "posthoc cloze x freq")
  a2 <- must_fit("dv_cp", m6b, raw, FALSE, "posthoc cloze x freq x group")
  r1 <- must_fit("dv_cp", m6a, raw, TRUE, "posthoc cloze x freq REML")
  r2 <- must_fit("dv_cp", m6b, raw, TRUE, "posthoc 3way REML")
  tab <- rbind(
    ml_row(a1$model, "M6_cloze_x_freq", "vs M6", lrt_pair(a0$model, a1$model)),
    ml_row(a2$model, "M6_cloze_x_freq_x_group", "vs M6_cloze_x_freq", lrt_pair(a1$model, a2$model))
  )
  write_table(tab, file.path(out_dir, "posthoc_lrt.csv"))
  write_table(rbind(coef_frame(r1$model, "cloze_x_freq"), coef_frame(r2$model, "cloze_x_freq_x_group")),
              file.path(out_dir, "posthoc_coef.csv"))
  stage_mark("posthoc")
}

# ---- CAP ----
if (!stage_done("cap")) {
  terms <- readRDS(file.path(out_dir, "back_terms_dyslexia.rds"))
  dx <- raw[raw$g01 == 1 & !is.na(raw$cap01), ]
  both <- force_cloze(terms)
  forced <- !("cloze_z" %in% terms)
  m0 <- must_fit("dv_cp", c(both, "cap01"), dx, FALSE, "CAP main ML")
  m1 <- must_fit("dv_cp", c(both, "cap01", "cap01:cloze_z"), dx, FALSE, "CAP x cloze ML")
  m0_only <- must_fit("dv_cp", c(terms, "cap01"), dx, FALSE, "CAP main on backwards ML")
  r1 <- must_fit("dv_cp", c(both, "cap01", "cap01:cloze_z"), dx, TRUE, "CAP x cloze REML")
  base_ml <- must_fit("dv_cp", terms, dx, FALSE, "CAP backwards baseline")
  lrt_cap <- lrt_pair(base_ml$model, m0_only$model)
  lrt_x <- lrt_pair(m0$model, m1$model)
  slopes <- rbind(
    cbind(group = "nCAP", wald_combo(r1$model, "cloze_z"), cloze_forced = forced),
    cbind(group = "aCAP", wald_combo(r1$model, c("cloze_z", "cap01:cloze_z")), cloze_forced = forced)
  )
  tab <- rbind(
    data.frame(test = "cap_main", chisq = lrt_cap$chisq, df = lrt_cap$df, p = lrt_cap$p,
               cloze_forced = FALSE, n_subjects = length(unique(dx$subject)), n_trials = nrow(dx)),
    data.frame(test = "cap_x_cloze", chisq = lrt_x$chisq, df = lrt_x$df, p = lrt_x$p,
               cloze_forced = forced, n_subjects = length(unique(dx$subject)), n_trials = nrow(dx))
  )
  write_table(tab, file.path(out_dir, "cap_lrt.csv"))
  write_table(slopes, file.path(out_dir, "cap_slopes.csv"))
  write_table(coef_frame(r1$model, "cap"), file.path(out_dir, "cap_coef.csv"))
  stage_mark("cap")
}

# ---- behaviour ----
if (!stage_done("behaviour")) {
  terms <- readRDS(file.path(out_dir, "back_terms_dyslexia.rds"))
  build_group <- function(d) {
    d <- attach_z(d, "age", "age_z")
    d <- attach_z(d, "literacy", "literacy_z")
    d <- attach_z(d, "fluency", "fluency_z")
    d <- attach_z(d, "gap", "gap_z")
    d <- attach_z(d, "dichotic", "dichotic_z")
    d <- attach_z(d, c("pa_tone", "pa_phon", "pa_syl"), "pa_z")
    d <- attach_z(d, c("ran_digit", "ran_mixed", "ran_object", "ran_color"), "ran_z")
    d
  }
  primary <- c(age = "age_z", literacy = "literacy_z", pa = "pa_z", ran = "ran_z", fluency = "fluency_z")
  secondary <- c(gap = "gap_z", dichotic = "dichotic_z")
  dx <- build_group(raw[raw$g01 == 1, ])
  td <- build_group(raw[raw$g01 == 0, ])
  dx_tab <- rbind(
    moderator_block(dx, terms, primary, "primary", "dyslexia"),
    moderator_block(dx, terms, secondary, "secondary", "dyslexia")
  )
  td_tab <- rbind(
    moderator_block(td, terms, primary, "primary_descriptive", "TD"),
    moderator_block(td, terms, secondary, "secondary_descriptive", "TD")
  )
  dx_tab$p_holm <- NA_real_
  for (fam in c("primary", "secondary")) {
    ix <- dx_tab$family == fam & dx_tab$note == "x_cloze_lrt" & is.finite(dx_tab$p)
    if (any(ix)) dx_tab$p_holm[ix] <- p.adjust(dx_tab$p[ix], method = "holm")
  }
  td_tab$p_holm <- NA_real_
  all_tab <- rbind(dx_tab, td_tab)
  write_table(all_tab, file.path(out_dir, "behaviour_lrt.csv"))
  stage_mark("behaviour")
}

# ---- Fz ladder ----
if (!stage_done("ladder_fz")) {
  base <- screen_base(raw, "dv_fz", "Fz")
  write_table(base$screen, file.path(out_dir, "base_screen_fz.csv"))
  write_table(base$coef, file.path(out_dir, "base_fz_coef.csv"))
  lad <- ladder(raw, "dv_fz", base$terms, "Fz")
  write_table(lad$table, file.path(out_dir, "ladder_fz.csv"))
  write_table(lad$coef, file.path(out_dir, "ladder_fz_coef.csv"))
  saveRDS(lad$bic, file.path(out_dir, "ladder_fz_bic.rds"))
  stage_mark("ladder_fz")
}

# ---- BIC Bayes factors and VIF ----
if (!stage_done("bf_vif")) {
  bic <- readRDS(file.path(out_dir, "ladder_cp_bic.rds"))
  bf_one <- function(null_name, alt_name, effect) {
    b0 <- unname(bic[[null_name]])
    b1 <- unname(bic[[alt_name]])
    data.frame(
      effect = effect, null_model = null_name, alt_model = alt_name,
      BIC_null = b0, BIC_alt = b1,
      BF01 = exp((b1 - b0) / 2),
      BF10 = exp((b0 - b1) / 2),
      stringsAsFactors = FALSE
    )
  }
  bf <- rbind(
    bf_one("M2", "M3", "LSA"),
    bf_one("M1", "M2", "Freq x Group"),
    bf_one("M5", "M6", "Cloze x Group")
  )
  write_table(bf, file.path(out_dir, "bayes_factors.csv"))
  m6 <- readRDS(file.path(out_dir, "m6_terms_cp.rds"))
  vif <- vif_table(raw, m6)
  vif$above_2_5 <- vif$vif > 2.5
  write_table(vif, file.path(out_dir, "vif_m6.csv"))
  stage_mark("bf_vif")
}

log_line("all stages complete")
writeLines(format(Sys.time()), file.path(out_dir, "DONE"))
