#' @title generate cross-validation folds
#' @description
#' Helper function to generate cross-validation folds that can be used by \code{cv.SPQR}.
#'
#' @param Y The response vector.
#' @param nfold The number of cross-validation folds.
#' @param stratified If \code{TRUE}, stratified folds based on quantiles of `Y` are generated.
#'
#' @return A list of size \code{nfold} containing indices of the observations for each fold.
#'
#' @export
createFolds.SPQR <- function(Y, nfold, stratified=FALSE) {

  nrows <- length(Y)
  rnd_idx <- sample.int(nrows)
  if (stratified) {
    Y <- Y[rnd_idx]
    cuts <- floor(length(Y) / nfold)
    if (cuts < 2) cuts <- 2
    if (cuts > 5) cuts <- 5
    Y <- cut(Y,
             unique(stats::quantile(Y, probs = seq(0, 1, length = cuts))),
             include.lowest = TRUE)

    if (nfold < length(Y)) {
      Y <- factor(as.character(Y))
      numInClass <- table(Y)
      foldVector <- vector(mode = "integer", length(Y))
      for (i in seq_along(numInClass)) {
        seqVector <- rep(seq_len(nfold), numInClass[i] %/% nfold)
        if (numInClass[i] %% nfold > 0) seqVector <- c(seqVector, sample.int(nfold, numInClass[i] %% nfold))
        foldVector[Y == dimnames(numInClass)$Y[i]] <- seqVector[sample.int(length(seqVector))]
      }
    } else {
      foldVector <- seq(along = Y)
    }

    folds <- split(seq(along = Y), foldVector)
    names(folds) <- NULL
  } else {
    kstep <- length(rnd_idx) %/% nfold
    folds <- list()
    for (i in seq_len(nfold - 1)) {
      folds[[i]] <- rnd_idx[seq_len(kstep)]
      rnd_idx <- rnd_idx[-seq_len(kstep)]
    }
    folds[[nfold]] <- rnd_idx
  }
  return(folds)
}
