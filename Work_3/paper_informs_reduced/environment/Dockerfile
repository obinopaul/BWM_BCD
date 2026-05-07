# hash:sha256:dda92073c4d354f49570764109cedeefca8cde42a494fd1b9a5245b221f0ec9b
FROM registry.codeocean.com/codeocean/r-studio:2022.07.0-548-r4.2.1-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        cmake=3.10.2-1ubuntu2.18.04.2 \
        libnlopt-dev=2.4.2+dfsg-4 \
        pkg-config=0.29.1-0ubuntu2 \
    && rm -rf /var/lib/apt/lists/*

RUN Rscript -e 'remotes::install_version("DescTools", "0.99.49")' \
    && Rscript -e 'remotes::install_version("Hmisc", "5.1-0")' \
    && Rscript -e 'remotes::install_version("MASS", "7.3-60")' \
    && Rscript -e 'remotes::install_version("Metrics", "0.1.4")' \
    && Rscript -e 'remotes::install_version("OneR", "2.2")' \
    && Rscript -e 'remotes::install_version("Rcpp", "1.0.10")' \
    && Rscript -e 'remotes::install_version("VIM", "6.2.2")' \
    && Rscript -e 'remotes::install_version("car", "3.1-2")' \
    && Rscript -e 'remotes::install_version("caret", "6.0-94")' \
    && Rscript -e 'remotes::install_version("data.table", "1.14.8")' \
    && Rscript -e 'remotes::install_version("devtools")' \
    && Rscript -e 'remotes::install_version("dplyr", "1.1.2")' \
    && Rscript -e 'remotes::install_version("e1071", "1.7-13")' \
    && Rscript -e 'remotes::install_version("earth", "5.3.2")' \
    && Rscript -e 'remotes::install_version("gbm", "2.1.8.1")' \
    && Rscript -e 'remotes::install_version("glmnet", "4.1-7")' \
    && Rscript -e 'remotes::install_version("grplasso", "0.4-7")' \
    && Rscript -e 'remotes::install_version("gtools", "3.9.4")' \
    && Rscript -e 'remotes::install_version("hot.deck", "1.2")' \
    && Rscript -e 'remotes::install_version("ipred", "0.9-14")' \
    && Rscript -e 'remotes::install_version("janitor", "2.2.0")' \
    && Rscript -e 'remotes::install_version("mice", "3.15.0")' \
    && Rscript -e 'remotes::install_version("partykit", "1.2-20")' \
    && Rscript -e 'remotes::install_version("purrr", "1.0.1")' \
    && Rscript -e 'remotes::install_version("randomForest", "4.7-1.1")' \
    && Rscript -e 'remotes::install_version("ranger", "0.15.1")' \
    && Rscript -e 'remotes::install_version("reticulate", "1.28")' \
    && Rscript -e 'remotes::install_version("rpart", "4.1.19")' \
    && Rscript -e 'remotes::install_version("scout", "1.0.4")' \
    && Rscript -e 'remotes::install_version("softImpute", "1.4-1")' \
    && Rscript -e 'remotes::install_version("tensorflow", "2.11.0")' \
    && Rscript -e 'remotes::install_version("tree", "1.0-43")'

COPY postInstall /
RUN /postInstall
