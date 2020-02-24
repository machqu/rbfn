library(data.table)
library(ggplot2)

dt = fread('output/traineval_combined/traineval.csv')

dt = dt[, .(mean = mean(rmse),
            sd = sd(rmse)),
        by = .(surface, model, num_rbfn_centroids)]

dt = dt[order(surface, model, num_rbfn_centroids)]

dt[, res := sprintf("%.4f ± %.4f", mean, sd)]

show(dt)

for (v_surface in unique(dt$surface)) {
  tmp = dt[surface == v_surface]

  xgb_mean = tmp[model == "xgb", mean]
  xgb_sd = tmp[model == "xgb", sd]

  dnn_mean = tmp[model == "dnn", mean]
  dnn_sd = tmp[model == "dnn", sd]

  tmp = tmp[model == "rbfn"]
  p = ggplot(tmp)
  p = p + theme_minimal()
  p = p + theme(text = element_text(size=10))
  p = p + geom_line(aes(num_rbfn_centroids, mean))
  p = p + geom_point(aes(num_rbfn_centroids, mean))
  p = p + geom_errorbar(aes(x=num_rbfn_centroids, ymax=mean+sd, ymin=mean-sd))

  p = p + geom_hline(yintercept=xgb_mean, linetype="solid")
  p = p + geom_hline(yintercept=xgb_mean-xgb_sd, linetype="dotted")
  p = p + geom_hline(yintercept=xgb_mean+xgb_sd, linetype="dotted")
  p = p + expand_limits(y = xgb_mean - xgb_sd)

  p = p + geom_hline(yintercept=dnn_mean, linetype="longdash")
  p = p + geom_hline(yintercept=dnn_mean-dnn_sd, linetype="dotted")
  p = p + geom_hline(yintercept=dnn_mean+dnn_sd, linetype="dotted")
  p = p + expand_limits(y = dnn_mean - dnn_sd)

  p = p + labs(x = "RBFN centroids", y = "RMSE (eV)")
  p = p + labs(title = sprintf('{%s}', v_surface))

  p = p + scale_x_continuous(limits = c(0, 4096), breaks=c(0, 2048, 4096))

  print(p)

  fn = sprintf("output/traineval_combined/plot_rbfn_xgb_%s.pdf", v_surface)
  ggsave(fn, p, width=5.3/3, height=5.3/3)
}
