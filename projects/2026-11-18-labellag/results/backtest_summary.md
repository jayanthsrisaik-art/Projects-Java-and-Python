seeds=[0, 1, 2, 3, 4] cutoffs=[240.0, 270.0, 300.0, 330.0] test_days=21.0 window_days=180.0 budget=0.005 txns_per_day=1200

                      avg_precision  recall_at_budget  precision_at_budget   brier  pred_rate_over_true_rate  ap_minus_naive  ap_minus_naive_se ap_beats_naive
method                                                                                                                                                        
rules                        0.0625            0.0772               0.1328     NaN                       NaN         -0.0849             0.0230           5/20
naive                        0.1474            0.1117               0.2447  0.0094                    1.0720          0.0000             0.0000              -
mature_only                  0.0734            0.0640               0.1355  0.0104                    1.2524         -0.0740             0.0137           0/20
drop_young_negatives         0.1396            0.1035               0.2297  0.0106                    2.3143         -0.0079             0.0016           2/20
ipw                          0.1570            0.1196               0.2589  0.0093                    1.2337          0.0095             0.0023          18/20
soft_label                   0.1450            0.1103               0.2423  0.0095                    1.3140         -0.0025             0.0016           6/20
oracle                       0.1755            0.1295               0.2799  0.0092                    1.2927          0.0281             0.0052          20/20

label completeness (mean over runs): training window 0.801, last 30 days 0.309
delay curve F(14d): estimated 0.310, realised 0.306
