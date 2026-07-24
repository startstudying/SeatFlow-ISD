# R3.5 A–F final diagnosis

Task 25 remains research-only with no selected objective and Task 26 disabled.

## Structured decision

| Category | Status | Evidence |
| --- | --- | --- |
| A_optimization_randomness | `unsupported` | L2 Top-5 Jaccard 0.563502 exceeds L0 0.410380; L2 failed the frozen net-spread gate in all six runs |
| B_top5_boundary_noise | `unsupported` | L2 Top-5 gross spread=0.00041265; L2 Top-20 gross spread=0.00268269 |
| C_regime_drift | `insufficient_evidence` | Task 25 contains descriptive market-regime slices but no frozen causal or predeclared regime gate |
| D_turnover_cost | `confirmed` | L2 original gross spread=0.00041265; L2 original cost difference=0.00062063; L2 original net spread=-0.00020798 |
| E_graph_no_increment | `supported` | L2 without Seat-Stock mean absolute score change=0.17090353; L2 without Seat-Stock Top-5 Jaccard=0.725482; L2 without Seat-Stock gross-spread delta=0.00039214; Removing Seat-Stock improved gross spread, consistent with harmful relation propagation; this is diagnostic, not a trained ablation |
| F_target_misalignment | `unsupported` | label and trade-return Oracle gross spreads are positive; T+1 open to T+2 open for horizon=1; Task 25 evaluation consumes the same frozen horizon=1 excess return |

## Next route

`D_cost_aware_seatrank`

No GRU, FiLM, seat-profile model, hypergraph, new loss, new seed, or new training checkpoint was produced.

## Objective evidence

```json
{
  "L0": {
    "cost_attribution": {
      "mean_candidate_turnover": 0.9257861635220127,
      "mean_continuation_rate": 0.07232704402515723,
      "mean_cost_difference": 0.0004830907060997716,
      "mean_gross_spread": 0.0023291970495949122,
      "mean_net_spread": 0.0018461063434951407
    },
    "label_oracle": {
      "mean_candidate_turnover": 0.9067924528301887,
      "mean_continuation_rate": 0.09132075471698113,
      "mean_cost_difference": 0.0003906705104520135,
      "mean_gross_spread": 0.15286261155672945,
      "mean_net_spread": 0.15247194104627745
    },
    "original_score": {
      "mean_candidate_turnover": 0.9257861635220127,
      "mean_continuation_rate": 0.07232704402515723,
      "mean_cost_difference": 0.0004830907060997716,
      "mean_gross_spread": 0.0023291970495949122,
      "mean_net_spread": 0.0018461063434951407
    },
    "reversed_score": {
      "mean_candidate_turnover": 0.969811320754717,
      "mean_continuation_rate": 0.028301886792452834,
      "mean_cost_difference": 0.0006861562087042882,
      "mean_gross_spread": -0.0017164695697570855,
      "mean_net_spread": -0.0024026257784613737
    },
    "topk_curve": {
      "top_1": {
        "mean_gross_spread": 0.00306766128968579,
        "mean_net_spread": 0.002364084335345485
      },
      "top_10": {
        "mean_gross_spread": 0.0025482775594524902,
        "mean_net_spread": 0.0022945037453301173
      },
      "top_10pct": {
        "mean_gross_spread": 0.0014538902371199404,
        "mean_net_spread": 0.0009992920137582163
      },
      "top_20": {
        "mean_gross_spread": 0.002992581225972773,
        "mean_net_spread": 0.0031570914108763116
      },
      "top_20pct": {
        "mean_gross_spread": 0.0027931490123155743,
        "mean_net_spread": 0.0025607626124014484
      },
      "top_3": {
        "mean_gross_spread": 0.002162080325991545,
        "mean_net_spread": 0.001576319879405221
      },
      "top_5": {
        "mean_gross_spread": 0.0023291970495949122,
        "mean_net_spread": 0.0018461063434951407
      }
    },
    "trade_return_oracle": {
      "mean_candidate_turnover": 0.9067924528301887,
      "mean_continuation_rate": 0.09132075471698113,
      "mean_cost_difference": 0.0003906705104520135,
      "mean_gross_spread": 0.15286261155672945,
      "mean_net_spread": 0.15247194104627745
    }
  },
  "L1": {
    "cost_attribution": {
      "mean_candidate_turnover": 0.9533333333333333,
      "mean_continuation_rate": 0.04477987421383648,
      "mean_cost_difference": 0.0005666718909666854,
      "mean_gross_spread": 0.0007860078296775297,
      "mean_net_spread": 0.00021933593871084462
    },
    "label_oracle": {
      "mean_candidate_turnover": 0.9067924528301887,
      "mean_continuation_rate": 0.09132075471698113,
      "mean_cost_difference": 0.0003906705104520135,
      "mean_gross_spread": 0.15286261155672945,
      "mean_net_spread": 0.15247194104627745
    },
    "original_score": {
      "mean_candidate_turnover": 0.9533333333333333,
      "mean_continuation_rate": 0.04477987421383648,
      "mean_cost_difference": 0.0005666718909666854,
      "mean_gross_spread": 0.0007860078296775297,
      "mean_net_spread": 0.00021933593871084462
    },
    "reversed_score": {
      "mean_candidate_turnover": 0.9767295597484277,
      "mean_continuation_rate": 0.021383647798742137,
      "mean_cost_difference": 0.0006977422591694706,
      "mean_gross_spread": -0.0009515912989788861,
      "mean_net_spread": -0.0016493335581483565
    },
    "topk_curve": {
      "top_1": {
        "mean_gross_spread": -0.001377099760102928,
        "mean_net_spread": -0.002108755798304423
      },
      "top_10": {
        "mean_gross_spread": 0.0019403529525372821,
        "mean_net_spread": 0.0016064604769087368
      },
      "top_10pct": {
        "mean_gross_spread": 0.0008698987731363637,
        "mean_net_spread": 0.0003291304261091571
      },
      "top_20": {
        "mean_gross_spread": 0.0036871008434365576,
        "mean_net_spread": 0.0038162533430422738
      },
      "top_20pct": {
        "mean_gross_spread": 0.0024856534608727106,
        "mean_net_spread": 0.002183829335349794
      },
      "top_3": {
        "mean_gross_spread": -0.0004982970695641364,
        "mean_net_spread": -0.0011476058672013024
      },
      "top_5": {
        "mean_gross_spread": 0.0007860078296775297,
        "mean_net_spread": 0.00021933593871084462
      }
    },
    "trade_return_oracle": {
      "mean_candidate_turnover": 0.9067924528301887,
      "mean_continuation_rate": 0.09132075471698113,
      "mean_cost_difference": 0.0003906705104520135,
      "mean_gross_spread": 0.15286261155672945,
      "mean_net_spread": 0.15247194104627745
    }
  },
  "L2": {
    "cost_attribution": {
      "mean_candidate_turnover": 0.9611320754716982,
      "mean_continuation_rate": 0.036981132075471705,
      "mean_cost_difference": 0.0006206283400849693,
      "mean_gross_spread": 0.0004126454377513112,
      "mean_net_spread": -0.0002079829023336578
    },
    "label_oracle": {
      "mean_candidate_turnover": 0.9067924528301887,
      "mean_continuation_rate": 0.09132075471698113,
      "mean_cost_difference": 0.0003906705104520135,
      "mean_gross_spread": 0.15286261155672945,
      "mean_net_spread": 0.15247194104627745
    },
    "original_score": {
      "mean_candidate_turnover": 0.9611320754716982,
      "mean_continuation_rate": 0.036981132075471705,
      "mean_cost_difference": 0.0006206283400849693,
      "mean_gross_spread": 0.0004126454377513112,
      "mean_net_spread": -0.0002079829023336578
    },
    "reversed_score": {
      "mean_candidate_turnover": 0.9690566037735849,
      "mean_continuation_rate": 0.029056603773584908,
      "mean_cost_difference": 0.0006700834322854519,
      "mean_gross_spread": 0.0005979423701960878,
      "mean_net_spread": -7.214106208936387e-05
    },
    "topk_curve": {
      "top_1": {
        "mean_gross_spread": -0.0011413883748114609,
        "mean_net_spread": -0.0018733910725309336
      },
      "top_10": {
        "mean_gross_spread": 0.0018005301560170483,
        "mean_net_spread": 0.0013693044506688503
      },
      "top_10pct": {
        "mean_gross_spread": 0.0011136175661954346,
        "mean_net_spread": 0.0005054164279797287
      },
      "top_20": {
        "mean_gross_spread": 0.0026826882010337286,
        "mean_net_spread": 0.002669136330865234
      },
      "top_20pct": {
        "mean_gross_spread": 0.0021156072188098104,
        "mean_net_spread": 0.0017080277887998785
      },
      "top_3": {
        "mean_gross_spread": -0.00015807144003151708,
        "mean_net_spread": -0.0008483785899134717
      },
      "top_5": {
        "mean_gross_spread": 0.0004126454377513112,
        "mean_net_spread": -0.0002079829023336578
      }
    },
    "trade_return_oracle": {
      "mean_candidate_turnover": 0.9067924528301887,
      "mean_continuation_rate": 0.09132075471698113,
      "mean_cost_difference": 0.0003906705104520135,
      "mean_gross_spread": 0.15286261155672945,
      "mean_net_spread": 0.15247194104627745
    }
  }
}
```
